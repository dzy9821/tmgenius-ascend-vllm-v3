from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from fastapi import WebSocket

from src.config import get_settings
from src.core.logging import trace_id_var
from src.core.opus_decoder import OpusDecoder
from src.services.asr_service import get_online_client, get_offline_client, strip_trailing_punct
from src.services.license_plate import normalize_license_plates
from src.services.vad_service import TenVADSession

logger = logging.getLogger(__name__)

FINALIZE_SENTINEL = object()
ONLINE_TRIGGER_SAMPLES: int = get_settings().online_trigger_ms * 16  # 400ms * 16 = 6400
ONLINE_VAD_PAUSE_SEC: float = get_settings().online_vad_pause_ms / 1000.0  # 500ms = 0.5s
ONLINE_MAX_SPEECH_SAMPLES: int = get_settings().online_max_speech_ms * 16

_itn_service: Any = None
_rnnoise_service_ref: Any = None


def set_itn_service(service: Any) -> None:
    global _itn_service
    _itn_service = service


def set_rnnoise_service(service: Any) -> None:
    global _rnnoise_service_ref
    _rnnoise_service_ref = service


# ── data structures ────────────────────────────────────────────

@dataclass
class QueueMsg:
    seg_id: int
    msgtype: str
    text: str
    bg: int
    ed: int
    final: bool = False


@dataclass
class FrameState:
    online_buffer: list = field(default_factory=list)
    online_total: int = 0
    online_last_trigger: int = 0
    online_epoch: int = 0
    online_busy: bool = False
    seg_id: int = 0
    # Online VAD 子分段：cursor 之后的音频才送入在线 ASR
    online_cut_cursor: int = 0
    # 与 online_buffer 一一对应的 VAD flag，1=语音 0=静音
    online_speech_flags: list = field(default_factory=list)
    # 当前在线子分段内 flag=1 的累计采样数，用于 10s 时长切
    online_speech_samples: int = 0
    online_last_text: str = ""
    online_accumulated_text: str = ""
    # 绝对样本时钟（自连接开始累计），用于计算 bg/ed
    abs_samples: int = 0
    seg_start_abs: int = 0
    # 每连接独立的 Opus 解码器（仅当 encoding=="opus" 时惰性创建）
    opus_decoder: Optional[OpusDecoder] = None


def _merge_hotwords(default: str, user: str) -> str:
    """按 , 或 | 切分默认热词与客户端热词，去重合并（保留顺序），以逗号连接。"""
    seen: list[str] = []
    for part in re.split(r"[,|]", f"{default},{user}"):
        w = part.strip()
        if w and w not in seen:
            seen.append(w)
    return ",".join(seen)


@dataclass
class ReorderState:
    next_seg_id_to_send: int = 0
    pending: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    final_seg_id: Optional[int] = None


# ── main WebSocket handler ─────────────────────────────────────

async def handle_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    sm = websocket.app.state.session_manager
    if not sm.acquire():
        await websocket.close(code=1013)
        return

    sid = str(uuid.uuid4())
    fs = FrameState()
    reorder = ReorderState()
    result_queue: asyncio.Queue = asyncio.Queue()
    all_tasks: list = []
    vad = TenVADSession(sid)

    rn = _rnnoise_service_ref
    rn_proc = rn.new_processor() if rn is not None else None

    sender_task: Optional[asyncio.Task] = None
    settings = get_settings()

    try:
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=settings.handshake_timeout,
            )
        except asyncio.TimeoutError:
            await websocket.close(code=1008)
            return

        frame = json.loads(raw)
        header = frame.get("header", {})
        trace_id: str = header.get("traceId", sid[:8])
        trace_id_var.set(trace_id)
        status: int = header.get("status", 0)

        payload = frame.get("payload", {})
        user_hotwords: str = payload.get("text", {}).get("text", "") or ""
        hotwords: str = _merge_hotwords(settings.hotwords, user_hotwords)

        sender_task = asyncio.create_task(
            _result_sender(websocket, result_queue, sid, trace_id)
        )

        if status != 2:
            await _process_audio_frame(
                frame, fs, vad, rn, rn_proc, hotwords, result_queue, all_tasks, reorder
            )

        if status != 2:
            async for raw_msg in websocket.iter_text():
                frame = json.loads(raw_msg)
                status = frame.get("header", {}).get("status", 1)
                if status == 2:
                    break
                await _process_audio_frame(
                    frame, fs, vad, rn, rn_proc, hotwords, result_queue, all_tasks, reorder
                )

        # End of stream: flush VAD and create tail offline tasks
        tail = vad.flush()
        if tail:
            _do_trigger_offline(
                tail["audio"], tail["start_sample"], tail["end_sample"],
                fs, hotwords, result_queue, reorder, all_tasks,
                is_final=True,
            )
        elif fs.online_buffer:
            audio = np.concatenate(fs.online_buffer)
            _do_trigger_offline(
                audio, 0, len(audio), fs, hotwords, result_queue, reorder, all_tasks,
                is_final=True,
            )

        await asyncio.gather(*all_tasks, return_exceptions=True)
        await result_queue.put(FINALIZE_SENTINEL)
        await sender_task

    except Exception as exc:
        logger.exception("WebSocket handler error sid=%s: %s", sid, exc)
    finally:
        if sender_task is not None and not sender_task.done():
            sender_task.cancel()
            try:
                await sender_task
            except (asyncio.CancelledError, Exception):
                pass
        vad.close()
        if fs.opus_decoder is not None:
            fs.opus_decoder.close()
        sm.release()


# ── audio frame processing ─────────────────────────────────────

async def _process_audio_frame(
    frame: dict,
    fs: FrameState,
    vad: Any,
    rn: Any,
    rn_proc: Any,
    hotwords: str,
    result_queue: asyncio.Queue,
    all_tasks: list,
    reorder: ReorderState,
) -> None:
    payload = frame.get("payload", {})
    audio_obj = payload.get("audio", {})
    audio_b64: str = audio_obj.get("audio", "")
    if not audio_b64:
        return

    raw_bytes = base64.b64decode(audio_b64)
    encoding = (audio_obj.get("encoding") or "").lower()
    if encoding == "opus":
        if fs.opus_decoder is None:
            fs.opus_decoder = OpusDecoder(sample_rate=16000, channels=1)
        pcm = fs.opus_decoder.decode(raw_bytes)
    else:
        pcm = np.frombuffer(raw_bytes, dtype=np.int16).copy()
    if len(pcm) == 0:
        return

    if rn is not None and rn_proc is not None:
        pcm = await rn.denoise(rn_proc, pcm)

    logger.debug("audio: %dms total=%dms", len(pcm) * 1000 // 16000, fs.online_total * 1000 // 16000)
    fs.online_buffer.append(pcm)
    fs.online_total += len(pcm)
    fs.abs_samples += len(pcm)

    segs, frame_flags = await vad.feed_audio(pcm)
    fs.online_speech_flags.extend(frame_flags)
    for flag in frame_flags:
        if flag == 1:
            fs.online_speech_samples += vad.hop_size

    for seg in segs:
        _do_trigger_offline(
            seg["audio"], seg["start_sample"], seg["end_sample"],
            fs, hotwords, result_queue, reorder, all_tasks,
        )

    # Online VAD 子分段：静音 ≥ 0.5s 时推进 cursor，后续在线推理只用新音频
    triggered_cut = False
    if vad.in_speech and vad.silence_duration >= ONLINE_VAD_PAUSE_SEC:
        silence_samples = int(vad.silence_duration * 16000)
        if fs.online_total - fs.online_cut_cursor > silence_samples:
            logger.debug("online vad cut (silence): cursor=%d -> %d epoch=%d",
                         fs.online_cut_cursor, fs.online_total, fs.online_epoch)
            if fs.online_last_text:
                sep = "，" if fs.online_last_text.rstrip()[-1:] not in "，。！？、；：,.!?;:" else ""
                fs.online_accumulated_text = fs.online_accumulated_text + fs.online_last_text + sep
            fs.online_cut_cursor = fs.online_total
            fs.online_epoch += 1
            fs.online_last_trigger = fs.online_total
            fs.online_speech_samples = 0
            triggered_cut = True

    # 有效语音 ≥ 10s 强制在线切分
    if not triggered_cut and fs.online_speech_samples >= ONLINE_MAX_SPEECH_SAMPLES:
        logger.debug("online vad cut (duration): samples=%d epoch=%d",
                     fs.online_speech_samples, fs.online_epoch)
        if fs.online_last_text:
            sep = "，" if fs.online_last_text.rstrip()[-1:] not in "，。！？、；：,.!?;:" else ""
            fs.online_accumulated_text = fs.online_accumulated_text + fs.online_last_text + sep
        fs.online_cut_cursor = fs.online_total
        fs.online_epoch += 1
        fs.online_last_trigger = fs.online_total
        fs.online_speech_samples = 0

    _maybe_trigger_online(fs, hotwords, result_queue, all_tasks, vad)


def _do_trigger_offline(
    audio: np.ndarray,
    start_sample: int,
    end_sample: int,
    fs: FrameState,
    hotwords: str,
    result_queue: asyncio.Queue,
    reorder: ReorderState,
    all_tasks: list,
    is_final: bool = False,
) -> None:
    if is_final:
        reorder.final_seg_id = fs.seg_id

    bg = start_sample // 16
    ed = end_sample // 16
    logger.debug(
        "offline trigger: seg_id=%d bg=%d ed=%d is_final=%s",
        fs.seg_id, bg, ed, is_final,
    )
    task = asyncio.create_task(
        _do_offline_asr(audio, fs.seg_id, bg, ed, hotwords, result_queue, reorder)
    )
    all_tasks.append(task)

    fs.online_buffer.clear()
    fs.online_speech_flags.clear()
    fs.online_speech_samples = 0
    fs.online_total = 0
    fs.online_last_trigger = 0
    fs.online_epoch += 1
    fs.online_busy = False
    fs.online_cut_cursor = 0
    fs.online_last_text = ""
    fs.online_accumulated_text = ""
    fs.seg_id += 1
    fs.seg_start_abs = fs.abs_samples


def _maybe_trigger_online(
    fs: FrameState,
    hotwords: str,
    result_queue: asyncio.Queue,
    all_tasks: list,
    vad: Any,
) -> None:
    if fs.online_busy:
        return
    if (fs.online_total - fs.online_last_trigger) < ONLINE_TRIGGER_SAMPLES:
        return

    epoch_snap = fs.online_epoch
    cursor_idx = fs.online_cut_cursor // vad.hop_size
    speech_frames = [
        fs.online_buffer[i]
        for i in range(cursor_idx, len(fs.online_buffer))
        if i < len(fs.online_speech_flags) and fs.online_speech_flags[i] == 1
    ]
    if not speech_frames:
        return
    audio_snap = np.concatenate(speech_frames)
    seg_id_snap = fs.seg_id
    bg_snap = fs.seg_start_abs // 16
    ed_snap = fs.abs_samples // 16
    fs.online_busy = True
    fs.online_last_trigger = fs.online_total

    logger.debug("online trigger: seg_id=%d samples=%d epoch=%d", seg_id_snap, len(audio_snap), epoch_snap)

    task = asyncio.create_task(
        _do_online_asr(audio_snap, seg_id_snap, epoch_snap, bg_snap, ed_snap, hotwords, result_queue, fs)
    )
    all_tasks.append(task)


# ── ASR coroutines ─────────────────────────────────────────────

async def _do_online_asr(
    audio: np.ndarray,
    seg_id_snap: int,
    epoch_snap: int,
    bg: int,
    ed: int,
    hotwords: str,
    result_queue: asyncio.Queue,
    fs: FrameState,
) -> None:
    try:
        text = await get_online_client().transcribe(audio, hotwords="")
        if fs.online_epoch != epoch_snap:
            # 过期结果：VAD cut（同 segment）的文本需累加；离线触发（跨 segment）则丢弃
            if text:
                text = strip_trailing_punct(text)
                if text and fs.seg_id == seg_id_snap:
                    fs.online_last_text = text
                    candidate = fs.online_accumulated_text + text
                    if candidate.count("，") > get_settings().online_comma_limit:
                        logger.debug("online comma limit (stale): %d commas, discarding",
                                     candidate.count("，"))
                        fs.online_accumulated_text = ""
                        fs.online_last_text = ""
                    else:
                        sep = "，" if text[-1] not in "，。！？、；：,.!?;:" else ""
                        fs.online_accumulated_text = candidate + sep
            fs.online_busy = False
            logger.debug("online stale: seg=%d epoch=%d current=%d", seg_id_snap, epoch_snap, fs.online_epoch)
            return
        text = strip_trailing_punct(text)
        if text:
            full_text = (fs.online_accumulated_text + text) if fs.online_accumulated_text else text
            if full_text.count("，") > get_settings().online_comma_limit:
                logger.debug("online comma limit: %d commas, discarding full_text",
                             full_text.count("，"))
                fs.online_accumulated_text = ""
                fs.online_last_text = ""
            else:
                fs.online_last_text = text
                logger.debug("online result: seg=%d epoch=%d text=%s", seg_id_snap, epoch_snap, full_text)
                await result_queue.put(QueueMsg(seg_id_snap, "Progressive", full_text, bg, ed))
    finally:
        if fs.online_epoch == epoch_snap:
            fs.online_busy = False


async def _do_offline_asr(
    audio: np.ndarray,
    seg_id: int,
    bg: int,
    ed: int,
    hotwords: str,
    result_queue: asyncio.Queue,
    reorder: ReorderState,
) -> None:
    try:
        text = await get_offline_client().transcribe(audio, hotwords=hotwords, skip_length_check=True)
        if text and _itn_service is not None:
            try:
                text = await _itn_service.process(text)
            except Exception as exc:
                logger.warning("ITN failed seg_id=%d: %s", seg_id, exc)
            text = normalize_license_plates(text)
        logger.debug("offline result: seg=%d text=%s", seg_id, text or "")
        await _advance_reorder_pointer(seg_id, text or None, bg, ed, result_queue, reorder)
    except Exception as exc:
        logger.exception("Offline ASR error seg_id=%d: %s", seg_id, exc)
        await _advance_reorder_pointer(seg_id, None, bg, ed, result_queue, reorder)


async def _advance_reorder_pointer(
    seg_id: int,
    text: Optional[str],
    bg: int,
    ed: int,
    result_queue: asyncio.Queue,
    reorder: ReorderState,
) -> None:
    async with reorder.lock:
        reorder.pending[seg_id] = (text, bg, ed) if text else None
        logger.debug("reorder: recv seg=%d text=%s next=%d final_seg=%s",
                     seg_id, text or "", reorder.next_seg_id_to_send, reorder.final_seg_id)
        while reorder.next_seg_id_to_send in reorder.pending:
            entry = reorder.pending.pop(reorder.next_seg_id_to_send)
            if entry:
                t, t_bg, t_ed = entry
                is_final = reorder.next_seg_id_to_send == reorder.final_seg_id
                logger.debug("reorder: emit seg=%d next=%d final=%s",
                             reorder.next_seg_id_to_send, reorder.next_seg_id_to_send + 1, is_final)
                await result_queue.put(
                    QueueMsg(
                        reorder.next_seg_id_to_send, "sentence", t, t_bg, t_ed,
                        final=is_final,
                    )
                )
            reorder.next_seg_id_to_send += 1


# ── result sender ──────────────────────────────────────────────

async def _result_sender(
    websocket: WebSocket,
    result_queue: asyncio.Queue,
    sid: str,
    trace_id: str,
) -> None:
    last_sent: Optional[QueueMsg] = None
    sent_final = False
    while True:
        item = await result_queue.get()
        if item is FINALIZE_SENTINEL:
            logger.debug("sender finalize: sent_final=%s last_sent_seg=%s",
                         sent_final, last_sent.seg_id if last_sent else None)
            if not sent_final and last_sent is not None:
                # 尾段无识别文本（静音/被过滤）时的兜底：仅补发一个空 payload 的
                # 终态帧（status=2，ws=[]）作为结束信号，沿用最后一句的 segId。
                # 不重发最后一句内容，避免同一句被推送两次（status=1 后又 status=2）。
                term = QueueMsg(last_sent.seg_id, last_sent.msgtype, "",
                                last_sent.bg, last_sent.ed, final=True)
                await _send_msg(websocket, term, status=2, sid=sid, trace_id=trace_id)
            break
        status = 2 if item.final else (0 if last_sent is None else 1)
        await _send_msg(websocket, item, status=status, sid=sid, trace_id=trace_id)
        last_sent = item
        sent_final = sent_final or item.final
    try:
        await websocket.close()
    except Exception:
        pass


async def _send_msg(
    websocket: WebSocket,
    msg: QueueMsg,
    status: int,
    sid: str,
    trace_id: str,
) -> None:
    data = {
        "header": {
            "status": status,
            "sid": sid,
            "traceId": trace_id,
        },
        "payload": {
            "result": {
                "msgtype": msg.msgtype,
                "segId": msg.seg_id,
                "bg": msg.bg,
                "ed": msg.ed,
                "ws": [{"cw": [{"w": msg.text, "rl": 0}]}] if msg.text else [],
            }
        },
    }
    json_str = json.dumps(data, ensure_ascii=False)
    logger.debug("send: status=%d seg=%d type=%s text=%s", status, msg.seg_id, msg.msgtype, msg.text)
    try:
        await websocket.send_text(json_str)
    except Exception as exc:
        logger.debug("send_msg failed: %s", exc)
