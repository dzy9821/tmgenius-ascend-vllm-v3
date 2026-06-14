from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from fastapi import WebSocket

from src.config import get_settings
from src.core.logging import trace_id_var
from src.services.asr_service import get_online_client, get_offline_client
from src.services.vad_service import TenVADSession

logger = logging.getLogger(__name__)

FINALIZE_SENTINEL = object()
ONLINE_TRIGGER_SAMPLES: int = get_settings().online_trigger_ms * 16  # 400ms * 16 = 6400

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
        user_hotwords: str = payload.get("hotwords", "") or ""
        hotwords: str = user_hotwords if user_hotwords else settings.hotwords

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
            audio = np.concatenate(fs.online_buffer) if fs.online_buffer else tail["audio"]
            _do_trigger_offline(
                audio, tail["start_sample"], tail["end_sample"],
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
    audio_b64: str = payload.get("audio", {}).get("audio", "")
    if not audio_b64:
        return

    pcm_bytes = base64.b64decode(audio_b64)
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
    if len(pcm) == 0:
        return

    if rn is not None and rn_proc is not None:
        pcm = await rn.denoise(rn_proc, pcm)

    logger.debug("audio frame: samples=%d online_total=%d", len(pcm), fs.online_total)
    fs.online_buffer.append(pcm)
    fs.online_total += len(pcm)

    segs = await vad.feed_audio(pcm)

    for seg in segs:
        audio = np.concatenate(fs.online_buffer) if fs.online_buffer else seg["audio"]
        _do_trigger_offline(
            audio, seg["start_sample"], seg["end_sample"],
            fs, hotwords, result_queue, reorder, all_tasks,
        )

    _maybe_trigger_online(fs, hotwords, result_queue, all_tasks)


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

    logger.debug(
        "offline trigger: seg_id=%d bg=%d ed=%d is_final=%s",
        fs.seg_id, start_sample // 16, end_sample // 16, is_final,
    )
    task = asyncio.create_task(
        _do_offline_asr(audio, fs.seg_id, hotwords, result_queue, reorder)
    )
    all_tasks.append(task)

    fs.online_buffer.clear()
    fs.online_total = 0
    fs.online_last_trigger = 0
    fs.online_epoch += 1
    fs.online_busy = False
    fs.seg_id += 1


def _maybe_trigger_online(
    fs: FrameState,
    hotwords: str,
    result_queue: asyncio.Queue,
    all_tasks: list,
) -> None:
    if fs.online_busy:
        return
    if (fs.online_total - fs.online_last_trigger) < ONLINE_TRIGGER_SAMPLES:
        return

    epoch_snap = fs.online_epoch
    audio_snap = np.concatenate(fs.online_buffer).copy()
    seg_id_snap = fs.seg_id
    fs.online_busy = True
    fs.online_last_trigger = fs.online_total

    logger.debug("online trigger: seg_id=%d samples=%d epoch=%d", seg_id_snap, len(audio_snap), epoch_snap)

    task = asyncio.create_task(
        _do_online_asr(audio_snap, seg_id_snap, epoch_snap, hotwords, result_queue, fs)
    )
    all_tasks.append(task)


# ── ASR coroutines ─────────────────────────────────────────────

async def _do_online_asr(
    audio: np.ndarray,
    seg_id_snap: int,
    epoch_snap: int,
    hotwords: str,
    result_queue: asyncio.Queue,
    fs: FrameState,
) -> None:
    try:
        text = await get_online_client().transcribe(audio, hotwords="")
        if fs.online_epoch != epoch_snap:
            logger.debug("online stale: seg=%d epoch=%d current=%d", seg_id_snap, epoch_snap, fs.online_epoch)
            return
        if text:
            logger.debug("online result: seg=%d epoch=%d text=%s", seg_id_snap, epoch_snap, text)
            await result_queue.put(QueueMsg(seg_id_snap, "Progressive", text, 0, 0))
    finally:
        if fs.online_epoch == epoch_snap:
            fs.online_busy = False


async def _do_offline_asr(
    audio: np.ndarray,
    seg_id: int,
    hotwords: str,
    result_queue: asyncio.Queue,
    reorder: ReorderState,
) -> None:
    try:
        text = await get_offline_client().transcribe(audio, hotwords=hotwords)
        if text and _itn_service is not None:
            try:
                text = await _itn_service.process(text)
            except Exception as exc:
                logger.warning("ITN failed seg_id=%d: %s", seg_id, exc)
        logger.debug("offline result: seg=%d text=%s", seg_id, text or "")
        await _advance_reorder_pointer(seg_id, text or None, result_queue, reorder)
    except Exception as exc:
        logger.exception("Offline ASR error seg_id=%d: %s", seg_id, exc)
        await _advance_reorder_pointer(seg_id, None, result_queue, reorder)


async def _advance_reorder_pointer(
    seg_id: int,
    text: Optional[str],
    result_queue: asyncio.Queue,
    reorder: ReorderState,
) -> None:
    async with reorder.lock:
        reorder.pending[seg_id] = text
        logger.debug("reorder: recv seg=%d text=%s next=%d final_seg=%s",
                     seg_id, text or "", reorder.next_seg_id_to_send, reorder.final_seg_id)
        while reorder.next_seg_id_to_send in reorder.pending:
            t = reorder.pending.pop(reorder.next_seg_id_to_send)
            if t:
                is_final = reorder.next_seg_id_to_send == reorder.final_seg_id
                logger.debug("reorder: emit seg=%d next=%d final=%s",
                             reorder.next_seg_id_to_send, reorder.next_seg_id_to_send + 1, is_final)
                await result_queue.put(
                    QueueMsg(
                        reorder.next_seg_id_to_send, "sentence", t, 0, 0,
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
                await _send_msg(websocket, last_sent, status=2, sid=sid, trace_id=trace_id)
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
                "ws": [{"cw": [{"w": msg.text}]}] if msg.text else [],
            }
        },
    }
    json_str = json.dumps(data, ensure_ascii=False)
    logger.debug("send: status=%d seg=%d type=%s text=%s", status, msg.seg_id, msg.msgtype, msg.text)
    try:
        await websocket.send_text(json_str)
    except Exception as exc:
        logger.debug("send_msg failed: %s", exc)
