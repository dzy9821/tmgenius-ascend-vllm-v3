"""
流式 VAD 服务 —— 基于 TEN-VAD 的每连接独立实例架构。

架构：
  - 每个 WebSocket 连接持有独立的 TenVad 实例（hop_size=640=40ms@16kHz）
  - process() 为同步调用、CPU 极轻（RTF ~0.01），通过 asyncio.to_thread 避免阻塞
  - 固定阈值断句状态机与旧版 VAD 完全一致
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Optional

import numpy as np


class Settings:
    """VAD 相关配置，通过环境变量注入。"""

    # ---- ASR 音频填充 ----
    ASR_PAD_FRAMES: int = int(os.getenv("ASR_PAD_FRAMES", "5"))

    # ---- VAD 断句阈值 ----
    VAD_HOP_SIZE: int = int(os.getenv("VAD_HOP_SIZE", "640"))
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.4"))
    VAD_PAUSE_MIN: float = float(os.getenv("VAD_PAUSE_MIN", "0.9"))
    VAD_MIN_SPEECH: float = float(os.getenv("VAD_MIN_SPEECH", "0.5"))
    VAD_MAX_SPEECH: float = float(os.getenv("VAD_MAX_SPEECH", "60.0"))


# 全局单例
settings = Settings()

logger = logging.getLogger(__name__)

# ---- 导入 TEN-VAD（本地 weights/vad/ten-vad/） ----
_vad_include = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "weights", "vad", "ten-vad", "include")
)
if _vad_include not in sys.path:
    sys.path.insert(0, _vad_include)
from ten_vad import TenVad  # noqa: E402

# ---- 固定阈值参数（从环境变量读取，VAD 实现无关） ----
PAUSE_THRESHOLD = settings.VAD_PAUSE_MIN
MIN_SPEECH_DURATION = settings.VAD_MIN_SPEECH
MAX_SPEECH_DURATION = settings.VAD_MAX_SPEECH

# ---- TEN-VAD 参数 ----
HOP_SIZE = settings.VAD_HOP_SIZE              # 640 samples = 40ms @ 16kHz
VAD_THRESHOLD = settings.VAD_THRESHOLD        # 语音概率阈值
SAMPLE_RATE = 16000


# ============================================================
# 流式 VAD 会话（每连接一个）
# ============================================================


class TenVADSession:
    """
    流式 VAD 会话 —— 每连接持有一个独立 TenVad 实例。

    逐帧接收 PCM int16 音频，通过 TenVad 获取语音概率，
    并在满足动态阈值条件时返回完整的语音片段。
    """

    def __init__(self, sid: str) -> None:
        self._sid = sid
        self._vad = TenVad(hop_size=HOP_SIZE, threshold=VAD_THRESHOLD)
        logger.info("TenVAD instance created: sid=%s, vad_id=%s", sid, id(self._vad))
        self.hop_size = HOP_SIZE
        self.frame_duration = self.hop_size / SAMPLE_RATE  # 秒

        self._pad_frames = settings.ASR_PAD_FRAMES

        # 样本缓冲（不足一帧时暂存）
        self._chunks: list[np.ndarray] = []
        self._chunk_total: int = 0

        # 滑动窗口：始终保留最近 N 帧，用于语音开始时作为前导上下文
        self._pre_buffer: list[np.ndarray] = []
        self._pre_snapshot: list[np.ndarray] = []

        # 当前语音段：pre_snapshot（前导）+ segment_frames（入段后原始流逐帧，含静默）
        self._segment_frames: list[np.ndarray] = []
        self._in_speech = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0

        # 全局采样计数
        self._total_samples: int = 0
        self._speech_start_sample: int = 0
        self._last_speech_end_sample: int = 0

        # gap window state
        self._gap_active: bool = False
        self._gap_base_silence: int = 0
        self._gap_speech: int = 0
        self._gap_buffer: list[np.ndarray] = []

    # ---- 公开属性 ----

    @property
    def in_speech(self) -> bool:
        """当前是否处于语音段中（供 progressive 调度使用）。"""
        return self._in_speech

    @property
    def speech_start_sample(self) -> int:
        """当前语音段起始采样位置（供 progressive bg/ed 计算）。"""
        return self._speech_start_sample

    # ---- 公开接口 ----

    async def feed_audio(self, pcm_int16: np.ndarray) -> list[dict]:
        """
        喂入 PCM int16 音频样本。

        Returns:
            触发的语音段列表，每项为
            {"audio": np.ndarray (int16), "start_sample": int, "end_sample": int}
        """
        self._chunks.append(pcm_int16)
        self._chunk_total += len(pcm_int16)
        segments: list[dict] = []

        while self._chunk_total >= self.hop_size:
            buffer = np.concatenate(self._chunks)
            frame = buffer[: self.hop_size]
            remainder = buffer[self.hop_size :]
            self._chunks = [remainder] if len(remainder) > 0 else []
            self._chunk_total = len(remainder)

            result = await self._process_frame(frame)
            if result is not None:
                segments.append(result)

        return segments

    def flush(self) -> Optional[dict]:
        """强制刷出剩余语音段（客户端发送 status=2 时调用）。"""
        if not self._segment_frames:
            return None

        speech_duration = self._speech_frame_count * self.frame_duration
        return self._finalize_segment(speech_duration)

    def close(self) -> None:
        """释放 TenVad 实例。"""
        if self._vad is not None:
            vad_id = id(self._vad)
            del self._vad
            self._vad = None
            logger.info("TenVAD instance released: sid=%s, vad_id=%s", self._sid, vad_id)
        else:
            logger.warning("TenVAD instance already released: sid=%s", self._sid)

    # ---- 内部逻辑 ----

    async def _process_frame(self, frame: np.ndarray) -> Optional[dict]:
        # TenVad.process 为同步调用，通过线程池执行避免阻塞事件循环
        prob, flag_i = await asyncio.to_thread(self._vad.process, frame)
        self._total_samples += self.hop_size
        flag = int(flag_i)

        # 语音开始时，先快照当前 pre_buffer 作为前导上下文（不含本帧）
        if flag == 1 and not self._in_speech:
            self._pre_snapshot = list(self._pre_buffer)

        # 维护前导帧滑动窗口（追加本帧后再限长）
        self._pre_buffer.append(frame)
        while len(self._pre_buffer) > self._pad_frames:
            self._pre_buffer.pop(0)

        if flag == 1:  # 语音
            if not self._in_speech:
                # 新语音段开始
                self._in_speech = True
                self._speech_frame_count = 0
                self._silence_frame_count = 0
                self._segment_frames = []
                self._gap_active = False
                self._gap_base_silence = 0
                self._gap_speech = 0
                self._gap_buffer.clear()
                self._speech_start_sample = (
                    self._total_samples - self.hop_size
                    - len(self._pre_snapshot) * self.hop_size
                )

            # 静音后出现语音 → 进入 gap 窗口
            if self._silence_frame_count > 0:
                if not self._gap_active:
                    self._gap_active = True
                    self._gap_base_silence = self._silence_frame_count
                self._gap_speech = 0
                self._gap_buffer.clear()

            if self._gap_active:
                # gap 窗口内：帧进 _gap_buffer，不进 _segment_frames
                self._gap_speech += 1
                self._gap_buffer.append(frame)
            else:
                # 正常语音
                self._speech_frame_count += 1
                self._segment_frames.append(frame)
                self._last_speech_end_sample = self._total_samples
                self._silence_frame_count = 0
        else:  # 静默
            if self._in_speech:
                # gap burst 结束判断
                if self._gap_active and self._gap_speech > 0:
                    gap_speech_dur = self._gap_speech * self.frame_duration
                    if gap_speech_dur < MIN_SPEECH_DURATION:
                        # 合并：gap 语音算作静音，gap_buffer 丢弃
                        self._silence_frame_count = (
                            self._gap_base_silence + self._gap_speech
                        )
                    else:
                        # 真实语音：flush gap_buffer 进 _segment_frames
                        self._segment_frames.extend(self._gap_buffer)
                        self._speech_frame_count += self._gap_speech
                        self._last_speech_end_sample = (
                            self._total_samples - self.hop_size
                        )
                        self._gap_active = False
                        self._gap_base_silence = 0
                        self._silence_frame_count = 0
                    self._gap_speech = 0
                    self._gap_buffer.clear()

                self._segment_frames.append(frame)
                self._silence_frame_count += 1

                speech_dur = self._speech_frame_count * self.frame_duration
                pause_dur = self._silence_frame_count * self.frame_duration

                if _should_cut_segment(speech_dur, pause_dur):
                    return self._finalize_segment(speech_dur)

        # 强制触发：语音过长（含 gap buffer）
        if self._in_speech:
            total_frames = len(self._segment_frames) + len(self._gap_buffer)
            total_dur = total_frames * self.frame_duration
            if total_dur > MAX_SPEECH_DURATION:
                if self._gap_active and self._gap_buffer:
                    self._segment_frames.extend(self._gap_buffer)
                    self._speech_frame_count += self._gap_speech
                    self._last_speech_end_sample = (
                        self._total_samples - self.hop_size
                    )
                    self._gap_active = False
                    self._gap_speech = 0
                    self._gap_buffer.clear()
                speech_dur = self._speech_frame_count * self.frame_duration
                return self._finalize_segment(speech_dur)

        return None

    def _compute_segment_end(self) -> int:
        """段尾 = 最后语音帧结束 + 后置 pad（不超过已缓冲的原始流）。"""
        pad_samples = self._pad_frames * self.hop_size
        target_end = self._last_speech_end_sample + pad_samples
        buffered_end = self._speech_start_sample + (
            len(self._pre_snapshot) + len(self._segment_frames)
        ) * self.hop_size
        return min(target_end, buffered_end)

    def _finalize_segment(self, speech_duration: float) -> Optional[dict]:
        """切分当前段；有效语音不足 MIN_SPEECH 时丢弃。"""
        seg = self._extract_and_reset()
        if speech_duration < MIN_SPEECH_DURATION:
            logger.debug(
                "VAD segment discarded: sid=%s, speech=%.0fms < min=%.0fms",
                self._sid,
                speech_duration * 1000,
                MIN_SPEECH_DURATION * 1000,
            )
            return None
        return seg

    def _extract_and_reset(self) -> dict:
        start = self._speech_start_sample
        end = self._compute_segment_end()
        all_frames = self._pre_snapshot + self._segment_frames
        audio = np.concatenate(all_frames)
        num_samples = end - start
        if len(audio) > num_samples:
            audio = audio[:num_samples]
        self._reset()
        return {"audio": audio, "start_sample": start, "end_sample": end}

    def _reset(self) -> None:
        self._segment_frames = []
        self._pre_snapshot = []
        self._in_speech = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._last_speech_end_sample = 0
        self._gap_active = False
        self._gap_base_silence = 0
        self._gap_speech = 0
        self._gap_buffer.clear()


# ---- 固定阈值判定（独立函数，方便单元测试） ----


def _pause_threshold(speech_duration: float) -> float:  # noqa: ARG001
    """返回触发切分所需的固定停顿阈值（秒）。"""
    return PAUSE_THRESHOLD


def _should_cut_segment(speech_duration: float, pause_duration: float) -> bool:
    """
    判断是否应切分当前语音段（与有效语音是否够长无关）。

    切分规则：
      - speech >= MAX_SPEECH → 立即切分（强制上限）
      - 停顿 >= PAUSE_THRESHOLD → 切分
    """
    if speech_duration >= MAX_SPEECH_DURATION:
        return True
    return pause_duration >= PAUSE_THRESHOLD


def _should_transcribe(speech_duration: float, pause_duration: float) -> bool:
    """兼容旧调用：满足切分条件且有效语音达到 MIN_SPEECH 才转发 ASR。"""
    if speech_duration < MIN_SPEECH_DURATION:
        return False
    return _should_cut_segment(speech_duration, pause_duration)


# ============================================================
# 测试 Demo
# ============================================================

import wave
import time


def _load_wav(path: str) -> np.ndarray:
    """读取 16kHz mono int16 WAV 文件，返回 int16 numpy 数组。"""
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "仅支持单声道"
        assert wf.getsampwidth() == 2, "仅支持 16-bit"
        assert wf.getframerate() == SAMPLE_RATE, f"仅支持 {SAMPLE_RATE}Hz"
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    return np.frombuffer(raw, dtype=np.int16).copy()


async def _run_demo(wav_path: str) -> None:
    """运行 VAD 测试 Demo。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 加载音频
    audio = _load_wav(wav_path)
    total_dur = len(audio) / SAMPLE_RATE

    print("=" * 60)
    print("TEN-VAD 流式服务测试 Demo")
    print("=" * 60)
    print(f"  音频文件:    {os.path.basename(wav_path)}")
    print(f"  采样率:      {SAMPLE_RATE} Hz")
    print(f"  总时长:      {total_dur:.1f}s")
    print(f"  帧大小:      {HOP_SIZE} samples ({HOP_SIZE / SAMPLE_RATE * 1000:.0f} ms)")
    print(f"  VAD 阈值:    {VAD_THRESHOLD}")
    print(f"  停顿阈值:    {PAUSE_THRESHOLD:.2f}s (固定)")
    print(f"  最小语音:    {MIN_SPEECH_DURATION:.2f}s")
    print(f"  最大语音:    {MAX_SPEECH_DURATION:.2f}s")
    print()

    # 创建 VAD 会话
    session = TenVADSession(sid="demo-001")
    segments: list[dict] = []

    # 分块喂入（模拟流式，每次喂 ~100ms）
    chunk_size = int(0.1 * SAMPLE_RATE)  # 100ms
    total_fed = 0
    t_start = time.perf_counter()
    last_report = 0.0

    while total_fed < len(audio):
        chunk = audio[total_fed : total_fed + chunk_size]
        total_fed += len(chunk)
        results = await session.feed_audio(chunk)
        for seg in results:
            segments.append(seg)
            seg_dur = len(seg["audio"]) / SAMPLE_RATE
            seg_start_s = seg["start_sample"] / SAMPLE_RATE
            seg_end_s = seg["end_sample"] / SAMPLE_RATE
            print(
                f"  → 语音段 #{len(segments)}: "
                f"[{seg_start_s:.2f}s .. {seg_end_s:.2f}s], "
                f"时长={seg_dur:.2f}s"
            )

        # 进度显示（每秒更新一次）
        progress = total_fed / len(audio) * 100
        if progress - last_report >= 5.0:
            elapsed = time.perf_counter() - t_start
            rtf = elapsed / (total_fed / SAMPLE_RATE) if total_fed > 0 else 0
            print(f"  进度: {progress:.0f}% ({total_fed / SAMPLE_RATE:.1f}s/{total_dur:.1f}s), "
                  f"RTF={rtf:.4f}")
            last_report = progress

    # 刷出尾部
    tail = session.flush()
    if tail is not None:
        segments.append(tail)
        seg_dur = len(tail["audio"]) / SAMPLE_RATE
        seg_start_s = tail["start_sample"] / SAMPLE_RATE
        seg_end_s = tail["end_sample"] / SAMPLE_RATE
        print(
            f"  → flush 语音段 #{len(segments)}: "
            f"[{seg_start_s:.2f}s .. {seg_end_s:.2f}s], "
            f"时长={seg_dur:.2f}s"
        )

    session.close()
    elapsed = time.perf_counter() - t_start
    print()
    print(f"处理完成，总耗时 {elapsed:.2f}s，RTF={elapsed / total_dur:.4f}")
    print(f"共检测到 {len(segments)} 个语音段：")
    print("-" * 50)
    for i, seg in enumerate(segments):
        seg_dur = len(seg["audio"]) / SAMPLE_RATE
        seg_start_s = seg["start_sample"] / SAMPLE_RATE
        seg_end_s = seg["end_sample"] / SAMPLE_RATE
        print(f"  段 {i + 1:2d}: "
              f"[{seg_start_s:7.2f}s .. {seg_end_s:7.2f}s] "
              f"时长 {seg_dur:5.2f}s "
              f"({len(seg['audio'])} samples)")
    print("-" * 50)
    total_speech = sum(len(s["audio"]) for s in segments) / SAMPLE_RATE
    print(f"  语音总时长: {total_speech:.1f}s / {total_dur:.1f}s "
          f"({total_speech / total_dur * 100:.1f}%)")
    print()
    print("Demo 完成。")


if __name__ == "__main__":
    wav_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "120报警电话16k.wav",
    )
    asyncio.run(_run_demo(wav_path))
