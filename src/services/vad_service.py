from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Optional

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

_vad_include = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "weights", "vad", "ten-vad", "include")
)
if _vad_include not in sys.path:
    sys.path.insert(0, _vad_include)
from ten_vad import TenVad  # noqa: E402

HOP_SIZE: int = _settings.vad_hop_size
VAD_THRESHOLD: float = _settings.vad_threshold
PAUSE_THRESHOLD: float = _settings.vad_min_speech
MAX_SPEECH_DURATION: float = _settings.vad_max_speech
MIN_SPEECH_DURATION: float = 0.5
SAMPLE_RATE = 16000


class TenVADSession:
    def __init__(self, sid: str) -> None:
        self._sid = sid
        self._vad = TenVad(hop_size=HOP_SIZE, threshold=VAD_THRESHOLD)
        logger.info("TenVAD instance created: sid=%s, vad_id=%s", sid, id(self._vad))
        self.hop_size = HOP_SIZE
        self.frame_duration = self.hop_size / SAMPLE_RATE

        self._pad_frames = _settings.asr_pad_frames

        self._chunks: list[np.ndarray] = []
        self._chunk_total: int = 0

        self._pre_buffer: list[np.ndarray] = []
        self._pre_snapshot: list[np.ndarray] = []

        self._segment_frames: list[np.ndarray] = []
        self._in_speech = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0

        self._total_samples: int = 0
        self._speech_start_sample: int = 0
        self._last_speech_end_sample: int = 0

        # gap window state
        self._gap_active: bool = False
        self._gap_base_silence: int = 0
        self._gap_speech: int = 0
        self._gap_buffer: list[np.ndarray] = []
        self._merged_frames: int = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def speech_start_sample(self) -> int:
        return self._speech_start_sample

    async def feed_audio(self, pcm_int16: np.ndarray) -> list[dict]:
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
        if not self._segment_frames:
            return None
        speech_duration = self._speech_frame_count * self.frame_duration
        logger.debug(
            "vad cut: flush speech=%.0fms silence=%.0fms(%df: orig=%d + merged=%d)",
            speech_duration * 1000,
            self._silence_frame_count * self.frame_duration * 1000,
            self._silence_frame_count,
            self._silence_frame_count - self._merged_frames,
            self._merged_frames,
        )
        return self._finalize_segment(speech_duration)

    def close(self) -> None:
        if self._vad is not None:
            vad_id = id(self._vad)
            del self._vad
            self._vad = None
            logger.info("TenVAD instance released: sid=%s, vad_id=%s", self._sid, vad_id)
        else:
            logger.warning("TenVAD instance already released: sid=%s", self._sid)

    async def _process_frame(self, frame: np.ndarray) -> Optional[dict]:
        prob, flag_i = await asyncio.to_thread(self._vad.process, frame)
        self._total_samples += self.hop_size
        flag = int(flag_i)

        if flag == 1 and not self._in_speech:
            self._pre_snapshot = list(self._pre_buffer)

        self._pre_buffer.append(frame)
        while len(self._pre_buffer) > self._pad_frames:
            self._pre_buffer.pop(0)

        if flag == 1:
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
        else:
            if self._in_speech:
                # gap burst 结束判断
                if self._gap_active and self._gap_speech > 0:
                    gap_speech_dur = self._gap_speech * self.frame_duration
                    if gap_speech_dur < MIN_SPEECH_DURATION:
                        # 合并：gap 语音算作静音，gap_buffer 丢弃
                        logger.debug(
                            "vad gap merge: dur=%.0fms < min=%.0fms "
                            "base_silence=%df speech=%df",
                            gap_speech_dur * 1000, MIN_SPEECH_DURATION * 1000,
                            self._gap_base_silence, self._gap_speech,
                        )
                        self._silence_frame_count = (
                            self._gap_base_silence + self._gap_speech
                        )
                        self._merged_frames += self._gap_speech
                    else:
                        # 真实语音：flush gap_buffer 进 _segment_frames
                        logger.debug(
                            "vad gap keep: dur=%.0fms >= min=%.0fms flushing",
                            gap_speech_dur * 1000, MIN_SPEECH_DURATION * 1000,
                        )
                        self._segment_frames.extend(self._gap_buffer)
                        self._speech_frame_count += self._gap_speech
                        self._last_speech_end_sample = (
                            self._total_samples - self.hop_size
                        )
                        self._gap_active = False
                        self._gap_base_silence = 0
                        self._silence_frame_count = 0
                        self._merged_frames = 0
                    self._gap_speech = 0
                    self._gap_buffer.clear()

                self._segment_frames.append(frame)
                self._silence_frame_count += 1

                speech_dur = self._speech_frame_count * self.frame_duration
                pause_dur = self._silence_frame_count * self.frame_duration

                if _should_cut_segment(speech_dur, pause_dur):
                    _log_vad_cut(
                        "silence", speech_dur, pause_dur,
                        self._silence_frame_count, self._merged_frames,
                    )
                    return self._finalize_segment(speech_dur)

        # 强制上限：用实际段长（含 gap buffer）做检查
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
                logger.debug("vad cut: max_speech speech=%.0fms", speech_dur * 1000)
                return self._finalize_segment(speech_dur)

        return None

    def _compute_segment_end(self) -> int:
        pad_samples = self._pad_frames * self.hop_size
        target_end = self._last_speech_end_sample + pad_samples
        buffered_end = self._speech_start_sample + (
            len(self._pre_snapshot) + len(self._segment_frames)
        ) * self.hop_size
        return min(target_end, buffered_end)

    def _finalize_segment(self, speech_duration: float) -> Optional[dict]:
        seg = self._extract_and_reset()
        if speech_duration < MIN_SPEECH_DURATION:
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
        self._merged_frames = 0


def _log_vad_cut(
    tag: str,
    speech_dur: float,
    pause_dur: float,
    silence_frames: int,
    merged_frames: int,
) -> None:
    logger.debug(
        "vad cut: %s speech=%.0fms silence=%.0fms(%df: orig=%d + merged=%d)",
        tag,
        speech_dur * 1000,
        pause_dur * 1000,
        silence_frames,
        silence_frames - merged_frames,
        merged_frames,
    )


def _should_cut_segment(speech_duration: float, pause_duration: float) -> bool:
    if speech_duration >= MAX_SPEECH_DURATION:
        return True
    return pause_duration >= PAUSE_THRESHOLD
