"""
Opus 解码器 — 基于 ctypes 封装系统 libopus（libopus.so.0）。

设计文档要求：当客户端 `payload.audio.encoding == "opus"` 时，对每帧音频做 Opus 解码；
每个 WebSocket 连接持有一个独立的解码器实例。本模块零新增 Python 依赖，直接通过 ctypes
调用系统已安装的 libopus 共享库，风格与项目内 ten-vad 的原生库加载方式一致。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

import numpy as np

logger = logging.getLogger(__name__)

# libopus 常量
_OPUS_OK = 0
# 输出缓冲区上限：120ms @ 48kHz 单声道 = 5760 采样，足以容纳任意合法 Opus 帧
_MAX_FRAME_SIZE = 5760


def _load_libopus() -> ctypes.CDLL:
    for name in ("libopus.so.0", "libopus.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    found = ctypes.util.find_library("opus")
    if found:
        return ctypes.CDLL(found)
    raise OSError("无法加载 libopus（请确认已安装 libopus0）")


_lib = _load_libopus()

# int opus_decoder_get_size(int channels) —— 仅用于校验，可选
# OpusDecoder *opus_decoder_create(opus_int32 Fs, int channels, int *error)
_lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_lib.opus_decoder_create.restype = ctypes.c_void_p

# int opus_decode(OpusDecoder *st, const unsigned char *data, opus_int32 len,
#                 opus_int16 *pcm, int frame_size, int decode_fec)
_lib.opus_decode.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.c_int,
]
_lib.opus_decode.restype = ctypes.c_int

# void opus_decoder_destroy(OpusDecoder *st)
_lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
_lib.opus_decoder_destroy.restype = None


class OpusDecoder:
    """单声道 Opus 解码器，逐帧解码为 int16 PCM。非线程安全，按连接独立持有。"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self._channels = channels
        err = ctypes.c_int(0)
        self._dec = _lib.opus_decoder_create(sample_rate, channels, ctypes.byref(err))
        if not self._dec or err.value != _OPUS_OK:
            raise RuntimeError(f"opus_decoder_create 失败: error={err.value}")
        self._pcm_buf = (ctypes.c_int16 * (_MAX_FRAME_SIZE * channels))()

    def decode(self, data: bytes) -> np.ndarray:
        """将一帧 Opus 数据解码为 int16 PCM（单声道返回一维数组）。"""
        if self._dec is None:
            raise RuntimeError("OpusDecoder 已关闭")
        n = _lib.opus_decode(
            self._dec, data, len(data), self._pcm_buf, _MAX_FRAME_SIZE, 0
        )
        if n < 0:
            raise RuntimeError(f"opus_decode 失败: code={n}")
        total = n * self._channels
        return np.frombuffer(self._pcm_buf, dtype=np.int16, count=total).copy()

    def close(self) -> None:
        if self._dec is not None:
            _lib.opus_decoder_destroy(self._dec)
            self._dec = None
