from __future__ import annotations

import logging
import struct
import re
from functools import lru_cache
from typing import Optional

import httpx
import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)

_HALLUCINATION_BLACKLIST = [
    "transcribe the audio to text accurately",
    "pay special attention to these words",
    "thank you for watching",
    "please subscribe",
]

SAMPLE_RATE = 16000


def _build_wav_bytes(pcm_int16: np.ndarray) -> bytes:
    data = pcm_int16.astype("<i2").tobytes()
    num_samples = len(pcm_int16)
    num_channels = 1
    bits_per_sample = 16
    byte_rate = SAMPLE_RATE * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = len(data)
    riff_chunk_size = 36 + data_chunk_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_chunk_size,
        b"WAVE",
        b"fmt ",
        16,            # fmt chunk size
        1,             # PCM format
        num_channels,
        SAMPLE_RATE,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_chunk_size,
    )
    return header + data


_ASR_TAG_RE = re.compile(r'language\s+\w+\s*<asr_text>', re.IGNORECASE)


def _parse_asr_response(content: str) -> str:
    """Strip Qwen3-ASR format tags from transcription output.

    Handles these Qwen3-ASR output formats:
      - "language Chinese<asr_text>..." → "..."
      - "language None<asr_text>" → ""
      - Plain text without tag → text as-is
      - Multiple tags (model hallucination) → all stripped

    Uses re.sub to remove ALL occurrences, preventing leakage when
    the model hallucinates extra "language X<asr_text>" prefixes mid-text.
    """
    if not content:
        return ""

    text = _ASR_TAG_RE.sub('', content).strip()
    return text


# 末尾标点（中英文）字符集，用于 Online Progressive 结果展示优化
_TRAILING_PUNCT = "，。！？、；：,.!?;: \t　…—"


def strip_trailing_punct(text: str) -> str:
    """去除末尾的中英文标点符号，改善前端实时展示效果（仅 Online 阶段使用）。"""
    return text.rstrip(_TRAILING_PUNCT)


def _filter_hallucination(text: str) -> str:
    lower = text.lower()
    for phrase in _HALLUCINATION_BLACKLIST:
        if phrase in lower:
            return ""
    # 规则：以 "热词：" 开头的文本视为热词 prompt 泄漏，丢弃
    if text.lstrip().startswith("热词："):
        return ""
    return text


class VLLMASRClient:
    def __init__(self, api_base: str, model_name: str, api_key: str) -> None:
        self._api_base = api_base.rstrip("/")
        self._model_name = model_name
        # vLLM 的 /health 在服务根路径，不带 /v1 前缀
        root = self._api_base[: -len("/v1")] if self._api_base.endswith("/v1") else self._api_base
        self._health_url = f"{root}/health"
        self._client = httpx.AsyncClient(
            base_url=self._api_base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0),
            trust_env=False,
        )

    async def transcribe(self, audio: np.ndarray, hotwords: str = "") -> str:
        wav_bytes = _build_wav_bytes(audio)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data: dict = {"model": self._model_name}
        if hotwords:
            data["hotwords"] = hotwords

        response = await self._client.post("/audio/transcriptions", files=files, data=data)
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("text", "")
        return _filter_hallucination(_parse_asr_response(raw_text))

    async def check_health(self) -> bool:
        try:
            response = await self._client.get(self._health_url, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


@lru_cache(maxsize=1)
def get_online_client() -> VLLMASRClient:
    s = get_settings()
    return VLLMASRClient(s.online_api_base, s.online_model_name, s.vllm_api_key)


@lru_cache(maxsize=1)
def get_offline_client() -> VLLMASRClient:
    s = get_settings()
    return VLLMASRClient(s.offline_api_base, s.offline_model_name, s.vllm_api_key)


async def close_asr_clients() -> None:
    for fn in (get_online_client, get_offline_client):
        try:
            client = fn()
            await client.close()
        except Exception:
            pass
