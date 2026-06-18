from __future__ import annotations

import asyncio
import base64
import logging
import struct
import re

import httpx
import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _build_hotword_context(hotwords: str) -> str:
    """将热词构建为系统提示词，多个热词以中文顿号分隔。

    Qwen3-ASR 通过 chat 接口的 system 消息（"热词：xxx、yyy"）来注入热词偏置，
    /audio/transcriptions 接口并不支持自定义 hotwords 字段（会被静默忽略，
    在部分严格的 vLLM 构建上还会因未知表单字段而报错）。
    """
    words = list(dict.fromkeys(
        w.strip() for w in hotwords.replace("|", ",").split(",") if w.strip()
    ))
    return f"热词：{'、'.join(words)}" if words else ""


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


def _filter_hallucination(text: str, hotwords: str = "") -> str:
    """过滤离线 ASR 的 prompt 泄漏幻觉。

    规则：
    1. 以 "热词：" 开头 → 视为 system prompt 泄漏，返回 ""
    2. 以热词列表中相邻两个热词（顿号分隔）开头 → 返回 ""
    """
    logger.debug("_filter_hallucination 过滤前: text=%r, hotwords=%r", text, hotwords)
    if not text or not hotwords:
        return text
    stripped = text.lstrip()
    if stripped.startswith("热词："):
        logger.debug("_filter_hallucination 命中: starts with 热词：")
        return ""
    words = [w.strip() for w in hotwords.replace("|", ",").split(",") if w.strip()]
    if len(words) < 2:
        return text
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        if stripped.startswith(f"{a}、{b}"):
            logger.debug("_filter_hallucination 命中: text starts with %r", f"{a}、{b}")
            return ""
    return text


def _prepare_transcribe_request(audio: np.ndarray, hotwords: str) -> list:
    """在后台线程中构建 WAV/base64/消息体，避免 event loop 阻塞。"""
    wav_bytes = _build_wav_bytes(audio)
    audio_b64 = base64.b64encode(wav_bytes).decode()

    messages: list = []
    hotword_ctx = _build_hotword_context(hotwords)
    if hotword_ctx:
        messages.append({"role": "system", "content": hotword_ctx})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
            }
        ],
    })
    return messages


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
        self._outstanding: int = 0

    async def transcribe(self, audio: np.ndarray, hotwords: str = "") -> str:
        messages = await asyncio.to_thread(_prepare_transcribe_request, audio, hotwords)

        self._outstanding += 1
        try:
            response = await self._client.post(
                "/chat/completions",
                json={"model": self._model_name, "messages": messages},
            )
            response.raise_for_status()
            result = response.json()
            raw_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return _filter_hallucination(_parse_asr_response(raw_text), hotwords)
        finally:
            self._outstanding -= 1

    async def check_health(self) -> bool:
        try:
            response = await self._client.get(self._health_url, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


_online_clients: list[VLLMASRClient] = []
_offline_clients: list[VLLMASRClient] = []


def _init_clients() -> None:
    global _online_clients, _offline_clients
    if _online_clients or _offline_clients:
        return
    s = get_settings()
    _online_clients = [
        VLLMASRClient(url, s.online_model_name, s.vllm_api_key)
        for url in s.online_api_bases
    ]
    _offline_clients = [
        VLLMASRClient(url, s.offline_model_name, s.vllm_api_key)
        for url in s.offline_api_bases
    ]


def get_online_client() -> VLLMASRClient:
    _init_clients()
    return min(_online_clients, key=lambda c: c._outstanding)


def get_offline_client() -> VLLMASRClient:
    _init_clients()
    return min(_offline_clients, key=lambda c: c._outstanding)


async def close_asr_clients() -> None:
    for client in _online_clients + _offline_clients:
        try:
            await client.close()
        except Exception:
            pass
