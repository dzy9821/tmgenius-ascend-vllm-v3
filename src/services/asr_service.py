from __future__ import annotations

import asyncio
import base64
import itertools
import logging
import struct
import re
from functools import lru_cache
from typing import Optional

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


def _detect_and_fix_repetitions(text: str) -> str:
    threshold = get_settings().asr_rep_threshold
    def _fix_char_repeats(s: str, thresh: int) -> str:
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
            else:
                res.append(s[i : i + count])
            i += count
        return "".join(res)

    def _fix_pattern_repeats(s: str, thresh: int, max_len: int = 30) -> str:
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
        i = 0
        result = []
        found = False
        while i <= n - min_repeat_chars:
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i : i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx : start_idx + k] != pattern:
                        valid = False
                        break
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index : end_index + k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(_fix_pattern_repeats(s[end_index:], thresh, max_len))
                    found = True
                    break
            if found:
                break
            result.append(s[i])
            i += 1
        if not found:
            result.append(s[i:])
        return "".join(result)

    text = _fix_char_repeats(text, threshold)
    text = _fix_pattern_repeats(text, threshold)
    return text


def _parse_asr_response(content: str) -> str:
    """Strip Qwen3-ASR format tags and fix repetition hallucinations."""
    if not content:
        return ""
    text = _ASR_TAG_RE.sub('', content).strip()
    text = _detect_and_fix_repetitions(text)
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

        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model_name,
                "messages": messages,
                "frequency_penalty": 0.8,
                "presence_penalty": 0.8,
            },
        )
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        text = _filter_hallucination(_parse_asr_response(raw_text), hotwords)
        if text:
            s = get_settings()
            audio_sec = len(audio) / SAMPLE_RATE
            max_chars = max(audio_sec * s.asr_max_chars_per_sec, 10)
            if len(text) > max_chars:
                logger.warning(
                    "text length exceeded: audio=%.1fs text=%d chars max=%.0f ratio=%.1f",
                    audio_sec, len(text), max_chars, s.asr_max_chars_per_sec,
                )
                return ""
        return text

    async def check_health(self) -> bool:
        try:
            response = await self._client.get(self._health_url, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


class RoundRobinASRClient:
    """轮询分发在线请求到多个 vLLM 实例。"""

    def __init__(self, api_bases: list[str], model_name: str, api_key: str) -> None:
        self._clients = [
            VLLMASRClient(base, model_name, api_key) for base in api_bases
        ]
        self._idx_cycle = itertools.cycle(range(len(self._clients)))
        self._lock = asyncio.Lock()

    async def transcribe(self, audio: "np.ndarray", hotwords: str = "") -> str:
        async with self._lock:
            idx = next(self._idx_cycle)
        return await self._clients[idx].transcribe(audio, hotwords)

    async def check_health(self) -> bool:
        results = await asyncio.gather(
            *[c.check_health() for c in self._clients], return_exceptions=True
        )
        return all(r is True for r in results)

    async def close(self) -> None:
        await asyncio.gather(
            *[c.close() for c in self._clients], return_exceptions=True
        )


@lru_cache(maxsize=1)
def get_online_client() -> RoundRobinASRClient:
    s = get_settings()
    return RoundRobinASRClient(s.online_api_bases, s.online_model_name, s.vllm_api_key)


@lru_cache(maxsize=1)
def get_offline_client() -> RoundRobinASRClient:
    s = get_settings()
    return RoundRobinASRClient(s.offline_api_bases, s.offline_model_name, s.vllm_api_key)


async def close_asr_clients() -> None:
    for fn in (get_online_client, get_offline_client):
        try:
            client = fn()
            await client.close()
        except Exception:
            pass
