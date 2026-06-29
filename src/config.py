from __future__ import annotations

import os
from functools import lru_cache

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings:
    def __init__(self) -> None:
        # WebSocket 服务
        self.ws_host: str = os.getenv("WS_HOST", "0.0.0.0")
        self.ws_port: int = int(os.getenv("WS_PORT", "8856"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.max_connections: int = int(os.getenv("MAX_CONNECTIONS", "64"))
        self.handshake_timeout: float = float(os.getenv("HANDSHAKE_TIMEOUT", "300"))
        self.ws_ping_interval: int = int(os.getenv("WS_PING_INTERVAL", "10"))
        self.ws_ping_timeout: int = int(os.getenv("WS_PING_TIMEOUT", "300"))

        # VAD 参数
        self.vad_hop_size: int = int(os.getenv("VAD_HOP_SIZE", "640"))
        self.vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.75"))
        self.vad_min_speech: float = float(os.getenv("VAD_MIN_SPEECH", "0.8"))
        self.vad_max_speech: float = float(os.getenv("VAD_MAX_SPEECH", "60.0"))
        self.asr_pad_frames: int = int(os.getenv("ASR_PAD_FRAMES", "5"))

        # Online ASR 触发阈值
        self.online_trigger_ms: int = int(os.getenv("ONLINE_TRIGGER_MS", "200"))
        self.online_vad_pause_ms: int = int(os.getenv("ONLINE_VAD_PAUSE_MS", "500"))
        self.online_max_speech_ms: int = int(os.getenv("ONLINE_MAX_SPEECH_MS", "10000"))
        self.online_comma_limit: int = int(os.getenv("ONLINE_COMMA_LIMIT", "5"))

        # ASR 幻觉过滤
        self.asr_rep_threshold: int = int(os.getenv("ASR_REP_THRESHOLD", "4"))
        self.asr_max_chars_per_sec: float = float(os.getenv("ASR_MAX_CHARS_PER_SEC", "6.0"))

        # ITN 多进程池
        self.itn_workers: int = int(os.getenv("ITN_WORKERS", "8"))
        self.fst_itn_zh_path: str = os.getenv(
            "FST_ITN_ZH_PATH",
            os.path.join(_PROJECT_ROOT, "weights", "fst_itn_zh"),
        )
        self.mp_queue_log_interval_sec: int = int(
            os.getenv("MP_QUEUE_LOG_INTERVAL_SEC", "10")
        )

        # vLLM 配置
        self.offline_api_base: str = os.getenv(
            "OFFLINE_API_BASE", "http://127.0.0.1:15002/v1"
        )
        self.online_api_base: str = os.getenv(
            "ONLINE_API_BASE", "http://127.0.0.1:15004/v1"
        )
        # 多个 online 实例的 API base，逗号分隔；未设置时回退到 ONLINE_API_BASE
        _online_bases = os.getenv("ONLINE_API_BASES", "")
        self.online_api_bases: list[str] = (
            [u.strip() for u in _online_bases.split(",") if u.strip()]
            if _online_bases
            else [self.online_api_base]
        )
        # 多个 offline 实例的 API base，逗号分隔；未设置时回退到 OFFLINE_API_BASE
        _offline_bases = os.getenv("OFFLINE_API_BASES", "")
        self.offline_api_bases: list[str] = (
            [u.strip() for u in _offline_bases.split(",") if u.strip()]
            if _offline_bases
            else [self.offline_api_base]
        )
        self.offline_model_name: str = os.getenv(
            "OFFLINE_MODEL_NAME", "Qwen3-ASR-1.7B"
        )
        self.online_model_name: str = os.getenv(
            "ONLINE_MODEL_NAME", "Qwen3-ASR-0.6B"
        )
        self.vllm_api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")
        self.offline_model_path: str = os.getenv(
            "OFFLINE_MODEL_PATH", "/weights/Qwen3-ASR-1.7B"
        )
        self.online_model_path: str = os.getenv(
            "ONLINE_MODEL_PATH", "/weights/Qwen3-ASR-0.6B"
        )
        self.offline_max_model_len: int = int(
            os.getenv("OFFLINE_MAX_MODEL_LEN", "4096")
        )
        self.online_max_model_len: int = int(
            os.getenv("ONLINE_MAX_MODEL_LEN", "4096")
        )
        self.vllm_health_check_interval: int = int(
            os.getenv("VLLM_HEALTH_CHECK_INTERVAL", "30")
        )

        # RNNoise 降噪
        _rn = os.getenv("RNNOISE_ENABLED", "false").lower()
        self.rnnoise_enabled: bool = _rn not in ("false", "0", "no")
        self.rnnoise_reduce_db: float = float(os.getenv("RNNOISE_REDUCE_DB", "12.0"))
        self.rnnoise_workers: int = int(os.getenv("RNNOISE_WORKERS", "4"))

        # 热词配置
        self.hotwords: str = os.getenv("HOTWORDS", "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
