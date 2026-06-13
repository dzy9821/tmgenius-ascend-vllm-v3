import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.api.health import router as health_router
from src.api.http_endpoints import router as http_router
from src.api.websocket_handler import handle_websocket
from src.config import get_settings
from src.metrics import (
    connections_current as asr_connections_current,
    processing_latency as asr_processing_latency,
    queue_depth as asr_queue_depth,
    segments_total as asr_segments_total,
    errors_total as asr_errors_total,
)
from src.services.asr_service import close_asr_clients, get_offline_client, get_online_client
from src.services.itn_service import ITNService
from src.services.session_manager import SessionManager

from src.core.logging import log_buffer


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            entry["trace_id"] = record.trace_id
        for key, value in getattr(record, "extra", {}).items():
            entry[key] = value
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    # 将日志同时写入内存环形缓冲区，供 /logs/stream SSE 端点消费
    log_buffer.setFormatter(JSONFormatter())
    root.addHandler(log_buffer)

    # 降低第三方库日志级别
    for name in ("uvicorn", "uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    setup_logging(settings.log_level)
    logger = logging.getLogger("asr_service")
    logger.info("Starting ASR service", extra={"port": settings.ws_port})

    app.state.session_manager = SessionManager(settings.max_connections)

    # 触发 lazy 初始化
    get_online_client()
    get_offline_client()

    logger.info("Starting ITN process pool")
    itn_service = ITNService(settings.itn_workers, settings.fst_itn_zh_path)
    await itn_service.start()
    app.state.itn_service = itn_service

    from src.api.websocket_handler import set_itn_service, set_rnnoise_service
    set_itn_service(itn_service)

    from src.services.rnnoise_service import RnnoiseService
    rnnoise_service = RnnoiseService(
        enabled=settings.rnnoise_enabled,
        reduce_db=settings.rnnoise_reduce_db,
        max_workers=settings.rnnoise_workers,
    )
    app.state.rnnoise_service = rnnoise_service
    set_rnnoise_service(rnnoise_service)

    # 后台 vLLM 健康检查
    async def _health_monitor():
        consecutive_fails = 0
        while True:
            await asyncio.sleep(settings.vllm_health_check_interval)
            online_ok = await get_online_client().check_health()
            offline_ok = await get_offline_client().check_health()
            if not (online_ok and offline_ok):
                consecutive_fails += 1
                if consecutive_fails >= 3:
                    logger.error(
                        "vLLM health check failed 3 times consecutively"
                    )
            else:
                consecutive_fails = 0
            app.state.vllm_healthy = online_ok and offline_ok

    app.state.vllm_healthy = True
    asyncio.create_task(_health_monitor())

    yield

    # ---- Shutdown ----
    logger.info("Shutting down ASR service")
    app.state.session_manager.shutting_down = True
    await app.state.session_manager.shutdown(timeout=10.0)
    await itn_service.shutdown()
    rnnoise_service.shutdown(wait=False)
    await close_asr_clients()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_api_websocket_route("/ast/v1", handle_websocket)
    app.include_router(http_router)
    app.include_router(health_router)

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.ws_host,
        port=settings.ws_port,
        log_level=settings.log_level.lower(),
        ws_ping_interval=settings.ws_ping_interval,
        ws_ping_timeout=settings.ws_ping_timeout,
    )
