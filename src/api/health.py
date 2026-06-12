from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from core.logging import log_buffer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

_BACKLOG_LINES = 50
_KEEPALIVE_INTERVAL = 30


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict:
    healthy = getattr(request.app.state, "vllm_healthy", True)
    if not healthy:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}


@router.get("/logs/stream")
async def logs_stream(request: Request) -> StreamingResponse:
    async def _generate():
        q = log_buffer.subscribe()
        try:
            for line in log_buffer.get_recent(_BACKLOG_LINES):
                yield f"data: {line}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=_KEEPALIVE_INTERVAL)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            log_buffer.unsubscribe(q)

    return StreamingResponse(_generate(), media_type="text/event-stream")
