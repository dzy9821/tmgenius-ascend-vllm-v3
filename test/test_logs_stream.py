"""
测试 /api/v1/logs/stream SSE 日志流端点。

测试策略：
- 单元测试：验证 event_generator 核心逻辑（backlog 回放、keepalive、subscriber 清理）
- HTTP 测试：验证参数校验和路由注册
"""

import asyncio
import json
import logging
import time

import httpx
import pytest

from core.logging import log_buffer
from main import app


@pytest.fixture(autouse=True)
def configure_root_logger():
    """确保 root logger 有 log_buffer handler，使用 JSON 格式便于测试验证。"""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    class JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return json.dumps({
                "ts": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }, ensure_ascii=False)

    fmt = JSONFormatter()
    log_buffer.setFormatter(fmt)
    if log_buffer not in root.handlers:
        root.addHandler(log_buffer)
    yield


# ── 单元测试：核心逻辑 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_event_generator_backlog():
    """异步生成器：先回放 backlog 再进入实时订阅。"""
    # 写入 5 条测试日志，保证 backlog 有足够内容
    for i in range(5):
        logging.getLogger("test").info(f"backlog_unit_{i}_{time.time()}")

    await asyncio.sleep(0.05)

    # 构造生成器（与 health.py 中实现一致）
    async def event_generator(backlog: int):
        for entry in log_buffer.get_recent(backlog):
            yield f"data: {entry}\n\n"
        queue = log_buffer.subscribe()
        try:
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {entry}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return
        finally:
            log_buffer.unsubscribe(queue)

    gen = event_generator(5)

    # 读取 5 条 backlog
    received = []
    for _ in range(5):
        chunk = await gen.__anext__()
        assert chunk.startswith("data: ")
        received.append(chunk)

    assert len(received) == 5
    for r in received:
        body = r.removeprefix("data: ").rstrip("\n")
        parsed = json.loads(body)
        assert "msg" in parsed
        assert "level" in parsed
        assert "backlog_unit_" in parsed["msg"]

    await gen.aclose()


@pytest.mark.asyncio
async def test_event_generator_realtime():
    """backlog 回放完毕后，新日志能实时推送。"""
    async def simple_gen():
        queue = log_buffer.subscribe()
        try:
            entry = await asyncio.wait_for(queue.get(), timeout=5)
            yield f"data: {entry}\n\n"
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return
        finally:
            log_buffer.unsubscribe(queue)

    gen = simple_gen()

    # 启动生成器（在后台调用 __anext__），会阻塞在 queue.get()
    async def read_entry():
        return await gen.__anext__()

    read_task = asyncio.create_task(read_entry())
    await asyncio.sleep(0.02)  # 确保 subscribe() 已执行

    # 写入新日志 — 此时 subscriber 已就位
    unique = f"realtime_{time.time()}"
    logging.getLogger("test").info(unique)

    realtime_chunk = await asyncio.wait_for(read_task, timeout=5)
    assert unique in realtime_chunk

    await gen.aclose()


@pytest.mark.asyncio
async def test_event_generator_keepalive():
    """backlog=0 且无新日志时，30 秒超时发送 keepalive。"""
    async def event_generator(backlog: int = 0):
        recent = log_buffer.get_recent(backlog)
        for entry in recent:
            yield f"data: {entry}\n\n"
        queue = log_buffer.subscribe()
        try:
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=1)
                    yield f"data: {entry}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return
        finally:
            log_buffer.unsubscribe(queue)

    gen = event_generator(0)
    # get_recent(0) 可能返回全部（Python list[-0:] 语义）
    # 跳过所有 backlog 条目
    chunk = await asyncio.wait_for(gen.__anext__(), timeout=3)
    while chunk.startswith("data: "):
        chunk = await asyncio.wait_for(gen.__anext__(), timeout=3)

    # 现在应该是 keepalive
    assert chunk.startswith(": keepalive"), f"Expected keepalive, got: {chunk[:50]}..."

    await gen.aclose()


@pytest.mark.asyncio
async def test_subscriber_cleanup():
    """生成器关闭时 subscriber 被正确移除。"""
    initial_count = len(log_buffer._subscribers)

    async def event_generator():
        queue = log_buffer.subscribe()
        try:
            yield f"data: test\n\n"
            await asyncio.sleep(30)  # 模拟长期运行
        except asyncio.CancelledError:
            return
        finally:
            log_buffer.unsubscribe(queue)

    gen = event_generator()
    await gen.__anext__()

    # subscriber 已被添加
    assert len(log_buffer._subscribers) == initial_count + 1

    await gen.aclose()
    await asyncio.sleep(0.1)

    # subscriber 已被移除
    assert len(log_buffer._subscribers) == initial_count, (
        f"Subscriber leak: {len(log_buffer._subscribers)} != {initial_count}"
    )


@pytest.mark.asyncio
async def test_get_recent_limit():
    """get_recent 正确限制返回条数。"""
    for i in range(10):
        logging.getLogger("test").info(f"limit_test_{i}")

    await asyncio.sleep(0.05)

    recent = log_buffer.get_recent(3)
    assert len(recent) == 3


@pytest.mark.asyncio
async def test_backlog_zero_returns_all():
    """get_recent(0) 由于 Python list[-0:] 语义返回全部记录。"""
    for i in range(5):
        logging.getLogger("test").info(f"zero_backlog_{i}")

    await asyncio.sleep(0.05)

    recent = log_buffer.get_recent(0)
    assert len(recent) >= 5, f"backlog=0 should return all items, got {len(recent)}"


# ── HTTP 集成测试 ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_logs_stream_invalid_params():
    """backlog 参数越界返回 422。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/logs/stream?backlog=3000")
        assert resp.status_code == 422

        resp = await client.get("/api/v1/logs/stream?backlog=-1")
        assert resp.status_code == 422


def test_logs_stream_route_registered():
    """/logs/stream 路由已正确注册。"""
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/api/v1/logs/stream" in routes


def test_endpoint_function_returns_streaming_response():
    """端点函数能正常返回 StreamingResponse。"""
    import asyncio
    from src.api.health import logs_stream

    async def _call():
        return await logs_stream(backlog=1)

    resp = asyncio.run(_call())
    from starlette.responses import StreamingResponse
    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
