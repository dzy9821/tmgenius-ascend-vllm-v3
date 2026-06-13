"""
JSON 结构化日志配置。

- 输出至 stdout
- 每条日志自动注入 trace_id 字段
- 不记录 Base64 音频原文
- 内存环形缓冲区供 HTTP 流式查询
"""

import asyncio
import collections
import logging
import json
from contextvars import ContextVar
from typing import Optional
import threading

# 用于在异步上下文中传递 trace_id
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


class JSONFormatter(logging.Formatter):
    """将日志格式化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get("-"),
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class InMemoryLogHandler(logging.Handler):
    """将日志写入内存环形缓冲区，并通知所有 SSE 订阅者。"""

    def __init__(self, capacity: int = 2000):
        super().__init__()
        self._buffer: collections.deque[str] = collections.deque(maxlen=capacity)
        self._buffer_lock = threading.Lock()
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._subs_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._buffer_lock:
                self._buffer.append(msg)

            loop = self._loop
            if loop is not None:
                with self._subs_lock:
                    subs = list(self._subscribers)
                for q in subs:
                    loop.call_soon_threadsafe(self._safe_put, q, msg)
        except Exception:
            self.handleError(record)

    @staticmethod
    def _safe_put(q: asyncio.Queue[str], msg: str) -> None:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        with self._subs_lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        with self._subs_lock:
            self._subscribers.discard(q)

    def get_recent(self, n: int = 50) -> list[str]:
        with self._buffer_lock:
            items = list(self._buffer)
        return items[-n:]


log_buffer = InMemoryLogHandler(capacity=2000)