from __future__ import annotations

import asyncio


class SessionManager:
    def __init__(self, max_connections: int) -> None:
        self._max = max_connections
        self._count = 0
        self.shutting_down: bool = False

    @property
    def current_count(self) -> int:
        return self._count

    def acquire(self) -> bool:
        if self.shutting_down or self._count >= self._max:
            return False
        self._count += 1
        return True

    def release(self) -> None:
        if self._count > 0:
            self._count -= 1

    async def shutdown(self, timeout: float = 30.0) -> None:
        self.shutting_down = True
        deadline = asyncio.get_event_loop().time() + timeout
        while self._count > 0:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(0.1)
