from __future__ import annotations

import asyncio
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

_itn_service: Optional["ITNService"] = None

_processor = None

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker_init(model_path: str) -> None:
    global _processor
    itn_dir = os.path.join(_PROJECT_ROOT, "weights", "itn")
    if itn_dir not in sys.path:
        sys.path.insert(0, itn_dir)
    from itn_wrapper import ITNProcessor  # noqa: PLC0415
    _processor = ITNProcessor(model_path=model_path)


def _do_itn(text: str) -> str:
    global _processor
    if _processor is None:
        return text
    try:
        return _processor.process(text)
    except Exception:
        return text


class ITNService:
    def __init__(self, max_workers: int, model_path: str) -> None:
        self._model_path = model_path
        self._max_workers = max_workers
        self._executor: Optional[ProcessPoolExecutor] = None

    async def start(self) -> None:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(self._model_path,),
        )

    async def process(self, text: str) -> str:
        if not text or self._executor is None:
            return text
        loop = asyncio.get_event_loop()
        try:
            result: str = await loop.run_in_executor(self._executor, _do_itn, text)
            return result
        except Exception as exc:
            logger.warning("ITN failed: %s", exc)
            return text

    async def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


def set_itn_service(service: Optional[ITNService]) -> None:
    global _itn_service
    _itn_service = service


def get_itn_service() -> Optional[ITNService]:
    return _itn_service
