from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from src.core.rnnoise_denoise import RnnoiseStreamProcessor

logger = logging.getLogger(__name__)

_rnnoise_service: Optional["RnnoiseService"] = None


class RnnoiseService:
    def __init__(self, enabled: bool, reduce_db: float, max_workers: int) -> None:
        self._enabled = enabled
        self._reduce_db = reduce_db
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rnnoise")

    def new_processor(self) -> Optional[RnnoiseStreamProcessor]:
        if not self._enabled:
            return None
        return RnnoiseStreamProcessor(noise_reduce_db=self._reduce_db)

    async def denoise(
        self,
        processor: RnnoiseStreamProcessor,
        pcm_int16: np.ndarray,
    ) -> np.ndarray:
        loop = asyncio.get_event_loop()
        pcm_f32 = pcm_int16.astype(np.float32) / 32768.0

        def _run() -> np.ndarray:
            return processor.process(pcm_f32)

        result_f32: np.ndarray = await loop.run_in_executor(self._executor, _run)
        return (result_f32 * 32768.0).clip(-32768, 32767).astype(np.int16)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


def set_rnnoise_service(service: Optional[RnnoiseService]) -> None:
    global _rnnoise_service
    _rnnoise_service = service


def get_rnnoise_service() -> Optional[RnnoiseService]:
    return _rnnoise_service
