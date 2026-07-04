"""
Model manager — loads and exposes all ML models as a single object
attached to app.state.models at startup.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.measurement_engine.models.pose import MediaPipePoseWrapper
from app.measurement_engine.models.segmentation import MediaPipeSegmenter
from app.measurement_engine.models.smpl import SMPLFitter

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.pose      = MediaPipePoseWrapper()
        self.segmenter = MediaPipeSegmenter()
        self.smpl      = SMPLFitter(device=settings.DEVICE)
        self._loaded   = False

    async def load(self) -> None:
        await self.pose.load()
        await self.segmenter.load()
        await self.smpl.load()
        self._loaded = True
        logger.info("All models loaded (device=%s)", settings.DEVICE)

    async def unload(self) -> None:
        self.pose.unload()
        self.segmenter.unload()
        self.smpl.unload()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded
