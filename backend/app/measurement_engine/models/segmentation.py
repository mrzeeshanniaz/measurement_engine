"""
MediaPipe body segmenter — Tasks API (mediapipe >= 0.10.14).

Produces a binary body mask (uint8, 255=person, 0=background) from a
single PIL image using the selfie_segmenter model.  The mask is used
downstream for occlusion scoring and silhouette-based measurements.

Fallback: returns None when the model or Tasks API is unavailable, so
all downstream code that consumes the mask must handle None gracefully.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
_SEG_PATH = Path(__file__).parents[3] / "models" / "selfie_segmenter.tflite"

# Pixels with confidence above this threshold are classified as foreground.
_CONF_THRESHOLD = 0.5


class MediaPipeSegmenter:

    def __init__(self):
        self._segmenter = None
        self._fallback = False
        self.is_loaded = False

    async def load(self) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = _ensure_model()
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_confidence_masks=True,
            )
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
            self.is_loaded = True
            logger.info("MediaPipe ImageSegmenter loaded (%s)", model_path.name)
        except Exception as exc:
            logger.warning(
                "MediaPipe ImageSegmenter unavailable (%s) — body masks disabled", exc
            )
            self._fallback = True
            self.is_loaded = True

    def segment(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Returns uint8 mask (H, W): 255 = person, 0 = background.
        Returns None when the model is unavailable or segmentation fails.
        """
        if not self.is_loaded:
            raise RuntimeError("Segmenter not loaded")
        if self._fallback or self._segmenter is None:
            return None
        try:
            import mediapipe as mp

            rgb = np.array(image.convert("RGB"))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._segmenter.segment(mp_image)

            if not result.confidence_masks:
                return None

            # confidence_masks[0] is the foreground (person) confidence: (H, W) float32
            conf_map = result.confidence_masks[0].numpy_view()
            return (conf_map > _CONF_THRESHOLD).astype(np.uint8) * 255
        except Exception as exc:
            logger.error("Segmentation failed: %s", exc)
            return None

    def unload(self) -> None:
        if self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception:
                pass
        self._segmenter = None
        self.is_loaded = False


def _ensure_model() -> Path:
    if _SEG_PATH.exists():
        return _SEG_PATH
    _SEG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading selfie segmenter model (~1.5 MB) ...")
    urllib.request.urlretrieve(_SEG_URL, _SEG_PATH)
    logger.info("Segmenter model saved to %s", _SEG_PATH)
    return _SEG_PATH
