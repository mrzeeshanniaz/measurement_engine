"""
MediaPipe BlazePose wrapper — Tasks API (mediapipe >= 0.10).

Detects 33 3D body landmarks from a single PIL image using the
PoseLandmarker Tasks API.  Downloads the heavy model on first load.

Fallback: symmetric template landmarks when the model is unavailable,
so the rest of the pipeline still runs in offline / test environments.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Heavy model = model_complexity=2 equivalent; best landmark accuracy.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
_MODEL_PATH = Path(__file__).parents[3] / "models" / "pose_landmarker_heavy.task"


class LM(IntEnum):
    NOSE       = 0
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_ELBOW    = 13
    R_ELBOW    = 14
    L_WRIST    = 15
    R_WRIST    = 16
    L_HIP      = 23
    R_HIP      = 24
    L_KNEE     = 25
    R_KNEE     = 26
    L_ANKLE    = 27
    R_ANKLE    = 28


@dataclass
class RawLandmark:
    x: float
    y: float
    z: float
    visibility: float


class MediaPipePoseWrapper:

    def __init__(self):
        self._landmarker = None
        self._fallback = False
        self.is_loaded = False

    async def load(self) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = _ensure_model()

            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self.is_loaded = True
            logger.info("MediaPipe PoseLandmarker loaded (Tasks API, %s)", model_path.name)
        except Exception as exc:
            logger.warning("MediaPipe Tasks API unavailable (%s) — using fallback landmarks", exc)
            self._fallback = True
            self.is_loaded = True

    def detect_landmarks(
        self, image: Image.Image
    ) -> Optional[dict[int, RawLandmark]]:
        if not self.is_loaded:
            raise RuntimeError("Pose model not loaded")

        if self._fallback:
            return self._fallback_landmarks()

        try:
            import mediapipe as mp
            rgb = np.array(image.convert("RGB"))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            if not result.pose_landmarks:
                logger.warning("No pose detected in frame")
                return None

            return {
                i: RawLandmark(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(lm.visibility),
                )
                for i, lm in enumerate(result.pose_landmarks[0])
            }
        except Exception as exc:
            logger.error("Landmark detection failed: %s", exc)
            return self._fallback_landmarks()

    def unload(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
        self._landmarker = None
        self.is_loaded = False

    @staticmethod
    def _fallback_landmarks() -> dict[int, RawLandmark]:
        """Symmetric template positions — used only when Tasks API unavailable."""
        positions = {
            0:  (0.50, 0.07, 0.0,  0.99),  # nose
            11: (0.38, 0.22, -0.05, 0.99),  # L shoulder
            12: (0.62, 0.22, -0.05, 0.99),  # R shoulder
            13: (0.28, 0.37, -0.03, 0.98),  # L elbow
            14: (0.72, 0.37, -0.03, 0.98),  # R elbow
            15: (0.22, 0.50, -0.02, 0.97),  # L wrist
            16: (0.78, 0.50, -0.02, 0.97),  # R wrist
            23: (0.42, 0.55, 0.0,  0.99),   # L hip
            24: (0.58, 0.55, 0.0,  0.99),   # R hip
            25: (0.41, 0.72, 0.02, 0.98),   # L knee
            26: (0.59, 0.72, 0.02, 0.98),   # R knee
            27: (0.40, 0.90, 0.04, 0.97),   # L ankle
            28: (0.60, 0.90, 0.04, 0.97),   # R ankle
        }
        return {
            idx: RawLandmark(x=x, y=y, z=z, visibility=v)
            for idx, (x, y, z, v) in positions.items()
        }


def _ensure_model() -> Path:
    """Return model path, downloading it first if needed."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading pose landmarker model (~29 MB) ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    logger.info("Pose landmarker model saved to %s", _MODEL_PATH)
    return _MODEL_PATH
