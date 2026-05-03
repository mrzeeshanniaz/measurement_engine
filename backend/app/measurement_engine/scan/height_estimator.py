"""
Automatic height estimation.

Resolution order:
  1. user_input      — height_cm provided directly → HIGH confidence
  2. sensor_fusion   — pinhole camera model using phone accelerometer + focal
                       length → MEDIUM confidence (±3–5 cm)
  3. population_mean — global average fallback → LOW confidence

The sensor fusion model (pinhole camera geometry):
  - Camera is at height H cm above the floor, tilted θ degrees below horizontal.
  - Subject stands at horizontal distance D from the camera.
  - Vertical field of view is derived from focal_length_px or assumed 65°.
  - Ankle landmarks give the floor-plane angle; nose landmark gives head angle.
  - Subject height = camera_height − D·tan(α_nose) + crown_offset
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from app.measurement_engine.scan.schemas import CameraMetadata, Confidence
from app.measurement_engine.scan.measurements import LandmarkPoint

logger = logging.getLogger(__name__)

_POPULATION_MEAN_CM = 170.0
_DEFAULT_VFOV_DEG   = 65.0   # typical smartphone vertical FOV
_CROWN_OFFSET_CM    = 3.5    # vertical distance from nose landmark to crown of head
_PLAUSIBLE_RANGE    = (100.0, 250.0)

# BlazePose landmark indices used for height estimation
_NOSE    = 0
_L_ANKLE = 27
_R_ANKLE = 28


@dataclass
class HeightEstimate:
    value_cm: float
    source: str          # "user_input" | "sensor_fusion" | "population_mean"
    confidence: Confidence
    note: Optional[str] = None


class HeightEstimator:
    """Resolves the height anchor from available inputs."""

    def estimate(
        self,
        height_cm: Optional[float],
        camera_metadata: Optional[CameraMetadata],
        landmarks: Optional[dict[int, LandmarkPoint]],
        image_height_px: int = 1080,
    ) -> HeightEstimate:
        if height_cm is not None:
            return HeightEstimate(
                value_cm=round(height_cm, 1),
                source="user_input",
                confidence=Confidence.HIGH,
            )

        if camera_metadata and landmarks:
            try:
                h = self._sensor_fusion(camera_metadata, landmarks, image_height_px)
                logger.info("Sensor-fusion height estimate: %.1f cm", h)
                return HeightEstimate(
                    value_cm=round(h, 1),
                    source="sensor_fusion",
                    confidence=Confidence.MEDIUM,
                    note="Auto-estimated ±3–5 cm — tailor should verify before cutting",
                )
            except Exception as exc:
                logger.warning("Sensor fusion height estimation failed: %s", exc)

        logger.warning("Falling back to population mean height (%.1f cm)", _POPULATION_MEAN_CM)
        return HeightEstimate(
            value_cm=_POPULATION_MEAN_CM,
            source="population_mean",
            confidence=Confidence.LOW,
            note="Population average used — tailor must confirm height before order",
        )

    # ------------------------------------------------------------------
    # Pinhole camera model
    # ------------------------------------------------------------------

    def _sensor_fusion(
        self,
        meta: CameraMetadata,
        landmarks: dict[int, LandmarkPoint],
        image_height_px: int,
    ) -> float:
        nose   = landmarks.get(_NOSE)
        l_ankl = landmarks.get(_L_ANKLE)
        r_ankl = landmarks.get(_R_ANKLE)
        ankle  = l_ankl or r_ankl

        if not nose or not ankle:
            raise ValueError("Nose or ankle landmarks not detected")
        if nose.visibility < 0.5 or ankle.visibility < 0.5:
            raise ValueError("Nose or ankle landmarks have low visibility")

        # Vertical field of view
        if meta.focal_length_px:
            vfov_rad = 2.0 * math.atan(image_height_px / (2.0 * meta.focal_length_px))
        else:
            vfov_rad = math.radians(_DEFAULT_VFOV_DEG)

        θ = math.radians(meta.tilt_angle_deg)  # positive = looking downward

        # Average ankle Y if both are visible
        ankle_y = (
            (l_ankl.y + r_ankl.y) / 2.0
            if l_ankl and r_ankl
            else ankle.y
        )

        # Angle below horizontal for each landmark
        # y_norm=0 → top of frame (above optical axis); y_norm=1 → bottom (below)
        # Pixel at y_norm is offset (y_norm − 0.5) × vfov_rad from the optical axis
        α_ankle = θ + (ankle_y      - 0.5) * vfov_rad
        α_nose  = θ + (nose.y       - 0.5) * vfov_rad

        if α_ankle <= 0.01:
            raise ValueError(
                f"Camera tilt {meta.tilt_angle_deg}° too shallow — cannot resolve floor plane"
            )

        # Horizontal distance from camera to subject
        D = meta.camera_height_cm / math.tan(α_ankle)

        # Nose height above floor: camera_height − D·tan(α_nose)
        # tan(α_nose) is negative when nose is above the optical axis, giving a
        # larger value — which is correct (nose is above camera level).
        nose_height = meta.camera_height_cm - D * math.tan(α_nose)

        subject_height = nose_height + _CROWN_OFFSET_CM

        lo, hi = _PLAUSIBLE_RANGE
        if not (lo <= subject_height <= hi):
            raise ValueError(
                f"Computed height {subject_height:.1f} cm outside plausible range "
                f"[{lo}, {hi}] — check camera_height_cm and tilt_angle_deg"
            )

        return subject_height
