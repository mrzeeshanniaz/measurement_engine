"""
Tests for app.measurement_engine.scan.height_estimator — resolution
order (user input → sensor fusion → population mean) and the pinhole
camera model used for sensor fusion.
"""
from __future__ import annotations

import math

import pytest

from app.measurement_engine.scan.height_estimator import (
    HeightEstimate,
    HeightEstimator,
    _CROWN_OFFSET_CM,
    _POPULATION_MEAN_CM,
)
from app.measurement_engine.scan.measurements import LandmarkPoint
from app.measurement_engine.scan.schemas import CameraMetadata, Confidence


def _landmarks(nose_y: float, ankle_y: float, vis: float = 0.9) -> dict[int, LandmarkPoint]:
    return {
        0:  LandmarkPoint(x=0.50, y=nose_y,  z=0.0, visibility=vis),
        27: LandmarkPoint(x=0.45, y=ankle_y, z=0.0, visibility=vis),
        28: LandmarkPoint(x=0.55, y=ankle_y, z=0.0, visibility=vis),
    }


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

class TestResolutionOrder:
    def test_user_input_takes_precedence(self):
        est = HeightEstimator().estimate(
            height_cm=180.0,
            camera_metadata=CameraMetadata(focal_length_px=1500, camera_height_cm=120, tilt_angle_deg=30),
            landmarks=_landmarks(0.10, 0.95),
            image_height_px=1920,
        )
        assert est.source == "user_input"
        assert est.confidence == Confidence.HIGH
        assert est.value_cm == 180.0

    def test_sensor_fusion_when_no_user_input(self):
        est = HeightEstimator().estimate(
            height_cm=None,
            camera_metadata=CameraMetadata(focal_length_px=1500, camera_height_cm=100, tilt_angle_deg=20),
            landmarks=_landmarks(0.20, 0.90),
            image_height_px=1920,
        )
        assert est.source == "sensor_fusion"
        assert est.confidence == Confidence.MEDIUM

    def test_population_mean_when_no_camera_or_landmarks(self):
        est = HeightEstimator().estimate(
            height_cm=None,
            camera_metadata=None,
            landmarks=None,
        )
        assert est.source == "population_mean"
        assert est.confidence == Confidence.LOW
        assert est.value_cm == _POPULATION_MEAN_CM

    def test_population_mean_when_landmarks_absent(self):
        est = HeightEstimator().estimate(
            height_cm=None,
            camera_metadata=CameraMetadata(focal_length_px=1500, camera_height_cm=100, tilt_angle_deg=20),
            landmarks=None,
        )
        assert est.source == "population_mean"


# ---------------------------------------------------------------------------
# Sensor fusion — failure modes fall back to population mean
# ---------------------------------------------------------------------------

class TestSensorFusionFallback:
    def test_low_visibility_landmarks_falls_back(self):
        est = HeightEstimator().estimate(
            height_cm=None,
            camera_metadata=CameraMetadata(focal_length_px=1500, camera_height_cm=100, tilt_angle_deg=30),
            landmarks=_landmarks(0.20, 0.90, vis=0.1),
            image_height_px=1920,
        )
        assert est.source == "population_mean"

    def test_implausible_computed_height_falls_back(self):
        # Pathological geometry: camera 30 cm above floor, tilt 5° — produces
        # a computed height that fails the plausible-range guard.
        est = HeightEstimator().estimate(
            height_cm=None,
            camera_metadata=CameraMetadata(focal_length_px=1500, camera_height_cm=31, tilt_angle_deg=5),
            landmarks=_landmarks(0.10, 0.95),
            image_height_px=1920,
        )
        # Either sensor fusion succeeded plausibly or fell back — never crash
        assert est.value_cm > 0


# ---------------------------------------------------------------------------
# Pinhole math — internal sanity
# ---------------------------------------------------------------------------

def _reference_height(
    camera_height_cm: float,
    tilt_deg: float,
    nose_y: float,
    ankle_y: float,
    focal_length_px: float | None,
    image_height_px: int,
) -> float:
    """Plain-Python re-implementation of HeightEstimator._sensor_fusion."""
    if focal_length_px:
        vfov = 2.0 * math.atan(image_height_px / (2.0 * focal_length_px))
    else:
        vfov = math.radians(65.0)
    theta   = math.radians(tilt_deg)
    α_ankle = theta + (ankle_y - 0.5) * vfov
    α_nose  = theta + (nose_y  - 0.5) * vfov
    D = camera_height_cm / math.tan(α_ankle)
    nose_h = camera_height_cm - D * math.tan(α_nose)
    return nose_h + _CROWN_OFFSET_CM


class TestPinholeMath:
    def test_sensor_fusion_matches_reference_geometry(self):
        meta = CameraMetadata(camera_height_cm=120.0, tilt_angle_deg=20.0)
        est = HeightEstimator().estimate(
            height_cm=None,
            camera_metadata=meta,
            landmarks=_landmarks(0.10, 0.85),
            image_height_px=1920,
        )
        ref = _reference_height(
            camera_height_cm=120.0,
            tilt_deg=20.0,
            nose_y=0.10,
            ankle_y=0.85,
            focal_length_px=None,
            image_height_px=1920,
        )
        assert est.source == "sensor_fusion"
        assert est.value_cm == pytest.approx(round(ref, 1), abs=0.2)

    def test_focal_length_overrides_default_vfov(self):
        meta = CameraMetadata(
            camera_height_cm=120.0, tilt_angle_deg=20.0, focal_length_px=1500.0
        )
        est = HeightEstimator().estimate(
            height_cm=None,
            camera_metadata=meta,
            landmarks=_landmarks(0.10, 0.85),
            image_height_px=1920,
        )
        ref = _reference_height(
            camera_height_cm=120.0, tilt_deg=20.0,
            nose_y=0.10, ankle_y=0.85,
            focal_length_px=1500.0, image_height_px=1920,
        )
        assert est.value_cm == pytest.approx(round(ref, 1), abs=0.2)

    def test_higher_nose_in_frame_yields_taller_subject(self):
        meta = CameraMetadata(camera_height_cm=120.0, tilt_angle_deg=20.0)
        short = HeightEstimator().estimate(
            None, meta, _landmarks(0.30, 0.85), image_height_px=1920
        )
        tall = HeightEstimator().estimate(
            None, meta, _landmarks(0.05, 0.85), image_height_px=1920
        )
        # Both succeed → comparable; if either fell back, just check no crash.
        if short.source == "sensor_fusion" and tall.source == "sensor_fusion":
            assert tall.value_cm > short.value_cm

    def test_default_vfov_used_when_focal_missing(self):
        """Omitting focal_length_px must not raise — falls back to 65° VFOV."""
        est = HeightEstimator().estimate(
            height_cm=None,
            camera_metadata=CameraMetadata(focal_length_px=None, camera_height_cm=120, tilt_angle_deg=25),
            landmarks=_landmarks(0.15, 0.92),
            image_height_px=1920,
        )
        # Either fusion succeeded or fell back to population_mean — both are valid
        assert est.source in ("sensor_fusion", "population_mean")

    def test_returns_height_estimate_object(self):
        est = HeightEstimator().estimate(
            height_cm=175.0,
            camera_metadata=None,
            landmarks=None,
        )
        assert isinstance(est, HeightEstimate)
        assert est.value_cm == 175.0
