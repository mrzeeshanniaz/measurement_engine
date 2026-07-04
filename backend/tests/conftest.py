"""
Shared fixtures for TailorSync scan module tests.
"""
from __future__ import annotations

import pytest

from app.measurement_engine.scan.schemas import (
    Confidence,
    MeasurementField,
    ScanMeasurements,
)


def _field(value: float, confidence: Confidence = Confidence.HIGH, source: str = "smpl_mesh") -> MeasurementField:
    return MeasurementField(value_cm=value, confidence=confidence, source=source)


def _low(value: float) -> MeasurementField:
    return _field(value, Confidence.LOW, "height_ratio")


def _med(value: float) -> MeasurementField:
    return _field(value, Confidence.MEDIUM, "landmark")


def _null() -> MeasurementField:
    return MeasurementField(value_cm=None, confidence=Confidence.LOW, source="height_ratio")


@pytest.fixture
def typical_male_measurements() -> ScanMeasurements:
    """Realistic male measurements for a 175 cm person."""
    return ScanMeasurements(
        M01_chest=_field(100.0),
        M02_under_bust=_field(90.0),
        M03_waist=_field(84.0),
        M04_abdomen=_field(88.0),
        M05_hips=_field(98.0),
        M06_neck=_field(38.0),
        M07_bicep=_field(34.0),
        M08_wrist=_field(17.0),
        M09_thigh=_field(56.0),
        M10_mid_thigh=_field(46.0),
        M11_knee=_field(38.0),
        M12_calf=_field(36.0),
        M13_ankle=_field(23.0),
        M14_total_height=_field(175.0, Confidence.HIGH, "user_input"),
        M15_shoulder_to_waist_front=_field(43.0),
        M16_shoulder_to_waist_back=_field(42.0),
        M17_kameez_length=_field(105.0),
        M18_dress_length=_field(130.0),
        M19_sleeve_length=_field(62.0),
        M20_sleeve_length_elbow=_field(36.0),
        M21_inseam=_field(78.0),
        M22_outseam=_field(105.0),
        M23_crotch_depth_front=_field(28.0),
        M24_crotch_depth_back=_field(32.0),
        M25_torso_length=_field(55.0),
        M26_shoulder_width=_field(44.0),
        M27_chest_width=_field(38.0),
        M28_back_width=_field(36.0),
        M29_hip_width=_field(36.0),
        M30_chest_depth=_field(22.0),
        M31_waist_depth=_field(19.0),
        M32_armhole_depth=_field(22.0),
    )
