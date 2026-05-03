"""
Tests for MeasurementCalculator and MeasurementResult.

All tests use synthetic PoseResult data so that MediaPipe is not required.
"""

import pytest

from measurement_engine.measurements import MeasurementCalculator, MeasurementResult
from measurement_engine.pose_detector import Landmark, PoseResult


def _full_pose(image_width: int = 480, image_height: int = 640) -> PoseResult:
    """Synthetic full-body pose for a person viewed from the front."""
    landmarks = {
        0:  Landmark(x=0.50, y=0.05, z=0.00, visibility=0.95),  # Nose
        11: Landmark(x=0.38, y=0.25, z=0.00, visibility=0.95),  # Left shoulder
        12: Landmark(x=0.62, y=0.25, z=0.00, visibility=0.95),  # Right shoulder
        13: Landmark(x=0.32, y=0.40, z=0.00, visibility=0.90),  # Left elbow
        14: Landmark(x=0.68, y=0.40, z=0.00, visibility=0.90),  # Right elbow
        15: Landmark(x=0.30, y=0.55, z=0.00, visibility=0.85),  # Left wrist
        16: Landmark(x=0.70, y=0.55, z=0.00, visibility=0.85),  # Right wrist
        23: Landmark(x=0.41, y=0.56, z=0.00, visibility=0.90),  # Left hip
        24: Landmark(x=0.59, y=0.56, z=0.00, visibility=0.90),  # Right hip
        25: Landmark(x=0.42, y=0.72, z=0.00, visibility=0.85),  # Left knee
        26: Landmark(x=0.58, y=0.72, z=0.00, visibility=0.85),  # Right knee
        27: Landmark(x=0.43, y=0.90, z=0.00, visibility=0.80),  # Left ankle
        28: Landmark(x=0.57, y=0.90, z=0.00, visibility=0.80),  # Right ankle
    }
    return PoseResult(
        landmarks=landmarks,
        image_width=image_width,
        image_height=image_height,
    )


def _no_legs_pose() -> PoseResult:
    """Pose with only upper-body landmarks visible (no legs)."""
    landmarks = {
        0:  Landmark(x=0.50, y=0.05, z=0.00, visibility=0.95),
        11: Landmark(x=0.38, y=0.25, z=0.00, visibility=0.95),
        12: Landmark(x=0.62, y=0.25, z=0.00, visibility=0.95),
        23: Landmark(x=0.41, y=0.56, z=0.00, visibility=0.90),
        24: Landmark(x=0.59, y=0.56, z=0.00, visibility=0.90),
        # Knees and ankles absent
    }
    return PoseResult(landmarks=landmarks, image_width=480, image_height=640)


# ------------------------------------------------------------------ #
# MeasurementResult                                                    #
# ------------------------------------------------------------------ #

class TestMeasurementResult:
    def test_defaults(self):
        r = MeasurementResult()
        assert r.units == "pixels"
        assert r.confidence == 0.0
        assert r.shoulder_width is None

    def test_to_dict_keys(self):
        r = MeasurementResult(shoulder_width=45.0, units="cm", confidence=0.9)
        d = r.to_dict()
        expected_keys = {
            "shoulder_width", "chest_circumference", "waist_circumference",
            "hip_circumference", "inseam_length", "sleeve_length",
            "torso_length", "back_length", "total_height", "neck_circumference",
            "units", "confidence",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        r = MeasurementResult(shoulder_width=45.0, units="cm", confidence=0.9)
        d = r.to_dict()
        assert d["shoulder_width"] == 45.0
        assert d["units"] == "cm"
        assert d["confidence"] == 0.9

    def test_repr_contains_class_name(self):
        r = MeasurementResult(shoulder_width=45.0, units="cm", confidence=0.8)
        assert "MeasurementResult" in repr(r)

    def test_repr_shows_non_none_values(self):
        r = MeasurementResult(shoulder_width=45.0, units="cm", confidence=0.8)
        assert "shoulder_width" in repr(r)

    def test_repr_omits_none_values(self):
        r = MeasurementResult(shoulder_width=None, units="cm", confidence=0.8)
        assert "shoulder_width" not in repr(r)


# ------------------------------------------------------------------ #
# MeasurementCalculator – confidence                                   #
# ------------------------------------------------------------------ #

class TestMeasurementCalculatorConfidence:
    def test_full_pose_high_confidence(self):
        calc = MeasurementCalculator(_full_pose())
        result = calc.calculate()
        assert result.confidence >= 0.75

    def test_no_legs_lower_confidence(self):
        calc_full = MeasurementCalculator(_full_pose())
        calc_partial = MeasurementCalculator(_no_legs_pose())
        assert calc_partial.calculate().confidence < calc_full.calculate().confidence

    def test_confidence_in_range(self):
        result = MeasurementCalculator(_full_pose()).calculate()
        assert 0.0 <= result.confidence <= 1.0


# ------------------------------------------------------------------ #
# MeasurementCalculator – pixel mode                                   #
# ------------------------------------------------------------------ #

class TestMeasurementCalculatorPixelMode:
    def setup_method(self):
        self.result = MeasurementCalculator(_full_pose()).calculate()

    def test_units_is_pixels(self):
        assert self.result.units == "pixels"

    def test_scale_factor_is_none(self):
        assert self.result.scale_factor is None

    def test_shoulder_width_positive(self):
        assert self.result.shoulder_width is not None
        assert self.result.shoulder_width > 0

    def test_torso_length_positive(self):
        assert self.result.torso_length is not None
        assert self.result.torso_length > 0

    def test_back_length_equals_torso_length(self):
        assert self.result.back_length == self.result.torso_length

    def test_inseam_length_positive(self):
        assert self.result.inseam_length is not None
        assert self.result.inseam_length > 0

    def test_sleeve_length_positive(self):
        assert self.result.sleeve_length is not None
        assert self.result.sleeve_length > 0

    def test_total_height_positive(self):
        assert self.result.total_height is not None
        assert self.result.total_height > 0

    def test_chest_circumference_positive(self):
        assert self.result.chest_circumference is not None
        assert self.result.chest_circumference > 0

    def test_waist_circumference_positive(self):
        assert self.result.waist_circumference is not None
        assert self.result.waist_circumference > 0

    def test_hip_circumference_positive(self):
        assert self.result.hip_circumference is not None
        assert self.result.hip_circumference > 0

    def test_neck_circumference_positive(self):
        assert self.result.neck_circumference is not None
        assert self.result.neck_circumference > 0

    def test_hip_larger_than_waist(self):
        # Hip circumference is typically larger than waist circumference
        assert self.result.hip_circumference > self.result.waist_circumference

    def test_inseam_longer_than_torso(self):
        # For a typical person, legs are longer than the torso
        assert self.result.inseam_length > self.result.torso_length


# ------------------------------------------------------------------ #
# MeasurementCalculator – centimetre mode                              #
# ------------------------------------------------------------------ #

class TestMeasurementCalculatorCmMode:
    def setup_method(self):
        self.result = MeasurementCalculator(_full_pose()).calculate(
            person_height_cm=175.0
        )

    def test_units_is_cm(self):
        assert self.result.units == "cm"

    def test_scale_factor_set(self):
        assert self.result.scale_factor is not None
        assert self.result.scale_factor > 0

    def test_total_height_close_to_175(self):
        # The engine calibrates using the detected pixel height against 175 cm.
        assert self.result.total_height == pytest.approx(175.0, abs=5.0)

    def test_shoulder_width_realistic_cm(self):
        # Typical shoulder width for a 175 cm person: 38–50 cm
        assert 30 < self.result.shoulder_width < 60

    def test_chest_circumference_realistic_cm(self):
        # Realistic range: 80–130 cm
        assert 75 < self.result.chest_circumference < 140

    def test_hip_circumference_realistic_cm(self):
        assert 60 < self.result.hip_circumference < 140

    def test_inseam_realistic_cm(self):
        assert 60 < self.result.inseam_length < 110


# ------------------------------------------------------------------ #
# MeasurementCalculator – missing landmarks                            #
# ------------------------------------------------------------------ #

class TestMeasurementCalculatorMissingLandmarks:
    def test_no_legs_inseam_none(self):
        result = MeasurementCalculator(_no_legs_pose()).calculate()
        assert result.inseam_length is None

    def test_no_legs_total_height_none(self):
        result = MeasurementCalculator(_no_legs_pose()).calculate()
        assert result.total_height is None

    def test_no_legs_shoulder_width_still_computed(self):
        result = MeasurementCalculator(_no_legs_pose()).calculate()
        assert result.shoulder_width is not None
