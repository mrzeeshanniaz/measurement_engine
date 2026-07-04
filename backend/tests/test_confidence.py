"""
Tests for app.measurement_engine.scan.confidence — score_field and overall_confidence.
"""
from __future__ import annotations

import pytest

from app.measurement_engine.scan.confidence import overall_confidence, score_field
from app.measurement_engine.scan.schemas import Confidence, ScanMeasurements
from tests.conftest import _field, _low


class TestScoreField:
    # ------------------------------------------------------------------
    # None value → always LOW
    # ------------------------------------------------------------------
    def test_none_value_always_low(self):
        f = score_field(None, "smpl_mesh", frame_composite=0.95)
        assert f.confidence == Confidence.LOW
        assert f.value_cm is None

    # ------------------------------------------------------------------
    # smpl_anthro_full — best-accuracy path
    # ------------------------------------------------------------------
    def test_smpl_anthro_full_high_composite_is_high(self):
        f = score_field(95.0, "smpl_anthro_full", frame_composite=0.80)
        assert f.confidence == Confidence.HIGH

    def test_smpl_anthro_full_low_composite_is_medium(self):
        f = score_field(95.0, "smpl_anthro_full", frame_composite=0.40)
        assert f.confidence == Confidence.MEDIUM

    # ------------------------------------------------------------------
    # smpl_mesh — standard mesh path
    # ------------------------------------------------------------------
    def test_smpl_mesh_high_composite_is_high(self):
        f = score_field(95.0, "smpl_mesh", frame_composite=0.75)
        assert f.confidence == Confidence.HIGH

    def test_smpl_mesh_medium_composite_is_medium(self):
        f = score_field(95.0, "smpl_mesh", frame_composite=0.55)
        assert f.confidence == Confidence.MEDIUM

    def test_smpl_mesh_poor_composite_is_low(self):
        f = score_field(95.0, "smpl_mesh", frame_composite=0.30)
        assert f.confidence == Confidence.LOW

    # ------------------------------------------------------------------
    # Mesh fit score ceiling
    # ------------------------------------------------------------------
    def test_poor_mesh_fit_caps_smpl_mesh_to_medium(self):
        # mesh_fit_score=0.20 → ceiling = 0.30 + 0.70*0.20 = 0.44
        # frame_composite=0.85 → effective = min(0.85, 0.44) = 0.44 → MEDIUM
        f = score_field(95.0, "smpl_mesh", frame_composite=0.85, mesh_fit_score=0.20)
        assert f.confidence in (Confidence.MEDIUM, Confidence.LOW)

    def test_perfect_mesh_fit_does_not_degrade(self):
        f = score_field(95.0, "smpl_mesh", frame_composite=0.80, mesh_fit_score=1.0)
        assert f.confidence == Confidence.HIGH

    # ------------------------------------------------------------------
    # landmark source
    # ------------------------------------------------------------------
    def test_landmark_good_visibility_and_composite_is_high(self):
        f = score_field(40.0, "landmark", frame_composite=0.70, landmark_visibility=0.85)
        assert f.confidence == Confidence.HIGH

    def test_landmark_poor_visibility_is_low(self):
        f = score_field(40.0, "landmark", frame_composite=0.70, landmark_visibility=0.30)
        assert f.confidence == Confidence.LOW

    # ------------------------------------------------------------------
    # height_ratio fallback
    # ------------------------------------------------------------------
    def test_height_ratio_always_low(self):
        f = score_field(40.0, "height_ratio", frame_composite=0.99)
        assert f.confidence == Confidence.LOW

    # ------------------------------------------------------------------
    # Value rounding
    # ------------------------------------------------------------------
    def test_value_rounded_to_1dp(self):
        f = score_field(95.123456, "smpl_mesh", frame_composite=0.80)
        assert f.value_cm == 95.1


class TestOverallConfidence:
    def test_all_high_fields_returns_high(self, typical_male_measurements):
        conf = overall_confidence(typical_male_measurements)
        assert conf == Confidence.HIGH

    def test_many_low_fields_returns_low(self, typical_male_measurements):
        updates = {}
        low_field = _low(50.0)
        for fname in list(ScanMeasurements.model_fields.keys())[:12]:
            updates[fname] = low_field
        m = typical_male_measurements.model_copy(update=updates)
        conf = overall_confidence(m)
        assert conf == Confidence.LOW

    def test_moderate_low_count_returns_medium(self, typical_male_measurements):
        updates = {}
        low_field = _low(50.0)
        for fname in list(ScanMeasurements.model_fields.keys())[:5]:
            updates[fname] = low_field
        m = typical_male_measurements.model_copy(update=updates)
        conf = overall_confidence(m)
        assert conf in (Confidence.MEDIUM, Confidence.LOW)
