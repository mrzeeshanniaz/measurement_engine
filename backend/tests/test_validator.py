"""
Tests for app.measurement_engine.scan.validator — all 5 passes.
"""
from __future__ import annotations

import pytest

from app.measurement_engine.scan.schemas import (
    Confidence,
    GarmentType,
    MeasurementField,
    ScanMeasurements,
)
from app.measurement_engine.scan.validator import validate
from tests.conftest import _field, _low, _null


# ---------------------------------------------------------------------------
# Pass 1 — Hard limits
# ---------------------------------------------------------------------------

class TestHardLimits:
    def test_within_limits_no_error(self, typical_male_measurements):
        result = validate(typical_male_measurements, 175.0)
        hard_errs = [i for i in result.issues if i.code.startswith("hard_limit")]
        assert hard_errs == []

    def test_waist_too_small_raises_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M03_waist": _field(5.0)}
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert "hard_limit_m03" in codes
        assert not result.is_valid

    def test_waist_too_large_raises_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M03_waist": _field(500.0)}
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert "hard_limit_m03" in codes

    def test_null_value_skipped(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M07_bicep": _null()}
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert not any("hard_limit_m07" in c for c in codes)


# ---------------------------------------------------------------------------
# Pass 2 — Population norms
# ---------------------------------------------------------------------------

class TestNorms:
    def test_typical_measurements_no_outlier_errors(self, typical_male_measurements):
        result = validate(typical_male_measurements, 175.0)
        norm_errs = [i for i in result.issues if "outlier_error" in i.code]
        assert norm_errs == [], f"Unexpected outlier errors: {norm_errs}"

    def test_extremely_large_chest_is_outlier_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M01_chest": _field(250.0)}
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert "outlier_error_m01" in codes

    def test_moderately_large_chest_is_outlier_warning(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M01_chest": _field(145.0)}
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert any("outlier_warn_m01" in c or "outlier_error_m01" in c for c in codes)


# ---------------------------------------------------------------------------
# Pass 3 — Cross-measurement rules
# ---------------------------------------------------------------------------

class TestCrossRules:
    def test_waist_gt_chest_raises_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={
                "M01_chest": _field(80.0),
                "M03_waist": _field(95.0),
            }
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert "waist_lt_chest" in codes

    def test_inseam_gt_outseam_raises_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={
                "M21_inseam": _field(110.0),
                "M22_outseam": _field(105.0),
            }
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert "inseam_lt_outseam" in codes

    def test_sleeve_shorter_than_elbow_sleeve_raises_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={
                "M19_sleeve_length": _field(30.0),
                "M20_sleeve_length_elbow": _field(36.0),
            }
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert "sleeve_gt_elbow_sleeve" in codes

    def test_valid_measurements_no_cross_errors(self, typical_male_measurements):
        result = validate(typical_male_measurements, 175.0)
        cross_errs = [i for i in result.issues if i.severity == "error" and "hard_limit" not in i.code and "outlier" not in i.code and "confidence" not in i.code and "mesh" not in i.code and "missing" not in i.code]
        assert cross_errs == []

    def test_thigh_gt_hip_raises_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={
                "M09_thigh": _field(120.0),
                "M05_hips": _field(98.0),
            }
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert "thigh_lt_hip" in codes

    def test_neck_gt_chest_raises_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={
                "M06_neck": _field(60.0),
                "M01_chest": _field(100.0),
            }
        )
        result = validate(m, 175.0)
        codes = [i.code for i in result.issues]
        assert "neck_lt_chest" in codes


# ---------------------------------------------------------------------------
# Pass 4 — Garment required fields
# ---------------------------------------------------------------------------

class TestGarmentRequired:
    def test_kameez_missing_required_field_raises_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M17_kameez_length": _null()}
        )
        result = validate(m, 175.0, garment_type=GarmentType.KAMEEZ)
        codes = [i.code for i in result.issues]
        assert any("missing_required_m17" in c for c in codes)
        assert not result.is_valid

    def test_no_garment_type_skips_pass4(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M17_kameez_length": _null()}
        )
        result = validate(m, 175.0, garment_type=None)
        codes = [i.code for i in result.issues]
        assert not any("missing_required" in c for c in codes)

    def test_trouser_missing_inseam_raises_error(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M21_inseam": _null()}
        )
        result = validate(m, 175.0, garment_type=GarmentType.TROUSER)
        codes = [i.code for i in result.issues]
        assert any("missing_required_m21" in c for c in codes)


# ---------------------------------------------------------------------------
# Pass 5 — Mesh quality gate
# ---------------------------------------------------------------------------

class TestMeshQualityGate:
    def test_high_fit_score_no_issue(self, typical_male_measurements):
        result = validate(typical_male_measurements, 175.0, mesh_fit_score=0.90)
        codes = [i.code for i in result.issues]
        assert "mesh_fit_low" not in codes
        assert "mesh_fit_poor" not in codes

    def test_default_fit_score_no_issue(self, typical_male_measurements):
        result = validate(typical_male_measurements, 175.0)
        codes = [i.code for i in result.issues]
        assert "mesh_fit_low" not in codes

    def test_low_fit_score_warning(self, typical_male_measurements):
        result = validate(typical_male_measurements, 175.0, mesh_fit_score=0.45)
        codes = [i.code for i in result.issues]
        assert "mesh_fit_low" in codes

    def test_very_low_fit_score_error(self, typical_male_measurements):
        result = validate(typical_male_measurements, 175.0, mesh_fit_score=0.20)
        codes = [i.code for i in result.issues]
        assert "mesh_fit_poor" in codes
        assert not result.is_valid

    def test_borderline_fit_score(self, typical_male_measurements):
        result = validate(typical_male_measurements, 175.0, mesh_fit_score=0.55)
        codes = [i.code for i in result.issues]
        assert "mesh_fit_low" not in codes
        assert "mesh_fit_poor" not in codes


# ---------------------------------------------------------------------------
# ValidationResult shape
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_can_order_false_when_critical_field_low(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M01_chest": MeasurementField(
                value_cm=5.0, confidence=Confidence.LOW, source="height_ratio"
            )}
        )
        result = validate(m, 175.0)
        assert not result.can_order

    def test_can_order_true_for_clean_scan(self, typical_male_measurements):
        result = validate(typical_male_measurements, 175.0)
        assert result.can_order

    def test_rescan_poses_deduplicated(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={
                "M01_chest": _field(5.0),
                "M03_waist": _field(5.0),
            }
        )
        result = validate(m, 175.0)
        assert len(result.rescan_poses) == len(set(result.rescan_poses))

    def test_summary_contains_error_count(self, typical_male_measurements):
        m = typical_male_measurements.model_copy(
            update={"M03_waist": _field(5.0)}
        )
        result = validate(m, 175.0)
        assert "error" in result.summary.lower()
