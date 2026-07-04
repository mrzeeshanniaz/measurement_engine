"""
Tests for app.measurement_engine.scan.garments — garment profiles and ease allowances.
"""
from __future__ import annotations

import pytest

from app.measurement_engine.scan.garments import (
    EASE_ALLOWANCES,
    GARMENT_REQUIRED_FIELDS,
    apply_garment_profile,
)
from app.measurement_engine.scan.schemas import FitStyle, GarmentType
from tests.conftest import _field


# ---------------------------------------------------------------------------
# GARMENT_REQUIRED_FIELDS
# ---------------------------------------------------------------------------

class TestGarmentRequiredFields:
    def test_all_12_garments_have_required_fields(self):
        assert len(GARMENT_REQUIRED_FIELDS) == 12

    def test_kameez_requires_chest_and_length(self):
        req = GARMENT_REQUIRED_FIELDS[GarmentType.KAMEEZ]
        assert "M01" in req   # chest
        assert "M17" in req   # kameez length

    def test_trouser_requires_waist_inseam_outseam(self):
        req = GARMENT_REQUIRED_FIELDS[GarmentType.TROUSER]
        assert "M03" in req   # waist
        assert "M21" in req   # inseam
        assert "M22" in req   # outseam

    def test_sherwani_requires_sleeve_length(self):
        req = GARMENT_REQUIRED_FIELDS[GarmentType.SHERWANI]
        assert "M19" in req   # sleeve length

    def test_blouse_does_not_require_inseam(self):
        req = GARMENT_REQUIRED_FIELDS[GarmentType.BLOUSE]
        assert "M21" not in req


# ---------------------------------------------------------------------------
# apply_garment_profile — required flags
# ---------------------------------------------------------------------------

class TestApplyGarmentProfileRequiredFlags:
    def test_no_garment_type_leaves_flags_none(self, typical_male_measurements):
        result = apply_garment_profile(typical_male_measurements, None, None)
        assert result.M01_chest.is_required_for_garment is None
        assert result.M17_kameez_length.is_required_for_garment is None

    def test_kameez_flags_required_fields_true(self, typical_male_measurements):
        result = apply_garment_profile(
            typical_male_measurements, GarmentType.KAMEEZ, None
        )
        assert result.M01_chest.is_required_for_garment is True
        assert result.M17_kameez_length.is_required_for_garment is True

    def test_kameez_flags_non_required_fields_false(self, typical_male_measurements):
        result = apply_garment_profile(
            typical_male_measurements, GarmentType.KAMEEZ, None
        )
        req = GARMENT_REQUIRED_FIELDS[GarmentType.KAMEEZ]
        from app.measurement_engine.scan.schemas import ScanMeasurements
        for fname in ScanMeasurements.model_fields:
            f = getattr(result, fname)
            code = fname.split("_")[0]
            if code not in req:
                assert f.is_required_for_garment is False, (
                    f"{fname} should be False but is {f.is_required_for_garment}"
                )


# ---------------------------------------------------------------------------
# apply_garment_profile — ease allowances
# ---------------------------------------------------------------------------

class TestApplyGarmentProfileEase:
    def test_no_fit_style_leaves_ease_none(self, typical_male_measurements):
        result = apply_garment_profile(
            typical_male_measurements, GarmentType.KAMEEZ, None
        )
        assert result.M01_chest.ease_cm is None
        assert result.M01_chest.cutting_value_cm is None

    def test_regular_fit_adds_ease_to_chest(self, typical_male_measurements):
        result = apply_garment_profile(
            typical_male_measurements, GarmentType.KAMEEZ, FitStyle.REGULAR
        )
        chest = result.M01_chest
        assert chest.ease_cm is not None
        assert chest.ease_cm > 0
        assert chest.cutting_value_cm == round(chest.value_cm + chest.ease_cm, 1)

    def test_relaxed_ease_gt_regular_ease_gt_fitted_ease(self, typical_male_measurements):
        def _ease(fit_style: FitStyle) -> float:
            r = apply_garment_profile(
                typical_male_measurements, GarmentType.KAMEEZ, fit_style
            )
            return r.M01_chest.ease_cm or 0.0

        assert _ease(FitStyle.RELAXED) > _ease(FitStyle.REGULAR) > _ease(FitStyle.FITTED)

    def test_cutting_value_equals_value_plus_ease(self, typical_male_measurements):
        result = apply_garment_profile(
            typical_male_measurements, GarmentType.KAMEEZ, FitStyle.REGULAR
        )
        for fname in type(result).model_fields:
            f = getattr(result, fname)
            if f.ease_cm is not None and f.value_cm is not None:
                expected = round(f.value_cm + f.ease_cm, 1)
                assert f.cutting_value_cm == expected, (
                    f"{fname}: {f.value_cm} + {f.ease_cm} ≠ {f.cutting_value_cm}"
                )

    def test_length_fields_have_no_ease(self, typical_male_measurements):
        result = apply_garment_profile(
            typical_male_measurements, GarmentType.KAMEEZ, FitStyle.REGULAR
        )
        # M14 total height, M17 kameez length — lengths don't get circumference ease
        # M17 may get a hem allowance; M14 never gets ease
        assert result.M14_total_height.ease_cm is None


# ---------------------------------------------------------------------------
# EASE_ALLOWANCES structure
# ---------------------------------------------------------------------------

class TestEaseAllowances:
    def test_all_fit_styles_present_for_circumferences(self):
        for code in ("M01", "M03", "M05"):
            assert code in EASE_ALLOWANCES, f"{code} missing from EASE_ALLOWANCES"
            for fit in FitStyle:
                assert fit in EASE_ALLOWANCES[code], (
                    f"{code}/{fit} missing from EASE_ALLOWANCES"
                )

    def test_ease_values_are_non_negative(self):
        for code, by_fit in EASE_ALLOWANCES.items():
            for fit, val in by_fit.items():
                assert val >= 0, f"Negative ease for {code}/{fit}: {val}"
