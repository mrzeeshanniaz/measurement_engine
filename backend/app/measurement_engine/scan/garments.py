"""
Garment profile data for F8 (required-field filtering) and F9 (ease allowances).

GARMENT_REQUIRED_FIELDS maps each GarmentType to the set of M-codes that
must be present for the tailor to cut that garment.  Missing required fields
become ValidationIssue ERRORs in Pass 4 of the validator.

EASE_ALLOWANCES maps each M-code to {FitStyle → cm} so that cutting_value_cm
= value_cm + ease for each circumference or width measurement.

Ease values (cm) follow standard South Asian tailoring practice:
  FITTED   — body-con; minimal ease so garment just clears the body
  REGULAR  — comfortable everyday fit
  RELAXED  — loose / flowy; maximum ease

Length and depth measurements carry 0 ease (cut to exact body dimension)
except kameez/dress/coat lengths which get +2 cm for a standard hem allowance.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.measurement_engine.scan.schemas import ScanMeasurements

from app.measurement_engine.scan.schemas import (
    GarmentType,
    FitStyle,
    MeasurementField,
)


# ---------------------------------------------------------------------------
# Required measurements per garment type
# ---------------------------------------------------------------------------

GARMENT_REQUIRED_FIELDS: dict[GarmentType, set[str]] = {
    GarmentType.KAMEEZ: {
        "M01",  # chest
        "M03",  # waist
        "M05",  # hips
        "M06",  # neck
        "M15",  # shoulder to waist front
        "M16",  # shoulder to waist back
        "M17",  # kameez length
        "M19",  # sleeve length
        "M26",  # shoulder width
    },
    GarmentType.KURTA: {
        "M01",  # chest
        "M03",  # waist
        "M06",  # neck
        "M17",  # kameez length
        "M19",  # sleeve length
        "M26",  # shoulder width
    },
    GarmentType.SHALWAR: {
        "M03",  # waist
        "M05",  # hips
        "M09",  # thigh
        "M21",  # inseam
        "M22",  # outseam
        "M23",  # crotch depth front
    },
    GarmentType.TROUSER: {
        "M03",  # waist
        "M05",  # hips
        "M09",  # thigh
        "M11",  # knee
        "M21",  # inseam
        "M22",  # outseam
        "M23",  # crotch depth front
    },
    GarmentType.SHIRT: {
        "M01",  # chest
        "M03",  # waist
        "M06",  # neck
        "M17",  # kameez length
        "M19",  # sleeve length
        "M26",  # shoulder width
    },
    GarmentType.SHERWANI: {
        "M01",  # chest
        "M03",  # waist
        "M05",  # hips
        "M06",  # neck
        "M15",  # shoulder to waist front
        "M16",  # shoulder to waist back
        "M17",  # kameez length
        "M19",  # sleeve length
        "M26",  # shoulder width
    },
    GarmentType.DRESS: {
        "M01",  # chest
        "M02",  # under-bust
        "M03",  # waist
        "M05",  # hips
        "M15",  # shoulder to waist front
        "M18",  # dress length
        "M26",  # shoulder width
    },
    GarmentType.SUIT_JACKET: {
        "M01",  # chest
        "M03",  # waist
        "M15",  # shoulder to waist front
        "M16",  # shoulder to waist back
        "M17",  # kameez length
        "M19",  # sleeve length
        "M26",  # shoulder width
        "M27",  # chest width
        "M28",  # back width
    },
    GarmentType.BLOUSE: {
        "M01",  # chest
        "M02",  # under-bust
        "M03",  # waist
        "M15",  # shoulder to waist front
        "M26",  # shoulder width
    },
    GarmentType.SKIRT: {
        "M03",  # waist
        "M05",  # hips
        "M18",  # dress / skirt length
    },
    GarmentType.LEHENGA_SKIRT: {
        "M03",  # waist
        "M05",  # hips
        "M18",  # skirt length
    },
    GarmentType.COAT: {
        "M01",  # chest
        "M03",  # waist
        "M17",  # kameez/coat length
        "M19",  # sleeve length
        "M26",  # shoulder width
    },
}


# ---------------------------------------------------------------------------
# Ease allowances (cm) per measurement code and fit style
# ---------------------------------------------------------------------------
# Only measurements that have non-zero ease for at least one fit style are listed.
# Measurements absent from this table get 0 ease for all fit styles.

EASE_ALLOWANCES: dict[str, dict[FitStyle, float]] = {
    # Circumferences -------------------------------------------------------
    "M01": {FitStyle.FITTED: 4.0,  FitStyle.REGULAR: 8.0,  FitStyle.RELAXED: 14.0},  # chest
    "M02": {FitStyle.FITTED: 2.0,  FitStyle.REGULAR: 4.0,  FitStyle.RELAXED: 6.0},   # under-bust
    "M03": {FitStyle.FITTED: 2.0,  FitStyle.REGULAR: 4.0,  FitStyle.RELAXED: 8.0},   # waist
    "M04": {FitStyle.FITTED: 2.0,  FitStyle.REGULAR: 4.0,  FitStyle.RELAXED: 8.0},   # abdomen
    "M05": {FitStyle.FITTED: 3.0,  FitStyle.REGULAR: 6.0,  FitStyle.RELAXED: 10.0},  # hips
    "M06": {FitStyle.FITTED: 1.0,  FitStyle.REGULAR: 2.0,  FitStyle.RELAXED: 3.0},   # neck
    "M07": {FitStyle.FITTED: 3.0,  FitStyle.REGULAR: 5.0,  FitStyle.RELAXED: 8.0},   # bicep
    "M09": {FitStyle.FITTED: 4.0,  FitStyle.REGULAR: 6.0,  FitStyle.RELAXED: 10.0},  # thigh
    "M10": {FitStyle.FITTED: 3.0,  FitStyle.REGULAR: 5.0,  FitStyle.RELAXED: 8.0},   # mid-thigh
    "M11": {FitStyle.FITTED: 2.0,  FitStyle.REGULAR: 3.0,  FitStyle.RELAXED: 5.0},   # knee
    "M12": {FitStyle.FITTED: 2.0,  FitStyle.REGULAR: 3.0,  FitStyle.RELAXED: 5.0},   # calf

    # Widths ---------------------------------------------------------------
    "M26": {FitStyle.FITTED: 0.5,  FitStyle.REGULAR: 1.0,  FitStyle.RELAXED: 2.0},   # shoulder width
    "M27": {FitStyle.FITTED: 0.5,  FitStyle.REGULAR: 1.5,  FitStyle.RELAXED: 3.0},   # chest width
    "M28": {FitStyle.FITTED: 0.5,  FitStyle.REGULAR: 1.5,  FitStyle.RELAXED: 3.0},   # back width
    "M29": {FitStyle.FITTED: 0.5,  FitStyle.REGULAR: 1.0,  FitStyle.RELAXED: 2.0},   # hip width

    # Depths ---------------------------------------------------------------
    "M32": {FitStyle.FITTED: 1.0,  FitStyle.REGULAR: 1.5,  FitStyle.RELAXED: 2.0},   # armhole depth

    # Length hem allowances (independent of fit style — fabric needed for the hem)
    "M17": {FitStyle.FITTED: 2.0,  FitStyle.REGULAR: 2.0,  FitStyle.RELAXED: 2.0},   # kameez length
    "M18": {FitStyle.FITTED: 2.0,  FitStyle.REGULAR: 2.0,  FitStyle.RELAXED: 2.0},   # dress length
    "M19": {FitStyle.FITTED: 2.5,  FitStyle.REGULAR: 2.5,  FitStyle.RELAXED: 2.5},   # sleeve length (full)
    "M20": {FitStyle.FITTED: 2.0,  FitStyle.REGULAR: 2.0,  FitStyle.RELAXED: 2.0},   # sleeve length (elbow)
    "M21": {FitStyle.FITTED: 3.0,  FitStyle.REGULAR: 3.0,  FitStyle.RELAXED: 3.0},   # inseam hem
    "M22": {FitStyle.FITTED: 3.0,  FitStyle.REGULAR: 3.0,  FitStyle.RELAXED: 3.0},   # outseam hem
}


# ---------------------------------------------------------------------------
# Core annotator — called by pipeline and manual endpoint
# ---------------------------------------------------------------------------

def apply_garment_profile(
    measurements: "ScanMeasurements",
    garment_type: Optional[GarmentType],
    fit_style: Optional[FitStyle],
) -> "ScanMeasurements":
    """
    Annotate each MeasurementField with:
      - is_required_for_garment: bool (True when this code is required for garment_type)
      - ease_cm: float | None   (ease for fit_style; None when no fit_style or no ease)
      - cutting_value_cm: float | None  (value_cm + ease; the dimension the tailor cuts to)

    When garment_type is None the measurements are returned unchanged.
    When fit_style is None, ease_cm / cutting_value_cm remain None.
    """
    if garment_type is None:
        return measurements

    from app.measurement_engine.scan.schemas import ScanMeasurements

    required_codes = GARMENT_REQUIRED_FIELDS.get(garment_type, set())
    updated: dict[str, MeasurementField] = {}

    for attr_name in ScanMeasurements.model_fields:
        field: MeasurementField = getattr(measurements, attr_name)
        code = attr_name.split("_")[0]   # "M01_chest" → "M01"

        required = code in required_codes

        # cutting_value_cm and ease_cm are only set when the measurement actually
        # has an ease entry for this fit_style. M14 (height) and fields with no
        # ease are not "cut to" a dimension, so both stay None.
        ease_cm: Optional[float] = None
        cutting_value_cm: Optional[float] = None
        if fit_style is not None and field.value_cm is not None:
            ease = EASE_ALLOWANCES.get(code, {}).get(fit_style)
            if ease is not None:
                ease_cm = round(ease, 1)
                cutting_value_cm = round(field.value_cm + ease, 1)

        updated[attr_name] = field.model_copy(update={
            "is_required_for_garment": required,
            "ease_cm": ease_cm,
            "cutting_value_cm": cutting_value_cm,
        })

    return ScanMeasurements(**updated)
