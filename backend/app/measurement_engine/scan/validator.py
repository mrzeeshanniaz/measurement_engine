"""
TailorSync measurement self-validator.

Runs three independent validation passes and merges results:

  Pass 1 — Hard limits
    Absolute physiological impossibilities (e.g. waist = 10 cm).
    These are almost certainly scan errors. Severity: ERROR.

  Pass 2 — Population norms (Z-score outlier detection)
    Each measurement is compared to the ANSUR II South-Asian-adjusted
    population distribution for this height.  Measurements beyond
    Z ± 2.0 produce a WARNING; beyond Z ± 3.0 produce an ERROR.

  Pass 3 — Cross-measurement consistency rules
    Physiological and garment-specific constraints that must hold
    regardless of body type (e.g. inseam < outseam, waist < chest).

Each issue carries:
  - severity:      ERROR | WARNING
  - code:          machine-readable rule ID (for Flutter to localise)
  - message:       human-readable explanation with actual values
  - fields:        measurement codes involved (e.g. ["M03", "M01"])
  - rescan_poses:  which pose(s) to redo to fix this (empty for manual issues)
  - suggestion:    one-line actionable fix for the customer

ValidationResult:
  is_valid      — True only when zero ERRORs
  can_order     — True only when no critical ERRORs on stitching-grade fields
  issues        — all ERRORs and WARNINGs
  rescan_poses  — deduplicated union of poses across all issues
  summary       — one sentence for display in the app
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from app.measurement_engine.scan.schemas import (
    GarmentType,
    PoseID,
    ScanMeasurements,
    Severity,
    ValidationIssue,
    ValidationResult,
)
from app.measurement_engine.scan.norms import NORMS, HARD_LIMITS


# Stitching-critical fields — LOW confidence on any of these blocks order (ORDER-03)
_CRITICAL_FIELDS = {"M01", "M03", "M05", "M26", "M21", "M19"}


# ---------------------------------------------------------------------------
# Helper to safely read a measurement value
# ---------------------------------------------------------------------------

def _v(measurements: ScanMeasurements, attr: str) -> Optional[float]:
    field_obj = getattr(measurements, attr, None)
    return field_obj.value_cm if field_obj else None


# ---------------------------------------------------------------------------
# Pass 1 — Hard limits
# ---------------------------------------------------------------------------

def _pass_hard_limits(
    measurements: ScanMeasurements,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    field_map = {
        "M01": "M01_chest",       "M02": "M02_under_bust",
        "M03": "M03_waist",       "M04": "M04_abdomen",
        "M05": "M05_hips",        "M06": "M06_neck",
        "M07": "M07_bicep",       "M08": "M08_wrist",
        "M09": "M09_thigh",       "M10": "M10_mid_thigh",
        "M11": "M11_knee",        "M12": "M12_calf",
        "M13": "M13_ankle",       "M15": "M15_shoulder_to_waist_front",
        "M16": "M16_shoulder_to_waist_back", "M17": "M17_kameez_length",
        "M18": "M18_dress_length","M19": "M19_sleeve_length",
        "M20": "M20_sleeve_length_elbow",    "M21": "M21_inseam",
        "M22": "M22_outseam",     "M23": "M23_crotch_depth_front",
        "M24": "M24_crotch_depth_back",      "M25": "M25_torso_length",
        "M26": "M26_shoulder_width",         "M27": "M27_chest_width",
        "M28": "M28_back_width",  "M29": "M29_hip_width",
        "M30": "M30_chest_depth", "M31": "M31_waist_depth",
        "M32": "M32_armhole_depth",
    }
    for code, attr in field_map.items():
        val = _v(measurements, attr)
        if val is None:
            continue
        lo, hi = HARD_LIMITS.get(code, (0.0, 9999.0))
        if val < lo or val > hi:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                code=f"hard_limit_{code.lower()}",
                message=(
                    f"{attr} = {val} cm is outside the physiologically plausible "
                    f"range [{lo}–{hi} cm]. This is almost certainly a scan error."
                ),
                fields=[code],
                rescan_poses=[PoseID.FRONT.value],
                suggestion="Please redo the scan ensuring your full body is in frame and your clothes fit closely.",
            ))
    return issues


# ---------------------------------------------------------------------------
# Pass 2 — Population norms / Z-score outlier detection
# ---------------------------------------------------------------------------

def _pass_norms(
    measurements: ScanMeasurements,
    height_cm: float,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    field_map = {
        "M01": "M01_chest",   "M02": "M02_under_bust",
        "M03": "M03_waist",   "M04": "M04_abdomen",
        "M05": "M05_hips",    "M06": "M06_neck",
        "M07": "M07_bicep",   "M08": "M08_wrist",
        "M09": "M09_thigh",   "M10": "M10_mid_thigh",
        "M11": "M11_knee",    "M12": "M12_calf",
        "M13": "M13_ankle",   "M15": "M15_shoulder_to_waist_front",
        "M16": "M16_shoulder_to_waist_back", "M17": "M17_kameez_length",
        "M18": "M18_dress_length",           "M19": "M19_sleeve_length",
        "M20": "M20_sleeve_length_elbow",    "M21": "M21_inseam",
        "M22": "M22_outseam",                "M23": "M23_crotch_depth_front",
        "M24": "M24_crotch_depth_back",      "M25": "M25_torso_length",
        "M26": "M26_shoulder_width",         "M27": "M27_chest_width",
        "M28": "M28_back_width",             "M29": "M29_hip_width",
        "M30": "M30_chest_depth",            "M31": "M31_waist_depth",
        "M32": "M32_armhole_depth",
    }

    for code, attr in field_map.items():
        val = _v(measurements, attr)
        norm = NORMS.get(code)
        if val is None or norm is None:
            continue

        z = norm.z_score(val, height_cm)
        abs_z = abs(z)
        direction = "large" if z > 0 else "small"
        expected = round(norm.mean_ratio * height_cm, 1)

        if abs_z >= norm.z_error:
            lo, hi = norm.plausible_range(height_cm, norm.z_error)
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                code=f"outlier_error_{code.lower()}",
                message=(
                    f"{attr} = {val} cm is unusually {direction} for your height "
                    f"({height_cm} cm). Expected ~{expected} cm, range [{lo}–{hi} cm]. "
                    f"Z-score: {z:+.1f}."
                ),
                fields=[code],
                rescan_poses=_poses_for(code),
                suggestion=_outlier_suggestion(code, direction),
            ))
        elif abs_z >= norm.z_warn:
            lo, hi = norm.plausible_range(height_cm, norm.z_warn)
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                code=f"outlier_warn_{code.lower()}",
                message=(
                    f"{attr} = {val} cm is somewhat {direction} for your height. "
                    f"Typical range: [{lo}–{hi} cm]."
                ),
                fields=[code],
                rescan_poses=_poses_for(code),
                suggestion="If this looks wrong, redo that pose with tighter-fitting clothing.",
            ))

    return issues


# ---------------------------------------------------------------------------
# Pass 3 — Cross-measurement consistency rules
# ---------------------------------------------------------------------------

@dataclass
class _Rule:
    code:         str
    severity:     Severity
    message_tmpl: str   # uses {a}, {b}, {diff} placeholders
    fields:       list[str]
    rescan_poses: list[str]
    suggestion:   str
    check:        object  # callable(a, b) → bool, True = PASS


def _cross_rules() -> list[_Rule]:
    F = PoseID.FRONT.value
    SL = PoseID.SIDE_LEFT.value
    AO = PoseID.ARMS_OUT.value
    BK = PoseID.BACK.value

    return [
        # Structural body rules
        _Rule(
            code="waist_lt_chest",
            severity=Severity.ERROR,
            message_tmpl="Waist ({a} cm) must be smaller than chest ({b} cm). Difference: {diff} cm.",
            fields=["M03", "M01"],
            rescan_poses=[F, SL],
            suggestion="Redo the front and side poses. Ensure your clothes fit closely — loose fabric inflates the chest.",
            check=lambda a, b: a < b + 2,  # 2 cm tolerance
        ),
        _Rule(
            code="waist_le_hips",
            severity=Severity.WARNING,
            message_tmpl="Waist ({a} cm) is larger than hips ({b} cm) — unusual. Difference: {diff} cm.",
            fields=["M03", "M05"],
            rescan_poses=[F, SL],
            suggestion="This is uncommon. Please verify your waist and hip measurements or redo the scan.",
            check=lambda a, b: a <= b + 5,
        ),
        _Rule(
            code="chest_le_hips",
            severity=Severity.WARNING,
            message_tmpl="Chest ({a} cm) is much larger than hips ({b} cm) — unusual. Difference: {diff} cm.",
            fields=["M01", "M05"],
            rescan_poses=[F, SL],
            suggestion="Ensure the hip frame (back/side poses) captured your widest point.",
            check=lambda a, b: a <= b + 15,
        ),
        _Rule(
            code="inseam_lt_outseam",
            severity=Severity.ERROR,
            message_tmpl="Inseam ({a} cm) must be less than outseam ({b} cm). Difference: {diff} cm.",
            fields=["M21", "M22"],
            rescan_poses=[F, SL],
            suggestion="Redo the front and side profile scans ensuring your full legs are visible.",
            check=lambda a, b: a < b,
        ),
        _Rule(
            code="outseam_inseam_gap",
            severity=Severity.WARNING,
            message_tmpl="Outseam ({b} cm) minus inseam ({a} cm) = {diff} cm — rise seems unusual (expected 10–18 cm).",
            fields=["M21", "M22"],
            rescan_poses=[F, SL],
            suggestion="The crotch rise looks off. Redo front and side poses ensuring hips are clearly visible.",
            check=lambda a, b: 8 <= (b - a) <= 22,
        ),
        _Rule(
            code="sleeve_gt_elbow_sleeve",
            severity=Severity.ERROR,
            message_tmpl="Full sleeve ({a} cm) must be longer than elbow sleeve ({b} cm). Got {diff} cm difference.",
            fields=["M19", "M20"],
            rescan_poses=[AO],
            suggestion="Redo the arms-out pose with arms fully extended at shoulder height.",
            check=lambda a, b: a > b,
        ),
        _Rule(
            code="kameez_gt_torso",
            severity=Severity.ERROR,
            message_tmpl="Kameez length ({a} cm) must exceed torso length ({b} cm). Difference: {diff} cm.",
            fields=["M17", "M25"],
            rescan_poses=[F],
            suggestion="The kameez length or torso length seems wrong. Redo the front pose.",
            check=lambda a, b: a > b,
        ),
        _Rule(
            code="dress_gt_kameez",
            severity=Severity.WARNING,
            message_tmpl="Dress/suit length ({a} cm) should be longer than kameez length ({b} cm).",
            fields=["M18", "M17"],
            rescan_poses=[F],
            suggestion="Verify dress and kameez length measurements — dress is expected to be longer.",
            check=lambda a, b: a >= b,
        ),
        _Rule(
            code="shoulder_width_vs_chest",
            severity=Severity.WARNING,
            message_tmpl="Shoulder width ({a} cm) is unusually wide relative to chest ({b} cm).",
            fields=["M26", "M01"],
            rescan_poses=[F, AO],
            suggestion="Redo the front and arms-out poses. Wide shoulders vs chest may indicate a scanning angle issue.",
            check=lambda a, b: a < b * 0.55,
        ),
        _Rule(
            code="back_width_le_chest_width",
            severity=Severity.WARNING,
            message_tmpl="Back width ({a} cm) is larger than chest width ({b} cm) — unusual.",
            fields=["M28", "M27"],
            rescan_poses=[F, BK],
            suggestion="Redo the front and back poses standing symmetrically.",
            check=lambda a, b: a <= b + 3,
        ),
        _Rule(
            code="torso_vs_height",
            severity=Severity.WARNING,
            message_tmpl="Torso length ({a} cm) looks short relative to height. Expected ~31% of height.",
            fields=["M25", "M14"],
            rescan_poses=[F],
            suggestion="Ensure your full torso (neck to crotch) is visible in the front pose.",
            check=lambda a, b: a >= b * 0.24,
        ),
        _Rule(
            code="neck_lt_chest",
            severity=Severity.ERROR,
            message_tmpl="Neck circumference ({a} cm) must be smaller than chest ({b} cm).",
            fields=["M06", "M01"],
            rescan_poses=[F],
            suggestion="Neck circumference looks incorrect. Ensure your neck is visible in the front pose.",
            check=lambda a, b: a < b * 0.55,
        ),
        _Rule(
            code="thigh_lt_hip",
            severity=Severity.ERROR,
            message_tmpl="Thigh circumference ({a} cm) must be less than hip circumference ({b} cm).",
            fields=["M09", "M05"],
            rescan_poses=[F, SL],
            suggestion="Redo the front pose ensuring your hips and upper thighs are clearly in frame.",
            check=lambda a, b: a < b,
        ),
        _Rule(
            code="calf_lt_thigh",
            severity=Severity.ERROR,
            message_tmpl="Calf circumference ({a} cm) must be less than thigh circumference ({b} cm).",
            fields=["M12", "M09"],
            rescan_poses=[F, SL],
            suggestion="Redo the side pose ensuring your full legs are visible.",
            check=lambda a, b: a < b,
        ),
        _Rule(
            code="shoulder_to_waist_vs_torso",
            severity=Severity.WARNING,
            message_tmpl=(
                "Shoulder-to-waist ({a} cm) is larger than torso length ({b} cm). "
                "Torso includes crotch — this may be inconsistent."
            ),
            fields=["M15", "M25"],
            rescan_poses=[F],
            suggestion="Redo the front pose. Ensure the camera captures from head to hip.",
            check=lambda a, b: a <= b,
        ),
        _Rule(
            code="sleeve_vs_height",
            severity=Severity.WARNING,
            message_tmpl="Sleeve length ({a} cm) is unusually long for height ({b} cm). Expected 30–36% of height.",
            fields=["M19", "M14"],
            rescan_poses=[AO],
            suggestion="Redo the arms-out pose with arms straight at shoulder height.",
            check=lambda a, b: a <= b * 0.40,
        ),
        _Rule(
            code="chest_depth_lt_chest_width",
            severity=Severity.WARNING,
            message_tmpl="Chest depth ({a} cm) is larger than chest width ({b} cm) — unexpected body geometry.",
            fields=["M30", "M27"],
            rescan_poses=[SL, F],
            suggestion="Redo the side pose. Chest depth is front-to-back; it should be less than left-to-right width.",
            check=lambda a, b: a <= b + 5,
        ),
    ]


def _pass_cross_rules(
    measurements: ScanMeasurements,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    field_map = {
        "M01": "M01_chest",          "M02": "M02_under_bust",
        "M03": "M03_waist",          "M04": "M04_abdomen",
        "M05": "M05_hips",           "M06": "M06_neck",
        "M07": "M07_bicep",          "M09": "M09_thigh",
        "M11": "M11_knee",           "M12": "M12_calf",
        "M14": "M14_total_height",   "M15": "M15_shoulder_to_waist_front",
        "M16": "M16_shoulder_to_waist_back", "M17": "M17_kameez_length",
        "M18": "M18_dress_length",   "M19": "M19_sleeve_length",
        "M20": "M20_sleeve_length_elbow",    "M21": "M21_inseam",
        "M22": "M22_outseam",        "M25": "M25_torso_length",
        "M26": "M26_shoulder_width", "M27": "M27_chest_width",
        "M28": "M28_back_width",     "M30": "M30_chest_depth",
    }

    for rule in _cross_rules():
        code_a, code_b = rule.fields[0], rule.fields[1]
        a = _v(measurements, field_map.get(code_a, ""))
        b = _v(measurements, field_map.get(code_b, ""))
        if a is None or b is None:
            continue
        if not rule.check(a, b):
            diff = round(abs(a - b), 1)
            issues.append(ValidationIssue(
                severity=rule.severity,
                code=rule.code,
                message=rule.message_tmpl.format(a=a, b=b, diff=diff),
                fields=rule.fields,
                rescan_poses=rule.rescan_poses,
                suggestion=rule.suggestion,
            ))
    return issues


# ---------------------------------------------------------------------------
# Confidence audit — warn when too many LOW fields are order-blocking
# ---------------------------------------------------------------------------

def _pass_confidence(measurements: ScanMeasurements) -> list[ValidationIssue]:
    from app.measurement_engine.scan.schemas import Confidence

    low_critical = [
        code for code in _CRITICAL_FIELDS
        if getattr(measurements, _code_to_attr(code), None) is not None
        and getattr(measurements, _code_to_attr(code)).confidence == Confidence.LOW
    ]

    issues: list[ValidationIssue] = []
    if low_critical:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            code="low_confidence_critical_fields",
            message=(
                f"The following stitching-critical measurements have LOW confidence: "
                f"{', '.join(low_critical)}. The tailor cannot safely cut fabric with "
                f"these values."
            ),
            fields=low_critical,
            rescan_poses=[PoseID.FRONT.value, PoseID.SIDE_LEFT.value],
            suggestion="Please redo the scan with tight-fitting clothing in a well-lit room.",
        ))

    # Count total LOW fields
    all_attrs = [a for a in ScanMeasurements.model_fields if a.startswith("M")]
    low_count = sum(
        1 for attr in all_attrs
        if (f := getattr(measurements, attr, None)) is not None
        and f.confidence == Confidence.LOW
    )
    if low_count >= 10 and not low_critical:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            code="many_low_confidence_fields",
            message=f"{low_count} measurements have LOW confidence. Measurement accuracy may be insufficient for tailoring.",
            fields=[],
            rescan_poses=[PoseID.FRONT.value, PoseID.SIDE_LEFT.value, PoseID.ARMS_OUT.value],
            suggestion="Add more pose frames (especially side and arms-out) for a more accurate scan.",
        ))
    return issues


# ---------------------------------------------------------------------------
# Pass 4 — Garment-specific required-field check (F8)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pass 5 — Mesh quality gate (B8 / spec 5.1.7)
# ---------------------------------------------------------------------------

# Convex-hull silhouette IoU equivalent of the spec's 88% threshold.
# Our IoU is a looser metric than per-pixel silhouette overlap, so 0.55 is
# calibrated to roughly correspond to the 88% coverage threshold.
_MESH_FIT_WARN_THRESHOLD  = 0.55
_MESH_FIT_ERROR_THRESHOLD = 0.30


def _pass_mesh_quality(mesh_fit_score: float) -> list[ValidationIssue]:
    """
    Warn (or error) when the fitted SMPL mesh silhouette doesn't match the
    body well enough to trust depth-derived measurements (chest/waist depth,
    circumferences).  Spec 5.1.7 requires ≥ 88% global fit score.
    """
    if mesh_fit_score >= _MESH_FIT_WARN_THRESHOLD:
        return []

    if mesh_fit_score < _MESH_FIT_ERROR_THRESHOLD:
        return [ValidationIssue(
            severity=Severity.ERROR,
            code="mesh_fit_poor",
            message=(
                f"The body mesh fit score is very low ({mesh_fit_score:.2f}). "
                "Depth-derived measurements (chest/waist depth, circumferences) are "
                "unreliable and cannot be used for garment cutting."
            ),
            fields=["M01", "M03", "M05", "M30", "M31"],
            rescan_poses=[PoseID.FRONT.value, PoseID.SIDE_LEFT.value],
            suggestion=(
                "Redo the scan in better lighting with tight-fitting clothes. "
                "Ensure your full body is visible against a plain background."
            ),
        )]

    return [ValidationIssue(
        severity=Severity.WARNING,
        code="mesh_fit_low",
        message=(
            f"The body mesh fit score is below the recommended threshold "
            f"({mesh_fit_score:.2f} < {_MESH_FIT_WARN_THRESHOLD}). "
            "Depth-derived measurements may be less accurate than usual."
        ),
        fields=["M30", "M31"],
        rescan_poses=[PoseID.FRONT.value, PoseID.SIDE_LEFT.value],
        suggestion=(
            "For best results, redo the scan with tight-fitting clothes "
            "against a plain, well-lit background."
        ),
    )]

def _pass_garment_required(
    measurements: ScanMeasurements,
    garment_type: Optional[GarmentType],
) -> list[ValidationIssue]:
    """Raise an ERROR for each measurement that is required for garment_type but has no value."""
    if garment_type is None:
        return []

    from app.measurement_engine.scan.garments import GARMENT_REQUIRED_FIELDS

    required_codes = GARMENT_REQUIRED_FIELDS.get(garment_type, set())
    issues: list[ValidationIssue] = []

    for code in sorted(required_codes):
        attr = _code_to_attr(code)
        if _v(measurements, attr) is None:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                code=f"missing_required_{code.lower()}_{garment_type.value}",
                message=(
                    f"{code} is required to cut a {garment_type.value} but was not measured. "
                    f"Ensure this body area is fully visible in the scan."
                ),
                fields=[code],
                rescan_poses=_poses_for(code),
                suggestion=f"Redo the scan so that your {attr.replace('_', ' ')} is clearly captured.",
            ))

    return issues


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate(
    measurements: ScanMeasurements,
    height_cm: float,
    garment_type: Optional[GarmentType] = None,
    mesh_fit_score: float = 1.0,
) -> ValidationResult:
    """
    Run all validation passes and return a consolidated ValidationResult.
    Pass 4 (garment-required) is skipped when garment_type is None.
    Pass 5 (mesh quality gate) is skipped when mesh_fit_score == 1.0 (no mesh).
    """
    issues: list[ValidationIssue] = []
    issues += _pass_hard_limits(measurements)
    issues += _pass_norms(measurements, height_cm)
    issues += _pass_cross_rules(measurements)
    issues += _pass_confidence(measurements)
    issues += _pass_garment_required(measurements, garment_type)
    issues += _pass_mesh_quality(mesh_fit_score)

    errors = [i for i in issues if i.severity == Severity.ERROR]
    is_valid = len(errors) == 0

    # can_order = no ERRORs on critical stitching fields (ORDER-03)
    critical_errors = [
        i for i in errors
        if any(f in _CRITICAL_FIELDS for f in i.fields)
    ]
    can_order = len(critical_errors) == 0

    # Deduplicate rescan poses across all issues
    rescan: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        for pose in issue.rescan_poses:
            if pose not in seen:
                seen.add(pose)
                rescan.append(pose)

    n_err  = len(errors)
    n_warn = len(issues) - n_err
    if n_err == 0 and n_warn == 0:
        summary = "All measurements look valid and consistent."
    elif n_err == 0:
        summary = f"{n_warn} warning(s) detected — review before ordering."
    else:
        summary = f"{n_err} error(s) detected — rescan required before placing an order."

    return ValidationResult(
        is_valid=is_valid,
        can_order=can_order,
        issues=issues,
        rescan_poses=rescan,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Public: export rules as a dict for the frontend validation-rules endpoint
# ---------------------------------------------------------------------------

def export_rules_for_frontend() -> dict:
    """
    Serialise all validation rules to a JSON-compatible dict.
    The Flutter app fetches this once and uses it for client-side validation
    on manual entry forms, before submitting to the server.
    """
    field_ranges = {
        code: {"min": lo, "max": hi, "unit": "cm"}
        for code, (lo, hi) in HARD_LIMITS.items()
    }

    norm_ranges = {
        code: {
            "mean_cm_per_height": round(norm.mean_ratio, 4),
            "sd_cm_per_height":   round(norm.sd_ratio, 4),
            "warn_z": norm.z_warn,
            "error_z": norm.z_error,
        }
        for code, norm in NORMS.items()
    }

    cross = [
        {
            "id":           r.code,
            "severity":     r.severity.value,
            "fields":       r.fields,
            "description":  r.message_tmpl,
            "suggestion":   r.suggestion,
            "rescan_poses": r.rescan_poses,
        }
        for r in _cross_rules()
    ]

    return {
        "version": "1.0",
        "field_ranges": field_ranges,
        "population_norms": norm_ranges,
        "cross_rules": cross,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CODE_TO_ATTR: dict[str, str] = {
    "M01": "M01_chest",          "M02": "M02_under_bust",
    "M03": "M03_waist",          "M04": "M04_abdomen",
    "M05": "M05_hips",           "M06": "M06_neck",
    "M07": "M07_bicep",          "M08": "M08_wrist",
    "M09": "M09_thigh",          "M10": "M10_mid_thigh",
    "M11": "M11_knee",           "M12": "M12_calf",
    "M13": "M13_ankle",          "M14": "M14_total_height",
    "M15": "M15_shoulder_to_waist_front",
    "M16": "M16_shoulder_to_waist_back",
    "M17": "M17_kameez_length",  "M18": "M18_dress_length",
    "M19": "M19_sleeve_length",  "M20": "M20_sleeve_length_elbow",
    "M21": "M21_inseam",         "M22": "M22_outseam",
    "M23": "M23_crotch_depth_front", "M24": "M24_crotch_depth_back",
    "M25": "M25_torso_length",   "M26": "M26_shoulder_width",
    "M27": "M27_chest_width",    "M28": "M28_back_width",
    "M29": "M29_hip_width",      "M30": "M30_chest_depth",
    "M31": "M31_waist_depth",    "M32": "M32_armhole_depth",
}


def _code_to_attr(code: str) -> str:
    return _CODE_TO_ATTR.get(code, code)


# Which poses are most relevant to fix each measurement
_POSE_FOR_CODE: dict[str, list[str]] = {
    "M01": [PoseID.FRONT.value, PoseID.SIDE_LEFT.value],
    "M03": [PoseID.FRONT.value, PoseID.SIDE_LEFT.value],
    "M05": [PoseID.FRONT.value, PoseID.BACK.value],
    "M06": [PoseID.FRONT.value],
    "M07": [PoseID.ARMS_OUT.value],
    "M08": [PoseID.ARMS_OUT.value],
    "M09": [PoseID.FRONT.value, PoseID.SIDE_LEFT.value],
    "M12": [PoseID.SIDE_LEFT.value],
    "M15": [PoseID.FRONT.value],
    "M16": [PoseID.BACK.value, PoseID.SIDE_LEFT.value],
    "M17": [PoseID.FRONT.value],
    "M19": [PoseID.ARMS_OUT.value],
    "M21": [PoseID.FRONT.value],
    "M22": [PoseID.SIDE_LEFT.value],
    "M26": [PoseID.FRONT.value, PoseID.BACK.value],
    "M30": [PoseID.SIDE_LEFT.value],
    "M31": [PoseID.SIDE_LEFT.value],
}


def _poses_for(code: str) -> list[str]:
    return _POSE_FOR_CODE.get(code, [PoseID.FRONT.value])


def _outlier_suggestion(code: str, direction: str) -> str:
    if direction == "large":
        suggestions = {
            "M01": "Wearing loose clothing? Switch to a fitted t-shirt and redo the scan.",
            "M03": "Waist looks wide. Try the side pose again standing straight.",
            "M05": "Hip measurement seems large. Redo the back and side poses.",
            "M07": "Bicep looks wide — redo the arms-out pose with arms fully extended.",
            "M09": "Thigh seems wide. Redo front pose with feet hip-width apart.",
            "M19": "Sleeve is very long. Redo arms-out pose with arms straight at shoulder height.",
        }
    else:
        suggestions = {
            "M01": "Chest seems small. Ensure your full torso is visible in the front frame.",
            "M03": "Waist looks very small. Redo the scan standing naturally, not holding in.",
            "M21": "Inseam seems short. Ensure your legs reach the bottom of the frame.",
            "M26": "Shoulder width is small. Redo front pose with arms slightly away from body.",
        }
    return suggestions.get(code, "Redo the scan with tight-fitting clothes in good lighting.")
