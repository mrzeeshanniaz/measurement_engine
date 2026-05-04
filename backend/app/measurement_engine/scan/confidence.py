"""
Per-measurement confidence scorer.

Rules (applied in order; first match wins):
  HIGH   — primary source is smpl_mesh AND frame composite ≥ 0.70
         — OR source is landmark AND landmark visibility ≥ 0.80 AND composite ≥ 0.65
  MEDIUM — source is smpl_mesh OR landmark with composite ≥ 0.45
  LOW    — anything else (height_ratio fallback, or poor frame quality)

The overall scan confidence is the majority vote across all 32 fields,
with LOW dragging the result down if more than 4 fields are LOW.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from app.measurement_engine.scan.schemas import (
    Confidence,
    MeasurementField,
    ScanMeasurements,
)
from app.measurement_engine.scan.measurements import RawMeasurements


def score_field(
    value_cm: Optional[float],
    source: str,
    frame_composite: float,
    landmark_visibility: float = 1.0,
    mesh_fit_score: float = 1.0,
) -> MeasurementField:
    """
    Assign a Confidence level to one measurement field.

    Args:
        value_cm:           Extracted value (None → LOW with no value).
        source:             'smpl_mesh', 'landmark', 'derived', or 'height_ratio'.
        frame_composite:    Best composite score of the frame(s) that contributed.
        landmark_visibility: Mean visibility of the landmarks used (0-1).
        mesh_fit_score:     Silhouette IoU of the fitted SMPL mesh (0-1, 1.0 = no info).
    """
    if value_cm is None:
        return MeasurementField(value_cm=None, confidence=Confidence.LOW, source=source)

    # For SMPL-derived measurements the effective quality is the product of
    # frame quality and how well the mesh silhouette matches the actual body.
    # Formula: cap composite at (0.30 + 0.70 × mesh_fit_score) so a poor mesh
    # fit (IoU=0.30) cannot yield a composite above 0.51 (≤ MEDIUM threshold).
    if source in ("smpl_anthro_full", "smpl_anthro_trimesh", "smpl_mesh"):
        mesh_ceiling = 0.30 + 0.70 * mesh_fit_score
        effective_composite = min(frame_composite, mesh_ceiling)
    else:
        effective_composite = frame_composite

    if source == "smpl_anthro_full":
        # Best-accuracy path — SMPL-Anthropometry with PKL body model
        conf = Confidence.HIGH if effective_composite >= 0.55 else Confidence.MEDIUM

    elif source == "smpl_anthro_trimesh":
        # Trimesh plane-intersection path — good but not full SMPL accuracy
        conf = Confidence.HIGH if effective_composite >= 0.70 else Confidence.MEDIUM

    elif source == "smpl_mesh":
        if effective_composite >= 0.70:
            conf = Confidence.HIGH
        elif effective_composite >= 0.45:
            conf = Confidence.MEDIUM
        else:
            conf = Confidence.LOW

    elif source == "landmark":
        if landmark_visibility >= 0.80 and frame_composite >= 0.65:
            conf = Confidence.HIGH
        elif landmark_visibility >= 0.55 and frame_composite >= 0.45:
            conf = Confidence.MEDIUM
        else:
            conf = Confidence.LOW

    elif source == "derived":
        # Derived from another measurement — inherit one level down
        if frame_composite >= 0.65:
            conf = Confidence.MEDIUM
        else:
            conf = Confidence.LOW

    else:  # height_ratio
        conf = Confidence.LOW

    return MeasurementField(
        value_cm=round(value_cm, 1),
        confidence=conf,
        source=source,
    )


def overall_confidence(measurements: ScanMeasurements) -> Confidence:
    """Compute overall scan confidence from field distribution."""
    fields = [
        measurements.M01_chest, measurements.M02_under_bust,
        measurements.M03_waist, measurements.M04_abdomen, measurements.M05_hips,
        measurements.M06_neck, measurements.M07_bicep, measurements.M08_wrist,
        measurements.M09_thigh, measurements.M10_mid_thigh, measurements.M11_knee,
        measurements.M12_calf, measurements.M13_ankle,
        measurements.M15_shoulder_to_waist_front, measurements.M16_shoulder_to_waist_back,
        measurements.M17_kameez_length, measurements.M18_dress_length,
        measurements.M19_sleeve_length, measurements.M20_sleeve_length_elbow,
        measurements.M21_inseam, measurements.M22_outseam,
        measurements.M23_crotch_depth_front, measurements.M24_crotch_depth_back,
        measurements.M25_torso_length,
        measurements.M26_shoulder_width, measurements.M27_chest_width,
        measurements.M28_back_width, measurements.M29_hip_width,
        measurements.M30_chest_depth, measurements.M31_waist_depth,
        measurements.M32_armhole_depth,
    ]
    counts: Counter = Counter(f.confidence for f in fields)
    n_low = counts[Confidence.LOW]

    if n_low > 8:
        return Confidence.LOW
    if n_low > 4 or counts[Confidence.HIGH] < 10:
        return Confidence.MEDIUM
    return Confidence.HIGH


def _cap_confidence(conf: Confidence, ceiling: Confidence) -> Confidence:
    """Lower conf to ceiling when the height anchor is uncertain."""
    _rank = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
    return conf if _rank[conf] >= _rank[ceiling] else ceiling


def build_scan_measurements(
    raw: RawMeasurements,
    frame_composites: dict[str, float],   # pose_id → composite score
    landmark_visibilities: dict[str, float],  # "M01", "M26", … → visibility
    height_source: str = "user_input",
    height_confidence: Confidence = Confidence.HIGH,
    mesh_fit_score: float = 1.0,
) -> ScanMeasurements:
    """
    Convert RawMeasurements + frame quality data into a scored ScanMeasurements.

    frame_composites should contain entries for each PoseID that was submitted.
    landmark_visibilities is a dict keyed by measurement code → mean visibility
    of the landmarks that contributed to that measurement.

    height_source / height_confidence: when height was auto-estimated, all
    measurement fields are capped at height_confidence since every derived
    value inherits the scale-anchor uncertainty.

    mesh_fit_score: silhouette IoU from F6 — gates maximum confidence for
    SMPL-derived measurements when the mesh doesn't fit the actual body well.
    """
    conf_ceiling = height_confidence

    def _field(key: str, value: Optional[float]) -> MeasurementField:
        source = raw.sources.get(key, "height_ratio")
        composite = _best_composite(key, frame_composites)
        vis = landmark_visibilities.get(key, 1.0)
        field = score_field(value, source, composite, vis, mesh_fit_score)
        # Every measurement is scaled by height_cm — propagate uncertainty
        if conf_ceiling != Confidence.HIGH:
            field = MeasurementField(
                value_cm=field.value_cm,
                confidence=_cap_confidence(field.confidence, conf_ceiling),
                source=field.source,
            )
        return field

    m14 = MeasurementField(
        value_cm=round(raw.height_cm, 1),
        confidence=height_confidence,
        source=height_source,
    )

    m = ScanMeasurements(
        M01_chest=_field("M01", raw.M01_chest),
        M02_under_bust=_field("M02", raw.M02_under_bust),
        M03_waist=_field("M03", raw.M03_waist),
        M04_abdomen=_field("M04", raw.M04_abdomen),
        M05_hips=_field("M05", raw.M05_hips),
        M06_neck=_field("M06", raw.M06_neck),
        M07_bicep=_field("M07", raw.M07_bicep),
        M08_wrist=_field("M08", raw.M08_wrist),
        M09_thigh=_field("M09", raw.M09_thigh),
        M10_mid_thigh=_field("M10", raw.M10_mid_thigh),
        M11_knee=_field("M11", raw.M11_knee),
        M12_calf=_field("M12", raw.M12_calf),
        M13_ankle=_field("M13", raw.M13_ankle),
        M14_total_height=m14,
        M15_shoulder_to_waist_front=_field("M15", raw.M15_shoulder_to_waist_front),
        M16_shoulder_to_waist_back=_field("M16", raw.M16_shoulder_to_waist_back),
        M17_kameez_length=_field("M17", raw.M17_kameez_length),
        M18_dress_length=_field("M18", raw.M18_dress_length),
        M19_sleeve_length=_field("M19", raw.M19_sleeve_length),
        M20_sleeve_length_elbow=_field("M20", raw.M20_sleeve_length_elbow),
        M21_inseam=_field("M21", raw.M21_inseam),
        M22_outseam=_field("M22", raw.M22_outseam),
        M23_crotch_depth_front=_field("M23", raw.M23_crotch_depth_front),
        M24_crotch_depth_back=_field("M24", raw.M24_crotch_depth_back),
        M25_torso_length=_field("M25", raw.M25_torso_length),
        M26_shoulder_width=_field("M26", raw.M26_shoulder_width),
        M27_chest_width=_field("M27", raw.M27_chest_width),
        M28_back_width=_field("M28", raw.M28_back_width),
        M29_hip_width=_field("M29", raw.M29_hip_width),
        M30_chest_depth=_field("M30", raw.M30_chest_depth),
        M31_waist_depth=_field("M31", raw.M31_waist_depth),
        M32_armhole_depth=_field("M32", raw.M32_armhole_depth),
    )
    return m


# ---------------------------------------------------------------------------
# Internal: which pose contributes to which measurement
# ---------------------------------------------------------------------------

# Measurement code → list of relevant PoseIDs (in priority order)
_POSE_RELEVANCE: dict[str, list[str]] = {
    # circumferences — best from FRONT (multi-view SMPL covers all)
    "M01": ["front", "quarter_left", "side_left"],
    "M02": ["front", "quarter_left", "side_left"],
    "M03": ["front", "quarter_left", "side_left"],
    "M04": ["front", "side_left"],
    "M05": ["front", "back"],
    "M06": ["front"],
    "M07": ["arms_out"],
    "M08": ["arms_out"],
    "M09": ["front", "back"],
    "M10": ["front"],
    "M11": ["front"],
    "M12": ["side_left"],
    "M13": ["front"],
    # lengths
    "M15": ["front"],
    "M16": ["back", "side_left"],
    "M17": ["front", "side_left"],
    "M18": ["front"],
    "M19": ["arms_out", "front"],
    "M20": ["arms_out"],
    "M21": ["front"],
    "M22": ["side_left"],
    "M23": ["side_left"],
    "M24": ["side_left", "back"],
    "M25": ["front"],
    # widths & depths
    "M26": ["front", "back"],
    "M27": ["front"],
    "M28": ["back"],
    "M29": ["front", "back"],
    "M30": ["side_left", "three_quarter"],
    "M31": ["side_left"],
    "M32": ["front", "arms_out"],
}


def _best_composite(
    mcode: str,
    frame_composites: dict[str, float],
) -> float:
    """Return the highest composite score among frames relevant to this measurement."""
    poses = _POSE_RELEVANCE.get(mcode, ["front"])
    scores = [frame_composites.get(p, 0.0) for p in poses]
    return max(scores) if scores else 0.0
