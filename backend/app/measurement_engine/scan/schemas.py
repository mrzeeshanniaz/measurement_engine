"""
Schemas for the TailorSync video-guided body scan pipeline.

Flow:
  Client submits height_cm (or camera_metadata for auto-estimation) + 6-8
  base64-encoded frames (one per pose checkpoint).
  Engine returns 32 measurements with per-field confidence levels.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Pose checkpoint IDs — match the 7-pose guided scan spec
# ---------------------------------------------------------------------------

class PoseID(str, Enum):
    FRONT         = "front"          # 0°   natural standing, facing camera
    QUARTER_LEFT  = "quarter_left"   # 45°  quarter turn left
    SIDE_LEFT     = "side_left"      # 90°  full left profile
    THREE_QUARTER = "three_quarter"  # 135° three-quarter turn
    BACK          = "back"           # 180° facing away
    SIDE_RIGHT    = "side_right"     # 270° right profile (auto-captured mid-rotation)
    ARMS_OUT      = "arms_out"       # 0°   T-pose arms extended


# ---------------------------------------------------------------------------
# Garment type + fit style (F8 / F9)
# ---------------------------------------------------------------------------

class GarmentType(str, Enum):
    KAMEEZ        = "kameez"        # long tunic (women / unisex)
    KURTA         = "kurta"         # shorter tunic (men)
    SHALWAR       = "shalwar"       # traditional baggy trousers
    TROUSER       = "trouser"       # fitted trousers / pants
    SHIRT         = "shirt"         # men's formal / casual shirt
    SHERWANI      = "sherwani"      # men's formal long coat
    DRESS         = "dress"         # full / midi-length dress
    SUIT_JACKET   = "suit_jacket"   # men's suit jacket / blazer
    BLOUSE        = "blouse"        # women's short top
    SKIRT         = "skirt"         # skirt (knee / ankle length)
    LEHENGA_SKIRT = "lehenga_skirt" # full circular skirt
    COAT          = "coat"          # overcoat / long coat


class FitStyle(str, Enum):
    FITTED  = "fitted"   # body-con; minimum ease
    REGULAR = "regular"  # standard comfortable fit
    RELAXED = "relaxed"  # loose / flowy; maximum ease


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CameraMetadata(BaseModel):
    """Phone sensor data sent by the mobile app for automatic height estimation."""
    focal_length_px: Optional[float] = Field(
        None, gt=0,
        description=(
            "Focal length in pixels (from EXIF or device camera API). "
            "If omitted, a 65° vertical field-of-view is assumed."
        ),
    )
    camera_height_cm: float = Field(
        ..., gt=30.0, lt=250.0,
        description="Distance from floor to camera lens in cm (user sets phone on a surface or mount).",
    )
    tilt_angle_deg: float = Field(
        ..., ge=5.0, le=85.0,
        description="Downward tilt of camera from horizontal in degrees (from accelerometer pitch).",
    )


class PoseFrame(BaseModel):
    pose_id: PoseID
    image_b64: str = Field(
        ...,
        description="Base64-encoded JPEG/PNG frame selected by the on-device scorer",
    )
    quality_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="On-device frame quality score reported by the mobile app",
    )

    @field_validator("image_b64")
    @classmethod
    def _enforce_max_size(cls, v: str) -> str:
        # Lazy import to avoid pulling settings at module import time.
        # We only enforce the maximum here — the real image-decode happens in
        # the pipeline, which raises on malformed/tiny inputs naturally.
        from app.config import settings

        if len(v) > settings.MAX_FRAME_B64_BYTES:
            raise ValueError(
                f"image_b64 exceeds maximum of {settings.MAX_FRAME_B64_BYTES} bytes"
            )
        return v


class ScaleTier(str, Enum):
    """Processing quality tier sent by the mobile client."""
    TIER1 = "TIER1"   # standard — CPU inference, single-view betas
    TIER2 = "TIER2"   # enhanced — multi-view silhouette optimization (default)
    TIER3 = "TIER3"   # maximum — reserved for future higher-accuracy models


class ScanSubmitRequest(BaseModel):
    # SCAN-10: client generates this on-device for idempotency + resume support
    client_scan_id: Optional[str] = Field(
        None,
        description="UUID generated on-device at scan start (SCAN-10). Used for idempotency and abandoned-scan resume.",
    )
    # A2: when provided the completed scan is persisted as a MeasurementProfile
    customer_id: Optional[str] = Field(
        None,
        description="Opaque customer / user identifier. When set, the completed scan is automatically saved as a measurement profile.",
    )
    height_cm: Optional[float] = Field(
        None, gt=50.0, lt=300.0,
        description=(
            "Customer-entered height in centimetres — the scale anchor. "
            "Omit to enable automatic height estimation via camera_metadata."
        ),
    )
    camera_metadata: Optional[CameraMetadata] = Field(
        None,
        description="Phone sensor data for automatic height estimation (required when height_cm is omitted).",
    )
    frames: list[PoseFrame] = Field(
        ..., min_length=1, max_length=8,
        description="Selected frames from the guided video scan (1-8 frames)",
    )
    scale_tier: ScaleTier = Field(
        ScaleTier.TIER2,
        description="Processing quality tier. TIER1 = fast/standard, TIER2 = multi-view optimization (default), TIER3 = reserved.",
    )
    # F8 / F9: optional garment context
    garment_type: Optional[GarmentType] = Field(
        None,
        description="Garment being made. When set, required measurements are flagged and missing ones raise validation errors.",
    )
    fit_style: Optional[FitStyle] = Field(
        None,
        description="Desired fit style. When set, ease allowances and cutting dimensions are added to each relevant measurement.",
    )

    @model_validator(mode="after")
    def require_height_or_camera(self) -> "ScanSubmitRequest":
        if self.height_cm is None and self.camera_metadata is None:
            raise ValueError("Provide either height_cm or camera_metadata for height estimation")
        return self

    @field_validator("frames")
    @classmethod
    def require_front_frame(cls, frames: list[PoseFrame]) -> list[PoseFrame]:
        ids = {f.pose_id for f in frames}
        if PoseID.FRONT not in ids:
            raise ValueError("At least a FRONT pose frame is required")
        return frames


# ---------------------------------------------------------------------------
# Confidence levels (per measurement field)
# ---------------------------------------------------------------------------

class Confidence(str, Enum):
    HIGH   = "HIGH"    # ±0.5–1.0 cm  — tailor can stitch directly
    MEDIUM = "MEDIUM"  # ±1.0–2.0 cm  — acceptable, flag on tailor sheet
    LOW    = "LOW"     # >2.0 cm      — require manual confirmation before order


# ---------------------------------------------------------------------------
# Single measurement value
# ---------------------------------------------------------------------------

class MeasurementField(BaseModel):
    value_cm: Optional[float] = Field(None, description="Measurement value (unit indicated by ScanResponse.response_unit)")
    unit: str = Field("cm", description="Unit of value_cm: 'cm' or 'in'")
    confidence: Confidence
    source: str = Field(
        ...,
        description="How this measurement was derived (e.g. 'smpl_mesh', 'landmark', 'height_ratio', 'manual')",
    )
    # SCAN-09: set True when the customer entered this value manually
    is_manual_override: bool = Field(
        False,
        description="True when the customer manually entered or corrected this measurement (SCAN-09)",
    )
    # F8: garment relevance flag (None when no garment_type was specified)
    is_required_for_garment: Optional[bool] = Field(
        None,
        description="True when this measurement is required for the requested garment type; None when no garment_type was specified",
    )
    # F9: ease allowance fields (None when no fit_style was specified)
    ease_cm: Optional[float] = Field(
        None,
        description="Ease allowance added to body measurement for the requested fit_style (same unit as value_cm)",
    )
    cutting_value_cm: Optional[float] = Field(
        None,
        description="value_cm + ease_cm — the dimension the tailor cuts to (same unit as value_cm)",
    )


# ---------------------------------------------------------------------------
# Full 32-measurement result — mirrors the spec taxonomy exactly
# ---------------------------------------------------------------------------

class ScanMeasurements(BaseModel):
    # Section A — Upper body circumferences
    M01_chest: MeasurementField
    M02_under_bust: MeasurementField        # women only; None value if skipped
    M03_waist: MeasurementField
    M04_abdomen: MeasurementField
    M05_hips: MeasurementField
    M06_neck: MeasurementField
    M07_bicep: MeasurementField
    M08_wrist: MeasurementField

    # Section B — Lower body circumferences
    M09_thigh: MeasurementField
    M10_mid_thigh: MeasurementField
    M11_knee: MeasurementField
    M12_calf: MeasurementField
    M13_ankle: MeasurementField

    # Section C — Lengths & heights
    M14_total_height: MeasurementField      # user-provided; always HIGH
    M15_shoulder_to_waist_front: MeasurementField
    M16_shoulder_to_waist_back: MeasurementField
    M17_kameez_length: MeasurementField
    M18_dress_length: MeasurementField
    M19_sleeve_length: MeasurementField
    M20_sleeve_length_elbow: MeasurementField
    M21_inseam: MeasurementField
    M22_outseam: MeasurementField
    M23_crotch_depth_front: MeasurementField
    M24_crotch_depth_back: MeasurementField
    M25_torso_length: MeasurementField

    # Section D — Widths & depths
    M26_shoulder_width: MeasurementField
    M27_chest_width: MeasurementField
    M28_back_width: MeasurementField
    M29_hip_width: MeasurementField
    M30_chest_depth: MeasurementField
    M31_waist_depth: MeasurementField
    M32_armhole_depth: MeasurementField


# ---------------------------------------------------------------------------
# Scan response
# ---------------------------------------------------------------------------

class ScanStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETE   = "complete"
    FAILED     = "failed"


# ---------------------------------------------------------------------------
# Async job schemas — POST /submit returns immediately; app polls /status
# ---------------------------------------------------------------------------

class ScanSubmitResponse(BaseModel):
    """Immediate acknowledgement returned by POST /submit."""
    session_id: str = Field(..., description="Server-assigned session ID — use for /status and /result polling")
    client_scan_id: Optional[str] = Field(None, description="Echo of the client_scan_id if supplied")
    status: Literal["QUEUED"] = "QUEUED"
    eta_seconds: int = Field(8, description="Estimated processing time in seconds")


class JobStatus(str, Enum):
    QUEUED     = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETE   = "COMPLETE"
    FAILED     = "FAILED"


class ScanStatusResponse(BaseModel):
    """Response from GET /status/{session_id} — poll every 2 s until COMPLETE."""
    session_id: str
    status: JobStatus
    progress_pct: int = Field(0, ge=0, le=100, description="0–100 processing progress")
    error: Optional[str] = Field(None, description="Set when status is FAILED")


# ---------------------------------------------------------------------------
# Validation result (returned alongside measurements in ScanResponse)
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    ERROR   = "error"    # blocks order placement
    WARNING = "warning"  # advisory only


class ValidationIssue(BaseModel):
    severity: Severity = Field(..., description="ERROR blocks order placement; WARNING is advisory only")
    code: str = Field(..., description="Machine-readable rule ID for Flutter localisation")
    message: str = Field(..., description="Human-readable explanation with actual values")
    fields: list[str] = Field(default_factory=list, description="Measurement codes involved")
    rescan_poses: list[str] = Field(default_factory=list, description="Pose IDs to redo to fix this issue")
    suggestion: str = Field("", description="Actionable one-liner for the customer")


class ValidationResult(BaseModel):
    is_valid: bool = Field(..., description="True only when zero ERRORs")
    can_order: bool = Field(..., description="True when no ERRORs on stitching-critical fields (M01/M03/M05/M26/M21/M19)")
    issues: list[ValidationIssue] = Field(default_factory=list)
    rescan_poses: list[str] = Field(default_factory=list, description="Deduplicated union of poses across all issues")
    summary: str = Field("", description="One-sentence display string for the app")

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


class ScanResponse(BaseModel):
    scan_id: str
    status: ScanStatus
    overall_confidence: Confidence
    frames_received: int
    height_cm: Optional[float] = Field(None, description="Resolved height used as scale anchor")
    height_source: str = Field(
        "user_input",
        description="How height was determined: 'user_input', 'sensor_fusion', or 'population_mean'",
    )
    response_unit: str = Field("cm", description="Unit used for all measurement values: 'cm' or 'in'")
    measurements: Optional[ScanMeasurements] = None
    validation: Optional[ValidationResult] = Field(None, description="Self-validation result — errors block ordering, warnings are advisory")
    garment_type: Optional[GarmentType] = Field(None, description="Garment type from the request, echoed back")
    fit_style: Optional[FitStyle] = Field(None, description="Fit style from the request, echoed back")
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal per-frame scoring result (not exposed to clients)
# ---------------------------------------------------------------------------

class FrameScore(BaseModel):
    pose_id: PoseID
    blur_score: float           # 0-1, higher = sharper
    pose_confidence: float      # 0-1, MediaPipe detection confidence
    angle_match: float          # 0-1, how well actual angle matches expected
    occlusion_score: float      # 0-1, 1 = no occlusion
    lighting_score: float       # 0-1
    composite: float            # weighted composite of above

    @property
    def is_usable(self) -> bool:
        # SCAN-04: reject frames scoring below 0.60
        return self.composite >= 0.60


# ---------------------------------------------------------------------------
# SCAN-09 — Manual measurement entry
# ---------------------------------------------------------------------------

class ManualMeasurementValue(BaseModel):
    """One manually entered measurement field with range validation."""
    value_cm: float = Field(..., gt=0.0, lt=500.0)


_MANUAL_RANGES: dict[str, tuple[float, float]] = {
    "M01": (50.0, 200.0), "M02": (40.0, 180.0), "M03": (40.0, 200.0),
    "M04": (40.0, 220.0), "M05": (50.0, 220.0), "M06": (20.0, 70.0),
    "M07": (15.0, 80.0),  "M08": (10.0, 40.0),  "M09": (25.0, 120.0),
    "M10": (20.0, 100.0), "M11": (18.0, 80.0),  "M12": (15.0, 70.0),
    "M13": (12.0, 50.0),  "M14": (100.0, 250.0),"M15": (20.0, 60.0),
    "M16": (18.0, 58.0),  "M17": (50.0, 150.0), "M18": (60.0, 180.0),
    "M19": (30.0, 100.0), "M20": (15.0, 55.0),  "M21": (50.0, 120.0),
    "M22": (60.0, 140.0), "M23": (15.0, 45.0),  "M24": (18.0, 50.0),
    "M25": (40.0, 100.0), "M26": (25.0, 70.0),  "M27": (20.0, 65.0),
    "M28": (20.0, 65.0),  "M29": (20.0, 70.0),  "M30": (10.0, 45.0),
    "M31": (8.0,  40.0),  "M32": (8.0,  30.0),
}


class ManualMeasurementRequest(BaseModel):
    """
    SCAN-09: Customer enters all 32 measurements manually.

    All fields are optional so partial overrides (e.g. correcting a single
    LOW-confidence field after an AI scan) are also supported.
    Each supplied value is validated against its physiological min/max range.
    """
    height_cm: float = Field(..., ge=100.0, le=250.0)
    garment_type: Optional[GarmentType] = Field(None, description="Optional garment context for required-field validation and ease allowances")
    fit_style: Optional[FitStyle] = Field(None, description="Optional fit style for ease allowance computation")

    M01_chest:       Optional[float] = Field(None, ge=_MANUAL_RANGES["M01"][0], le=_MANUAL_RANGES["M01"][1])
    M02_under_bust:  Optional[float] = Field(None, ge=_MANUAL_RANGES["M02"][0], le=_MANUAL_RANGES["M02"][1])
    M03_waist:       Optional[float] = Field(None, ge=_MANUAL_RANGES["M03"][0], le=_MANUAL_RANGES["M03"][1])
    M04_abdomen:     Optional[float] = Field(None, ge=_MANUAL_RANGES["M04"][0], le=_MANUAL_RANGES["M04"][1])
    M05_hips:        Optional[float] = Field(None, ge=_MANUAL_RANGES["M05"][0], le=_MANUAL_RANGES["M05"][1])
    M06_neck:        Optional[float] = Field(None, ge=_MANUAL_RANGES["M06"][0], le=_MANUAL_RANGES["M06"][1])
    M07_bicep:       Optional[float] = Field(None, ge=_MANUAL_RANGES["M07"][0], le=_MANUAL_RANGES["M07"][1])
    M08_wrist:       Optional[float] = Field(None, ge=_MANUAL_RANGES["M08"][0], le=_MANUAL_RANGES["M08"][1])
    M09_thigh:       Optional[float] = Field(None, ge=_MANUAL_RANGES["M09"][0], le=_MANUAL_RANGES["M09"][1])
    M10_mid_thigh:   Optional[float] = Field(None, ge=_MANUAL_RANGES["M10"][0], le=_MANUAL_RANGES["M10"][1])
    M11_knee:        Optional[float] = Field(None, ge=_MANUAL_RANGES["M11"][0], le=_MANUAL_RANGES["M11"][1])
    M12_calf:        Optional[float] = Field(None, ge=_MANUAL_RANGES["M12"][0], le=_MANUAL_RANGES["M12"][1])
    M13_ankle:       Optional[float] = Field(None, ge=_MANUAL_RANGES["M13"][0], le=_MANUAL_RANGES["M13"][1])
    M15_shoulder_to_waist_front: Optional[float] = Field(None, ge=_MANUAL_RANGES["M15"][0], le=_MANUAL_RANGES["M15"][1])
    M16_shoulder_to_waist_back:  Optional[float] = Field(None, ge=_MANUAL_RANGES["M16"][0], le=_MANUAL_RANGES["M16"][1])
    M17_kameez_length:           Optional[float] = Field(None, ge=_MANUAL_RANGES["M17"][0], le=_MANUAL_RANGES["M17"][1])
    M18_dress_length:            Optional[float] = Field(None, ge=_MANUAL_RANGES["M18"][0], le=_MANUAL_RANGES["M18"][1])
    M19_sleeve_length:           Optional[float] = Field(None, ge=_MANUAL_RANGES["M19"][0], le=_MANUAL_RANGES["M19"][1])
    M20_sleeve_length_elbow:     Optional[float] = Field(None, ge=_MANUAL_RANGES["M20"][0], le=_MANUAL_RANGES["M20"][1])
    M21_inseam:                  Optional[float] = Field(None, ge=_MANUAL_RANGES["M21"][0], le=_MANUAL_RANGES["M21"][1])
    M22_outseam:                 Optional[float] = Field(None, ge=_MANUAL_RANGES["M22"][0], le=_MANUAL_RANGES["M22"][1])
    M23_crotch_depth_front:      Optional[float] = Field(None, ge=_MANUAL_RANGES["M23"][0], le=_MANUAL_RANGES["M23"][1])
    M24_crotch_depth_back:       Optional[float] = Field(None, ge=_MANUAL_RANGES["M24"][0], le=_MANUAL_RANGES["M24"][1])
    M25_torso_length:            Optional[float] = Field(None, ge=_MANUAL_RANGES["M25"][0], le=_MANUAL_RANGES["M25"][1])
    M26_shoulder_width:          Optional[float] = Field(None, ge=_MANUAL_RANGES["M26"][0], le=_MANUAL_RANGES["M26"][1])
    M27_chest_width:             Optional[float] = Field(None, ge=_MANUAL_RANGES["M27"][0], le=_MANUAL_RANGES["M27"][1])
    M28_back_width:              Optional[float] = Field(None, ge=_MANUAL_RANGES["M28"][0], le=_MANUAL_RANGES["M28"][1])
    M29_hip_width:               Optional[float] = Field(None, ge=_MANUAL_RANGES["M29"][0], le=_MANUAL_RANGES["M29"][1])
    M30_chest_depth:             Optional[float] = Field(None, ge=_MANUAL_RANGES["M30"][0], le=_MANUAL_RANGES["M30"][1])
    M31_waist_depth:             Optional[float] = Field(None, ge=_MANUAL_RANGES["M31"][0], le=_MANUAL_RANGES["M31"][1])
    M32_armhole_depth:           Optional[float] = Field(None, ge=_MANUAL_RANGES["M32"][0], le=_MANUAL_RANGES["M32"][1])
