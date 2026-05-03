"""
POST /api/v1/scan/submit  — submit height + frames, receive 32 measurements
GET  /api/v1/scan/health  — liveness check for the pipeline
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

import uuid

from app.measurement_engine.scan.pipeline import ScanPipeline
from app.measurement_engine.scan.schemas import (
    Confidence,
    ManualMeasurementRequest,
    MeasurementField,
    ScanMeasurements,
    ScanResponse,
    ScanStatus,
    ScanSubmitRequest,
)
from app.measurement_engine.scan.validator import export_rules_for_frontend, validate

logger = logging.getLogger(__name__)
router = APIRouter()

_CM_TO_IN = 0.393701


def _to_inches(resp: ScanResponse) -> ScanResponse:
    """Return a copy of resp with all measurement values converted to inches."""
    new_height = round(resp.height_cm * _CM_TO_IN, 2) if resp.height_cm is not None else None

    if resp.measurements is None:
        return resp.model_copy(update={"response_unit": "in", "height_cm": new_height})

    converted: dict = {}
    for fname in resp.measurements.model_fields:
        mf: MeasurementField = getattr(resp.measurements, fname)
        new_val = round(mf.value_cm * _CM_TO_IN, 2) if mf.value_cm is not None else None
        converted[fname] = mf.model_copy(update={"value_cm": new_val, "unit": "in"})

    return resp.model_copy(update={
        "response_unit": "in",
        "height_cm": new_height,
        "measurements": ScanMeasurements(**converted),
    })


@router.post(
    "/submit",
    response_model=ScanResponse,
    summary="Submit a guided video scan for body measurement extraction",
    responses={
        400: {"description": "Invalid request (missing front frame, bad base64, etc.)"},
        503: {"description": "Models not yet loaded"},
    },
)
async def submit_scan(
    request: Request,
    body: ScanSubmitRequest,
    units: Literal["cm", "in"] = Query("cm", description="Response unit for all measurements ('cm' or 'in')"),
) -> ScanResponse:
    """
    Accepts the customer's height and a set of pose-labelled frames selected
    by the on-device scorer.  Returns all 32 body measurements with
    per-field confidence levels (HIGH / MEDIUM / LOW).

    **Minimum viable submission:** height_cm + one FRONT frame.
    For full 32-measurement extraction supply all 7 pose frames.
    """
    models = getattr(request.app.state, "models", None)
    if models is None or not models.is_loaded:
        raise HTTPException(status_code=503, detail="Models not ready")

    logger.info(
        "Scan submitted — height=%s cm, frames=%d",
        f"{body.height_cm:.1f}" if body.height_cm else "auto",
        len(body.frames),
    )

    pipeline = ScanPipeline(pose_model=models.pose, smpl_model=models.smpl)
    result = pipeline.run(
        frames=body.frames,
        height_cm=body.height_cm,
        camera_metadata=body.camera_metadata,
    )

    if result.status.value == "failed":
        raise HTTPException(status_code=500, detail=result.error or "Pipeline failed")

    return _to_inches(result) if units == "in" else result


@router.post(
    "/manual",
    response_model=ScanResponse,
    summary="Submit all 32 measurements entered manually (SCAN-09)",
)
async def submit_manual(
    body: ManualMeasurementRequest,
    units: Literal["cm", "in"] = Query("cm", description="Response unit for all measurements ('cm' or 'in')"),
) -> ScanResponse:
    """
    SCAN-09: Accept a full (or partial) set of manually entered measurements.

    - All supplied fields are flagged `is_manual_override=true` and given
      MEDIUM confidence (human measurement error ±1–2 cm is expected).
    - Missing fields default to None with LOW confidence.
    - The tailor sheet will display a MANUAL flag on each overridden field.
    """

    def _field(value: float | None) -> MeasurementField:
        if value is None:
            return MeasurementField(value_cm=None, confidence=Confidence.LOW, source="manual", is_manual_override=True)
        return MeasurementField(
            value_cm=round(value, 1),
            confidence=Confidence.MEDIUM,
            source="manual",
            is_manual_override=True,
        )

    measurements = ScanMeasurements(
        M01_chest=_field(body.M01_chest),
        M02_under_bust=_field(body.M02_under_bust),
        M03_waist=_field(body.M03_waist),
        M04_abdomen=_field(body.M04_abdomen),
        M05_hips=_field(body.M05_hips),
        M06_neck=_field(body.M06_neck),
        M07_bicep=_field(body.M07_bicep),
        M08_wrist=_field(body.M08_wrist),
        M09_thigh=_field(body.M09_thigh),
        M10_mid_thigh=_field(body.M10_mid_thigh),
        M11_knee=_field(body.M11_knee),
        M12_calf=_field(body.M12_calf),
        M13_ankle=_field(body.M13_ankle),
        M14_total_height=MeasurementField(
            value_cm=round(body.height_cm, 1),
            confidence=Confidence.HIGH,
            source="manual",
            is_manual_override=True,
        ),
        M15_shoulder_to_waist_front=_field(body.M15_shoulder_to_waist_front),
        M16_shoulder_to_waist_back=_field(body.M16_shoulder_to_waist_back),
        M17_kameez_length=_field(body.M17_kameez_length),
        M18_dress_length=_field(body.M18_dress_length),
        M19_sleeve_length=_field(body.M19_sleeve_length),
        M20_sleeve_length_elbow=_field(body.M20_sleeve_length_elbow),
        M21_inseam=_field(body.M21_inseam),
        M22_outseam=_field(body.M22_outseam),
        M23_crotch_depth_front=_field(body.M23_crotch_depth_front),
        M24_crotch_depth_back=_field(body.M24_crotch_depth_back),
        M25_torso_length=_field(body.M25_torso_length),
        M26_shoulder_width=_field(body.M26_shoulder_width),
        M27_chest_width=_field(body.M27_chest_width),
        M28_back_width=_field(body.M28_back_width),
        M29_hip_width=_field(body.M29_hip_width),
        M30_chest_depth=_field(body.M30_chest_depth),
        M31_waist_depth=_field(body.M31_waist_depth),
        M32_armhole_depth=_field(body.M32_armhole_depth),
    )

    validation = validate(measurements, body.height_cm)

    resp = ScanResponse(
        scan_id=str(uuid.uuid4()),
        status=ScanStatus.COMPLETE,
        overall_confidence=Confidence.MEDIUM,
        frames_received=0,
        height_cm=body.height_cm,
        height_source="manual",
        measurements=measurements,
        validation=validation,
    )
    return _to_inches(resp) if units == "in" else resp


@router.get(
    "/validation-rules",
    summary="Export validation rules for client-side checks (SCAN-09)",
    response_model=None,
)
async def validation_rules() -> dict:
    """
    Returns all validation rules as a JSON-serialisable dict so the Flutter app
    can run the same checks client-side before submitting a manual entry form.

    Payload sections:
    - **field_ranges**: absolute physiological min/max per measurement code
    - **population_norms**: ANSUR II height-relative mean/SD ratios and Z-score thresholds
    - **cross_rules**: ordered list of cross-measurement consistency constraints

    The Flutter SDK should fetch this once per app launch (or on update) and
    cache it locally.  The `version` field changes whenever rules are updated.
    """
    return export_rules_for_frontend()


@router.get("/health", summary="Pipeline health check")
async def scan_health(request: Request) -> dict:
    models = getattr(request.app.state, "models", None)
    return {
        "pipeline": "ok",
        "models_loaded": models.is_loaded if models else False,
        "pose_model": models.pose.is_loaded if models else False,
        "smpl_model": models.smpl.is_loaded if models else False,
    }
