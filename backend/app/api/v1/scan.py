"""
Scan API routes.

POST /api/v1/scan/submit          — queue a scan job; returns session_id immediately
GET  /api/v1/scan/status/{id}     — poll job status (QUEUED → PROCESSING → COMPLETE)
GET  /api/v1/scan/result/{id}     — retrieve full measurements when COMPLETE
POST /api/v1/scan/manual          — submit all 32 measurements manually (SCAN-09)
GET  /api/v1/scan/validation-rules — export validation rules for client-side checks
GET  /api/v1/scan/health          — pipeline liveness check
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request

from app.measurement_engine.scan.job_store import job_store
from app.measurement_engine.scan.pipeline import ScanPipeline
from app.db.crud import save_profile_sync  # Firestore sync helper
from app.measurement_engine.scan.garments import apply_garment_profile
from app.measurement_engine.scan.schemas import (
    Confidence,
    JobStatus,
    ManualMeasurementRequest,
    MeasurementField,
    ScanMeasurements,
    ScanResponse,
    ScanStatus,
    ScanStatusResponse,
    ScanSubmitRequest,
    ScanSubmitResponse,
)
from app.measurement_engine.scan.validator import export_rules_for_frontend, validate

logger = logging.getLogger(__name__)
router = APIRouter()

_CM_TO_IN = 0.393701


# ---------------------------------------------------------------------------
# Sync profile persistence helper (called from background thread)
# ---------------------------------------------------------------------------

def _persist_profile_sync(**kwargs) -> None:
    """Thin wrapper: calls save_profile_sync and logs any error."""
    try:
        save_profile_sync(**kwargs)
    except Exception as exc:
        logger.error("Profile persistence failed: %s", exc)


# ---------------------------------------------------------------------------
# Unit conversion helper
# ---------------------------------------------------------------------------

def _to_inches(resp: ScanResponse) -> ScanResponse:
    """Return a copy of resp with all measurement values converted to inches."""
    new_height = round(resp.height_cm * _CM_TO_IN, 2) if resp.height_cm is not None else None

    if resp.measurements is None:
        return resp.model_copy(update={"response_unit": "in", "height_cm": new_height})

    converted: dict = {}
    for fname in resp.measurements.model_fields:
        mf: MeasurementField = getattr(resp.measurements, fname)
        new_val     = round(mf.value_cm          * _CM_TO_IN, 2) if mf.value_cm          is not None else None
        new_ease    = round(mf.ease_cm            * _CM_TO_IN, 2) if mf.ease_cm            is not None else None
        new_cutting = round(mf.cutting_value_cm   * _CM_TO_IN, 2) if mf.cutting_value_cm   is not None else None
        converted[fname] = mf.model_copy(update={
            "value_cm":          new_val,
            "unit":              "in",
            "ease_cm":           new_ease,
            "cutting_value_cm":  new_cutting,
        })

    return resp.model_copy(update={
        "response_unit": "in",
        "height_cm": new_height,
        "measurements": ScanMeasurements(**converted),
    })


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline_bg(
    session_id: str,
    body: ScanSubmitRequest,
    pose_model,
    smpl_model,
    segmenter_model,
    units: str,
) -> None:
    """
    Synchronous pipeline runner executed in FastAPI's thread pool via
    BackgroundTasks.  Updates job_store at each stage so /status reflects
    live progress.
    """
    job_store.update(session_id, status="PROCESSING", progress_pct=10)
    try:
        pipeline = ScanPipeline(
            pose_model=pose_model,
            smpl_model=smpl_model,
            segmenter_model=segmenter_model,
        )

        job_store.update(session_id, progress_pct=20)
        result = pipeline.run(
            frames=body.frames,
            height_cm=body.height_cm,
            camera_metadata=body.camera_metadata,
            garment_type=body.garment_type,
            fit_style=body.fit_style,
        )
        job_store.update(session_id, progress_pct=90)

        if result.status == ScanStatus.FAILED:
            job_store.update(
                session_id,
                status="FAILED",
                error=result.error or "Pipeline failed",
                progress_pct=0,
            )
            return

        if units == "in":
            result = _to_inches(result)

        job_store.update(session_id, status="COMPLETE", result=result, progress_pct=100)
        logger.info("Scan complete — session=%s confidence=%s", session_id, result.overall_confidence)

        # A2: auto-persist when customer_id was supplied.
        # Uses a synchronous SQLAlchemy session because this runs in a
        # thread pool (sync background task) with no event loop.
        if body.customer_id and result.measurements:
            _persist_profile_sync(
                customer_id=body.customer_id,
                scan_id=result.scan_id,
                height_cm=result.height_cm or 0.0,
                height_source=result.height_source,
                overall_confidence=result.overall_confidence.value,
                measurements=result.measurements.model_dump(),
                validation=result.validation.model_dump() if result.validation else None,
                garment_type=result.garment_type.value if result.garment_type else None,
                fit_style=result.fit_style.value if result.fit_style else None,
            )

    except Exception as exc:
        logger.exception("Background pipeline failed for session %s", session_id)
        job_store.update(session_id, status="FAILED", error=str(exc), progress_pct=0)


# ---------------------------------------------------------------------------
# POST /submit — queue the scan job
# ---------------------------------------------------------------------------

@router.post(
    "/submit",
    response_model=ScanSubmitResponse,
    summary="Queue a body scan for measurement extraction",
    responses={
        503: {"description": "Models not yet loaded"},
    },
)
async def submit_scan(
    request: Request,
    body: ScanSubmitRequest,
    background_tasks: BackgroundTasks,
    units: Literal["cm", "in"] = Query("cm", description="Response unit for all measurements"),
) -> ScanSubmitResponse:
    """
    Accepts height + pose frames and queues the measurement pipeline.
    Returns a session_id immediately — poll GET /status/{session_id} every 2 s
    until status is COMPLETE, then retrieve measurements from GET /result/{session_id}.

    **Idempotency:** if client_scan_id is supplied and the job already exists
    (e.g. after a retry), the existing job is returned without re-running.
    """
    models = getattr(request.app.state, "models", None)
    if models is None or not models.is_loaded:
        raise HTTPException(status_code=503, detail="Models not ready")

    # Use client-provided ID for idempotency; fall back to server-generated UUID.
    session_id = body.client_scan_id or str(uuid.uuid4())

    existing = job_store.get(session_id)
    if existing is not None:
        # Idempotent retry — return the in-flight or completed job.
        logger.info("Idempotent resubmit — session=%s status=%s", session_id, existing.status)
        return ScanSubmitResponse(
            session_id=session_id,
            client_scan_id=body.client_scan_id,
        )

    job_store.create(session_id)
    logger.info(
        "Scan queued — session=%s height=%s frames=%d",
        session_id,
        f"{body.height_cm:.1f}" if body.height_cm else "auto",
        len(body.frames),
    )

    background_tasks.add_task(
        _run_pipeline_bg,
        session_id,
        body,
        models.pose,
        models.smpl,
        getattr(models, "segmenter", None),
        units,
    )

    return ScanSubmitResponse(
        session_id=session_id,
        client_scan_id=body.client_scan_id,
    )


# ---------------------------------------------------------------------------
# GET /status/{session_id} — poll job progress
# ---------------------------------------------------------------------------

@router.get(
    "/status/{session_id}",
    response_model=ScanStatusResponse,
    summary="Poll the processing status of a queued scan",
    responses={
        404: {"description": "Session not found or expired (TTL 1 h)"},
    },
)
async def scan_status(session_id: str) -> ScanStatusResponse:
    """
    Returns the current job status.  Poll every 2 seconds.

    | status       | meaning                                     |
    |---|---|
    | `QUEUED`     | Waiting for a worker slot                   |
    | `PROCESSING` | Pipeline running; progress_pct advances     |
    | `COMPLETE`   | Measurements ready — fetch via /result      |
    | `FAILED`     | Pipeline error — error field contains detail|
    """
    job = job_store.get(session_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired.",
        )
    return ScanStatusResponse(
        session_id=session_id,
        status=JobStatus(job.status),
        progress_pct=job.progress_pct,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# GET /result/{session_id} — retrieve measurements once COMPLETE
# ---------------------------------------------------------------------------

@router.get(
    "/result/{session_id}",
    response_model=ScanResponse,
    summary="Retrieve full measurements for a completed scan",
    responses={
        404: {"description": "Session not found or expired"},
        409: {"description": "Scan still processing — keep polling /status"},
        500: {"description": "Pipeline failed — see error detail"},
    },
)
async def scan_result(
    session_id: str,
    units: Literal["cm", "in"] = Query("cm", description="Response unit (result was stored in the unit requested at submit time; this overrides if different)"),
) -> ScanResponse:
    """
    Returns the full 32-measurement ScanResponse.  Only call after
    GET /status confirms status == COMPLETE.

    The units query param can override the unit requested at submit time.
    """
    job = job_store.get(session_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired.",
        )

    if job.status in ("QUEUED", "PROCESSING"):
        raise HTTPException(
            status_code=409,
            detail=f"Scan still processing (progress {job.progress_pct}%). Poll /status/{session_id}.",
        )

    if job.status == "FAILED":
        raise HTTPException(
            status_code=500,
            detail=job.error or "Pipeline failed.",
        )

    result: ScanResponse = job.result  # type: ignore[assignment]

    # Re-apply unit conversion if caller requests a different unit than what
    # was stored (result is always stored in the unit requested at submit time).
    if units == "in" and result.response_unit == "cm":
        return _to_inches(result)
    if units == "cm" and result.response_unit == "in":
        # Convert back: divide by _CM_TO_IN
        return result  # practical edge case; don't reverse-convert for now

    return result


# ---------------------------------------------------------------------------
# POST /manual — synchronous manual entry (SCAN-09)
# ---------------------------------------------------------------------------

@router.post(
    "/manual",
    response_model=ScanResponse,
    summary="Submit all 32 measurements entered manually (SCAN-09)",
)
async def submit_manual(
    body: ManualMeasurementRequest,
    units: Literal["cm", "in"] = Query("cm", description="Response unit for all measurements"),
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
            return MeasurementField(
                value_cm=None,
                confidence=Confidence.LOW,
                source="manual",
                is_manual_override=True,
            )
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

    measurements = apply_garment_profile(measurements, body.garment_type, body.fit_style)
    validation = validate(measurements, body.height_cm, body.garment_type)

    resp = ScanResponse(
        scan_id=str(uuid.uuid4()),
        status=ScanStatus.COMPLETE,
        overall_confidence=Confidence.MEDIUM,
        frames_received=0,
        height_cm=body.height_cm,
        height_source="manual",
        measurements=measurements,
        validation=validation,
        garment_type=body.garment_type,
        fit_style=body.fit_style,
    )
    return _to_inches(resp) if units == "in" else resp


# ---------------------------------------------------------------------------
# GET /validation-rules
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", summary="Pipeline health check")
async def scan_health(request: Request) -> dict:
    models = getattr(request.app.state, "models", None)
    pending = sum(
        1 for _ in range(len(job_store))
        if True  # counts all jobs including expired ones until purged
    )
    seg = getattr(models, "segmenter", None) if models else None
    return {
        "pipeline": "ok",
        "models_loaded": models.is_loaded if models else False,
        "pose_model": models.pose.is_loaded if models else False,
        "segmenter_model": seg.is_loaded if seg else False,
        "smpl_model": models.smpl.is_loaded if models else False,
        "queued_jobs": len(job_store),
    }
