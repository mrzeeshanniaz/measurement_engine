"""
Tests for request schema validation — boundary values, required fields, validators.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.measurement_engine.scan.schemas import (
    FitStyle,
    GarmentType,
    ManualMeasurementRequest,
    PoseFrame,
    PoseID,
    ScaleTier,
    ScanSubmitRequest,
)


def _minimal_submit(**kwargs) -> dict:
    # Dummy base64 padded to ≥100 chars so it passes PoseFrame's sanity check
    # (real production payloads are much larger; the floor protects workers).
    dummy_b64 = "dGVzdA==" * 20
    return {
        "height_cm": 175.0,
        "frames": [{"pose_id": "front", "image_b64": dummy_b64, "quality_score": 0.9}],
        **kwargs,
    }


# ---------------------------------------------------------------------------
# ManualMeasurementRequest — boundary value validation (ge/le)
# ---------------------------------------------------------------------------

class TestManualMeasurementBoundaries:
    def test_exact_minimum_accepted(self):
        req = ManualMeasurementRequest(height_cm=100.0, M01_chest=50.0)
        assert req.M01_chest == 50.0

    def test_exact_maximum_accepted(self):
        req = ManualMeasurementRequest(height_cm=175.0, M01_chest=200.0)
        assert req.M01_chest == 200.0

    def test_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            ManualMeasurementRequest(height_cm=175.0, M01_chest=49.9)

    def test_above_maximum_rejected(self):
        with pytest.raises(ValidationError):
            ManualMeasurementRequest(height_cm=175.0, M01_chest=200.1)

    def test_height_exact_minimum_accepted(self):
        req = ManualMeasurementRequest(height_cm=100.0)
        assert req.height_cm == 100.0

    def test_height_exact_maximum_accepted(self):
        req = ManualMeasurementRequest(height_cm=250.0)
        assert req.height_cm == 250.0

    def test_height_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            ManualMeasurementRequest(height_cm=99.9)

    def test_all_fields_optional(self):
        req = ManualMeasurementRequest(height_cm=175.0)
        assert req.M01_chest is None
        assert req.M21_inseam is None

    def test_garment_type_accepted(self):
        req = ManualMeasurementRequest(
            height_cm=175.0,
            garment_type=GarmentType.KAMEEZ,
            fit_style=FitStyle.REGULAR,
        )
        assert req.garment_type == GarmentType.KAMEEZ


# ---------------------------------------------------------------------------
# ScanSubmitRequest — validators
# ---------------------------------------------------------------------------

class TestScanSubmitRequest:
    def test_minimal_request_valid(self):
        req = ScanSubmitRequest(**_minimal_submit())
        assert req.height_cm == 175.0
        assert req.scale_tier == ScaleTier.TIER2  # default

    def test_requires_front_frame(self):
        with pytest.raises(ValidationError):
            ScanSubmitRequest(**_minimal_submit(
                frames=[{"pose_id": "back", "image_b64": "dGVzdA==", "quality_score": 0.9}]
            ))

    def test_requires_height_or_camera_metadata(self):
        with pytest.raises(ValidationError):
            ScanSubmitRequest(
                height_cm=None,
                camera_metadata=None,
                frames=[{"pose_id": "front", "image_b64": "dGVzdA==", "quality_score": 0.9}],
            )

    def test_scale_tier_enum(self):
        req = ScanSubmitRequest(**_minimal_submit(scale_tier="TIER1"))
        assert req.scale_tier == ScaleTier.TIER1

    def test_garment_type_and_fit_style_optional(self):
        req = ScanSubmitRequest(**_minimal_submit(
            garment_type="kameez",
            fit_style="regular",
        ))
        assert req.garment_type == GarmentType.KAMEEZ
        assert req.fit_style == FitStyle.REGULAR

    def test_client_scan_id_optional(self):
        req = ScanSubmitRequest(**_minimal_submit(client_scan_id="abc-123"))
        assert req.client_scan_id == "abc-123"

    def test_invalid_scale_tier_rejected(self):
        with pytest.raises(ValidationError):
            ScanSubmitRequest(**_minimal_submit(scale_tier="TIER9"))


# ---------------------------------------------------------------------------
# PoseID enum coverage
# ---------------------------------------------------------------------------

class TestPoseID:
    def test_all_7_pose_ids_defined(self):
        assert len(PoseID) == 7

    def test_pose_id_values(self):
        assert PoseID.FRONT.value == "front"
        assert PoseID.ARMS_OUT.value == "arms_out"
        assert PoseID.BACK.value == "back"
