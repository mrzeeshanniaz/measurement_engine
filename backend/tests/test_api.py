"""
Integration tests for scan API endpoints — uses FastAPI TestClient.

Tests that don't require loaded ML models:
  GET  /api/v1/scan/health
  GET  /api/v1/scan/validation-rules
  POST /api/v1/scan/manual
  GET  /api/v1/scan/status/{id}  (not-found case)
  GET  /api/v1/scan/result/{id}  (not-found case)
"""
from __future__ import annotations

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _dummy_b64(width: int = 64, height: int = 128) -> str:
    img = Image.new("RGB", (width, height), color=(200, 180, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# GET /api/v1/scan/health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/api/v1/scan/health")
        assert r.status_code == 200

    def test_health_has_required_keys(self, client):
        data = client.get("/api/v1/scan/health").json()
        assert "pipeline" in data
        assert "models_loaded" in data
        assert "jobs_queued" in data
        assert "jobs_processing" in data
        assert "jobs_complete" in data
        assert "jobs_failed" in data

    def test_root_health_liveness(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# GET /api/v1/scan/validation-rules
# ---------------------------------------------------------------------------

class TestValidationRules:
    def test_returns_200(self, client):
        r = client.get("/api/v1/scan/validation-rules")
        assert r.status_code == 200

    def test_has_expected_sections(self, client):
        data = client.get("/api/v1/scan/validation-rules").json()
        assert "version" in data
        assert "field_ranges" in data
        assert "population_norms" in data
        assert "cross_rules" in data

    def test_field_ranges_cover_all_m_codes(self, client):
        data = client.get("/api/v1/scan/validation-rules").json()
        ranges = data["field_ranges"]
        for code in ("M01", "M03", "M05", "M21", "M26"):
            assert code in ranges, f"{code} missing from field_ranges"

    def test_cross_rules_are_list(self, client):
        data = client.get("/api/v1/scan/validation-rules").json()
        assert isinstance(data["cross_rules"], list)
        assert len(data["cross_rules"]) >= 10

    def test_cross_rules_have_required_fields(self, client):
        data = client.get("/api/v1/scan/validation-rules").json()
        for rule in data["cross_rules"]:
            assert "id" in rule
            assert "severity" in rule
            assert "fields" in rule


# ---------------------------------------------------------------------------
# POST /api/v1/scan/manual
# ---------------------------------------------------------------------------

class TestManualScan:
    def _body(self, **kwargs) -> dict:
        return {"height_cm": 175.0, **kwargs}

    def test_minimal_request_returns_200(self, client):
        r = client.post("/api/v1/scan/manual", json=self._body())
        assert r.status_code == 200

    def test_response_has_scan_id(self, client):
        data = client.post("/api/v1/scan/manual", json=self._body()).json()
        assert "scan_id" in data
        assert data["scan_id"]

    def test_response_has_32_measurements(self, client):
        data = client.post("/api/v1/scan/manual", json=self._body()).json()
        meas = data["measurements"]
        assert len(meas) == 32

    def test_supplied_chest_appears_in_response(self, client):
        data = client.post("/api/v1/scan/manual", json=self._body(M01_chest=96.0)).json()
        assert data["measurements"]["M01_chest"]["value_cm"] == 96.0

    def test_supplied_field_is_manual_override(self, client):
        data = client.post("/api/v1/scan/manual", json=self._body(M01_chest=96.0)).json()
        assert data["measurements"]["M01_chest"]["is_manual_override"] is True

    def test_unsupplied_field_has_null_value(self, client):
        data = client.post("/api/v1/scan/manual", json=self._body()).json()
        assert data["measurements"]["M01_chest"]["value_cm"] is None

    def test_overall_confidence_is_medium(self, client):
        data = client.post("/api/v1/scan/manual", json=self._body(
            M01_chest=100.0, M03_waist=84.0, M05_hips=98.0
        )).json()
        assert data["overall_confidence"] in ("MEDIUM", "LOW")

    def test_inches_unit_conversion(self, client):
        data_cm = client.post("/api/v1/scan/manual?units=cm", json=self._body(M01_chest=96.0)).json()
        data_in = client.post("/api/v1/scan/manual?units=in", json=self._body(M01_chest=96.0)).json()
        assert data_in["response_unit"] == "in"
        chest_in = data_in["measurements"]["M01_chest"]["value_cm"]
        chest_cm = data_cm["measurements"]["M01_chest"]["value_cm"]
        assert chest_in is not None and chest_cm is not None
        assert abs(chest_in - round(chest_cm * 0.393701, 2)) < 0.01

    def test_garment_type_echoed_in_response(self, client):
        data = client.post(
            "/api/v1/scan/manual",
            json=self._body(garment_type="kameez", fit_style="regular"),
        ).json()
        assert data["garment_type"] == "kameez"
        assert data["fit_style"] == "regular"

    def test_below_min_height_returns_422(self, client):
        r = client.post("/api/v1/scan/manual", json={"height_cm": 50.0})
        assert r.status_code == 422

    def test_invalid_measurement_value_returns_422(self, client):
        r = client.post("/api/v1/scan/manual", json=self._body(M01_chest=999.0))
        assert r.status_code == 422

    def test_validation_result_present(self, client):
        data = client.post("/api/v1/scan/manual", json=self._body(M01_chest=96.0)).json()
        assert "validation" in data
        assert "is_valid" in data["validation"]
        assert "can_order" in data["validation"]
        assert "issues" in data["validation"]


# ---------------------------------------------------------------------------
# GET /api/v1/scan/status — not-found
# ---------------------------------------------------------------------------

class TestScanStatusNotFound:
    def test_unknown_session_returns_404(self, client):
        r = client.get("/api/v1/scan/status/nonexistent-session-id")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/scan/result — not-found
# ---------------------------------------------------------------------------

class TestScanResultNotFound:
    def test_unknown_session_returns_404(self, client):
        r = client.get("/api/v1/scan/result/nonexistent-session-id")
        assert r.status_code == 404
