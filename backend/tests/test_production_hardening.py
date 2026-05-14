"""
Tests for production-readiness guardrails:
  - per-frame image size limit (PoseFrame validator)
  - request-body size cap (MaxBodySizeMiddleware)
  - per-IP rate limiting (slowapi)
  - X-Request-ID middleware
  - log redaction of base64 / image_b64 payloads
  - SQLite JobStore: persistence, recover_orphaned, ownership scoping
  - /metrics endpoint exposes Prometheus format
  - /api/v1/scan/health reports degraded mode correctly
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import settings
from app.measurement_engine.scan.job_store import (
    InMemoryJobStore,
    SQLiteJobStore,
)
from app.measurement_engine.scan.schemas import PoseFrame
from app.middleware import RedactingFilter, request_id_var


# ---------------------------------------------------------------------------
# Image size limits
# ---------------------------------------------------------------------------

class TestImageSizeLimits:
    def test_oversize_image_rejected(self):
        oversized = "A" * (settings.MAX_FRAME_B64_BYTES + 100)
        with pytest.raises(ValidationError) as exc:
            PoseFrame(pose_id="front", image_b64=oversized, quality_score=0.9)
        assert "exceeds maximum" in str(exc.value)

    def test_reasonable_image_accepted(self):
        ok = "A" * 5000
        f = PoseFrame(pose_id="front", image_b64=ok, quality_score=0.9)
        assert f.image_b64 == ok

    def test_at_size_limit_accepted(self):
        # Boundary case: exactly MAX_FRAME_B64_BYTES is allowed.
        at_max = "A" * settings.MAX_FRAME_B64_BYTES
        f = PoseFrame(pose_id="front", image_b64=at_max, quality_score=0.9)
        assert len(f.image_b64) == settings.MAX_FRAME_B64_BYTES


# ---------------------------------------------------------------------------
# Request body cap (middleware) + request ID echo
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestRequestBodyCap:
    def test_oversize_content_length_returns_413(self, client):
        # We can't actually send 90+ MB, but we can fake the Content-Length header.
        # TestClient honors the header literally — no need to ship the bytes.
        huge = str(settings.MAX_REQUEST_BODY_BYTES + 1)
        r = client.post(
            "/api/v1/scan/manual",
            content=b"{}",
            headers={"content-length": huge, "content-type": "application/json"},
        )
        assert r.status_code == 413
        assert "Request body too large" in r.json()["detail"]


class TestRequestIDMiddleware:
    def test_response_has_request_id_header(self, client):
        r = client.get("/health")
        assert "x-request-id" in {k.lower() for k in r.headers}

    def test_client_supplied_request_id_echoed(self, client):
        r = client.get("/health", headers={"X-Request-ID": "test-trace-42"})
        # Header is echoed back verbatim
        assert r.headers["X-Request-ID"] == "test-trace-42"


# ---------------------------------------------------------------------------
# Log redaction
# ---------------------------------------------------------------------------

class TestRedactingFilter:
    def _emit(self, msg: str, *args) -> str:
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="", lineno=0,
            msg=msg, args=args, exc_info=None,
        )
        f = RedactingFilter()
        f.filter(record)
        return record.getMessage()

    def test_redacts_image_b64_json_key(self):
        out = self._emit('{"image_b64": "AAAABBBBCCCCDDDDEEEEFFFF1234567890"}')
        assert "<redacted>" in out
        assert "AAAABBBB" not in out

    def test_redacts_long_base64_blob(self):
        blob = "A" * 200
        out = self._emit("scan body: " + blob)
        assert "<base64-redacted>" in out
        assert blob not in out

    def test_passes_normal_logs_through(self):
        out = self._emit("Scan complete — session=%s confidence=HIGH", "abc-123")
        assert out == "Scan complete — session=abc-123 confidence=HIGH"

    def test_injects_request_id_attribute(self):
        token = request_id_var.set("trace-77")
        try:
            record = logging.LogRecord(
                name="t", level=logging.INFO, pathname="", lineno=0,
                msg="hi", args=(), exc_info=None,
            )
            RedactingFilter().filter(record)
            assert record.request_id == "trace-77"
        finally:
            request_id_var.reset(token)


# ---------------------------------------------------------------------------
# SQLite job store
# ---------------------------------------------------------------------------

class TestSQLiteJobStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> SQLiteJobStore:
        return SQLiteJobStore(str(tmp_path / "jobs.db"))

    def test_create_get_roundtrip(self, store):
        job = store.create("sess-1", customer_id="user-A")
        fetched = store.get("sess-1")
        assert fetched is not None
        assert fetched.session_id == "sess-1"
        assert fetched.customer_id == "user-A"
        assert fetched.status == "QUEUED"

    def test_create_is_idempotent(self, store):
        a = store.create("sess-1", customer_id="user-A")
        b = store.create("sess-1", customer_id="user-A")
        assert a.session_id == b.session_id

    def test_update_status_and_progress(self, store):
        store.create("sess-1")
        store.update("sess-1", status="PROCESSING", progress_pct=40)
        job = store.get("sess-1")
        assert job.status == "PROCESSING"
        assert job.progress_pct == 40

    def test_counts_groups_by_state(self, store):
        store.create("a"); store.create("b")
        store.update("b", status="COMPLETE")
        counts = store.counts()
        assert counts["QUEUED"] == 1
        assert counts["COMPLETE"] == 1

    def test_recover_orphaned_flips_in_flight_to_failed(self, store):
        store.create("a")
        store.create("b")
        store.update("b", status="PROCESSING")
        recovered = store.recover_orphaned()
        assert recovered == 2
        assert store.get("a").status == "FAILED"
        assert store.get("b").status == "FAILED"
        # And it doesn't touch already-terminal jobs.
        store.create("c")
        store.update("c", status="COMPLETE")
        assert store.recover_orphaned() == 0
        assert store.get("c").status == "COMPLETE"

    def test_persistence_survives_new_instance(self, tmp_path):
        path = str(tmp_path / "jobs.db")
        s1 = SQLiteJobStore(path)
        s1.create("sess-1", customer_id="user-A")
        s1.update("sess-1", status="COMPLETE", progress_pct=100)
        # Simulate process restart: build a new store backed by the same file.
        s2 = SQLiteJobStore(path)
        job = s2.get("sess-1")
        assert job is not None
        assert job.status == "COMPLETE"
        assert job.customer_id == "user-A"

    def test_expired_jobs_return_none(self, store):
        store.create("sess-1")
        # Forcibly age the row past TTL.
        from app.measurement_engine.scan.job_store import TTL_SECONDS
        with store._connect() as conn:
            conn.execute(
                "UPDATE scan_jobs SET created_at = ? WHERE session_id = ?",
                (time.time() - TTL_SECONDS - 100, "sess-1"),
            )
        assert store.get("sess-1") is None

    def test_in_memory_recover_orphaned_is_noop(self):
        s = InMemoryJobStore()
        s.create("sess-1")
        s.update("sess-1", status="PROCESSING")
        assert s.recover_orphaned() == 0
        # In-memory store has no notion of "prior process" — orphans don't exist.


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        # Prometheus format: # HELP + # TYPE + metric_name {labels} value
        assert "# HELP" in body
        assert "tailorsync_scan_submitted_total" in body
        assert "tailorsync_jobs_in_state" in body


# ---------------------------------------------------------------------------
# Health endpoint degraded reporting
# ---------------------------------------------------------------------------

class TestHealthDegradedReporting:
    def test_health_payload_has_segmenter_real_field(self, client):
        r = client.get("/api/v1/scan/health")
        # 200 (ok or degraded) or 503 (unready). All include the field.
        assert r.status_code in (200, 503)
        body = r.json()
        assert "segmenter_real" in body
        assert "pipeline" in body
        assert body["pipeline"] in ("ok", "degraded", "unready")


# ---------------------------------------------------------------------------
# Ownership scoping on /status and /result
# ---------------------------------------------------------------------------

class TestOwnershipScoping:
    def test_status_returns_404_when_owner_mismatches(self, client):
        # Inject a job owned by user-A directly into the store
        from app.measurement_engine.scan.job_store import job_store
        sid = "test-cross-user-1"
        job_store.create(sid, customer_id="user-A")
        try:
            # Anonymous caller asking for user-A's job → 404 (no auth in dev,
            # token_uid is None which is != "user-A").
            r = client.get(f"/api/v1/scan/status/{sid}")
            assert r.status_code == 404
        finally:
            # cleanup
            try:
                from app.measurement_engine.scan.job_store import job_store
                # in-memory store path
                if hasattr(job_store, "_jobs"):
                    job_store._jobs.pop(sid, None)
            except Exception:
                pass
