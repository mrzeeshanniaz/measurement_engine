"""
Prometheus metrics.

Exposes:
  - tailorsync_scan_submitted_total{status="queued"}
  - tailorsync_scan_completed_total{status="complete|failed"}
  - tailorsync_pipeline_seconds (Histogram of end-to-end pipeline durations)
  - tailorsync_mesh_fit_iou (Histogram of B8 silhouette IoU)
  - tailorsync_jobs_in_state{state="QUEUED|PROCESSING|..."} (Gauge sampled on scrape)
  - tailorsync_overall_confidence_total{level="HIGH|MEDIUM|LOW"}

Mount via mount_metrics(app) — exposes GET /metrics.
"""
from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

scan_submitted_total = Counter(
    "tailorsync_scan_submitted_total",
    "Total scan submissions accepted by /submit.",
    ["idempotent_replay"],
)

scan_completed_total = Counter(
    "tailorsync_scan_completed_total",
    "Total scans that reached a terminal state in the background worker.",
    ["status"],   # "complete" | "failed"
)

pipeline_seconds = Histogram(
    "tailorsync_pipeline_seconds",
    "End-to-end ScanPipeline.run() duration in seconds.",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 60.0),
)

mesh_fit_iou = Histogram(
    "tailorsync_mesh_fit_iou",
    "Silhouette IoU between the fitted SMPL mesh and the body mask (F6).",
    buckets=(0.20, 0.30, 0.40, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0),
)

overall_confidence_total = Counter(
    "tailorsync_overall_confidence_total",
    "Distribution of overall scan confidence labels returned to clients.",
    ["level"],
)

jobs_in_state = Gauge(
    "tailorsync_jobs_in_state",
    "Job store size by state (sampled on each /metrics scrape).",
    ["state"],
)


# ---------------------------------------------------------------------------
# Mount
# ---------------------------------------------------------------------------

def mount_metrics(app: FastAPI) -> None:
    """Attach a GET /metrics endpoint to `app`."""

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:  # noqa: D401
        # Sample the job-store gauge fresh on every scrape so a Prometheus poll
        # always reflects the current store contents (rather than the last
        # observed value).
        try:
            from app.measurement_engine.scan.job_store import job_store
            counts = job_store.counts()
            for state, n in counts.items():
                jobs_in_state.labels(state=state).set(n)
        except Exception:  # never let metrics scraping crash the app
            pass

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
