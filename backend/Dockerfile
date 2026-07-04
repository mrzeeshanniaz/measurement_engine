# syntax=docker/dockerfile:1.7
# ----------------------------------------------------------------------------
# TailorSync Measurement Engine — production container
#
# Build:
#   docker build -t tailorsync-engine:latest backend
#
# Run (mount SMPL pkl files at /smpl; map a writable model-cache volume):
#   docker run -p 8080:8080 \
#     -v $(pwd)/backend/app/measurement_engine/smpl_anthropometry/data/smpl:/app/app/measurement_engine/smpl_anthropometry/data/smpl:ro \
#     -v $(pwd)/backend/models:/app/models \
#     -e AUTH_ENABLED=true \
#     -e CORS_ORIGINS=https://tailorsync.app \
#     -e JOB_STORE_BACKEND=sqlite \
#     -e JOB_STORE_SQLITE_PATH=/app/data/job_store.db \
#     -v $(pwd)/_jobs:/app/data \
#     tailorsync-engine:latest
#
# Designed for Cloud Run (port 8080, single concurrency = workers * threads).
# ----------------------------------------------------------------------------

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    MODEL_CACHE_DIR=/app/models

WORKDIR /app

# System deps: libGL / libglib for OpenCV + MediaPipe; tini for clean signals.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tini \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in their own layer for caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application
COPY app ./app
COPY start.sh ./

# Cloud Run / Kubernetes: run as non-root user with a writable data dir
RUN useradd --uid 1000 --create-home appuser \
    && mkdir -p /app/models /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Container-level healthcheck used by docker / orchestrators not running k8s probes.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# tini reaps zombies and forwards SIGTERM cleanly so gunicorn workers can drain
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["gunicorn", \
     "--config", "/app/app/gunicorn_conf.py", \
     "app.main:app"]
