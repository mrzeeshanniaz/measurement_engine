"""
Gunicorn config for production.

Workers and threads are chosen for a CPU-bound ML workload:
  - Each worker loads MediaPipe + SMPL once (~1.5 GB RAM) so we keep
    workers low and let threads share a worker.
  - One pipeline.run() takes ~3 s and is CPU-bound — too many threads per
    worker would queue rather than parallelize.

Defaults are tuned for a 2 vCPU / 4 GB Cloud Run instance. Override via env
vars:  GUNICORN_WORKERS, GUNICORN_THREADS, GUNICORN_TIMEOUT.
"""
from __future__ import annotations

import multiprocessing
import os

# Pickup PORT (Cloud Run / generic) — default to 8080.
bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"

# Worker count — capped at 4 to avoid thrashing RAM on small instances. Each
# UvicornWorker loads its own copy of the ML models.
_default_workers = max(1, min(4, (multiprocessing.cpu_count() // 2) or 1))
workers = int(os.getenv("GUNICORN_WORKERS", _default_workers))

# Async worker so FastAPI's coroutines run natively. Threads = N gives N
# concurrent in-flight pipelines per worker; pipeline run is CPU-bound so we
# keep this small.
worker_class = "uvicorn.workers.UvicornWorker"
threads = int(os.getenv("GUNICORN_THREADS", "2"))

# Long timeout because a heavy scan can take 5–10 s end-to-end.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "60"))
keepalive = 5

# Restart workers periodically to defend against slow memory leaks in
# long-running ML processes.
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "5000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "500"))

# Send access + error logs to stdout/stderr for Cloud Logging / journald.
accesslog = "-"
errorlog = "-"
access_log_format = (
    '%(h)s "%(r)s" %(s)s %(b)sb %(D)sus '
    'ua="%(a)s" rid="%(\\{X-Request-ID\\}i)s"'
)

# Pre-load the app in the master process so workers fork with models already
# loaded. NOTE: this also fixes the model file paths *before* fork; if the
# image is built without SMPL pkl files mounted you'll see workers fail to
# start — easier-to-diagnose than a runtime crash.
preload_app = False  # keep False: model load is heavy and we want each worker
                    # to own its own state, especially for MediaPipe GL.
