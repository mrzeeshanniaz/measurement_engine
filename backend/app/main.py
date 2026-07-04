import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import settings
from app.db.firestore import dispose_firestore, init_firestore
from app.measurement_engine.models.model_manager import ModelManager
from app.middleware import (
    MaxBodySizeMiddleware,
    RedactingFilter,
    RequestIDMiddleware,
)
from app.rate_limit import limiter
from app.api.v1 import profiles, scan

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s [req=%(request_id)s] %(message)s",
)
# Install the redacting + request-id filter on every existing handler.
_redactor = RedactingFilter()
for _h in logging.getLogger().handlers:
    _h.addFilter(_redactor)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TailorSync Measurement Engine...")

    # Firestore (optional — silently skipped when credentials are absent)
    init_firestore()

    # Restart-safety: any QUEUED/PROCESSING job left over from a prior boot
    # gets a definitive FAILED state so clients polling /status receive an
    # answer instead of waiting forever.
    from app.measurement_engine.scan.job_store import job_store
    recovered = job_store.recover_orphaned()
    if recovered:
        logger.warning("Recovered %d orphaned scan job(s) from prior boot", recovered)

    manager = ModelManager()
    await manager.load()
    app.state.models = manager
    logger.info("Models ready")
    yield
    await manager.unload()
    await dispose_firestore()
    logger.info("Shutdown complete")


app = FastAPI(
    title="TailorSync Measurement Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate-limiting must be registered on the app before any decorated route is
# exercised. slowapi raises RateLimitExceeded → translate to HTTP 429.
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(_request, exc: RateLimitExceeded):  # noqa: ANN001
    from starlette.responses import JSONResponse
    retry_after = getattr(exc, "retry_after", None)
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
        headers=headers,
    )


# Order matters: outermost middleware runs first. Size check → rate limit →
# request-id → CORS.  Starlette/FastAPI add_middleware semantics wrap in
# *reverse* call order, so we add the innermost last.
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.MAX_REQUEST_BODY_BYTES)
app.add_middleware(RequestIDMiddleware)

_cors_origins = (
    ["*"] if settings.CORS_ORIGINS.strip() == "*"
    else [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scan.router, prefix="/api/v1/scan", tags=["Scan"])
app.include_router(profiles.router, prefix="/api/v1/profiles", tags=["Profiles"])

# Prometheus /metrics endpoint — exposes pipeline counters + job gauges
if settings.METRICS_ENABLED:
    from app.metrics import mount_metrics
    mount_metrics(app)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tailorsync-measurement-engine"}
