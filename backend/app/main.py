import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.firestore import dispose_firestore, init_firestore
from app.measurement_engine.models.model_manager import ModelManager
from app.api.v1 import profiles, scan

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TailorSync Measurement Engine...")

    # Firestore (optional — silently skipped when credentials are absent)
    init_firestore()

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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tailorsync-measurement-engine"}
