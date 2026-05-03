import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.measurement_engine.models.model_manager import ModelManager
from app.api.v1 import scan

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TailorSync Measurement Engine...")
    manager = ModelManager()
    await manager.load()
    app.state.models = manager
    logger.info("Models ready")
    yield
    await manager.unload()
    logger.info("Shutdown complete")


app = FastAPI(
    title="TailorSync Measurement Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scan.router, prefix="/api/v1/scan", tags=["Scan"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "tailorsync-measurement-engine"}
