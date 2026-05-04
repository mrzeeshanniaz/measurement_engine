from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEBUG: bool = False
    DEVICE: str = "cpu"
    MODEL_CACHE_DIR: str = "./models"

    DEFAULT_FOCAL_LENGTH_MM: float = 4.25
    DEFAULT_SENSOR_WIDTH_MM: float = 4.8

    # Firebase / Firestore — optional.  Persistence is silently disabled when not set.
    # Path to a Firebase service account JSON file.
    # When running on GCP / Cloud Run this can be omitted; ADC is used automatically.
    FIREBASE_CREDENTIALS_PATH: Optional[str] = None
    # GCP project ID (optional — inferred from credentials when not set).
    FIREBASE_PROJECT_ID: Optional[str] = None

    # JWT auth (A3) — required only when auth is enabled
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7   # 1 week

    class Config:
        env_file = ".env"


settings = Settings()
