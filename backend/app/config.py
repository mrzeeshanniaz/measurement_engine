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

    # Firebase Auth (A3)
    # AUTH_ENABLED=False in dev — unauthenticated requests are allowed.
    # Set AUTH_ENABLED=True in production; tokens are verified via Firebase Admin SDK.
    AUTH_ENABLED: bool = False

    # CORS — comma-separated list of allowed origins.
    # Set to specific domains in production (e.g. "https://tailorsync.app,https://www.tailorsync.app").
    # "*" allows all origins (dev default).
    CORS_ORIGINS: str = "*"

    class Config:
        env_file = ".env"


settings = Settings()
