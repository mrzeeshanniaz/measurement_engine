from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
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

    # --- Production hardening -------------------------------------------------
    # Maximum bytes per base64-encoded frame.  10 MB ≈ a 10 MP JPEG quality 90.
    MAX_FRAME_B64_BYTES: int = 10 * 1024 * 1024
    # Maximum total request body in bytes (8 frames × 10 MB + JSON overhead).
    MAX_REQUEST_BODY_BYTES: int = 90 * 1024 * 1024

    # Per-IP rate limits for the expensive endpoints.  Format: "<count>/<period>".
    RATE_LIMIT_SUBMIT: str = "30/minute"
    RATE_LIMIT_MANUAL: str = "60/minute"

    # Job store backend.  "memory" (single-instance, lost on restart) or
    # "sqlite" (file-based, restart-safe).  Production should use "sqlite".
    JOB_STORE_BACKEND: str = "memory"
    JOB_STORE_SQLITE_PATH: str = "./job_store.db"

    # Whether to expose Prometheus metrics at /metrics.
    METRICS_ENABLED: bool = True

    # ---- Image storage (S3 / GCS / in-memory) -----------------------------
    # IMAGE_STORE_BACKEND selects where pose frames live:
    #   "memory" — frames are sent inline as base64 in /submit (current default;
    #              fine for tests, breaks above ~5 MB payloads).
    #   "s3"     — mobile PUTs to presigned S3 URLs, /submit carries object
    #              keys only. Recommended for production.
    IMAGE_STORE_BACKEND: str = "memory"

    # S3 settings — only required when IMAGE_STORE_BACKEND=s3.
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: Optional[str] = None
    # Bucket the backend signs uploads into and downloads from.
    S3_UPLOAD_BUCKET: Optional[str] = None
    # Presigned URL lifetime — keep short so a leaked URL is worthless soon.
    S3_UPLOAD_URL_TTL_SECONDS: int = 600
    # Hard ceiling enforced in the presigned PUT policy.
    S3_UPLOAD_MAX_BYTES: int = 8 * 1024 * 1024
    # Optional custom endpoint (use for MinIO / R2 / LocalStack).
    S3_ENDPOINT_URL: Optional[str] = None


settings = Settings()
