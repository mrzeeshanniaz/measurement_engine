"""
Firebase Firestore client initialisation.

Credentials are resolved in this order:
  1. FIREBASE_CREDENTIALS_PATH env var → service account JSON file
  2. Application Default Credentials   → works on GCP / Cloud Run automatically

Persistence is silently disabled when neither credential source is available
so the service runs without Firebase in local / offline environments.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import google.cloud.firestore

logger = logging.getLogger(__name__)

_sync_client: Optional["google.cloud.firestore.Client"] = None

COLLECTION = "measurement_profiles"


def init_firestore() -> None:
    """Initialise Firebase Admin app and Firestore sync client."""
    global _sync_client
    from app.config import settings

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred_path = settings.FIREBASE_CREDENTIALS_PATH
            if cred_path:
                cred = credentials.Certificate(cred_path)
            else:
                cred = credentials.ApplicationDefault()

            kwargs: dict = {}
            if settings.FIREBASE_PROJECT_ID:
                kwargs["projectId"] = settings.FIREBASE_PROJECT_ID

            firebase_admin.initialize_app(cred, kwargs or {})

        _sync_client = firestore.client()
        logger.info("Firestore client initialised (project=%s)", _sync_client.project)

    except Exception as exc:
        logger.warning(
            "Firestore unavailable (%s) — measurement persistence disabled", exc
        )
        _sync_client = None


def get_firestore() -> Optional["google.cloud.firestore.Client"]:
    return _sync_client


async def dispose_firestore() -> None:
    """No-op — Firestore client has no explicit close/dispose."""
    pass


def get_db():
    """FastAPI dependency — yields the Firestore client or None when disabled."""
    yield get_firestore()
