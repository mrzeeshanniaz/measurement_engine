"""
CRUD helpers for measurement profiles stored in Firestore.

All functions use the synchronous Firestore client, which is safe to call:
  - directly from sync background tasks (thread pool)
  - via asyncio.run_in_executor() from async FastAPI route handlers

Functions return None / empty list when db is None (persistence disabled).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import google.cloud.firestore

from app.db.models import MeasurementProfile
from app.db.firestore import COLLECTION

logger = logging.getLogger(__name__)


def save_profile_sync(
    *,
    customer_id: str,
    scan_id: str,
    height_cm: float,
    height_source: str,
    overall_confidence: str,
    measurements: dict,
    validation: Optional[dict] = None,
    garment_type: Optional[str] = None,
    fit_style: Optional[str] = None,
    db: Optional["google.cloud.firestore.Client"] = None,
) -> Optional[MeasurementProfile]:
    """
    Persist a completed scan result synchronously.
    Idempotent: if a document with this scan_id already exists, returns it.
    """
    if db is None:
        from app.db.firestore import get_firestore
        db = get_firestore()
    if db is None:
        return None

    try:
        # Check for existing document by scan_id
        existing = (
            db.collection(COLLECTION)
            .where("scan_id", "==", scan_id)
            .limit(1)
            .stream()
        )
        for doc in existing:
            return MeasurementProfile.from_firestore(doc.id, doc.to_dict())

        profile_id = str(uuid.uuid4())
        profile = MeasurementProfile(
            id=profile_id,
            customer_id=customer_id,
            scan_id=scan_id,
            created_at=datetime.now(timezone.utc),
            height_cm=height_cm,
            height_source=height_source,
            overall_confidence=overall_confidence,
            measurements=measurements,
            validation=validation,
            garment_type=garment_type,
            fit_style=fit_style,
        )
        db.collection(COLLECTION).document(profile_id).set(profile.to_firestore())
        logger.info("Profile saved — id=%s customer=%s", profile_id[:8], customer_id)
        return profile
    except Exception as exc:
        logger.error("save_profile_sync failed for scan %s: %s", scan_id, exc)
        return None


def list_profiles_sync(
    db: Optional["google.cloud.firestore.Client"],
    customer_id: str,
    limit: int = 20,
    offset: int = 0,
) -> list[MeasurementProfile]:
    if db is None:
        return []
    try:
        from google.cloud.firestore import Query
        docs = (
            db.collection(COLLECTION)
            .where("customer_id", "==", customer_id)
            .order_by("created_at", direction=Query.DESCENDING)
            .limit(limit + offset)
            .stream()
        )
        all_docs = list(docs)
        return [
            MeasurementProfile.from_firestore(d.id, d.to_dict())
            for d in all_docs[offset:]
        ]
    except Exception as exc:
        logger.error("list_profiles_sync failed for customer %s: %s", customer_id, exc)
        return []


def get_profile_sync(
    db: Optional["google.cloud.firestore.Client"],
    profile_id: str,
) -> Optional[MeasurementProfile]:
    if db is None:
        return None
    try:
        doc = db.collection(COLLECTION).document(profile_id).get()
        if not doc.exists:
            return None
        return MeasurementProfile.from_firestore(doc.id, doc.to_dict())
    except Exception as exc:
        logger.error("get_profile_sync failed for id %s: %s", profile_id, exc)
        return None


def get_profile_by_scan_sync(
    db: Optional["google.cloud.firestore.Client"],
    scan_id: str,
) -> Optional[MeasurementProfile]:
    if db is None:
        return None
    try:
        docs = (
            db.collection(COLLECTION)
            .where("scan_id", "==", scan_id)
            .limit(1)
            .stream()
        )
        for doc in docs:
            return MeasurementProfile.from_firestore(doc.id, doc.to_dict())
        return None
    except Exception as exc:
        logger.error("get_profile_by_scan_sync failed for scan %s: %s", scan_id, exc)
        return None


def delete_profile_sync(
    db: Optional["google.cloud.firestore.Client"],
    profile_id: str,
) -> bool:
    if db is None:
        return False
    try:
        ref = db.collection(COLLECTION).document(profile_id)
        doc = ref.get()
        if not doc.exists:
            return False
        ref.delete()
        return True
    except Exception as exc:
        logger.error("delete_profile_sync failed for id %s: %s", profile_id, exc)
        return False
