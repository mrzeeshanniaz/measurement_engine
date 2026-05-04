"""
Measurement profile API routes.

GET    /api/v1/profiles                     — list caller's profiles (token-scoped)
GET    /api/v1/profiles/{profile_id}        — get one profile by ID
GET    /api/v1/profiles/by-scan/{scan_id}   — get profile linked to a scan session
DELETE /api/v1/profiles/{profile_id}        — delete a profile

Authentication:
  When AUTH_ENABLED=True a Firebase ID token (Bearer) is required.
  Profiles are automatically scoped to the token's UID so customers can only
  see and delete their own data.

  When AUTH_ENABLED=False (dev) a ?customer_id=<uid> query param is used
  as a fallback so the routes work without credentials.

Returns 503 when Firestore is not configured.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth import current_customer_id
from app.db.crud import (
    delete_profile_sync,
    get_profile_by_scan_sync,
    get_profile_sync,
    list_profiles_sync,
)
from app.db.firestore import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ProfileSummary(BaseModel):
    id: str
    customer_id: str
    scan_id: str
    created_at: str
    height_cm: float
    height_source: str
    garment_type: Optional[str] = None
    fit_style: Optional[str] = None
    overall_confidence: str


class ProfileDetail(ProfileSummary):
    measurements: dict
    validation: Optional[dict] = None


def _to_summary(p) -> ProfileSummary:
    return ProfileSummary(
        id=p.id,
        customer_id=p.customer_id,
        scan_id=p.scan_id,
        created_at=p.created_at.isoformat(),
        height_cm=p.height_cm,
        height_source=p.height_source,
        garment_type=p.garment_type,
        fit_style=p.fit_style,
        overall_confidence=p.overall_confidence,
    )


def _to_detail(p) -> ProfileDetail:
    return ProfileDetail(
        **_to_summary(p).model_dump(),
        measurements=p.measurements,
        validation=p.validation,
    )


def _require_db(db) -> None:
    if db is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Measurement persistence not configured. "
                "Set FIREBASE_CREDENTIALS_PATH (or use Application Default Credentials)."
            ),
        )


def _resolve_customer(
    token_uid: Optional[str],
    query_customer_id: Optional[str],
) -> str:
    """
    Token UID takes precedence; fallback to query param (dev mode only).
    Raises 401 when neither is available.
    """
    cid = token_uid or query_customer_id
    if not cid:
        raise HTTPException(
            status_code=401,
            detail="Provide a Firebase ID token or a customer_id query parameter.",
        )
    return cid


async def _run(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=list[ProfileSummary],
    summary="List measurement profiles for the authenticated customer",
)
async def list_customer_profiles(
    customer_id: Optional[str] = Query(None, description="Customer ID (dev fallback; token UID used when auth is enabled)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    token_uid: Optional[str] = Depends(current_customer_id),
    db=Depends(get_db),
) -> list[ProfileSummary]:
    _require_db(db)
    cid = _resolve_customer(token_uid, customer_id)
    profiles = await _run(list_profiles_sync, db, cid, limit=limit, offset=offset)
    return [_to_summary(p) for p in profiles]


@router.get(
    "/by-scan/{scan_id}",
    response_model=ProfileDetail,
    summary="Get the measurement profile linked to a scan session",
    responses={403: {"description": "Profile belongs to a different customer"}, 404: {"description": "Not found"}},
)
async def get_profile_by_scan_id(
    scan_id: str,
    customer_id: Optional[str] = Query(None, description="Customer ID (dev fallback)"),
    token_uid: Optional[str] = Depends(current_customer_id),
    db=Depends(get_db),
) -> ProfileDetail:
    _require_db(db)
    cid = _resolve_customer(token_uid, customer_id)
    profile = await _run(get_profile_by_scan_sync, db, scan_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile found for scan '{scan_id}'.")
    if profile.customer_id != cid:
        raise HTTPException(status_code=403, detail="Access denied.")
    return _to_detail(profile)


@router.get(
    "/{profile_id}",
    response_model=ProfileDetail,
    summary="Get a measurement profile by ID",
    responses={403: {"description": "Profile belongs to a different customer"}, 404: {"description": "Not found"}},
)
async def get_profile_by_id(
    profile_id: str,
    customer_id: Optional[str] = Query(None, description="Customer ID (dev fallback)"),
    token_uid: Optional[str] = Depends(current_customer_id),
    db=Depends(get_db),
) -> ProfileDetail:
    _require_db(db)
    cid = _resolve_customer(token_uid, customer_id)
    profile = await _run(get_profile_sync, db, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    if profile.customer_id != cid:
        raise HTTPException(status_code=403, detail="Access denied.")
    return _to_detail(profile)


@router.delete(
    "/{profile_id}",
    status_code=204,
    summary="Delete a measurement profile",
    responses={204: {"description": "Deleted"}, 403: {"description": "Access denied"}, 404: {"description": "Not found"}},
)
async def delete_profile_by_id(
    profile_id: str,
    customer_id: Optional[str] = Query(None, description="Customer ID (dev fallback)"),
    token_uid: Optional[str] = Depends(current_customer_id),
    db=Depends(get_db),
) -> None:
    _require_db(db)
    cid = _resolve_customer(token_uid, customer_id)
    # Verify ownership before deleting
    profile = await _run(get_profile_sync, db, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    if profile.customer_id != cid:
        raise HTTPException(status_code=403, detail="Access denied.")
    await _run(delete_profile_sync, db, profile_id)
