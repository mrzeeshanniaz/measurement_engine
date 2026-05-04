"""
Firebase Authentication middleware for TailorSync Measurement Engine.

The Flutter client authenticates with Firebase Auth (Google, email/password,
phone, etc.) and receives a Firebase ID token.  That token is sent as a
Bearer credential in every API request.  This module verifies the token
server-side using the Firebase Admin SDK and extracts the Firebase UID,
which is used as the customer identifier throughout the system.

Token flow:
  1. Flutter app  → firebase_auth.currentUser.getIdToken()  → ID token string
  2. Flutter app  → Authorization: Bearer <id-token>        → backend
  3. Backend      → firebase_admin.auth.verify_id_token()   → {uid, email, …}
  4. Backend uses uid as customer_id for profile persistence and scoping

AUTH_ENABLED (settings):
  False (default) — auth is skipped in dev; requests without a token are
                    allowed and customer_id comes from the request body.
  True            — a valid Firebase ID token is required on protected routes.

Protected routes:
  POST  /api/v1/scan/submit           (auto-populates customer_id from token)
  GET   /api/v1/profiles              (scoped to token uid)
  GET   /api/v1/profiles/{id}
  GET   /api/v1/profiles/by-scan/{id}
  DELETE /api/v1/profiles/{id}

Public routes (no token required):
  GET   /api/v1/scan/status/{id}
  GET   /api/v1/scan/result/{id}
  POST  /api/v1/scan/manual
  GET   /api/v1/scan/validation-rules
  GET   /api/v1/scan/health
  GET   /health
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


def _verify_firebase_token(id_token: str) -> dict:
    """
    Verify a Firebase ID token and return the decoded claims.
    Raises HTTP 401 on any verification failure.
    """
    try:
        import firebase_admin.auth as fb_auth
        return fb_auth.verify_id_token(id_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired Firebase ID token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def current_customer_id(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    """
    FastAPI dependency — extracts the Firebase UID from the Bearer token.

    Returns:
      str  — Firebase UID when a valid token is present.
      None — when no token is present AND AUTH_ENABLED=False (dev mode).

    Raises HTTP 401 when AUTH_ENABLED=True and the token is absent or invalid.
    """
    if creds is None:
        if settings.AUTH_ENABLED:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Send a Firebase ID token as a Bearer credential.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None  # dev mode — unauthenticated requests allowed

    decoded = _verify_firebase_token(creds.credentials)
    uid: str = decoded["uid"]
    return uid


async def require_customer_id(
    customer_id: Optional[str] = Depends(current_customer_id),
) -> str:
    """
    Stricter version of current_customer_id — always raises 401 when the
    identity is absent (i.e. AUTH_ENABLED=False and no token was sent).
    Use this on routes where a customer identity is structurally required
    even in dev mode.
    """
    if not customer_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return customer_id
