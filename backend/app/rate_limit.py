"""
slowapi rate-limiter setup.

Key strategy: when a Firebase ID token is present and verified upstream, the
limit applies per-uid; otherwise per-IP.  This prevents one IP behind a NAT
from blocking many legitimate users, while still throttling anonymous abuse.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded  # noqa: F401  (re-exported)
from slowapi.util import get_remote_address
from starlette.requests import Request


def _rate_limit_key(request: Request) -> str:
    """
    Prefer a per-token key over per-IP.  We use a short hash of the bearer
    token rather than the raw token (slowapi keys are stored verbatim in the
    limiter backend, so we don't want to leak credentials there).
    """
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(None, 1)[1].strip()
        if token:
            import hashlib
            return "tok:" + hashlib.sha256(token.encode()).hexdigest()[:16]
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_rate_limit_key)
