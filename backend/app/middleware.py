"""
Production middleware:
  - MaxBodySizeMiddleware  — reject requests whose Content-Length exceeds the
                             configured cap before any router or schema runs.
  - RequestIDMiddleware    — attach a per-request UUID for correlation in logs
                             and propagate it back as the X-Request-ID response
                             header.
"""
from __future__ import annotations

import logging
import re
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Populated by RequestIDMiddleware; available to background tasks and any
# logger handler via the RedactingFilter below.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """
    Reject any request whose Content-Length exceeds `max_bytes` with HTTP 413.
    Clients can skirt this by streaming without a Content-Length header, so the
    body-handler in scan.submit must enforce its own running cap as a backstop.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self._max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                f"Request body too large: {cl} bytes exceeds "
                                f"limit of {self._max_bytes} bytes."
                            )
                        },
                    )
            except ValueError:
                pass  # malformed header — let downstream handle
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attach a request_id to every request.  If the client sends X-Request-ID we
    honor it (handy for tracing through API gateways); otherwise we generate
    one.  The id is propagated to log records via the `request_id` context.
    """

    HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get(self.HEADER) or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        token = request_id_var.set(req_id)
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[self.HEADER] = req_id
        return response


# ---------------------------------------------------------------------------
# Logging hygiene — strip biometric blobs from logs and inject request_id
# ---------------------------------------------------------------------------

# Long base64 runs are a strong signal of image data.  ≥120 chars covers
# even tiny JPEGs and is well above any anticipated non-image base64 token.
_B64_RE = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")
_JSON_KEY_RES = (
    re.compile(r'("image_b64"\s*:\s*)"[^"]+"'),
    re.compile(r"('image_b64'\s*:\s*)'[^']+'"),
)


class RedactingFilter(logging.Filter):
    """
    Strip base64 image payloads from log records and inject `request_id`.

    Defence-in-depth: production code paths never log image data, but a stray
    `logger.debug(request_body)` would otherwise leak biometric data into
    log aggregation.  This filter is installed on the root logger so every
    handler (stdout, Cloud Logging, files) sees redacted output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)

        if "image_b64" in rendered or _B64_RE.search(rendered):
            redacted = rendered
            for pat in _JSON_KEY_RES:
                redacted = pat.sub(r'\1"<redacted>"', redacted)
            redacted = _B64_RE.sub("<base64-redacted>", redacted)
            # Replace msg with the redacted, already-rendered form so handlers
            # don't re-format with the original args.
            record.msg = redacted
            record.args = ()

        # Make request_id available to formatters as %(request_id)s.
        record.request_id = request_id_var.get()
        return True
