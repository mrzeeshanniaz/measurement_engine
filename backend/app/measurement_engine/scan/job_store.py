"""
In-memory scan job store.

Provides thread-safe job lifecycle management for async scan processing.
Each job transitions: QUEUED → PROCESSING → COMPLETE | FAILED.

Jobs expire after TTL_SECONDS (default 1 h) to bound memory growth.
For horizontal scaling, replace with a Redis-backed store that uses
the same JobStore interface.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.measurement_engine.scan.schemas import ScanResponse

TTL_SECONDS = 3600  # 1 hour


@dataclass
class ScanJob:
    session_id: str
    status: str = "QUEUED"          # QUEUED | PROCESSING | COMPLETE | FAILED
    progress_pct: int = 0
    result: Optional["ScanResponse"] = None
    error: Optional[str] = None
    _created_at: float = field(default_factory=time.monotonic, repr=False)

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self._created_at) > TTL_SECONDS


class JobStore:
    """Thread-safe in-memory job registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, ScanJob] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self, session_id: str) -> ScanJob:
        """Create a new QUEUED job. Returns the existing job if already present
        (idempotent — safe for client retries with the same client_scan_id)."""
        with self._lock:
            if session_id in self._jobs:
                return self._jobs[session_id]
            job = ScanJob(session_id=session_id)
            self._jobs[session_id] = job
            return job

    def get(self, session_id: str) -> Optional[ScanJob]:
        """Return the job or None if not found / expired."""
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                return None
            if job.is_expired:
                del self._jobs[session_id]
                return None
            return job

    def update(self, session_id: str, **kwargs) -> None:
        """Atomically update job fields."""
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_expired(self) -> int:
        """Remove expired jobs. Returns count removed."""
        with self._lock:
            expired = [sid for sid, j in self._jobs.items() if j.is_expired]
            for sid in expired:
                del self._jobs[sid]
            return len(expired)

    def counts(self) -> dict[str, int]:
        """Return a snapshot of job counts per status."""
        with self._lock:
            result: dict[str, int] = {"QUEUED": 0, "PROCESSING": 0, "COMPLETE": 0, "FAILED": 0}
            for job in self._jobs.values():
                result[job.status] = result.get(job.status, 0) + 1
            return result

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)


# Module-level singleton used by the scan router and background tasks.
job_store = JobStore()
