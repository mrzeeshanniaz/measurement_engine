"""
Scan job store.

Two backends share a common interface:

  - InMemoryJobStore   — process-local; lost on restart. Use for tests and dev.
  - SQLiteJobStore     — file-backed; restart-safe. Use for production single-
                          instance and small-pilot deployments.  Horizontal
                          scaling beyond one app process needs a Redis-backed
                          store (later upgrade — same interface).

Each job transitions: QUEUED → PROCESSING → COMPLETE | FAILED.  Jobs expire
after TTL_SECONDS (1 h) to bound storage growth.  On startup we sweep any
job left in QUEUED/PROCESSING from a previous crash and mark it FAILED so
clients polling /status get a definitive answer.

The active backend is selected at import time from settings.JOB_STORE_BACKEND
and exposed as the module-level `job_store` singleton.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Protocol

if TYPE_CHECKING:
    from app.measurement_engine.scan.schemas import ScanResponse

logger = logging.getLogger(__name__)

TTL_SECONDS = 3600  # 1 hour
_VALID_STATES = ("QUEUED", "PROCESSING", "COMPLETE", "FAILED")


# ---------------------------------------------------------------------------
# Job dataclass — shared by both backends
# ---------------------------------------------------------------------------

@dataclass
class ScanJob:
    session_id: str
    status: str = "QUEUED"
    progress_pct: int = 0
    result: Optional["ScanResponse"] = None
    error: Optional[str] = None
    # Token UID (or body.customer_id when auth disabled) that owns this job.
    # None means the job was submitted anonymously and is not owner-scoped.
    customer_id: Optional[str] = None
    _created_at: float = field(default_factory=time.time, repr=False)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self._created_at) > TTL_SECONDS


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class _JobStoreProtocol(Protocol):
    def create(self, session_id: str, customer_id: Optional[str] = None) -> ScanJob: ...
    def get(self, session_id: str) -> Optional[ScanJob]: ...
    def update(self, session_id: str, **kwargs) -> None: ...
    def purge_expired(self) -> int: ...
    def counts(self) -> dict[str, int]: ...
    def recover_orphaned(self) -> int: ...
    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
# In-memory backend (tests, dev)
# ---------------------------------------------------------------------------

class InMemoryJobStore:
    """Thread-safe in-process job registry — lost on restart."""

    def __init__(self) -> None:
        self._jobs: dict[str, ScanJob] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, customer_id: Optional[str] = None) -> ScanJob:
        with self._lock:
            if session_id in self._jobs:
                return self._jobs[session_id]
            job = ScanJob(session_id=session_id, customer_id=customer_id)
            self._jobs[session_id] = job
            return job

    def get(self, session_id: str) -> Optional[ScanJob]:
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                return None
            if job.is_expired:
                del self._jobs[session_id]
                return None
            return job

    def update(self, session_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)

    def purge_expired(self) -> int:
        with self._lock:
            expired = [sid for sid, j in self._jobs.items() if j.is_expired]
            for sid in expired:
                del self._jobs[sid]
            return len(expired)

    def counts(self) -> dict[str, int]:
        with self._lock:
            result: dict[str, int] = {s: 0 for s in _VALID_STATES}
            for job in self._jobs.values():
                result[job.status] = result.get(job.status, 0) + 1
            return result

    def recover_orphaned(self) -> int:
        """In-memory store starts empty on every process boot — nothing to do."""
        return 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)


# ---------------------------------------------------------------------------
# SQLite backend (production single-instance)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_jobs (
    session_id   TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    progress_pct INTEGER NOT NULL DEFAULT 0,
    result_json  TEXT,
    error        TEXT,
    customer_id  TEXT,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scan_jobs_status ON scan_jobs(status);
CREATE INDEX IF NOT EXISTS idx_scan_jobs_created ON scan_jobs(created_at);
"""


class SQLiteJobStore:
    """
    File-backed restart-safe job store.

    Trade-offs:
      - One process can write; readers OK with WAL.
      - For horizontal scaling, upgrade to Redis (same interface).
      - ScanResponse is serialised as JSON via pydantic .model_dump_json().
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Write-ahead logging keeps reads non-blocking under contention.
            conn.execute("PRAGMA journal_mode=WAL;")

    # ---- internal -----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False because we run from FastAPI thread pool.
        conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> ScanJob:
        from app.measurement_engine.scan.schemas import ScanResponse
        result = None
        if row["result_json"]:
            try:
                result = ScanResponse.model_validate_json(row["result_json"])
            except Exception as exc:
                logger.warning("Failed to deserialise stored ScanResponse: %s", exc)
        job = ScanJob(
            session_id=row["session_id"],
            status=row["status"],
            progress_pct=row["progress_pct"],
            result=result,
            error=row["error"],
            customer_id=row["customer_id"],
        )
        job._created_at = row["created_at"]
        return job

    # ---- public interface --------------------------------------------

    def create(self, session_id: str, customer_id: Optional[str] = None) -> ScanJob:
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM scan_jobs WHERE session_id = ?", (session_id,)
            ).fetchone()
            if existing is not None:
                return self._row_to_job(existing)
            now = time.time()
            conn.execute(
                "INSERT INTO scan_jobs (session_id, status, progress_pct, customer_id, created_at) "
                "VALUES (?, 'QUEUED', 0, ?, ?)",
                (session_id, customer_id, now),
            )
            return ScanJob(session_id=session_id, customer_id=customer_id, _created_at=now)

    def get(self, session_id: str) -> Optional[ScanJob]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scan_jobs WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            job = self._row_to_job(row)
            if job.is_expired:
                conn.execute("DELETE FROM scan_jobs WHERE session_id = ?", (session_id,))
                return None
            return job

    def update(self, session_id: str, **kwargs) -> None:
        if not kwargs:
            return
        with self._lock, self._connect() as conn:
            # Translate `result=<ScanResponse>` to JSON; pass through other fields.
            fields: list[str] = []
            values: list = []
            for k, v in kwargs.items():
                if k == "result":
                    fields.append("result_json = ?")
                    values.append(v.model_dump_json() if v is not None else None)
                elif k in ("status", "progress_pct", "error", "customer_id"):
                    fields.append(f"{k} = ?")
                    values.append(v)
                else:
                    logger.debug("Ignoring unsupported job-store field: %s", k)
            if not fields:
                return
            values.append(session_id)
            conn.execute(
                f"UPDATE scan_jobs SET {', '.join(fields)} WHERE session_id = ?",
                values,
            )

    def purge_expired(self) -> int:
        cutoff = time.time() - TTL_SECONDS
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM scan_jobs WHERE created_at < ?", (cutoff,))
            return cur.rowcount or 0

    def counts(self) -> dict[str, int]:
        result = {s: 0 for s in _VALID_STATES}
        with self._lock, self._connect() as conn:
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM scan_jobs GROUP BY status"
            ):
                result[row["status"]] = row["n"]
        return result

    def recover_orphaned(self) -> int:
        """
        Mark any job left in QUEUED or PROCESSING as FAILED.  Called once at
        startup so a crashed/restarted server gives clients a deterministic
        answer instead of stranding them on poll loops.
        """
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE scan_jobs SET status = 'FAILED', error = COALESCE(error, ?) "
                "WHERE status IN ('QUEUED', 'PROCESSING')",
                ("Pipeline aborted: server restarted before completion.",),
            )
            return cur.rowcount or 0

    def __len__(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM scan_jobs").fetchone()
            return int(row["n"])


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _build_default_store() -> _JobStoreProtocol:
    # Lazy import to avoid pulling config at module import for tests that
    # monkeypatch settings.
    from app.config import settings

    backend = (settings.JOB_STORE_BACKEND or "memory").lower()
    if backend == "sqlite":
        path = settings.JOB_STORE_SQLITE_PATH or "./job_store.db"
        logger.info("Using SQLite job store at %s", path)
        return SQLiteJobStore(path)
    if backend != "memory":
        logger.warning("Unknown JOB_STORE_BACKEND=%r — falling back to memory", backend)
    return InMemoryJobStore()


# Module-level singleton used by the scan router, background tasks, and tests.
job_store: _JobStoreProtocol = _build_default_store()


# Back-compat alias for any caller that still imports `JobStore`.
JobStore = InMemoryJobStore
