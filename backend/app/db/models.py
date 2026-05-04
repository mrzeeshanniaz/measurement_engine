"""
Pure-Python data model for a Measurement Profile (mirrors Firestore document).

Using a dataclass instead of an ORM model keeps the persistence layer
decoupled from SQLAlchemy while giving the API layer typed attribute access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class MeasurementProfile:
    id: str                          # Firestore document ID
    customer_id: str
    scan_id: str
    created_at: datetime
    height_cm: float
    height_source: str
    overall_confidence: str
    measurements: dict               # full 32-field serialised payload
    garment_type: Optional[str] = None
    fit_style: Optional[str] = None
    validation: Optional[dict] = None

    @classmethod
    def from_firestore(cls, doc_id: str, data: dict) -> "MeasurementProfile":
        created = data.get("created_at")
        if hasattr(created, "ToDatetime"):
            created = created.ToDatetime(tzinfo=timezone.utc)
        elif not isinstance(created, datetime):
            created = datetime.now(timezone.utc)
        return cls(
            id=doc_id,
            customer_id=data["customer_id"],
            scan_id=data["scan_id"],
            created_at=created,
            height_cm=float(data["height_cm"]),
            height_source=data.get("height_source", "user_input"),
            overall_confidence=data["overall_confidence"],
            measurements=data.get("measurements", {}),
            garment_type=data.get("garment_type"),
            fit_style=data.get("fit_style"),
            validation=data.get("validation"),
        )

    def to_firestore(self) -> dict:
        return {
            "customer_id":        self.customer_id,
            "scan_id":            self.scan_id,
            "created_at":         self.created_at,
            "height_cm":          self.height_cm,
            "height_source":      self.height_source,
            "overall_confidence": self.overall_confidence,
            "measurements":       self.measurements,
            "garment_type":       self.garment_type,
            "fit_style":          self.fit_style,
            "validation":         self.validation,
        }
