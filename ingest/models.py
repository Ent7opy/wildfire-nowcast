"""Common data structures for ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Dict, Mapping

from pydantic import BaseModel, ConfigDict, Field


def compute_dedupe_hash(source: str, lat: float, lon: float, acq_time: datetime) -> str:
    """Create a deterministic hash for deduplication."""
    normalized_time = acq_time.astimezone(timezone.utc).replace(microsecond=0)
    rounded_lat = round(lat, 4)
    rounded_lon = round(lon, 4)
    payload = f"{source}|{rounded_lat:.4f}|{rounded_lon:.4f}|{normalized_time.isoformat()}"
    return sha1(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DetectionRecord:
    """Normalized detection ready for DB insertion."""

    lat: float
    lon: float
    acq_time: datetime
    sensor: str | None
    source: str
    confidence: float | None
    confidence_score: float | None
    brightness: float | None
    bright_t31: float | None
    frp: float | None
    scan: float | None
    track: float | None
    raw_properties: Mapping[str, Any]
    ingest_batch_id: int

    @property
    def dedupe_hash(self) -> str:
        return compute_dedupe_hash(self.source, self.lat, self.lon, self.acq_time)

    def to_parameters(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "acq_time": self.acq_time,
            "sensor": self.sensor,
            "source": self.source,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "brightness": self.brightness,
            "bright_t31": self.bright_t31,
            "frp": self.frp,
            "scan": self.scan,
            "track": self.track,
            "raw_properties": dict(self.raw_properties),
            "ingest_batch_id": self.ingest_batch_id,
            "dedupe_hash": self.dedupe_hash,
        }


class FireDetection(BaseModel):
    """Standard API/ML representation of a fire detection record."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    lat: float
    lon: float
    acq_time: datetime
    sensor: str | None = None
    source: str
    confidence: float | None = None
    confidence_score: float | None = None
    brightness: float | None = None
    bright_t31: float | None = None
    frp: float | None = None
    scan: float | None = None
    track: float | None = None
    raw_properties: Mapping[str, Any] = Field(default_factory=dict)
    ingest_batch_id: int | None = None
    dedupe_hash: str | None = None
    denoised_score: float | None = None
    is_noise: bool | None = None
    created_at: datetime | None = None

    @classmethod
    def from_record(cls, record: DetectionRecord) -> "FireDetection":
        """Lift an ingest DetectionRecord into the standard shape."""
        return cls(
            lat=record.lat,
            lon=record.lon,
            acq_time=record.acq_time,
            sensor=record.sensor,
            source=record.source,
            confidence=record.confidence,
            confidence_score=record.confidence_score,
            brightness=record.brightness,
            bright_t31=record.bright_t31,
            frp=record.frp,
            scan=record.scan,
            track=record.track,
            raw_properties=record.raw_properties,
            ingest_batch_id=record.ingest_batch_id,
            dedupe_hash=record.dedupe_hash,
        )

