"""Common data structures for ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Dict, Mapping


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
            "brightness": self.brightness,
            "bright_t31": self.bright_t31,
            "frp": self.frp,
            "scan": self.scan,
            "track": self.track,
            "raw_properties": dict(self.raw_properties),
            "ingest_batch_id": self.ingest_batch_id,
            "dedupe_hash": self.dedupe_hash,
        }


