"""Helpers for downloading and parsing NASA FIRMS CSV feeds."""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import httpx

from ingest.models import DetectionRecord

LOGGER = logging.getLogger(__name__)
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


class FIRMSClientError(RuntimeError):
    """Raised when the FIRMS API request fails."""


def build_firms_url(map_key: str, source: str, bbox: str, day_range: int, date: str | None = None) -> str:
    """Construct the FIRMS API URL for a given source and spatial window."""
    base = f"{FIRMS_BASE_URL}/{map_key}/{source}/{bbox}/{day_range}"
    return f"{base}/{date}" if date else base


def fetch_csv_rows(
    map_key: str,
    source: str,
    bbox: str,
    day_range: int,
    timeout_seconds: float,
    date: str | None = None,
) -> List[Dict[str, str]]:
    """Download FIRMS CSV rows."""
    url = build_firms_url(map_key, source, bbox, day_range, date=date)
    LOGGER.info("Requesting FIRMS CSV", extra={"source": source, "url": url})
    try:
        response = httpx.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - network error path
        raise FIRMSClientError(f"Failed to fetch FIRMS data: {exc}") from exc

    text_stream = io.StringIO(response.text)
    reader = csv.DictReader(text_stream)
    rows = list(reader)
    LOGGER.info("Fetched %s rows from FIRMS", len(rows), extra={"source": source})
    return rows


def parse_detection_rows(
    rows: Iterable[Dict[str, str]],
    source: str,
    ingest_batch_id: int,
) -> List[DetectionRecord]:
    """Normalize FIRMS CSV rows into `DetectionRecord`s."""
    detections: List[DetectionRecord] = []
    for row in rows:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            acq_time = _parse_acq_datetime(row)
        except (KeyError, ValueError) as exc:
            LOGGER.warning("Skipping row due to invalid coordinates/time: %s", exc)
            continue

        detection = DetectionRecord(
            lat=lat,
            lon=lon,
            acq_time=acq_time,
            sensor=_pick_sensor(row),
            source=source,
            confidence=_parse_confidence(row),
            brightness=_optional_float(row.get("brightness")),
            bright_t31=_optional_float(row.get("bright_t31")),
            frp=_optional_float(row.get("frp")),
            scan=_optional_float(row.get("scan")),
            track=_optional_float(row.get("track")),
            raw_properties={**row},
            ingest_batch_id=ingest_batch_id,
        )
        detections.append(detection)

    return detections


def _parse_acq_datetime(row: Dict[str, str]) -> datetime:
    date_str = row.get("acq_date")
    time_str = (row.get("acq_time") or "").zfill(4)
    if not date_str:
        raise ValueError("Missing acquisition date.")
    composite = f"{date_str} {time_str}"
    dt = datetime.strptime(composite, "%Y-%m-%d %H%M")
    return dt.replace(tzinfo=timezone.utc)


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_confidence(row: Dict[str, str]) -> float | None:
    numeric = row.get("confidence")
    if numeric:
        try:
            return float(numeric)
        except ValueError:
            pass

    confidence_text = (row.get("confidence_text") or "").lower()
    mapping = {"h": 90.0, "high": 90.0, "n": 50.0, "nominal": 50.0, "l": 10.0, "low": 10.0}
    return mapping.get(confidence_text) if confidence_text else None


def _pick_sensor(row: Dict[str, str]) -> str | None:
    for key in ("instrument", "satellite", "sensor"):
        if row.get(key):
            return row[key]
    return None


