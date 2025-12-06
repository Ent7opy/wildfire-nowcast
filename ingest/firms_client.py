"""Helpers for downloading and parsing NASA FIRMS CSV feeds."""

from __future__ import annotations

import csv
import io
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import httpx

from ingest.logging_utils import log_event
from ingest.models import DetectionRecord

LOGGER = logging.getLogger(__name__)
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
VALID_LAT_RANGE = (-90.0, 90.0)
VALID_LON_RANGE = (-180.0, 180.0)
CONFIDENCE_RANGE = (0.0, 100.0)
BRIGHTNESS_RANGE = (200.0, 500.0)


class FIRMSClientError(RuntimeError):
    """Raised when the FIRMS API request fails."""


@dataclass
class FirmsValidationSummary:
    """Capture FIRMS validation results for logging."""

    total_rows: int = 0
    parsed_rows: int = 0
    skipped_invalid_coord: int = 0
    skipped_invalid_time: int = 0
    missing_confidence: int = 0
    confidence_out_of_range: int = 0
    brightness_missing: int = 0
    brightness_out_of_range: int = 0
    sensor_counts: Counter[str] = field(default_factory=Counter)
    confidence_buckets: Counter[str] = field(default_factory=Counter)


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
) -> tuple[List[DetectionRecord], FirmsValidationSummary]:
    """Normalize FIRMS CSV rows into `DetectionRecord`s and validation details."""
    detections: List[DetectionRecord] = []
    rows_list = rows if isinstance(rows, list) else list(rows)
    summary = FirmsValidationSummary(total_rows=len(rows_list))

    for row in rows_list:
        lat_raw = row.get("latitude")
        lon_raw = row.get("longitude")
        if lat_raw is None or lon_raw is None:
            summary.skipped_invalid_coord += 1
            log_event(
                LOGGER,
                "firms.validation",
                "Skipping row with missing coordinates",
                level="warning",
                row_ref=_row_ref(row),
            )
            continue

        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except ValueError:
            summary.skipped_invalid_coord += 1
            log_event(
                LOGGER,
                "firms.validation",
                "Skipping row with non-numeric coordinates",
                level="warning",
                row_ref=_row_ref(row),
                latitude=lat_raw,
                longitude=lon_raw,
            )
            continue

        if not (VALID_LAT_RANGE[0] <= lat <= VALID_LAT_RANGE[1]) or not (
            VALID_LON_RANGE[0] <= lon <= VALID_LON_RANGE[1]
        ):
            summary.skipped_invalid_coord += 1
            log_event(
                LOGGER,
                "firms.validation",
                "Skipping row with out-of-range coordinates",
                level="warning",
                latitude=lat,
                longitude=lon,
                row_ref=_row_ref(row),
            )
            continue

        try:
            acq_time = _parse_acq_datetime(row)
        except (KeyError, ValueError) as exc:
            summary.skipped_invalid_time += 1
            log_event(
                LOGGER,
                "firms.validation",
                "Skipping row with invalid acquisition time",
                level="warning",
                error=str(exc),
                row_ref=_row_ref(row),
            )
            continue

        confidence = _parse_confidence(row)
        if confidence is None:
            summary.missing_confidence += 1
        elif not (CONFIDENCE_RANGE[0] <= confidence <= CONFIDENCE_RANGE[1]):
            summary.confidence_out_of_range += 1
            log_event(
                LOGGER,
                "firms.validation",
                "Confidence out of expected range; dropping value",
                level="warning",
                confidence=confidence,
                row_ref=_row_ref(row),
            )
            confidence = None

        brightness = _optional_float(row.get("brightness"))
        if brightness is None:
            summary.brightness_missing += 1
        elif not (BRIGHTNESS_RANGE[0] <= brightness <= BRIGHTNESS_RANGE[1]):
            summary.brightness_out_of_range += 1
            log_event(
                LOGGER,
                "firms.validation",
                "Brightness out of expected range; dropping value",
                level="warning",
                brightness=brightness,
                row_ref=_row_ref(row),
            )
            brightness = None

        detection = DetectionRecord(
            lat=lat,
            lon=lon,
            acq_time=acq_time,
            sensor=_pick_sensor(row),
            source=source,
            confidence=confidence,
            brightness=brightness,
            bright_t31=_optional_float(row.get("bright_t31")),
            frp=_optional_float(row.get("frp")),
            scan=_optional_float(row.get("scan")),
            track=_optional_float(row.get("track")),
            raw_properties={**row},
            ingest_batch_id=ingest_batch_id,
        )
        detections.append(detection)

        sensor_label = detection.sensor or "unknown"
        summary.sensor_counts[sensor_label] += 1
        summary.confidence_buckets[_bucket_confidence(detection.confidence)] += 1

    summary.parsed_rows = len(detections)
    return detections, summary


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


def _row_ref(row: Dict[str, str]) -> Dict[str, str | None]:
    """Small reference payload to avoid logging entire rows."""
    return {
        "acq_date": row.get("acq_date"),
        "acq_time": row.get("acq_time"),
        "sensor": row.get("sensor") or row.get("instrument") or row.get("satellite"),
        "latitude": row.get("latitude"),
        "longitude": row.get("longitude"),
    }


def _bucket_confidence(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 30:
        return "low"
    if value < 70:
        return "nominal"
    return "high"


