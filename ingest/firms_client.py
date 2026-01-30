"""Helpers for downloading and parsing NASA FIRMS CSV feeds."""

from __future__ import annotations

import csv
import io
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import httpx

from api.fires.service import normalize_firms_confidence
from ingest.logging_utils import log_event
from ingest.models import DetectionRecord

LOGGER = logging.getLogger(__name__)
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
VALID_LAT_RANGE = (-90.0, 90.0)
VALID_LON_RANGE = (-180.0, 180.0)
CONFIDENCE_RANGE = (0.0, 100.0)
BRIGHTNESS_RANGE = (200.0, 500.0)

# Rate limit handling constants
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0
RATE_LIMIT_STATUS_CODE = 429


class FIRMSClientError(RuntimeError):
    """Raised when the FIRMS API request fails."""


def redact_firms_url(url: str, map_key: str) -> str:
    """Remove the API key from a FIRMS URL for safe logging/storage."""
    return url.replace(f"/{map_key}/", "/<redacted>/")


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
    max_retries: int = MAX_RETRIES,
    backoff_base_seconds: float = BACKOFF_BASE_SECONDS,
) -> List[Dict[str, str]]:
    """Download FIRMS CSV rows with rate limit handling and exponential backoff.
    
    Args:
        map_key: FIRMS API key
        source: Data source (e.g., VIIRS_SNPP_NRT)
        bbox: Bounding box string "west,south,east,north"
        day_range: Number of days to fetch
        timeout_seconds: HTTP request timeout
        date: Optional specific date
        max_retries: Maximum number of retries for rate limit (429) errors
        backoff_base_seconds: Base backoff time, doubled each retry
        
    Returns:
        List of CSV rows as dictionaries
        
    Raises:
        FIRMSClientError: If the request fails after all retries
    """
    url = build_firms_url(map_key, source, bbox, day_range, date=date)
    # Avoid logging sensitive API tokens.
    safe_url = redact_firms_url(url, map_key)
    
    last_exception: Exception | None = None
    
    for attempt in range(max_retries + 1):
        LOGGER.info(
            "Requesting FIRMS CSV",
            extra={
                "source": source,
                "url": safe_url,
                "attempt": attempt + 1,
                "max_retries": max_retries,
            },
        )
        
        try:
            response = httpx.get(url, timeout=timeout_seconds)
            
            # Handle rate limit (429) with exponential backoff
            if response.status_code == RATE_LIMIT_STATUS_CODE:
                if attempt < max_retries:
                    sleep_seconds = backoff_base_seconds * (2 ** attempt)
                    LOGGER.warning(
                        "FIRMS API rate limit hit (429), retrying after %.1f seconds (attempt %d/%d)",
                        sleep_seconds,
                        attempt + 1,
                        max_retries,
                        extra={
                            "source": source,
                            "attempt": attempt + 1,
                            "backoff_seconds": sleep_seconds,
                        },
                    )
                    time.sleep(sleep_seconds)
                    continue
                else:
                    raise FIRMSClientError(
                        f"FIRMS API rate limit exceeded after {max_retries} retries. "
                        "Consider reducing request frequency or day_range."
                    )
            
            response.raise_for_status()
            
            text_stream = io.StringIO(response.text)
            reader = csv.DictReader(text_stream)
            rows = list(reader)
            LOGGER.info(
                "Fetched %s rows from FIRMS",
                len(rows),
                extra={
                    "source": source,
                    "attempts": attempt + 1,
                },
            )
            return rows
            
        except httpx.HTTPStatusError as exc:
            last_exception = exc
            # Don't retry on client errors (4xx) except rate limit (handled above)
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != RATE_LIMIT_STATUS_CODE:
                raise FIRMSClientError(f"FIRMS API client error: {exc}") from exc
            # Retry on server errors (5xx) if we have retries left
            if exc.response.status_code >= 500 and attempt < max_retries:
                sleep_seconds = backoff_base_seconds * (2 ** attempt)
                LOGGER.warning(
                    "FIRMS API server error (%s), retrying after %.1f seconds",
                    exc.response.status_code,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            raise FIRMSClientError(f"Failed to fetch FIRMS data: {exc}") from exc
            
        except httpx.HTTPError as exc:  # pragma: no cover - network error path
            last_exception = exc
            if attempt < max_retries:
                sleep_seconds = backoff_base_seconds * (2 ** attempt)
                LOGGER.warning(
                    "FIRMS request failed (%s), retrying after %.1f seconds",
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            raise FIRMSClientError(f"Failed to fetch FIRMS data after {max_retries} retries: {exc}") from exc
    
    # This should not be reached, but just in case
    raise FIRMSClientError(f"Failed to fetch FIRMS data: {last_exception}")


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

        sensor = _pick_sensor(row)
        confidence_score = normalize_firms_confidence(confidence, sensor)

        detection = DetectionRecord(
            lat=lat,
            lon=lon,
            acq_time=acq_time,
            sensor=sensor,
            source=source,
            confidence=confidence,
            confidence_score=confidence_score,
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


