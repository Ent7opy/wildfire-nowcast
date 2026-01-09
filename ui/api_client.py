"""Thin requests-based client for the FastAPI backend.

Keep this module backend-contract-aware and minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import requests

from runtime_config import api_base_url


JsonDict = Dict[str, Any]
BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
TimeRange = Tuple[datetime, datetime]  # (start_time, end_time)


@dataclass
class ApiError(Exception):
    message: str
    status_code: Optional[int] = None
    url: Optional[str] = None
    response_text: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"(status={self.status_code})")
        if self.url:
            parts.append(f"url={self.url}")
        return " ".join(parts)


class ApiUnavailableError(ApiError):
    """Backend is unreachable or timed out."""


def _isoformat(dt: datetime) -> str:
    # FastAPI parses RFC3339/ISO-8601; use 'Z' for UTC to avoid URL encoding issues with '+00:00'
    # Remove microseconds to avoid potential parsing issues
    if dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0:
        # UTC timezone - use 'Z' suffix instead of '+00:00' to avoid URL encoding issues
        # Remove microseconds for cleaner datetime strings
        dt_no_microseconds = dt.replace(microsecond=0)
        return dt_no_microseconds.replace(tzinfo=None).isoformat() + "Z"
    # Remove microseconds for non-UTC times too
    dt_no_microseconds = dt.replace(microsecond=0)
    return dt_no_microseconds.isoformat()


def _get_json(path: str, params: Mapping[str, Any]) -> JsonDict:
    base = api_base_url()
    url = f"{base}{path}"
    try:
        resp = requests.get(url, params=dict(params), timeout=(2.0, 5.0))
    except (requests.Timeout, requests.ConnectionError) as e:
        raise ApiUnavailableError(message=str(e), url=url) from e

    if resp.status_code != 200:
        raise ApiError(
            message="Non-200 response from API",
            status_code=resp.status_code,
            url=str(resp.url),
            response_text=resp.text,
        )

    try:
        return resp.json()
    except ValueError as e:
        raise ApiError(
            message="API returned non-JSON response",
            status_code=resp.status_code,
            url=str(resp.url),
            response_text=resp.text,
        ) from e


def get_fires(
    bbox: BBox,
    time_range: TimeRange,
    filters: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Fetch FIRMS detections from the backend.

    Backend contract: GET /fires (alias for /fires/detections)
      - min_lon, min_lat, max_lon, max_lat (float)
      - start_time, end_time (datetime)
      - min_confidence (float, optional)
      - include_noise (bool, optional)
      - include_denoiser_fields (bool, optional)
      - limit (int, optional)

    Response shape:
      { "count": int, "detections": [ { "lat": float, "lon": float, ... }, ... ] }
    """
    # Validate bbox
    if not bbox or len(bbox) != 4:
        raise ApiError(
            message="Invalid bbox: must be a 4-tuple (min_lon, min_lat, max_lon, max_lat)",
            url=None,
        )
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Validate bbox values
    if not all(isinstance(x, (int, float)) for x in bbox):
        raise ApiError(
            message="Invalid bbox: all values must be numbers",
            url=None,
        )
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ApiError(
            message="Invalid bbox: min values must be less than max values",
            url=None,
        )
    
    start_time, end_time = time_range

    params: Dict[str, Any] = {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "start_time": _isoformat(start_time),
        "end_time": _isoformat(end_time),
    }
    if filters:
        params.update(dict(filters))

    data = _get_json("/fires", params=params)
    if not isinstance(data, dict):
        raise ApiError(message="API returned invalid fires payload (not a JSON object)", url=None)
    detections = data.get("detections")
    if detections is None or not isinstance(detections, list):
        raise ApiError(
            message="API returned invalid fires payload (missing 'detections')",
            status_code=None,
            url=None,
            response_text=str(data)[:500],
        )
    return data


def get_forecast(
    bbox: BBox,
    horizons: Optional[Iterable[int]] = None,
    region_name: Optional[str] = None,
) -> JsonDict:
    """Fetch latest spread forecast metadata for a bbox.

    Backend contract: GET /forecast
      - min_lon, min_lat, max_lon, max_lat
      - region_name (optional - if None, uses location-based forecasting)

    The `horizons` argument is currently not used by the backend route; it is accepted
    here to keep the UI call-site explicit and future-compatible.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    params: Dict[str, Any] = {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }
    if region_name is not None:
        params["region_name"] = region_name
    elif horizons is not None:
        # Not currently consumed by the backend; safe to ignore server-side.
        params["horizons"] = ",".join(str(h) for h in horizons)

    data = _get_json("/forecast", params=params)
    if not isinstance(data, dict):
        raise ApiError(message="API returned invalid forecast payload (not a JSON object)", url=None)
    if "run" not in data:
        raise ApiError(
            message="API returned invalid forecast payload (missing 'run')",
            status_code=None,
            url=None,
            response_text=str(data)[:500],
        )
    return data


def generate_forecast(
    bbox: BBox,
    horizons: Optional[Iterable[int]] = None,
    region_name: Optional[str] = None,
    forecast_reference_time: Optional[datetime] = None,
) -> JsonDict:
    """Generate a spread forecast on-the-fly for a bbox.

    Backend contract: POST /forecast/generate
      Request body:
      - min_lon, min_lat, max_lon, max_lat
      - region_name (optional - if None, uses location-based forecasting)
      - forecast_reference_time (optional - ISO format string, defaults to now)
      - horizons_hours (optional - list of ints, defaults to [24,48,72])
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    body: Dict[str, Any] = {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }
    if region_name is not None:
        body["region_name"] = region_name
    if horizons is not None:
        body["horizons_hours"] = list(horizons)
    if forecast_reference_time is not None:
        body["forecast_reference_time"] = _isoformat(forecast_reference_time)

    base = api_base_url()
    url = f"{base}/forecast/generate"
    try:
        resp = requests.post(url, json=body, timeout=(5.0, 60.0))  # Longer timeout for forecast generation
    except (requests.Timeout, requests.ConnectionError) as e:
        raise ApiUnavailableError(message=str(e), url=url) from e

    if resp.status_code != 200:
        raise ApiError(
            message="Non-200 response from forecast generation API",
            status_code=resp.status_code,
            url=str(resp.url),
            response_text=resp.text,
        )

    try:
        return resp.json()
    except ValueError as e:
        raise ApiError(
            message="API returned non-JSON response",
            status_code=resp.status_code,
            url=str(resp.url),
            response_text=resp.text,
        ) from e

