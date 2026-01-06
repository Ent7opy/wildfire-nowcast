"""Thin requests-based client for the FastAPI backend.

Keep this module backend-contract-aware and minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import requests

from ui.runtime_config import api_base_url, forecast_region_name


JsonDict = Dict[str, Any]
BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
TimeRange = Tuple[datetime, datetime]  # (start_time, end_time)


@dataclass(frozen=True)
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
    # FastAPI parses RFC3339/ISO-8601; this keeps timezone if present.
    return dt.isoformat()


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
      - include_noise (bool, optional)
      - include_denoiser_fields (bool, optional)
      - limit (int, optional)

    Response shape:
      { "count": int, "detections": [ { "lat": float, "lon": float, ... }, ... ] }
    """
    min_lon, min_lat, max_lon, max_lat = bbox
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

    return _get_json("/fires", params=params)


def get_forecast(
    bbox: BBox,
    horizons: Optional[Iterable[int]] = None,
) -> JsonDict:
    """Fetch latest spread forecast metadata for a bbox.

    Backend contract: GET /forecast
      - region_name (required by backend)
      - min_lon, min_lat, max_lon, max_lat

    The `horizons` argument is currently not used by the backend route; it is accepted
    here to keep the UI call-site explicit and future-compatible.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    params: Dict[str, Any] = {
        "region_name": forecast_region_name(),
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }
    if horizons is not None:
        # Not currently consumed by the backend; safe to ignore server-side.
        params["horizons"] = ",".join(str(h) for h in horizons)

    return _get_json("/forecast", params=params)

