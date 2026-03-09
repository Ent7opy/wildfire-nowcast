"""Thin requests-based client for the FastAPI backend.

Keep this module backend-contract-aware and minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import requests

from runtime_config import api_base_url, api_base_url_candidates


__all__ = [
    "ApiError",
    "ApiUnavailableError",
    "get_fires",
    "get_fire_events",
    "get_fire_fronts",
    "get_forecast",
    "generate_forecast",
    "create_jit_forecast",
    "create_jit_forecast_from_front",
    "get_active_spread_model_id",
    "get_jit_forecast_status",
    "get_data_freshness_status",
]


JsonDict = Dict[str, Any]
BBox = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
TimeRange = Tuple[datetime, datetime]  # (start_time, end_time)
_GET_CONNECT_TIMEOUT = 2.0
_GET_READ_TIMEOUT = 8.0
_GET_RETRY_READ_TIMEOUT = 15.0
_SLOW_READ_RETRY_PATHS = (
    "/fires",
    "/fires/events",
    "/fires/fronts",
    "/health/data-freshness",
)


def get_jit_forecast_status(job_id: str) -> JsonDict:
    """Get JIT forecast job status.

    Backend contract: GET /forecast/jit/{job_id}

    Returns:
      {
        "job_id": UUID,
        "status": "pending|ingesting_terrain|ingesting_weather|running_forecast|completed|failed",
        "progress_message": str,
        "result": {...} (if completed),
        "error": str (if failed)
      }
    """
    base = api_base_url()
    url = f"{base}/forecast/jit/{job_id}"
    try:
        resp = requests.get(url, timeout=(2.0, 5.0))
    except (requests.Timeout, requests.ConnectionError) as e:
        raise ApiUnavailableError(message=str(e), url=url) from e

    if resp.status_code == 404:
        raise ApiError(
            message="Job not found",
            status_code=404,
            url=str(resp.url),
            response_text=resp.text,
        )

    if resp.status_code != 200:
        raise ApiError(
            message="Non-200 response from JIT status API",
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


def get_data_freshness_status() -> JsonDict:
    """Get data freshness, stale behavior, and idempotency dashboard snapshot."""
    return _get_json("/health/data-freshness", params={})


def get_active_spread_model_id() -> str:
    """Resolve the currently promoted spread model id from the backend registry."""
    payload = _get_json("/internal/models/active", params={})
    models = payload.get("models") if isinstance(payload, dict) else None
    spread = models.get("spread") if isinstance(models, dict) else None
    model_id = spread.get("model_id") if isinstance(spread, dict) else None
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    raise ApiError(
        message="No active spread model is promoted. Promote a spread model and retry.",
        status_code=422,
    )


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
    # Check if UTC: utcoffset() can return None even when tzinfo is set (custom tzinfo implementations)
    offset = dt.utcoffset() if dt.tzinfo is not None else None
    if offset is not None and offset.total_seconds() == 0:
        # UTC timezone - use 'Z' suffix instead of '+00:00' to avoid URL encoding issues
        # Remove microseconds for cleaner datetime strings
        dt_no_microseconds = dt.replace(microsecond=0)
        return dt_no_microseconds.replace(tzinfo=None).isoformat() + "Z"
    # Remove microseconds for non-UTC times too
    dt_no_microseconds = dt.replace(microsecond=0)
    return dt_no_microseconds.isoformat()


def _get_json(path: str, params: Mapping[str, Any]) -> JsonDict:
    params_dict = dict(params)
    last_error: ApiUnavailableError | None = None
    candidates = api_base_url_candidates() or [api_base_url()]

    resp: requests.Response | None = None
    for base in candidates:
        url = f"{base}{path}"
        try:
            resp = requests.get(url, params=params_dict, timeout=(_GET_CONNECT_TIMEOUT, _GET_READ_TIMEOUT))
            break
        except requests.Timeout as e:
            # Some endpoints are heavier; retry once with a longer read timeout.
            if any(path.startswith(prefix) for prefix in _SLOW_READ_RETRY_PATHS):
                try:
                    resp = requests.get(
                        url,
                        params=params_dict,
                        timeout=(_GET_CONNECT_TIMEOUT, _GET_RETRY_READ_TIMEOUT),
                    )
                    break
                except (requests.Timeout, requests.ConnectionError) as inner_exc:
                    last_error = ApiUnavailableError(message=str(inner_exc), url=url)
            else:
                last_error = ApiUnavailableError(message=str(e), url=url)
        except requests.ConnectionError as e:
            last_error = ApiUnavailableError(message=str(e), url=url)

    if resp is None:
        if last_error is None:
            last_error = ApiUnavailableError(message="API unavailable", url=None)
        raise last_error

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
        normalized_filters: Dict[str, Any] = {}
        for key, value in dict(filters).items():
            if isinstance(value, bool):
                normalized_filters[key] = "true" if value else "false"
            else:
                normalized_filters[key] = value
        params.update(normalized_filters)

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


def get_fire_events(
    bbox: BBox,
    time_range: TimeRange,
    filters: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Fetch fire events from the backend.

    Backend contract: GET /fires/events
      - min_lon, min_lat, max_lon, max_lat (float)
      - start_time, end_time (datetime)
      - min_event_score (float, optional)
      - include_review_required (bool, optional)
      - limit (int, optional)

    Response shape:
      { "count": int, "events": [ { "event_id": str, "lat": float, "lon": float, "geom_geojson": str, ... }, ... ] }
    """
    if not bbox or len(bbox) != 4:
        raise ApiError(
            message="Invalid bbox: must be a 4-tuple (min_lon, min_lat, max_lon, max_lat)",
            url=None,
        )
    min_lon, min_lat, max_lon, max_lat = bbox
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
        normalized_filters: Dict[str, Any] = {}
        for key, value in dict(filters).items():
            if isinstance(value, bool):
                normalized_filters[key] = "true" if value else "false"
            else:
                normalized_filters[key] = value
        params.update(normalized_filters)

    data = _get_json("/fires/events", params=params)
    if not isinstance(data, dict):
        raise ApiError(message="API returned invalid events payload (not a JSON object)", url=None)
    events = data.get("events")
    if events is None or not isinstance(events, list):
        raise ApiError(
            message="API returned invalid events payload (missing 'events')",
            status_code=None,
            url=None,
            response_text=str(data)[:500],
        )
    return data


def get_fire_fronts(
    bbox: BBox,
    time_range: TimeRange,
    filters: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Fetch fire fronts from the backend.

    Backend contract: GET /fires/fronts
      - min_lon, min_lat, max_lon, max_lat (float)
      - start_time, end_time (datetime)
      - min_event_score (float, optional)
      - include_review_required (bool, optional)
      - limit (int, optional)

    Response shape:
      { "count": int, "fronts": [ { "front_id": str, "geom_geojson": str, ... }, ... ] }
    """
    if not bbox or len(bbox) != 4:
        raise ApiError(
            message="Invalid bbox: must be a 4-tuple (min_lon, min_lat, max_lon, max_lat)",
            url=None,
        )
    min_lon, min_lat, max_lon, max_lat = bbox
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
        normalized_filters: Dict[str, Any] = {}
        for key, value in dict(filters).items():
            if isinstance(value, bool):
                normalized_filters[key] = "true" if value else "false"
            else:
                normalized_filters[key] = value
        params.update(normalized_filters)

    data = _get_json("/fires/fronts", params=params)
    if not isinstance(data, dict):
        raise ApiError(message="API returned invalid fronts payload (not a JSON object)", url=None)
    fronts = data.get("fronts")
    if fronts is None or not isinstance(fronts, list):
        raise ApiError(
            message="API returned invalid fronts payload (missing 'fronts')",
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


def create_jit_forecast(
    bbox: BBox,
    horizons: Optional[Iterable[int]] = None,
    forecast_reference_time: Optional[datetime] = None,
    model_id: Optional[str] = None,
) -> JsonDict:
    """Enqueue a JIT forecast pipeline for arbitrary bbox.

    Backend contract: POST /forecast/jit
      Request body:
      - bbox: [min_lon, min_lat, max_lon, max_lat]
      - forecast_reference_time (optional - ISO format string, defaults to now)
      - horizons_hours (optional - list of ints, defaults to [24,48,72])

    Returns:
      { "job_id": UUID, "status": "queued" }
    """
    body: Dict[str, Any] = {
        "bbox": list(bbox),
    }
    if horizons is not None:
        body["horizons_hours"] = list(horizons)
    if forecast_reference_time is not None:
        body["forecast_reference_time"] = _isoformat(forecast_reference_time)
    if model_id is not None:
        body["model_id"] = str(model_id)

    base = api_base_url()
    url = f"{base}/forecast/jit"
    try:
        resp = requests.post(url, json=body, timeout=(5.0, 10.0))
    except (requests.Timeout, requests.ConnectionError) as e:
        raise ApiUnavailableError(message=str(e), url=url) from e

    if resp.status_code != 202:
        message = "Non-202 response from JIT forecast API"
        try:
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("message"):
                message = str(payload["message"])
        except ValueError:
            payload = None
        raise ApiError(
            message=message,
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


def create_jit_forecast_from_front(
    front_id: str,
    *,
    buffer_km: float = 3.0,
    horizons: Optional[Iterable[int]] = None,
    forecast_reference_time: Optional[datetime] = None,
    model_id: Optional[str] = None,
) -> JsonDict:
    """Enqueue a front-driven JIT forecast pipeline.

    Backend contract: POST /forecast/jit/from-front
      Request body:
      - front_id: str
      - buffer_km: float (optional)
      - forecast_reference_time (optional)
      - horizons_hours (optional)

    Returns:
      { "job_id": UUID, "status": "queued", "front_id": str, "bbox": [...] }
    """
    body: Dict[str, Any] = {
        "front_id": str(front_id),
        "buffer_km": float(buffer_km),
    }
    if horizons is not None:
        body["horizons_hours"] = list(horizons)
    if forecast_reference_time is not None:
        body["forecast_reference_time"] = _isoformat(forecast_reference_time)
    if model_id is not None:
        body["model_id"] = str(model_id)

    base = api_base_url()
    url = f"{base}/forecast/jit/from-front"
    try:
        resp = requests.post(url, json=body, timeout=(5.0, 10.0))
    except (requests.Timeout, requests.ConnectionError) as e:
        raise ApiUnavailableError(message=str(e), url=url) from e

    if resp.status_code != 202:
        message = "Non-202 response from front JIT forecast API"
        try:
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("message"):
                message = str(payload["message"])
        except ValueError:
            payload = None
        raise ApiError(
            message=message,
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
