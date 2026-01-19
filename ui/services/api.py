"""API client for the wildfire dashboard."""

from typing import Any, Dict, Optional

from api_client import (
    ApiError,
    ApiUnavailableError,
    get_forecast,
)

def get_latest_forecast(
    region_name: str,
    bbox: tuple[float, float, float, float]
) -> Optional[Dict[str, Any]]:
    """Fetch the latest spread forecast for an AOI from the API."""
    try:
        data = get_forecast(bbox=bbox, horizons=None)
        # Preserve previous behavior: treat missing run as None.
        if not data.get("run"):
            return None
        return data
    except (ApiUnavailableError, ApiError):
        return None

