"""API client for the wildfire dashboard."""

import os
import httpx
from typing import Any, Dict, Optional

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def get_latest_forecast(
    region_name: str,
    bbox: tuple[float, float, float, float]
) -> Optional[Dict[str, Any]]:
    """Fetch the latest spread forecast for an AOI from the API."""
    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "region_name": region_name,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }
    
    try:
        response = httpx.get(f"{API_BASE_URL}/forecast", params=params, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if data.get("run"):
                return data
        return None
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        return None

