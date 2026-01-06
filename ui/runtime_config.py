"""Runtime configuration for the Streamlit UI.

Note: This file intentionally does NOT use the name `ui/config.py` because the repo
already has a `ui/config/` package for UI constants.
"""

from __future__ import annotations

import os


DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_FORECAST_REGION_NAME = "smoke_grid"


def api_base_url() -> str:
    """FastAPI base URL (no trailing slash)."""
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def forecast_region_name() -> str:
    """Region name required by the backend /forecast contract."""
    return os.getenv("FORECAST_REGION_NAME", DEFAULT_FORECAST_REGION_NAME)

