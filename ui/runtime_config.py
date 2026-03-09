"""Runtime configuration for the Streamlit UI.

Note: This file intentionally does NOT use the name `ui/config.py` because the repo
already has a `ui/config/` package for UI constants.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse


DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_VECTOR_TILES_BASE_URL = "http://localhost:7800"
DEFAULT_FORECAST_REGION_NAME = "smoke_grid"


def _rewrite_internal_service_host(base_url: str) -> str:
    """Rewrite internal Docker service hostnames to host-accessible localhost URLs."""
    value = (base_url or "").rstrip("/")
    if not value:
        return value
    try:
        parsed = urlparse(value)
    except Exception:
        return value

    host = parsed.hostname
    if host != "api":
        return value

    scheme = parsed.scheme or "http"
    port = parsed.port or 8000
    return urlunparse((scheme, f"localhost:{port}", "", "", "", ""))


def api_base_url() -> str:
    """FastAPI base URL (no trailing slash)."""
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def api_base_url_candidates() -> list[str]:
    """Candidate API base URLs ordered by preference.

    Helps local/browser dev when `API_BASE_URL` is set to Compose-internal host
    (`http://api:8000`) but the app runs outside that Docker network.
    """
    primary = api_base_url()
    public = os.getenv("API_PUBLIC_BASE_URL", "").rstrip("/")

    out: list[str] = []
    for value in (primary, public):
        if value and value not in out:
            out.append(value)

    try:
        parsed = urlparse(primary)
    except Exception:
        parsed = None

    host = parsed.hostname if parsed is not None else None
    port = parsed.port if parsed is not None else None
    scheme = (parsed.scheme if parsed is not None and parsed.scheme else "http")
    effective_port = port or 8000

    if host == "api":
        localhost = urlunparse((scheme, f"localhost:{effective_port}", "", "", "", ""))
        loopback = urlunparse((scheme, f"127.0.0.1:{effective_port}", "", "", "", ""))
        for value in (localhost, loopback):
            if value not in out:
                out.append(value)

    return out


def api_public_base_url() -> str:
    """FastAPI base URL for browser requests (no trailing slash)."""
    raw = os.getenv("API_PUBLIC_BASE_URL")
    if raw is not None and raw.strip():
        return _rewrite_internal_service_host(raw.strip())
    return _rewrite_internal_service_host(api_base_url())


def vector_tiles_base_url() -> str:
    """Vector tile server base URL (no trailing slash)."""
    return os.getenv("VECTOR_TILES_PUBLIC_BASE_URL", DEFAULT_VECTOR_TILES_BASE_URL).rstrip("/")


def forecast_region_name() -> str:
    """Region name required by the backend /forecast contract."""
    return os.getenv("FORECAST_REGION_NAME", DEFAULT_FORECAST_REGION_NAME)
