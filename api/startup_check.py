"""Startup configuration validation for the API.

Called from the FastAPI startup handler. Raises StartupError for fatal
misconfigurations that would cause cryptic failures later; logs WARNING
for optional config whose absence degrades features but doesn't break the API.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from api.config import AppSettings

LOGGER = logging.getLogger(__name__)


class StartupError(RuntimeError):
    """Raised when required configuration is missing or invalid at startup."""


def validate_database_url(database_url: str) -> None:
    """Check that the resolved database URL is a structurally valid postgresql:// URL."""
    if not database_url or not database_url.strip():
        raise StartupError(
            "DATABASE_URL is not set. "
            "Provide DATABASE_URL (or DATABASE_PRIVATE_URL) or set "
            "POSTGRES_HOST / POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB."
        )
    url = database_url.strip()
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise StartupError(f"DATABASE_URL is malformed: {exc}") from exc

    valid_schemes = {
        "postgresql",
        "postgresql+asyncpg",
        "postgresql+psycopg2",
        "postgresql+psycopg",
    }
    if parsed.scheme not in valid_schemes:
        raise StartupError(
            f"DATABASE_URL has unsupported scheme '{parsed.scheme}'. "
            f"Expected postgresql:// (e.g., postgresql://user:pass@host:5432/db). "
            f"Got: {url!r}"
        )
    if not parsed.hostname:
        raise StartupError(
            f"DATABASE_URL is missing a hostname: {url!r}. "
            "Expected format: postgresql://user:pass@host:5432/db."
        )
    db_name = (parsed.path or "").lstrip("/")
    if not db_name:
        raise StartupError(
            f"DATABASE_URL is missing a database name: {url!r}. "
            "Expected format: postgresql://user:pass@host:5432/db."
        )


def validate_spread_model_artifact_paths() -> None:
    """If SPREAD_MODEL_CATALOG_JSON is set, verify any referenced artifact dirs exist.

    JSON parse errors are intentionally not raised here — they surface with a
    clearer message when get_spread_model_catalog() is first called.
    """
    raw = os.getenv("SPREAD_MODEL_CATALOG_JSON")
    if not raw or not raw.strip():
        return

    try:
        catalog = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return  # deferred to catalog load

    if not isinstance(catalog, dict):
        return

    for model_id, entry in catalog.items():
        if not isinstance(entry, dict):
            continue
        params = entry.get("model_params")
        if not isinstance(params, dict):
            continue
        for key in ("model_run_dir", "calibrator_run_dir"):
            path_str = params.get(key)
            if path_str and isinstance(path_str, str):
                if not Path(path_str).is_dir():
                    raise StartupError(
                        f"Spread model artifact path for model_id={model_id!r} "
                        f"({key}={path_str!r}) does not exist or is not a directory. "
                        "Check SPREAD_MODEL_CATALOG_JSON and ensure model artifacts are mounted."
                    )


def warn_optional_config(gemini_api_key: str) -> None:
    """Log warnings for optional config that won't block startup."""
    if not gemini_api_key or not gemini_api_key.strip():
        LOGGER.warning(
            "GEMINI_API_KEY is not set; the AI assistant (/assistant) endpoint will be "
            "unavailable. Set GEMINI_API_KEY to enable it."
        )


def run_api_startup_checks(settings: AppSettings) -> None:
    """Run all API startup checks.

    Raises StartupError on fatal misconfiguration.
    Logs WARNING for optional missing config.
    """
    validate_database_url(settings.database_url)
    validate_spread_model_artifact_paths()
    warn_optional_config(settings.gemini_api_key)
