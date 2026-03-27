"""Startup configuration validation for the ingest orchestrator.

Called from orchestrator.main() before any jobs are built or run. Raises
StartupError for fatal misconfigurations; logs WARNING for optional issues.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

LOGGER = logging.getLogger("ingest_orchestrator")


class StartupError(RuntimeError):
    """Raised when required configuration is missing or invalid at startup."""


def validate_firms_map_key() -> None:
    """Check that FIRMS_MAP_KEY is present and non-empty."""
    key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise StartupError(
            "FIRMS_MAP_KEY is not set. "
            "Obtain an API key from https://firms.modaps.eosdis.nasa.gov/api/area/ "
            "and set FIRMS_MAP_KEY in your .env file."
        )


def validate_denoiser_model_path() -> None:
    """If DENOISER_ENABLED=true, verify DENOISER_MODEL_RUN_DIR is set and exists.

    Fatal (StartupError) when DENOISER_REQUIRED=true.
    Warning-only when DENOISER_REQUIRED=false so the orchestrator can still run
    without the denoiser on initial bootstrap.
    """
    enabled = os.getenv("DENOISER_ENABLED", "false").strip().lower() in ("1", "true", "yes")
    if not enabled:
        return

    required = os.getenv("DENOISER_REQUIRED", "true").strip().lower() in ("1", "true", "yes")
    run_dir = os.getenv("DENOISER_MODEL_RUN_DIR", "").strip()

    if not run_dir:
        msg = (
            "DENOISER_MODEL_RUN_DIR is not set but DENOISER_ENABLED=true. "
            "Set DENOISER_MODEL_RUN_DIR to the trained model artifact directory, "
            "or set DENOISER_REQUIRED=false to run without the denoiser."
        )
        if required:
            raise StartupError(msg)
        LOGGER.warning("%s Denoiser will be skipped (DENOISER_REQUIRED=false).", msg)
        return

    path = Path(run_dir)
    if not path.is_dir():
        msg = (
            f"DENOISER_MODEL_RUN_DIR={run_dir!r} does not exist or is not a directory. "
            "Ensure the model artifact directory is present before starting ingest."
        )
        if required:
            raise StartupError(msg)
        LOGGER.warning("%s Denoiser will be skipped (DENOISER_REQUIRED=false).", msg)


def run_ingest_startup_checks() -> None:
    """Run all ingest orchestrator startup checks.

    Raises StartupError on fatal misconfiguration.
    Logs WARNING for optional missing config.
    """
    validate_firms_map_key()
    validate_denoiser_model_path()
