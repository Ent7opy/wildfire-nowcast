from datetime import datetime, timezone

from fastapi import APIRouter

from api.config import settings
from api.data_status import build_data_status_snapshot
from api.model_registry import list_active_models

internal_router = APIRouter(tags=["internal"])


@internal_router.get("/health")
async def healthcheck() -> dict:
    """Simple health endpoint used for local dev and readiness checks."""
    return {"status": "ok"}


@internal_router.get("/health/data-freshness")
async def data_freshness_healthcheck() -> dict:
    """Return source freshness, stale-data policy, and idempotency dashboard metrics."""
    try:
        return build_data_status_snapshot()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "as_of": None,
            "overall_state": "unknown",
            "stale_sources": [],
            "critical_stale_sources": [],
            "forecast_inputs_ready": False,
            "forecast_gate": {
                "can_run": False,
                "would_block_if_fail_closed": True,
                "policy": "fail_closed",
                "reasons": ["snapshot_unavailable"],
                "missing_or_stale_sources": [],
                "as_of": None,
                "retry_hint": "retry after health snapshot recovers",
            },
            "stale_behavior": {
                "mode": "degraded",
                "policy": "serve_last_known_data_with_warning",
                "fires_api": "returns cached/latest detections and includes freshness status endpoint",
                "forecast_api": "deny_new_forecasts_until_weather_fresh_and_terrain_present",
                "ui": "show_stale_data_banner_when_state_not_healthy",
                "critical_sources": ["firms", "weather"],
            },
            "sources": {},
            "idempotency_dashboard": {},
            "error": str(exc),
        }


@internal_router.get("/version")
async def version() -> dict:
    """Return the current app version and deployment metadata."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "git_commit": settings.git_commit,
        "environment": settings.environment,
    }


@internal_router.get("/internal/models/active")
async def active_models() -> dict:
    """Return currently promoted (active) model per family."""
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        models = list_active_models()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"as_of": as_of, "models": {}, "error": str(exc)}

    return {"as_of": as_of, "models": models}
