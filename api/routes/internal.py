from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from api.config import settings
from api.data_status import build_data_status_snapshot
from api.model_registry import list_active_models
from api.fires.repo import (
    get_latest_denoiser_gate_report,
    get_latest_denoiser_coverage_status,
    list_recent_denoiser_drift,
    list_denoiser_review_queue,
    resolve_denoiser_review_event,
)

internal_router = APIRouter(tags=["internal"])


class DenoiserReviewResolveRequest(BaseModel):
    resolved_by: str = "system"
    resolved_notes: str | None = None


@internal_router.get("/health")
async def healthcheck() -> dict:
    """Simple health endpoint used for local dev and readiness checks."""
    return {"status": "ok"}


@internal_router.get("/health/data-freshness")
async def data_freshness_healthcheck() -> dict:
    """Return user-facing source freshness and stale-data policy."""
    try:
        return build_data_status_snapshot()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "as_of": None,
            "overall_state": "degraded",
            "stale_sources": [],
            "critical_stale_sources": [],
            "forecast_inputs_ready": True,
            "forecast_gate": {
                "can_run": True,
                "would_block_if_fail_closed": False,
                "policy": "on_demand",
                "reasons": [],
                "missing_or_stale_sources": [],
                "as_of": None,
                "retry_hint": None,
            },
            "stale_behavior": {
                "mode": "informational",
                "policy": "show_last_fetched_only",
                "fires_api": "returns latest detections with source timestamps",
                "forecast_api": "ingests missing weather/terrain on forecast request",
                "ui": "show_last_fetched_timestamps",
                "critical_sources": ["firms", "weather"],
            },
            "sources": {},
            "error": str(exc),
        }


@internal_router.get("/internal/health/data-freshness")
async def data_freshness_healthcheck_internal() -> dict:
    """Return internal freshness snapshot including idempotency diagnostics."""
    try:
        return build_data_status_snapshot(include_internal=True)
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


@internal_router.get("/internal/denoiser/gates/latest")
async def denoiser_latest_gate_report() -> dict:
    """Return latest stored denoiser gate report."""
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        gate = get_latest_denoiser_gate_report()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"as_of": as_of, "gate": None, "error": str(exc)}
    return {"as_of": as_of, "gate": gate}


@internal_router.get("/internal/denoiser/coverage/latest")
async def denoiser_latest_coverage_status(authority_profile: str = "wfigs_us") -> dict:
    """Return latest authoritative coverage ingest status."""
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        payload = get_latest_denoiser_coverage_status(authority_profile=authority_profile)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"as_of": as_of, "coverage": None, "error": str(exc)}
    return {"as_of": as_of, "coverage": payload}


@internal_router.get("/internal/denoiser/drift")
async def denoiser_drift(limit: int = 50) -> dict:
    """Return recent denoiser drift metrics."""
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        rows = list_recent_denoiser_drift(limit=limit)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"as_of": as_of, "rows": [], "error": str(exc)}
    return {"as_of": as_of, "rows": rows}


@internal_router.get("/internal/denoiser/review-queue")
async def denoiser_review_queue(limit: int = 200, status: str = "open") -> dict:
    """Return denoiser review queue items."""
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        rows = list_denoiser_review_queue(limit=limit, status=status)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"as_of": as_of, "rows": [], "error": str(exc)}
    return {"as_of": as_of, "rows": rows}


@internal_router.post("/internal/denoiser/review-queue/{event_id}/resolve")
async def denoiser_review_queue_resolve(event_id: str, request: DenoiserReviewResolveRequest) -> dict:
    """Resolve all open review rows for an event id."""
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        updated = resolve_denoiser_review_event(
            event_id=event_id,
            resolved_by=request.resolved_by,
            resolved_notes=request.resolved_notes,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"as_of": as_of, "event_id": event_id, "updated": 0, "error": str(exc)}
    return {"as_of": as_of, "event_id": event_id, "updated": updated}
