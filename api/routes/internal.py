from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from api.config import settings
from api.data_status import build_data_status_snapshot
from api.db_health import build_db_size_snapshot, read_orchestrator_dashboard
from api.model_registry import list_active_models
from api.terrain.features_repo import list_terrain_coverage_inventory
from api.fires.repo import (
    get_latest_denoiser_gate_report,
    get_latest_denoiser_coverage_status,
    get_latest_denoiser_industrial_coverage_status,
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


@internal_router.get("/internal/health/terrain-coverage")
async def terrain_coverage_inventory() -> dict:
    """Return DEM coverage inventory: regions with preprocessed terrain, bboxes, resolution, staleness.

    Operators use this to determine where forecasts will use real terrain vs the D1 flat-terrain fallback.
    Staleness is computed against DATA_STALE_TERRAIN_MINUTES (default 10080 = 7 days).
    """
    as_of = datetime.now(timezone.utc)
    try:
        rows = list_terrain_coverage_inventory()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {"as_of": as_of.isoformat(), "regions": [], "error": str(exc)}

    regions = []
    for row in rows:
        created_at_utc = row.created_at.replace(tzinfo=timezone.utc)
        age_minutes = (as_of - created_at_utc).total_seconds() / 60.0
        is_stale = age_minutes > settings.data_stale_terrain_minutes
        regions.append(
            {
                "region_name": row.region_name,
                "bbox": {
                    "min_lon": row.bbox[0],
                    "min_lat": row.bbox[1],
                    "max_lon": row.bbox[2],
                    "max_lat": row.bbox[3],
                },
                "resolution_deg": row.cell_size_deg,
                "crs_epsg": row.crs_epsg,
                "grid": {"n_lat": row.grid_n_lat, "n_lon": row.grid_n_lon},
                "terrain_fallback_used": row.terrain_fallback_used,
                "coverage_fraction": row.coverage_fraction,
                "preprocessed_at": created_at_utc.isoformat(),
                "age_minutes": round(age_minutes, 1),
                "is_stale": is_stale,
                "slope_path": row.slope_path,
                "aspect_path": row.aspect_path,
            }
        )

    return {
        "as_of": as_of.isoformat(),
        "stale_threshold_minutes": settings.data_stale_terrain_minutes,
        "region_count": len(regions),
        "regions": regions,
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


@internal_router.get("/internal/denoiser/industrial-coverage/latest")
async def denoiser_latest_industrial_coverage_status(
    source_profile: str | None = None,
    policy_version: str | None = None,
) -> dict:
    """Return latest authoritative industrial ingest and policy coverage status."""
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        payload = get_latest_denoiser_industrial_coverage_status(
            source_profile=source_profile,
            policy_version=policy_version,
        )
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


@internal_router.get("/internal/health/db-size")
async def db_size_health() -> dict:
    """Return per-table row counts, total DB size, and retention/cleanup policy status.

    Row counts are approximate (pg_stat_user_tables.n_live_tup, refreshed by
    autovacuum) — suitable for capacity alerting without a sequential scan.
    Cleanup timing is derived from the orchestrator dashboard JSON so callers
    see the same last-run/next-run values the orchestrator reports.

    All numeric leaf fields (size_bytes, row_count, interval_minutes, *_days)
    are raw numbers so monitors can compare thresholds without string parsing.
    """
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        return build_db_size_snapshot()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "as_of": as_of,
            "database": {"size_bytes": None, "size_pretty": None},
            "tables": {},
            "retention_policy": {
                "default_retention_days": None,
                "archive_retention_days": None,
            },
            "cleanup": {
                "last_run_at": None,
                "last_outcome": None,
                "next_run_at": None,
                "interval_minutes": None,
                "source": "error",
            },
            "error": str(exc),
        }


@internal_router.get("/internal/health/dashboard")
async def ingest_health_dashboard() -> dict:
    """Return consolidated operational snapshot: per-job metrics, data freshness, DB size, and cleanup status.

    Assembles in a single response what would otherwise require cross-referencing:
    - ``GET /internal/health/data-freshness`` (source freshness)
    - ``GET /internal/health/db-size`` (DB size + cleanup timing)
    - The raw orchestrator dashboard JSON (per-job success/failure timestamps)

    All section helpers are reused directly — no duplicate business logic.

    Schema is additive; individual sub-endpoints remain available for targeted queries.
    """
    as_of = datetime.now(timezone.utc).isoformat()

    # --- per-job metrics from orchestrator dashboard file ---
    raw = read_orchestrator_dashboard()
    jobs_section = raw.get("metrics") if raw else None
    orchestrator_generated_at = raw.get("generated_at") if raw else None

    # --- data freshness (reuse shared helper; include internal idempotency fields) ---
    try:
        freshness_snapshot = build_data_status_snapshot(include_internal=True)
        freshness_section = {
            "overall_state": freshness_snapshot.get("overall_state"),
            "forecast_inputs_ready": freshness_snapshot.get("forecast_inputs_ready"),
            "sources": freshness_snapshot.get("sources", {}),
        }
    except Exception as exc:  # pragma: no cover - defensive fallback
        freshness_section = {"overall_state": "unknown", "forecast_inputs_ready": False, "sources": {}, "error": str(exc)}

    # --- DB size + cleanup (reuse shared helper; pass pre-read dashboard to avoid a second file read) ---
    try:
        db_snapshot = build_db_size_snapshot(dashboard=raw)
        db_section = {
            "database": db_snapshot.get("database"),
            "tables": db_snapshot.get("tables"),
            "retention_policy": db_snapshot.get("retention_policy"),
        }
        cleanup_section = db_snapshot.get("cleanup")
    except Exception as exc:  # pragma: no cover - defensive fallback
        db_section = {"database": {"size_bytes": None, "size_pretty": None}, "tables": {}, "retention_policy": {}, "error": str(exc)}
        cleanup_section = {"last_run_at": None, "last_outcome": None, "next_run_at": None, "interval_minutes": None, "source": "error"}

    return {
        "as_of": as_of,
        "orchestrator": {
            "generated_at": orchestrator_generated_at,
            "jobs": jobs_section,
        },
        "data_freshness": freshness_section,
        "db_size": db_section,
        "cleanup": cleanup_section,
    }


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
