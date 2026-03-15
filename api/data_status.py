"""Data freshness and idempotency status helpers for API and ingest orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine, text

from api.config import settings
from api.db import get_engine


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _age_minutes(last_seen_at: datetime | None, now: datetime) -> float | None:
    if last_seen_at is None:
        return None
    delta = (now - last_seen_at).total_seconds() / 60.0
    return max(0.0, float(delta))


def _source_status(*, name: str, last_seen_at: datetime | None, threshold_minutes: int, now: datetime) -> dict[str, Any]:
    age_minutes = _age_minutes(last_seen_at, now)
    if age_minutes is None:
        state = "missing"
    elif age_minutes > float(threshold_minutes):
        state = "stale"
    else:
        state = "fresh"

    return {
        "source": name,
        "state": state,
        "last_seen_at": last_seen_at.isoformat() if last_seen_at else None,
        "age_minutes": round(age_minutes, 2) if age_minutes is not None else None,
        "stale_threshold_minutes": int(threshold_minutes),
        "is_stale": state in {"stale", "missing"},
    }


def _build_forecast_gate(
    *,
    sources: dict[str, Any],
    as_of: str | None,
) -> dict[str, Any]:
    """Build the forecast fail-closed gate decision from freshness source states."""
    reasons: list[str] = []
    missing_or_stale_sources: list[str] = []

    weather_state = str((sources.get("weather") or {}).get("state", "missing")).lower()
    terrain_state = str((sources.get("terrain") or {}).get("state", "missing")).lower()

    if weather_state != "fresh":
        reasons.append("weather_stale_or_missing")
        missing_or_stale_sources.append("weather")

    # Terrain is quasi-static: require presence, not strict freshness.
    if terrain_state == "missing":
        reasons.append("terrain_missing")
        missing_or_stale_sources.append("terrain")

    can_run_without_policy_override = len(reasons) == 0
    fail_closed = bool(settings.forecast_fail_closed_on_stale)
    can_run = can_run_without_policy_override if fail_closed else True

    retry_hints: list[str] = []
    if "weather" in missing_or_stale_sources:
        retry_hints.append("wait for weather refresh or trigger weather prewarm")
    if "terrain" in missing_or_stale_sources:
        retry_hints.append("trigger terrain prewarm for the selected coordinates")

    return {
        "can_run": can_run,
        "would_block_if_fail_closed": not can_run_without_policy_override,
        "policy": "fail_closed" if fail_closed else "best_effort",
        "reasons": reasons,
        "missing_or_stale_sources": missing_or_stale_sources,
        "as_of": as_of,
        "retry_hint": "; ".join(retry_hints) if retry_hints else None,
    }


def resolve_forecast_gate(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Resolve forecast gate payload from snapshot, rebuilding when absent."""
    gate = snapshot.get("forecast_gate")
    if isinstance(gate, dict):
        return gate
    return _build_forecast_gate(
        sources=snapshot.get("sources", {}),
        as_of=snapshot.get("as_of"),
    )


def _fetch_latest_firms_status(conn) -> dict[str, Any]:
    row = conn.execute(
        text(
            """
            SELECT
                id,
                source,
                completed_at,
                records_fetched,
                records_inserted,
                records_skipped_duplicates
            FROM ingest_batches
            WHERE status = 'succeeded'
            ORDER BY completed_at DESC NULLS LAST, id DESC
            LIMIT 1
            """
        )
    ).mappings().first()

    watermark_row = None
    try:
        watermark_row = conn.execute(
            text(
                """
                SELECT
                    MAX(last_acq_time_utc) AS latest_acq_time_utc,
                    MAX(updated_at) AS latest_watermark_updated_at
                FROM ingest_watermarks
                """
            )
        ).mappings().first()
    except Exception:
        watermark_row = None

    watermark_last_acq = _as_utc((watermark_row or {}).get("latest_acq_time_utc"))
    watermark_updated_at = _as_utc((watermark_row or {}).get("latest_watermark_updated_at"))

    if row is None:
        return {
            "last_seen_at": watermark_last_acq,
            "idempotency": {
                "latest_batch_id": None,
                "records_fetched": 0,
                "records_inserted": 0,
                "records_skipped_duplicates": 0,
                "duplicate_ratio": None,
                "latest_watermark_acq_time": watermark_last_acq.isoformat() if watermark_last_acq else None,
                "latest_watermark_updated_at": watermark_updated_at.isoformat() if watermark_updated_at else None,
            },
        }

    fetched = int(row.get("records_fetched") or 0)
    skipped = int(row.get("records_skipped_duplicates") or 0)
    duplicate_ratio = (float(skipped) / float(fetched)) if fetched > 0 else None

    return {
        "last_seen_at": watermark_last_acq or _as_utc(row.get("completed_at")),
        "idempotency": {
            "latest_batch_id": row.get("id"),
            "latest_source": row.get("source"),
            "records_fetched": fetched,
            "records_inserted": int(row.get("records_inserted") or 0),
            "records_skipped_duplicates": skipped,
            "duplicate_ratio": round(duplicate_ratio, 4) if duplicate_ratio is not None else None,
            "latest_watermark_acq_time": watermark_last_acq.isoformat() if watermark_last_acq else None,
            "latest_watermark_updated_at": watermark_updated_at.isoformat() if watermark_updated_at else None,
        },
    }


def _fetch_latest_weather_status(conn) -> dict[str, Any]:
    latest = conn.execute(
        text(
            """
            SELECT
                id,
                model,
                run_time,
                created_at,
                horizon_hours,
                step_hours
            FROM weather_runs
            WHERE status = 'completed'
            ORDER BY run_time DESC, id DESC
            LIMIT 1
            """
        )
    ).mappings().first()

    recent = conn.execute(
        text(
            """
            SELECT
                COUNT(*) AS total_runs,
                COUNT(DISTINCT run_time) AS unique_run_times
            FROM weather_runs
            WHERE status = 'completed'
              AND created_at >= NOW() - INTERVAL '24 hours'
            """
        )
    ).mappings().first()

    total_runs_24h = int((recent or {}).get("total_runs") or 0)
    unique_runs_24h = int((recent or {}).get("unique_run_times") or 0)
    duplicate_ratio_24h = (
        float(total_runs_24h - unique_runs_24h) / float(total_runs_24h)
        if total_runs_24h > 0
        else None
    )

    return {
        "last_seen_at": _as_utc((latest or {}).get("run_time")),
        "idempotency": {
            "latest_run_id": (latest or {}).get("id"),
            "latest_model": (latest or {}).get("model"),
            "latest_run_time": _as_utc((latest or {}).get("run_time")).isoformat()
            if (latest or {}).get("run_time")
            else None,
            "horizon_hours": (latest or {}).get("horizon_hours"),
            "step_hours": (latest or {}).get("step_hours"),
            "completed_runs_last_24h": total_runs_24h,
            "unique_run_times_last_24h": unique_runs_24h,
            "duplicate_run_ratio_last_24h": round(duplicate_ratio_24h, 4)
            if duplicate_ratio_24h is not None
            else None,
        },
    }


def _fetch_latest_terrain_status(conn) -> dict[str, Any]:
    latest = conn.execute(
        text(
            """
            SELECT id, region_name, created_at, source_dem_metadata_id
            FROM terrain_features_metadata
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
        )
    ).mappings().first()

    counts = conn.execute(
        text(
            """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT source_dem_metadata_id) AS distinct_source_dem_rows
            FROM terrain_features_metadata
            """
        )
    ).mappings().first()

    return {
        "last_seen_at": _as_utc((latest or {}).get("created_at")),
        "idempotency": {
            "latest_features_id": (latest or {}).get("id"),
            "latest_region_name": (latest or {}).get("region_name"),
            "latest_source_dem_metadata_id": (latest or {}).get("source_dem_metadata_id"),
            "total_rows": int((counts or {}).get("total_rows") or 0),
            "distinct_source_dem_rows": int((counts or {}).get("distinct_source_dem_rows") or 0),
        },
    }


def _fetch_latest_perimeters_status(conn) -> dict[str, Any]:
    stats = conn.execute(
        text(
            """
            SELECT
                MAX(created_at) AS latest_created_at,
                COUNT(*) AS total_rows,
                COUNT(DISTINCT source || ':' || COALESCE(source_id, '')) AS unique_rows
            FROM fire_perimeters
            """
        )
    ).mappings().first()

    total_rows = int((stats or {}).get("total_rows") or 0)
    unique_rows = int((stats or {}).get("unique_rows") or 0)
    conflict_ratio = (
        float(total_rows - unique_rows) / float(total_rows)
        if total_rows > 0
        else None
    )

    return {
        "last_seen_at": _as_utc((stats or {}).get("latest_created_at")),
        "idempotency": {
            "total_rows": total_rows,
            "unique_rows": unique_rows,
            "duplicate_key_ratio": round(conflict_ratio, 4) if conflict_ratio is not None else None,
        },
    }


def build_data_status_snapshot(
    *,
    now: datetime | None = None,
    engine: Engine | None = None,
    include_internal: bool = False,
) -> dict[str, Any]:
    """Build a snapshot of data freshness and idempotency metrics."""
    now_utc = _as_utc(now) or _utc_now()
    db_engine = engine or get_engine()

    with db_engine.begin() as conn:
        firms = _fetch_latest_firms_status(conn)
        weather = _fetch_latest_weather_status(conn)
        terrain = _fetch_latest_terrain_status(conn)
        perimeters = _fetch_latest_perimeters_status(conn)

    sources = {
        "firms": _source_status(
            name="firms",
            last_seen_at=firms["last_seen_at"],
            threshold_minutes=settings.data_stale_firms_minutes,
            now=now_utc,
        ),
        "weather": _source_status(
            name="weather",
            last_seen_at=weather["last_seen_at"],
            threshold_minutes=settings.data_stale_weather_minutes,
            now=now_utc,
        ),
        "terrain": _source_status(
            name="terrain",
            last_seen_at=terrain["last_seen_at"],
            threshold_minutes=settings.data_stale_terrain_minutes,
            now=now_utc,
        ),
        "perimeters": _source_status(
            name="perimeters",
            last_seen_at=perimeters["last_seen_at"],
            threshold_minutes=settings.data_stale_perimeters_minutes,
            now=now_utc,
        ),
    }

    critical_sources = settings.data_status_critical_sources_set
    critical_issues = [
        name
        for name in critical_sources
        if name in sources and sources[name]["state"] in {"stale", "missing"}
    ]
    all_issues = [
        name
        for name, details in sources.items()
        if details["state"] in {"stale", "missing"}
    ]

    if include_internal:
        if critical_issues:
            overall_state = "critical"
        elif all_issues:
            overall_state = "degraded"
        else:
            overall_state = "healthy"

        forecast_gate = _build_forecast_gate(sources=sources, as_of=now_utc.isoformat())
        forecast_inputs_ready = bool(forecast_gate["can_run"])
        stale_behavior = {
            "mode": "normal" if overall_state == "healthy" else "degraded",
            "policy": "serve_last_known_data_with_warning",
            "fires_api": "returns cached/latest detections and includes freshness status endpoint",
            "forecast_api": (
                "allow_forecast_generation"
                if forecast_gate["can_run"]
                else "deny_new_forecasts_until_weather_fresh_and_terrain_present"
            ),
            "ui": "show_stale_data_banner_when_state_not_healthy",
            "critical_sources": sorted(critical_sources),
            "forecast_retry_hint": forecast_gate.get("retry_hint"),
        }
    else:
        # Public/user-facing contract: informational only (last-fetched status), no stale signaling.
        missing_sources = [
            name
            for name, details in sources.items()
            if details["state"] == "missing"
        ]
        overall_state = "degraded" if missing_sources else "healthy"
        forecast_inputs_ready = True
        forecast_gate = {
            "can_run": True,
            "would_block_if_fail_closed": False,
            "policy": "on_demand",
            "reasons": [],
            "missing_or_stale_sources": [],
            "as_of": now_utc.isoformat(),
            "retry_hint": None,
        }
        stale_behavior = {
            "mode": "informational",
            "policy": "show_last_fetched_only",
            "fires_api": "returns latest detections with source timestamps",
            "forecast_api": "ingests missing weather/terrain on forecast request",
            "ui": "show_last_fetched_timestamps",
            "critical_sources": sorted(critical_sources),
            "forecast_retry_hint": None,
        }

    public_stale_sources = all_issues if include_internal else []
    public_critical_stale_sources = critical_issues if include_internal else []

    snapshot = {
        "as_of": now_utc.isoformat(),
        "overall_state": overall_state,
        "stale_sources": public_stale_sources,
        "critical_stale_sources": public_critical_stale_sources,
        "forecast_inputs_ready": forecast_inputs_ready,
        "forecast_gate": forecast_gate,
        "stale_behavior": stale_behavior,
        "sources": sources,
    }
    if include_internal:
        snapshot["idempotency_dashboard"] = {
            "firms": firms["idempotency"],
            "weather": weather["idempotency"],
            "terrain": terrain["idempotency"],
            "perimeters": perimeters["idempotency"],
        }
    return snapshot
