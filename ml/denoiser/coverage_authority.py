"""Helpers for authoritative coverage provenance and freshness checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.db import get_engine


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def get_latest_authoritative_run(
    *,
    authority_profile: str,
    engine: Engine | None = None,
) -> dict[str, Any] | None:
    stmt = text(
        """
        WITH run_candidates AS (
            SELECT
                air.run_id,
                air.source_profile,
                air.source_uri,
                air.source_layer,
                air.status,
                air.started_at,
                air.finished_at,
                air.source_last_edit,
                air.records_fetched,
                air.records_upserted,
                air.records_skipped,
                air.http_429_count,
                air.max_backoff_seconds
            FROM authoritative_perimeter_ingest_runs air
            JOIN perimeter_coverage_masks pcm
              ON pcm.run_id = air.run_id
            WHERE pcm.authority_profile = :authority_profile
              AND pcm.is_active
              AND air.status = 'succeeded'
            UNION
            SELECT
                run_id,
                source_profile,
                source_uri,
                source_layer,
                status,
                started_at,
                finished_at,
                source_last_edit,
                records_fetched,
                records_upserted,
                records_skipped,
                http_429_count,
                max_backoff_seconds
            FROM authoritative_perimeter_ingest_runs
            WHERE source_profile = :authority_profile
              AND status = 'succeeded'
        )
        SELECT
            run_id,
            source_profile,
            source_uri,
            source_layer,
            status,
            started_at,
            finished_at,
            source_last_edit,
            records_fetched,
            records_upserted,
            records_skipped,
            http_429_count,
            max_backoff_seconds
        FROM run_candidates
        ORDER BY finished_at DESC NULLS LAST, started_at DESC
        LIMIT 1
        """
    )
    active_engine = engine or get_engine()
    with active_engine.begin() as conn:
        row = conn.execute(stmt, {"authority_profile": authority_profile}).mappings().first()
    return dict(row) if row else None


def get_active_mask_ids(
    *,
    authority_profile: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    engine: Engine | None = None,
) -> list[str]:
    active_engine = engine or get_engine()
    params: dict[str, Any] = {"authority_profile": authority_profile}
    clauses = [
        "is_active",
        "authority_profile = :authority_profile",
    ]
    if start_time is not None:
        params["start_time"] = _as_utc(start_time)
        clauses.append("(valid_to IS NULL OR valid_to >= :start_time)")
    if end_time is not None:
        params["end_time"] = _as_utc(end_time)
        clauses.append("(valid_from IS NULL OR valid_from <= :end_time)")
    stmt = text(
        f"""
        SELECT mask_id
        FROM perimeter_coverage_masks
        WHERE {' AND '.join(clauses)}
        ORDER BY mask_id
        """
    )
    with active_engine.begin() as conn:
        rows = conn.execute(stmt, params).mappings().all()
    return [str(r["mask_id"]) for r in rows]


def get_coverage_freshness(
    *,
    authority_profile: str,
    max_age_hours: float,
    engine: Engine | None = None,
) -> dict[str, Any]:
    run = get_latest_authoritative_run(authority_profile=authority_profile, engine=engine)
    if run is None:
        return {
            "authority_profile": authority_profile,
            "available": False,
            "fresh": False,
            "age_hours": None,
            "run_id": None,
            "source_last_edit": None,
            "finished_at": None,
        }

    now = datetime.now(timezone.utc)
    finished_at = _as_utc(run.get("finished_at"))
    source_last_edit = _as_utc(run.get("source_last_edit"))
    anchor = source_last_edit or finished_at
    age_hours = None
    fresh = False
    if anchor is not None:
        age_hours = max(0.0, (now - anchor).total_seconds() / 3600.0)
        fresh = age_hours <= float(max_age_hours)

    return {
        "authority_profile": authority_profile,
        "available": True,
        "fresh": bool(fresh),
        "age_hours": age_hours,
        "run_id": run.get("run_id"),
        "source_last_edit": source_last_edit.isoformat() if source_last_edit else None,
        "finished_at": finished_at.isoformat() if finished_at else None,
    }


def require_coverage_freshness(
    *,
    authority_profile: str,
    max_age_hours: float,
    engine: Engine | None = None,
) -> dict[str, Any]:
    status = get_coverage_freshness(
        authority_profile=authority_profile,
        max_age_hours=max_age_hours,
        engine=engine,
    )
    if not status["available"]:
        raise ValueError(
            f"No successful authoritative perimeter ingest run for authority_profile={authority_profile!r}"
        )
    if not status["fresh"]:
        raise ValueError(
            "Authoritative perimeter ingest is stale for "
            f"authority_profile={authority_profile!r}; "
            f"age_hours={status['age_hours']}, max_age_hours={max_age_hours}"
        )
    return status


def default_coverage_window(max_age_hours: float) -> tuple[datetime, datetime]:
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=float(max_age_hours))
    return start_time, end_time
