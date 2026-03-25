"""DB size and retention health helpers for the /internal/health/db-size endpoint.

Querying strategy
-----------------
Row counts come from ``pg_stat_user_tables.n_live_tup``, which is an approximate
count maintained by autovacuum — no sequential scan, sub-millisecond.  Operators
who need exact counts can VACUUM ANALYZE first, but the approximation is adequate
for capacity alerting.

Cleanup timing
--------------
The orchestrator writes ``data/ingest/orchestrator_dashboard.json`` after every
job cycle.  We read ``metrics.cleanup.last_finished_at`` from that file to derive
both the last-run timestamp and the next scheduled time (last_run + interval).
If the file does not exist yet (first deploy, tests) we return explicit nulls
with a ``source`` field explaining why.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from api.config import REPO_ROOT
from api.db import get_engine

# ---------------------------------------------------------------------------
# Tables monitored for row counts.  Only these names are interpolated into SQL
# (via ANY(:tables) — not string concatenation — so injection is not possible).
# Ordering mirrors CASCADE priority in db_cleanup.py.
# ---------------------------------------------------------------------------
MONITORED_TABLES: list[str] = [
    "fire_detections",
    "denoiser_labels_v2",
    "fire_event_memberships",
    "fire_events",
    "fire_fronts",
    "denoiser_drift_metrics",
    "weather_runs",
    "spread_forecast_runs",
    "ingest_batches",
    "export_jobs",
    "reverse_geocode_cache",
]

# Per-table retention reference — mirrors TABLE_CONFIG in scripts/db_cleanup.py.
# fire_detections uses two-tier retention so its value is a nested dict.
_DEFAULT_RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "14"))
_DEFAULT_ARCHIVE_RETENTION_DAYS = int(os.environ.get("ARCHIVE_RETENTION_DAYS", "3"))

TABLE_RETENTION: dict[str, Any] = {
    "fire_detections": {
        "archive_days": _DEFAULT_ARCHIVE_RETENTION_DAYS,
        "nrt_days": _DEFAULT_RETENTION_DAYS,
    },
    "denoiser_labels_v2":     _DEFAULT_RETENTION_DAYS,
    "fire_event_memberships": _DEFAULT_RETENTION_DAYS,
    "fire_events":            _DEFAULT_RETENTION_DAYS,
    "fire_fronts":            _DEFAULT_RETENTION_DAYS,
    "denoiser_drift_metrics": _DEFAULT_RETENTION_DAYS,
    "weather_runs":           _DEFAULT_RETENTION_DAYS,
    "spread_forecast_runs":   _DEFAULT_RETENTION_DAYS,
    "ingest_batches":         30,
    "export_jobs":            7,
    "reverse_geocode_cache":  0,  # cutoff = now(); rows expire via expires_at
}

_DEFAULT_CLEANUP_INTERVAL_MINUTES = float(
    os.environ.get("CLEANUP_INTERVAL_MINUTES", "1440")
)

DEFAULT_DASHBOARD_PATH = REPO_ROOT / "data" / "ingest" / "orchestrator_dashboard.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_dashboard(dashboard_path: Path) -> dict | None:
    """Return parsed orchestrator dashboard JSON, or None on any error."""
    try:
        if dashboard_path.exists():
            return json.loads(dashboard_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return None


def _get_cleanup_status(dashboard_path: Path) -> dict:
    """Derive last-run timestamp and next-scheduled estimate from dashboard file.

    Returns a dict with numeric/string fields so monitors can scrape them directly.
    ``source`` explains the data provenance; consumers should check it when
    ``last_run_at`` is None.
    """
    interval_minutes = _DEFAULT_CLEANUP_INTERVAL_MINUTES
    dashboard = _read_dashboard(dashboard_path)

    if dashboard is None:
        return {
            "last_run_at": None,
            "last_outcome": None,
            "next_run_at": None,
            "interval_minutes": interval_minutes,
            "source": "dashboard_unavailable",
            "source_detail": str(dashboard_path),
        }

    cleanup_metrics: dict = (dashboard.get("metrics") or {}).get("cleanup") or {}
    last_finished_at: str | None = cleanup_metrics.get("last_finished_at")
    last_outcome: str | None = cleanup_metrics.get("last_outcome")

    next_run_at: str | None = None
    if last_finished_at:
        try:
            last_dt = datetime.fromisoformat(last_finished_at)
            next_run_at = (last_dt + timedelta(minutes=interval_minutes)).isoformat()
        except ValueError:
            pass  # malformed timestamp — leave next_run_at as None

    return {
        "last_run_at": last_finished_at,
        "last_outcome": last_outcome,
        "next_run_at": next_run_at,
        "interval_minutes": interval_minutes,
        "source": "orchestrator_dashboard",
    }


def _query_db_sizes() -> dict:
    """Run the DB size SQL queries.

    Returns:
        dict with keys ``database`` (size_bytes, size_pretty) and ``tables``
        (per-table row_count + retention reference).

    Raises:
        sqlalchemy.exc.* on DB connectivity failure — callers must handle.
    """
    table_stats: dict[str, dict] = {}
    db_size_bytes: int | None = None
    db_size_pretty: str | None = None

    with get_engine().connect() as conn:
        # Approximate row counts from autovacuum statistics — no seq scan.
        rows = conn.execute(
            text(
                "SELECT relname, n_live_tup "
                "FROM pg_stat_user_tables "
                "WHERE relname = ANY(:tables)"
            ),
            {"tables": MONITORED_TABLES},
        ).fetchall()

        for relname, n_live_tup in rows:
            table_stats[relname] = {
                "row_count": int(n_live_tup),
                "retention": TABLE_RETENTION.get(relname),
            }

        # Fill zeros for tables not yet seen in pg_stat (empty / never vacuumed).
        for table in MONITORED_TABLES:
            if table not in table_stats:
                table_stats[table] = {
                    "row_count": 0,
                    "retention": TABLE_RETENTION.get(table),
                }

        # Total cluster-level DB size.
        result = conn.execute(
            text(
                "SELECT pg_database_size(current_database()), "
                "       pg_size_pretty(pg_database_size(current_database()))"
            )
        ).fetchone()
        if result:
            db_size_bytes = int(result[0])
            db_size_pretty = result[1]

    return {
        "database": {
            "size_bytes": db_size_bytes,
            "size_pretty": db_size_pretty,
        },
        "tables": table_stats,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_db_size_snapshot(
    dashboard_path: Path | None = None,
) -> dict:
    """Return per-table row counts, total DB size, and retention/cleanup status.

    This is the single source of truth consumed by:
    - ``GET /internal/health/db-size`` (API)
    - ``orchestrator_dashboard.json`` (ingest, via ``db_size`` key)

    Args:
        dashboard_path: Path to orchestrator dashboard JSON.  Defaults to
            ``data/ingest/orchestrator_dashboard.json`` relative to repo root.

    Returns:
        Structured dict with ``as_of``, ``database``, ``tables``,
        ``retention_policy``, and ``cleanup`` fields.  All numeric leaf values
        are raw integers/floats so alerting systems can compare thresholds
        without parsing strings.
    """
    resolved_path = dashboard_path or DEFAULT_DASHBOARD_PATH

    as_of = datetime.now(timezone.utc).isoformat()
    db_data = _query_db_sizes()
    cleanup_status = _get_cleanup_status(resolved_path)

    return {
        "as_of": as_of,
        **db_data,
        "retention_policy": {
            "default_retention_days": _DEFAULT_RETENTION_DAYS,
            "archive_retention_days": _DEFAULT_ARCHIVE_RETENTION_DAYS,
        },
        "cleanup": cleanup_status,
    }
