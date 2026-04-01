"""Database helpers for weather ingestion."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence

from sqlalchemy import JSON, bindparam, text

from ingest.repository import get_engine

LOGGER = logging.getLogger(__name__)


def create_weather_run_record(
    *,
    model: str,
    run_time: datetime,
    horizon_hours: int,
    step_hours: int,
    bbox: tuple[float | None, float | None, float | None, float | None],
    variables: list[str],
) -> int:
    """Insert a new weather run row and return its ID."""
    metadata: Mapping[str, Any] = {"variables": variables}
    stmt = text(
        """
        INSERT INTO weather_runs (
            model,
            run_time,
            horizon_hours,
            step_hours,
            bbox_min_lon,
            bbox_min_lat,
            bbox_max_lon,
            bbox_max_lat,
            file_format,
            storage_path,
            status,
            metadata
        )
        VALUES (
            :model,
            :run_time,
            :horizon_hours,
            :step_hours,
            :bbox_min_lon,
            :bbox_min_lat,
            :bbox_max_lon,
            :bbox_max_lat,
            'netcdf',
            '',
            'running',
            :metadata
        )
        RETURNING id
        """
    ).bindparams(bindparam("metadata", type_=JSON))

    with get_engine().begin() as conn:
        result = conn.execute(
            stmt,
            {
                "model": model,
                "run_time": run_time,
                "horizon_hours": horizon_hours,
                "step_hours": step_hours,
                "bbox_min_lon": bbox[0],
                "bbox_min_lat": bbox[1],
                "bbox_max_lon": bbox[2],
                "bbox_max_lat": bbox[3],
                "metadata": metadata,
            },
        )
        run_id = result.scalar_one()

    return int(run_id)


def finalize_weather_run_record(
    *,
    run_id: int,
    storage_path: str,
    status: str,
    run_time: datetime | None = None,
    extra_metadata: Dict[str, Any] | None = None,
) -> None:
    """Update weather run status and storage details."""
    extra_metadata = extra_metadata or {}
    stmt = (
        text(
            """
            UPDATE weather_runs
            SET
                storage_path = :storage_path,
                status = :status,
                run_time = COALESCE(:run_time, run_time),
                metadata = COALESCE(metadata, '{}'::jsonb) || CAST(:extra_metadata AS jsonb)
            WHERE id = :run_id
            """
        ).bindparams(bindparam("extra_metadata", type_=JSON))
    )

    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": run_id,
                "storage_path": storage_path,
                "status": status,
                "run_time": run_time,
                "extra_metadata": extra_metadata,
            },
        )


# ---------------------------------------------------------------------------
# Point-cache helpers
# ---------------------------------------------------------------------------

#: GFS native resolution in degrees.
GFS_GRID_DEG = 0.25


def snap_to_gfs_grid(lat: float, lon: float) -> tuple[float, float]:
    """Snap a lat/lon pair to the nearest GFS 0.25° grid point."""
    return (
        round(round(lat / GFS_GRID_DEG) * GFS_GRID_DEG, 4),
        round(round(lon / GFS_GRID_DEG) * GFS_GRID_DEG, 4),
    )


def query_fire_detection_grid_points(
    lookback_hours: float = 48.0,
) -> list[tuple[float, float]]:
    """Return deduplicated GFS-snapped grid points near recent fire detections.

    Queries ``fire_detections`` for the last *lookback_hours*, snaps each
    detection to the nearest GFS 0.25° grid point, and returns unique pairs.
    """
    stmt = text(
        """
        SELECT DISTINCT
            round((lat  / :grid)::numeric) * :grid AS lat_grid,
            round((lon  / :grid)::numeric) * :grid AS lon_grid
        FROM fire_detections
        WHERE acq_time >= NOW() - INTERVAL '1 hour' * :lookback
        """
    )
    with get_engine().connect() as conn:
        rows = conn.execute(
            stmt, {"grid": GFS_GRID_DEG, "lookback": lookback_hours}
        ).fetchall()

    return [(float(r[0]), float(r[1])) for r in rows]


def bulk_insert_weather_point_cache(
    run_id: int,
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int = 5000,
) -> int:
    """Bulk-insert rows into ``weather_point_cache``.

    Each record must contain keys: ``forecast_hour``, ``lat_grid``,
    ``lon_grid``, and optionally ``u10``, ``v10``, ``t2m``, ``rh2m``, ``tp``.

    Returns the total number of rows inserted.
    """
    if not records:
        return 0

    stmt = text(
        """
        INSERT INTO weather_point_cache
            (run_id, forecast_hour, lat_grid, lon_grid, u10, v10, t2m, rh2m, tp)
        VALUES
            (:run_id, :forecast_hour, :lat_grid, :lon_grid,
             :u10, :v10, :t2m, :rh2m, :tp)
        """
    )

    total = 0
    with get_engine().begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            params = [
                {
                    "run_id": run_id,
                    "forecast_hour": r["forecast_hour"],
                    "lat_grid": r["lat_grid"],
                    "lon_grid": r["lon_grid"],
                    "u10": r.get("u10"),
                    "v10": r.get("v10"),
                    "t2m": r.get("t2m"),
                    "rh2m": r.get("rh2m"),
                    "tp": r.get("tp"),
                }
                for r in batch
            ]
            conn.execute(stmt, params)
            total += len(batch)

    LOGGER.info(
        "Inserted %d weather_point_cache rows for run_id=%d", total, run_id
    )
    return total

