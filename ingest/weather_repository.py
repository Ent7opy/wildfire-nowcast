"""Database helpers for weather ingestion."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping

from sqlalchemy import JSON, bindparam, text

from ingest.repository import get_engine


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

