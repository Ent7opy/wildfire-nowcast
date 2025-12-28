"""Database helpers for spread forecast products."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from sqlalchemy import JSON, bindparam, text

from ingest.repository import get_engine

BBox = tuple[float, float, float, float]


def create_spread_forecast_run(
    *,
    region_name: str,
    model_name: str,
    model_version: str,
    forecast_reference_time: datetime,
    bbox: BBox,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Insert a new spread forecast run and return its ID."""
    stmt = text(
        """
        INSERT INTO spread_forecast_runs (
            region_name,
            model_name,
            model_version,
            forecast_reference_time,
            bbox,
            status,
            metadata
        )
        VALUES (
            :region_name,
            :model_name,
            :model_version,
            :forecast_reference_time,
            ST_SetSRID(ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat), 4326),
            'running',
            :metadata
        )
        RETURNING id
        """
    ).bindparams(bindparam("metadata", type_=JSON))

    min_lon, min_lat, max_lon, max_lat = bbox
    with get_engine().begin() as conn:
        result = conn.execute(
            stmt,
            {
                "region_name": region_name,
                "model_name": model_name,
                "model_version": model_version,
                "forecast_reference_time": forecast_reference_time,
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "metadata": metadata or {},
            },
        )
        run_id = result.scalar_one()

    return int(run_id)


def finalize_spread_forecast_run(
    run_id: int,
    *,
    status: str,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Update spread forecast run status and metadata."""
    stmt = text(
        """
        UPDATE spread_forecast_runs
        SET
            status = :status,
            metadata = COALESCE(metadata, '{}'::jsonb) || CAST(:extra_metadata AS jsonb)
        WHERE id = :run_id
        """
    ).bindparams(bindparam("extra_metadata", type_=JSON))

    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": run_id,
                "status": status,
                "extra_metadata": extra_metadata or {},
            },
        )


def insert_spread_forecast_rasters(
    run_id: int,
    rasters: list[dict[str, Any]],
) -> None:
    """Bulk insert spread forecast raster asset records."""
    if not rasters:
        return

    stmt = text(
        """
        INSERT INTO spread_forecast_rasters (
            run_id,
            horizon_hours,
            file_format,
            storage_path
        )
        VALUES (
            :run_id,
            :horizon_hours,
            :file_format,
            :storage_path
        )
        """
    )

    params = []
    for r in rasters:
        params.append({
            "run_id": run_id,
            "horizon_hours": r["horizon_hours"],
            "file_format": r["file_format"],
            "storage_path": r["storage_path"],
        })

    with get_engine().begin() as conn:
        conn.execute(stmt, params)


def insert_spread_forecast_contours(
    run_id: int,
    contours: list[dict[str, Any]],
) -> None:
    """Bulk insert spread forecast contour geometries."""
    if not contours:
        return

    stmt = text(
        """
        INSERT INTO spread_forecast_contours (
            run_id,
            horizon_hours,
            threshold,
            geom
        )
        VALUES (
            :run_id,
            :horizon_hours,
            :threshold,
            ST_SetSRID(ST_GeomFromGeoJSON(:geom_geojson), 4326)
        )
        """
    )

    params = []
    for c in contours:
        params.append({
            "run_id": run_id,
            "horizon_hours": c["horizon_hours"],
            "threshold": c["threshold"],
            "geom_geojson": c["geom_geojson"],
        })

    with get_engine().begin() as conn:
        conn.execute(stmt, params)

