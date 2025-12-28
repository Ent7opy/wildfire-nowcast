"""DB queries for spread forecasts."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from api.db import get_engine

BBox = tuple[float, float, float, float]


def get_latest_forecast_run(
    region_name: str,
    bbox: BBox,
) -> dict[str, Any] | None:
    """Find the latest completed forecast run intersecting the AOI."""
    min_lon, min_lat, max_lon, max_lat = bbox

    stmt = text(
        """
        SELECT
            id,
            model_name,
            model_version,
            forecast_reference_time,
            region_name,
            status,
            metadata,
            ST_AsGeoJSON(bbox) AS bbox_geojson
        FROM spread_forecast_runs
        WHERE region_name = :region_name
          AND status = 'completed'
          AND bbox && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
          AND ST_Intersects(bbox, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))
        ORDER BY forecast_reference_time DESC, created_at DESC
        LIMIT 1
        """
    )

    with get_engine().begin() as conn:
        row = conn.execute(
            stmt,
            {
                "region_name": region_name,
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            },
        ).mappings().first()

    return dict(row) if row else None


def list_rasters_for_run(run_id: int) -> list[dict[str, Any]]:
    """List all raster assets for a forecast run."""
    stmt = text(
        """
        SELECT horizon_hours, file_format, storage_path
        FROM spread_forecast_rasters
        WHERE run_id = :run_id
        ORDER BY horizon_hours ASC
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, {"run_id": run_id}).mappings().all()
    return [dict(r) for r in rows]


def list_contours_for_run(run_id: int) -> list[dict[str, Any]]:
    """List all threshold contours for a forecast run."""
    stmt = text(
        """
        SELECT horizon_hours, threshold, ST_AsGeoJSON(geom) AS geom_geojson
        FROM spread_forecast_contours
        WHERE run_id = :run_id
        ORDER BY horizon_hours ASC, threshold ASC
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, {"run_id": run_id}).mappings().all()
    return [dict(r) for r in rows]

