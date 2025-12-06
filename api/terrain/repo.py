"""Repository helpers for terrain metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from api.db import get_engine


@dataclass(slots=True)
class TerrainMetadataCreate:
    region_name: str
    dem_source: str
    crs_epsg: int
    resolution_m: float
    bbox: tuple[float, float, float, float]
    raster_path: str


@dataclass(slots=True)
class TerrainMetadata(TerrainMetadataCreate):
    id: int
    created_at: datetime


def _row_to_metadata(row: dict) -> TerrainMetadata:
    return TerrainMetadata(
        id=int(row["id"]),
        region_name=row["region_name"],
        dem_source=row["dem_source"],
        crs_epsg=int(row["crs_epsg"]),
        resolution_m=float(row["resolution_m"]),
        bbox=(
            float(row["bbox_min_lon"]),
            float(row["bbox_min_lat"]),
            float(row["bbox_max_lon"]),
            float(row["bbox_max_lat"]),
        ),
        raster_path=row["raster_path"],
        created_at=row["created_at"],
    )


def insert_terrain_metadata(metadata: TerrainMetadataCreate) -> TerrainMetadata:
    """Insert a terrain_metadata row and return the stored record."""
    stmt = text(
        """
        INSERT INTO terrain_metadata (
            region_name,
            dem_source,
            crs_epsg,
            resolution_m,
            bbox,
            raster_path
        )
        VALUES (
            :region_name,
            :dem_source,
            :crs_epsg,
            :resolution_m,
            ST_SetSRID(ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat), 4326),
            :raster_path
        )
        RETURNING
            id,
            region_name,
            dem_source,
            crs_epsg,
            resolution_m,
            raster_path,
            created_at,
            ST_XMin(bbox) AS bbox_min_lon,
            ST_YMin(bbox) AS bbox_min_lat,
            ST_XMax(bbox) AS bbox_max_lon,
            ST_YMax(bbox) AS bbox_max_lat
        """
    )

    bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat = metadata.bbox
    with get_engine().begin() as conn:
        result = conn.execute(
            stmt,
            {
                "region_name": metadata.region_name,
                "dem_source": metadata.dem_source,
                "crs_epsg": metadata.crs_epsg,
                "resolution_m": metadata.resolution_m,
                "min_lon": bbox_min_lon,
                "min_lat": bbox_min_lat,
                "max_lon": bbox_max_lon,
                "max_lat": bbox_max_lat,
                "raster_path": metadata.raster_path,
            },
        )
        row = result.mappings().one()
    return _row_to_metadata(row)


def get_latest_dem_metadata_for_region(region_name: str) -> Optional[TerrainMetadata]:
    """Fetch the latest DEM metadata row for a region."""
    stmt = text(
        """
        SELECT
            id,
            region_name,
            dem_source,
            crs_epsg,
            resolution_m,
            raster_path,
            created_at,
            ST_XMin(bbox) AS bbox_min_lon,
            ST_YMin(bbox) AS bbox_min_lat,
            ST_XMax(bbox) AS bbox_max_lon,
            ST_YMax(bbox) AS bbox_max_lat
        FROM terrain_metadata
        WHERE region_name = :region_name
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    with get_engine().begin() as conn:
        result = conn.execute(stmt, {"region_name": region_name})
        row = result.mappings().first()
    if row is None:
        return None
    return _row_to_metadata(row)

