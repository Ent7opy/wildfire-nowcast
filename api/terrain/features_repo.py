"""Repository helpers for derived terrain feature rasters (slope/aspect)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import text

from api.db import get_engine


@dataclass(slots=True, kw_only=True)
class TerrainFeaturesMetadataCreate:
    region_name: str
    source_dem_metadata_id: int
    slope_path: str
    aspect_path: str
    crs_epsg: int
    cell_size_deg: float
    origin_lat: float
    origin_lon: float
    grid_n_lat: int
    grid_n_lon: int
    bbox: tuple[float, float, float, float]
    slope_units: str = "degrees"
    aspect_units: str = "degrees"
    aspect_convention: str = "clockwise_from_north_downslope"
    nodata_value: float = -9999.0
    slope_min: float | None = None
    slope_max: float | None = None
    aspect_min: float | None = None
    aspect_max: float | None = None
    coverage_fraction: float | None = None


@dataclass(slots=True, kw_only=True)
class TerrainFeaturesMetadata(TerrainFeaturesMetadataCreate):
    id: int
    created_at: datetime


def _row_to_features(row: dict) -> TerrainFeaturesMetadata:
    return TerrainFeaturesMetadata(
        id=int(row["id"]),
        region_name=row["region_name"],
        source_dem_metadata_id=int(row["source_dem_metadata_id"]),
        slope_path=row["slope_path"],
        aspect_path=row["aspect_path"],
        crs_epsg=int(row["crs_epsg"]),
        cell_size_deg=float(row["cell_size_deg"]),
        origin_lat=float(row["origin_lat"]),
        origin_lon=float(row["origin_lon"]),
        grid_n_lat=int(row["grid_n_lat"]),
        grid_n_lon=int(row["grid_n_lon"]),
        bbox=(
            float(row["bbox_min_lon"]),
            float(row["bbox_min_lat"]),
            float(row["bbox_max_lon"]),
            float(row["bbox_max_lat"]),
        ),
        slope_units=row["slope_units"],
        aspect_units=row["aspect_units"],
        aspect_convention=row["aspect_convention"],
        nodata_value=float(row["nodata_value"]),
        slope_min=(float(row["slope_min"]) if row.get("slope_min") is not None else None),
        slope_max=(float(row["slope_max"]) if row.get("slope_max") is not None else None),
        aspect_min=(float(row["aspect_min"]) if row.get("aspect_min") is not None else None),
        aspect_max=(float(row["aspect_max"]) if row.get("aspect_max") is not None else None),
        coverage_fraction=(
            float(row["coverage_fraction"]) if row.get("coverage_fraction") is not None else None
        ),
        created_at=row["created_at"],
    )


def insert_terrain_features_metadata(
    metadata: TerrainFeaturesMetadataCreate,
) -> TerrainFeaturesMetadata:
    """Insert a terrain_features_metadata row and return the stored record."""
    stmt = text(
        """
        INSERT INTO terrain_features_metadata (
            region_name,
            source_dem_metadata_id,
            slope_path,
            aspect_path,
            crs_epsg,
            cell_size_deg,
            origin_lat,
            origin_lon,
            grid_n_lat,
            grid_n_lon,
            bbox,
            slope_units,
            aspect_units,
            aspect_convention,
            nodata_value,
            slope_min,
            slope_max,
            aspect_min,
            aspect_max,
            coverage_fraction
        )
        VALUES (
            :region_name,
            :source_dem_metadata_id,
            :slope_path,
            :aspect_path,
            :crs_epsg,
            :cell_size_deg,
            :origin_lat,
            :origin_lon,
            :grid_n_lat,
            :grid_n_lon,
            ST_SetSRID(ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat), 4326),
            :slope_units,
            :aspect_units,
            :aspect_convention,
            :nodata_value,
            :slope_min,
            :slope_max,
            :aspect_min,
            :aspect_max,
            :coverage_fraction
        )
        RETURNING
            id,
            region_name,
            source_dem_metadata_id,
            slope_path,
            aspect_path,
            crs_epsg,
            cell_size_deg,
            origin_lat,
            origin_lon,
            grid_n_lat,
            grid_n_lon,
            slope_units,
            aspect_units,
            aspect_convention,
            nodata_value,
            slope_min,
            slope_max,
            aspect_min,
            aspect_max,
            coverage_fraction,
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
                "source_dem_metadata_id": metadata.source_dem_metadata_id,
                "slope_path": metadata.slope_path,
                "aspect_path": metadata.aspect_path,
                "crs_epsg": metadata.crs_epsg,
                "cell_size_deg": metadata.cell_size_deg,
                "origin_lat": metadata.origin_lat,
                "origin_lon": metadata.origin_lon,
                "grid_n_lat": metadata.grid_n_lat,
                "grid_n_lon": metadata.grid_n_lon,
                "min_lon": bbox_min_lon,
                "min_lat": bbox_min_lat,
                "max_lon": bbox_max_lon,
                "max_lat": bbox_max_lat,
                "slope_units": metadata.slope_units,
                "aspect_units": metadata.aspect_units,
                "aspect_convention": metadata.aspect_convention,
                "nodata_value": metadata.nodata_value,
                "slope_min": metadata.slope_min,
                "slope_max": metadata.slope_max,
                "aspect_min": metadata.aspect_min,
                "aspect_max": metadata.aspect_max,
                "coverage_fraction": metadata.coverage_fraction,
            },
        )
        row = result.mappings().one()
    return _row_to_features(row)


def get_latest_terrain_features_metadata_for_region(
    region_name: str,
) -> Optional[TerrainFeaturesMetadata]:
    """Fetch the latest terrain_features_metadata row for a region."""
    stmt = text(
        """
        SELECT
            id,
            region_name,
            source_dem_metadata_id,
            slope_path,
            aspect_path,
            crs_epsg,
            cell_size_deg,
            origin_lat,
            origin_lon,
            grid_n_lat,
            grid_n_lon,
            slope_units,
            aspect_units,
            aspect_convention,
            nodata_value,
            slope_min,
            slope_max,
            aspect_min,
            aspect_max,
            coverage_fraction,
            created_at,
            ST_XMin(bbox) AS bbox_min_lon,
            ST_YMin(bbox) AS bbox_min_lat,
            ST_XMax(bbox) AS bbox_max_lon,
            ST_YMax(bbox) AS bbox_max_lat
        FROM terrain_features_metadata
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
    return _row_to_features(row)

