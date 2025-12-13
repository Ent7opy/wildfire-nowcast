"""Terrain helpers for DEM metadata and loading."""

from .dem_loader import grid_spec_from_metadata, load_dem_for_bbox
from .features_loader import load_slope_aspect_for_bbox
from .window import TerrainWindow, load_terrain_for_aoi, load_terrain_window
from .features_repo import (
    TerrainFeaturesMetadata,
    TerrainFeaturesMetadataCreate,
    get_latest_terrain_features_metadata_for_region,
    insert_terrain_features_metadata,
)
from .repo import (
    TerrainMetadata,
    TerrainMetadataCreate,
    get_latest_dem_metadata_for_region,
    insert_terrain_metadata,
)

__all__ = [
    "TerrainMetadata",
    "TerrainMetadataCreate",
    "get_latest_dem_metadata_for_region",
    "insert_terrain_metadata",
    "load_dem_for_bbox",
    "grid_spec_from_metadata",
    "TerrainFeaturesMetadata",
    "TerrainFeaturesMetadataCreate",
    "get_latest_terrain_features_metadata_for_region",
    "insert_terrain_features_metadata",
    "load_slope_aspect_for_bbox",
    "TerrainWindow",
    "load_terrain_window",
    "load_terrain_for_aoi",
]

