from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from api.core.grid import GridSpec, get_grid_window_for_bbox, grid_bounds
from api.terrain.features_repo import TerrainFeaturesMetadata
from api.terrain.repo import TerrainMetadata
from api.terrain.window import load_terrain_for_aoi, load_terrain_window


def _write_test_geotiff(
    path: Path,
    *,
    grid: GridSpec,
    data_latlon: np.ndarray,
    nodata: float = -9999.0,
) -> None:
    """Write a north-up GeoTIFF for a grid-aligned array.

    `data_latlon` is in analysis convention (lat increasing). The written GeoTIFF is
    north-up (row 0 is north), so we flip rows when writing.
    """
    assert data_latlon.shape == (grid.n_lat, grid.n_lon)
    north = grid.origin_lat + grid.n_lat * grid.cell_size_deg
    transform = from_origin(
        west=grid.origin_lon,
        north=north,
        xsize=grid.cell_size_deg,
        ysize=grid.cell_size_deg,
    )
    raster_data = np.flipud(data_latlon).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": grid.n_lat,
        "width": grid.n_lon,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": transform,
        "nodata": float(nodata),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(raster_data, 1)


def test_get_grid_window_for_bbox_shapes_and_coords():
    grid = GridSpec(crs="EPSG:4326", cell_size_deg=0.1, origin_lat=0.0, origin_lon=0.0, n_lat=10, n_lon=20)
    bbox = (0.25, 0.15, 0.85, 0.55)  # (min_lon, min_lat, max_lon, max_lat)
    w = get_grid_window_for_bbox(grid, bbox)
    assert (w.i0, w.i1, w.j0, w.j1) == (1, 6, 2, 9)
    assert len(w.lat) == (w.i1 - w.i0)
    assert len(w.lon) == (w.j1 - w.j0)
    assert np.all(np.diff(w.lat) > 0)
    assert np.all(np.diff(w.lon) > 0)


def test_load_terrain_window_reads_by_window_and_normalizes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    grid = GridSpec(crs="EPSG:4326", cell_size_deg=0.1, origin_lat=0.0, origin_lon=0.0, n_lat=10, n_lon=20)
    min_lon, min_lat, max_lon, max_lat = grid_bounds(grid)

    slope = (np.arange(grid.n_lat)[:, None] * 1000 + np.arange(grid.n_lon)[None, :]).astype(np.float32)
    aspect = (slope + 7.0).astype(np.float32)
    dem = (slope + 100.0).astype(np.float32)

    slope_path = tmp_path / "slope.tif"
    aspect_path = tmp_path / "aspect.tif"
    dem_path = tmp_path / "dem.tif"
    _write_test_geotiff(slope_path, grid=grid, data_latlon=slope)
    _write_test_geotiff(aspect_path, grid=grid, data_latlon=aspect)
    _write_test_geotiff(dem_path, grid=grid, data_latlon=dem)

    features_md = TerrainFeaturesMetadata(
        id=1,
        created_at=datetime.now(timezone.utc),
        region_name="test",
        source_dem_metadata_id=1,
        slope_path=str(slope_path),
        aspect_path=str(aspect_path),
        crs_epsg=4326,
        cell_size_deg=grid.cell_size_deg,
        origin_lat=grid.origin_lat,
        origin_lon=grid.origin_lon,
        grid_n_lat=grid.n_lat,
        grid_n_lon=grid.n_lon,
        bbox=(min_lon, min_lat, max_lon, max_lat),
    )
    dem_md = TerrainMetadata(
        id=1,
        created_at=datetime.now(timezone.utc),
        region_name="test",
        dem_source="synthetic",
        crs_epsg=4326,
        resolution_m=11132.0,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        raster_path=str(dem_path),
        cell_size_deg=grid.cell_size_deg,
        origin_lat=grid.origin_lat,
        origin_lon=grid.origin_lon,
        grid_n_lat=grid.n_lat,
        grid_n_lon=grid.n_lon,
    )

    from api.terrain import window as window_mod

    monkeypatch.setattr(
        window_mod.features_repo,
        "get_latest_terrain_features_metadata_for_region",
        lambda region_name: features_md,
    )
    monkeypatch.setattr(
        window_mod.repo,
        "get_latest_dem_metadata_for_region",
        lambda region_name: dem_md,
    )

    bbox = (0.25, 0.15, 0.85, 0.55)
    tw = load_terrain_window("test", bbox, include_dem=True)
    assert tw.slope.shape == (5, 7)
    assert tw.aspect.shape == (5, 7)
    assert tw.elevation is not None and tw.elevation.shape == (5, 7)

    # Verify normalization: returned arrays match analysis (lat, lon) order.
    assert np.allclose(tw.slope, slope[tw.window.i0 : tw.window.i1, tw.window.j0 : tw.window.j1])
    assert np.allclose(tw.aspect, aspect[tw.window.i0 : tw.window.i1, tw.window.j0 : tw.window.j1])
    assert np.allclose(tw.elevation, dem[tw.window.i0 : tw.window.i1, tw.window.j0 : tw.window.j1])


def test_load_terrain_window_clips_partial_outside(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    grid = GridSpec(crs="EPSG:4326", cell_size_deg=0.1, origin_lat=0.0, origin_lon=0.0, n_lat=10, n_lon=10)
    min_lon, min_lat, max_lon, max_lat = grid_bounds(grid)
    slope = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float32)
    aspect = np.ones((grid.n_lat, grid.n_lon), dtype=np.float32)
    slope_path = tmp_path / "slope.tif"
    aspect_path = tmp_path / "aspect.tif"
    _write_test_geotiff(slope_path, grid=grid, data_latlon=slope)
    _write_test_geotiff(aspect_path, grid=grid, data_latlon=aspect)

    features_md = TerrainFeaturesMetadata(
        id=1,
        created_at=datetime.now(timezone.utc),
        region_name="test",
        source_dem_metadata_id=1,
        slope_path=str(slope_path),
        aspect_path=str(aspect_path),
        crs_epsg=4326,
        cell_size_deg=grid.cell_size_deg,
        origin_lat=grid.origin_lat,
        origin_lon=grid.origin_lon,
        grid_n_lat=grid.n_lat,
        grid_n_lon=grid.n_lon,
        bbox=(min_lon, min_lat, max_lon, max_lat),
    )

    from api.terrain import window as window_mod

    monkeypatch.setattr(
        window_mod.features_repo,
        "get_latest_terrain_features_metadata_for_region",
        lambda region_name: features_md,
    )

    # extends beyond grid; should clip
    tw = load_terrain_window("test", (-0.2, -0.2, 0.3, 0.3), clip=True)
    assert tw.window.i0 == 0 and tw.window.j0 == 0
    assert tw.slope.shape == (3, 3)


def test_load_terrain_window_empty_when_fully_outside(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    grid = GridSpec(crs="EPSG:4326", cell_size_deg=0.1, origin_lat=0.0, origin_lon=0.0, n_lat=10, n_lon=10)
    min_lon, min_lat, max_lon, max_lat = grid_bounds(grid)
    slope = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float32)
    aspect = np.ones((grid.n_lat, grid.n_lon), dtype=np.float32)
    slope_path = tmp_path / "slope.tif"
    aspect_path = tmp_path / "aspect.tif"
    _write_test_geotiff(slope_path, grid=grid, data_latlon=slope)
    _write_test_geotiff(aspect_path, grid=grid, data_latlon=aspect)

    features_md = TerrainFeaturesMetadata(
        id=1,
        created_at=datetime.now(timezone.utc),
        region_name="test",
        source_dem_metadata_id=1,
        slope_path=str(slope_path),
        aspect_path=str(aspect_path),
        crs_epsg=4326,
        cell_size_deg=grid.cell_size_deg,
        origin_lat=grid.origin_lat,
        origin_lon=grid.origin_lon,
        grid_n_lat=grid.n_lat,
        grid_n_lon=grid.n_lon,
        bbox=(min_lon, min_lat, max_lon, max_lat),
    )

    from api.terrain import window as window_mod

    monkeypatch.setattr(
        window_mod.features_repo,
        "get_latest_terrain_features_metadata_for_region",
        lambda region_name: features_md,
    )

    tw = load_terrain_window("test", (5.0, 5.0, 6.0, 6.0), clip=True)
    assert (tw.window.i0, tw.window.i1, tw.window.j0, tw.window.j1) == (grid.n_lat, grid.n_lat, grid.n_lon, grid.n_lon)
    assert tw.slope.shape == (0, 0)
    assert tw.aspect.shape == (0, 0)
    assert tw.window.lat.size == 0
    assert tw.window.lon.size == 0


def test_polygon_aoi_mask_has_true_and_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    grid = GridSpec(crs="EPSG:4326", cell_size_deg=0.1, origin_lat=0.0, origin_lon=0.0, n_lat=10, n_lon=10)
    min_lon, min_lat, max_lon, max_lat = grid_bounds(grid)
    slope = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float32)
    aspect = np.ones((grid.n_lat, grid.n_lon), dtype=np.float32)
    slope_path = tmp_path / "slope.tif"
    aspect_path = tmp_path / "aspect.tif"
    _write_test_geotiff(slope_path, grid=grid, data_latlon=slope)
    _write_test_geotiff(aspect_path, grid=grid, data_latlon=aspect)

    features_md = TerrainFeaturesMetadata(
        id=1,
        created_at=datetime.now(timezone.utc),
        region_name="test",
        source_dem_metadata_id=1,
        slope_path=str(slope_path),
        aspect_path=str(aspect_path),
        crs_epsg=4326,
        cell_size_deg=grid.cell_size_deg,
        origin_lat=grid.origin_lat,
        origin_lon=grid.origin_lon,
        grid_n_lat=grid.n_lat,
        grid_n_lon=grid.n_lon,
        bbox=(min_lon, min_lat, max_lon, max_lat),
    )

    from api.terrain import window as window_mod

    monkeypatch.setattr(
        window_mod.features_repo,
        "get_latest_terrain_features_metadata_for_region",
        lambda region_name: features_md,
    )

    # AOI polygon covering only part of the window
    aoi = box(0.25, 0.15, 0.55, 0.35)
    tw = load_terrain_for_aoi("test", aoi, return_mask=True)
    assert tw.aoi_mask is not None
    assert tw.aoi_mask.shape == tw.slope.shape
    assert bool(tw.aoi_mask.any()) is True
    assert bool((~tw.aoi_mask).any()) is True

