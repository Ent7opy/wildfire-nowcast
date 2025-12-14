from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from api.core.grid import GridSpec
from api.terrain.features_math import compute_slope_aspect
from api.terrain.validate import validate_raster_matches_grid, validate_terrain_stack


def _repo_root() -> Path:
    # api/tests/ -> api/ -> repo root
    return Path(__file__).resolve().parents[2]


def _grid_from_raster(path: Path) -> GridSpec:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise AssertionError(f"{path} missing CRS")
        cell = float(abs(src.transform.a))
        left, bottom, right, top = map(float, src.bounds)
        # origin_lat/lon are southern/western *edges* in this project.
        return GridSpec(
            crs=src.crs.to_string(),
            cell_size_deg=cell,
            origin_lat=bottom,
            origin_lon=left,
            n_lat=int(src.height),
            n_lon=int(src.width),
        )


def test_sample_rasters_align_to_grid_contract():
    root = _repo_root()
    dem = root / "data" / "dem" / "smoke_grid" / "dem_smoke_grid_epsg4326_0p01deg.tif"
    slope = root / "data" / "terrain" / "smoke_grid" / "slope_smoke_grid_epsg4326_0p01deg.tif"
    aspect = root / "data" / "terrain" / "smoke_grid" / "aspect_smoke_grid_epsg4326_0p01deg.tif"

    assert dem.exists()
    assert slope.exists()
    assert aspect.exists()

    grid = _grid_from_raster(dem)
    # All three should be exactly aligned.
    validate_terrain_stack(dem, slope, aspect, grid, strict=True)


def test_sample_slope_aspect_value_sanity():
    root = _repo_root()
    slope_path = root / "data" / "terrain" / "smoke_grid" / "slope_smoke_grid_epsg4326_0p01deg.tif"
    aspect_path = root / "data" / "terrain" / "smoke_grid" / "aspect_smoke_grid_epsg4326_0p01deg.tif"
    assert slope_path.exists()
    assert aspect_path.exists()

    with rasterio.open(slope_path) as ssrc:
        slope = ssrc.read(1, masked=True).filled(np.nan).astype(float)
    with rasterio.open(aspect_path) as asrc:
        aspect = asrc.read(1, masked=True).filled(np.nan).astype(float)

    slope_vals = slope[np.isfinite(slope)]
    assert slope_vals.size > 0
    assert float(np.nanmin(slope_vals)) >= 0.0
    assert float(np.nanmax(slope_vals)) <= 90.0

    # Aspect is defined where slope > 0; elsewhere it can be NaN (flat) or nodata.
    mask = np.isfinite(slope) & (slope > 0.0) & np.isfinite(aspect)
    assert bool(mask.any()) is True
    a = aspect[mask]
    assert float(np.nanmin(a)) >= 0.0
    assert float(np.nanmax(a)) < 360.0


def test_compute_slope_aspect_known_plane_has_constant_aspect_and_positive_slope():
    height, width = 64, 96
    cell_size_deg = 0.01

    # North-up raster convention: row 0 is north, row index increases southward.
    # Make elevation decrease southward so downslope points south (aspect ~ 180°).
    z = (-np.arange(height, dtype=float)[:, None] * 10.0) * np.ones((1, width), dtype=float)

    # Lat centers (north->south) decrease by row.
    lat_north = 40.0
    lat_centers = lat_north - (np.arange(height, dtype=float) + 0.5) * cell_size_deg

    slope, aspect = compute_slope_aspect(z, cell_size_deg=cell_size_deg, lat_centers_deg=lat_centers)
    core = (slice(1, -1), slice(1, -1))

    s = slope[core]
    a = aspect[core]
    assert float(np.nanmedian(s)) > 0.0
    assert float(np.nanmax(s)) <= 90.0
    assert np.isfinite(a).all()
    assert float(np.nanmedian(a)) == pytest.approx(180.0, abs=1.0)


def _write_geotiff_for_grid(path: Path, *, grid: GridSpec, data_latlon: np.ndarray) -> None:
    """Write a north-up GeoTIFF for an array in analysis convention (lat increasing)."""
    assert data_latlon.shape == (grid.n_lat, grid.n_lon)
    north = grid.origin_lat + grid.n_lat * grid.cell_size_deg
    transform = from_origin(
        west=grid.origin_lon,
        north=north,
        xsize=grid.cell_size_deg,
        ysize=grid.cell_size_deg,
    )
    # Stored GeoTIFF is north-up, so flip the analysis array on write.
    raster_data = np.flipud(data_latlon).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": grid.n_lat,
        "width": grid.n_lon,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(raster_data, 1)


def test_validate_raster_matches_grid_catches_misalignment(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    grid = GridSpec(crs="EPSG:4326", cell_size_deg=0.1, origin_lat=0.0, origin_lon=0.0, n_lat=5, n_lon=7)
    data = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float32)

    good = tmp_path / "good.tif"
    bad_shift_west = tmp_path / "bad_shift_west.tif"

    _write_geotiff_for_grid(good, grid=grid, data_latlon=data)

    # Shift by half a cell west: same shape/resolution but wrong origin edge/bounds.
    shifted_grid = GridSpec(
        crs=grid.crs,
        cell_size_deg=grid.cell_size_deg,
        origin_lat=grid.origin_lat,
        origin_lon=grid.origin_lon - grid.cell_size_deg / 2.0,
        n_lat=grid.n_lat,
        n_lon=grid.n_lon,
    )
    _write_geotiff_for_grid(bad_shift_west, grid=shifted_grid, data_latlon=data)

    validate_raster_matches_grid(good, grid, strict=True)

    with pytest.raises(ValueError):
        validate_raster_matches_grid(bad_shift_west, grid, strict=True)

    caplog.clear()
    validate_raster_matches_grid(bad_shift_west, grid, strict=False)
    assert any("mismatch" in rec.message.lower() for rec in caplog.records)

