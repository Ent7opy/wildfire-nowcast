"""Unit tests for the graceful DEM fallback in terrain_features.py and dem_loader.py.

These tests exercise the flat-terrain stub path without touching a real database
or filesystem DEM.  They mock only what is necessary to isolate the fallback logic.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

_BBOX = (-120.5, 39.5, -120.0, 40.0)


def _make_dem_metadata(
    *,
    raster_path: str = "/nonexistent/dem.tif",
    resolution_m: float = 30.0,
    region_name: str = "test_region",
) -> object:
    """Return a minimal TerrainMetadata-like object."""
    md = MagicMock()
    md.id = 1
    md.region_name = region_name
    md.raster_path = raster_path
    md.resolution_m = resolution_m
    md.bbox = _BBOX
    md.cell_size_deg = 0.01
    md.origin_lat = 39.5
    md.origin_lon = -120.5
    md.grid_n_lat = 50
    md.grid_n_lon = 50
    md.crs_epsg = 4326
    md.created_at = datetime(2026, 1, 1)
    return md


# ── dem_loader: flat stub when no file and no fallback available ──────────────

class TestLoadDemForBboxFallback:
    """Tests for load_dem_for_bbox flat-terrain fallback."""

    @patch("api.terrain.dem_loader.find_fallback_dem", return_value=None)
    @patch("api.terrain.dem_loader.get_latest_dem_metadata_for_region")
    def test_flat_stub_returned_when_file_missing_and_no_ladder(
        self, mock_latest, mock_find, caplog
    ):
        """When the DEM file is missing and no alternative exists, return a flat DataArray."""
        from api.terrain.dem_loader import load_dem_for_bbox

        md = _make_dem_metadata(raster_path="/nonexistent/dem.tif")
        mock_latest.side_effect = [md, None]  # first=region, second=global_base

        with caplog.at_level(logging.WARNING, logger="api.terrain.dem_loader"):
            result = load_dem_for_bbox("test_region", _BBOX)

        # Must return a DataArray, not raise.
        import xarray as xr
        assert isinstance(result, xr.DataArray)
        # Flag must be set.
        assert result.attrs.get("terrain_fallback_used") is True
        # All values should be zero (flat).
        assert float(result.values.max()) == 0.0
        # WARNING must have been emitted.
        assert any("flat-terrain stub" in r.message for r in caplog.records)
        assert any("Mitigation" in r.message for r in caplog.records)

    @patch("api.terrain.dem_loader.validate_raster_matches_grid")
    @patch("api.terrain.dem_loader.find_fallback_dem")
    @patch("api.terrain.dem_loader.get_latest_dem_metadata_for_region")
    def test_lower_res_fallback_used_when_primary_missing(
        self, mock_latest, mock_find, mock_validate, tmp_path, caplog
    ):
        """When primary DEM is missing but a lower-res DEM exists, use it with WARNING."""
        import rioxarray  # noqa: F401 — ensure import is available
        from api.terrain.dem_loader import load_dem_for_bbox

        # Create a real GeoTIFF that rioxarray can open.
        import rasterio
        from rasterio.transform import from_bounds

        fallback_tif = tmp_path / "dem_coarse.tif"
        transform = from_bounds(-120.5, 39.5, -120.0, 40.0, 5, 5)
        with rasterio.open(
            fallback_tif,
            "w",
            driver="GTiff",
            height=5,
            width=5,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(np.full((5, 5), 100.0, dtype=np.float32), 1)

        primary_md = _make_dem_metadata(raster_path="/nonexistent/dem_10m.tif", resolution_m=10.0)
        fallback_md = _make_dem_metadata(raster_path=str(fallback_tif), resolution_m=90.0)

        mock_latest.return_value = primary_md
        mock_find.return_value = fallback_md
        mock_validate.return_value = None  # skip grid alignment check

        with caplog.at_level(logging.WARNING, logger="api.terrain.dem_loader"):
            result = load_dem_for_bbox("test_region", _BBOX)

        import xarray as xr
        assert isinstance(result, xr.DataArray)
        # No fallback attr — real data was loaded.
        assert result.attrs.get("terrain_fallback_used") is not True
        assert any("lower-resolution" in r.message for r in caplog.records)


# ── terrain_features: flat stub written to disk when no DEM file ─────────────

class TestTerrainFeaturesFlatStub:
    """Tests for the ingest/terrain_features.py flat-terrain stub path."""

    @patch("ingest.terrain_features.insert_terrain_features_metadata")
    @patch("ingest.terrain_features.find_fallback_dem", return_value=None)
    @patch("ingest.terrain_features.get_latest_dem_metadata_for_region")
    def test_flat_stub_written_and_registered_when_dem_missing(
        self, mock_get_md, mock_find_fallback, mock_insert, tmp_path, caplog
    ):
        """main() writes flat slope/aspect and inserts metadata when no DEM file exists."""
        from ingest.terrain_features import TerrainFeaturesSettings, main

        # Patch settings to use tmp_path.
        md = _make_dem_metadata(raster_path=str(tmp_path / "nonexistent.tif"))
        mock_get_md.return_value = md

        # Mock insert to return an object with .id
        inserted = MagicMock()
        inserted.id = 99
        mock_insert.return_value = inserted

        settings_patch = {
            "region_name": "test_region",
            "data_dir": tmp_path,
            "output_dir": tmp_path,
            "recompute": False,
            "nodata_value": -9999.0,
        }

        with (
            patch("ingest.terrain_features.TerrainFeaturesSettings") as mock_settings_cls,
            caplog.at_level(logging.WARNING, logger="terrain_features"),
        ):
            settings_obj = MagicMock(spec=TerrainFeaturesSettings)
            settings_obj.region_name = "test_region"
            settings_obj.resolved_output_dir = tmp_path
            settings_obj.recompute = False
            settings_obj.nodata_value = -9999.0
            mock_settings_cls.return_value = settings_obj

            # grid_spec_from_metadata must return something sensible.
            from api.core.grid import GridSpec

            grid = GridSpec(
                crs="EPSG:4326",
                cell_size_deg=0.01,
                origin_lat=39.5,
                origin_lon=-120.5,
                n_lat=50,
                n_lon=50,
            )
            with patch("ingest.terrain_features.grid_spec_from_metadata", return_value=grid):
                main([])

        # insert must have been called with terrain_fallback_used=True.
        assert mock_insert.called
        call_arg = mock_insert.call_args[0][0]
        assert call_arg.terrain_fallback_used is True

        # WARNING log must include Mitigation hint.
        assert any("Mitigation" in r.message for r in caplog.records)

    @patch("ingest.terrain_features.get_latest_dem_metadata_for_region", return_value=None)
    def test_no_crash_when_no_dem_metadata(self, mock_get_md, caplog):
        """main() exits cleanly (no crash, no ValueError) when no DEM metadata exists."""
        from ingest.terrain_features import main

        with caplog.at_level(logging.WARNING, logger="terrain_features"):
            main([])  # should not raise

        assert any("No DEM metadata" in r.message for r in caplog.records)
        assert any("Mitigation" in r.message for r in caplog.records)


# ── dem_loader: flat stub shape matches bbox ──────────────────────────────────

class TestFlatDemStubShape:
    """Verify _flat_dem_stub produces correctly shaped/valued output."""

    def test_flat_stub_shape_and_values(self):
        from api.terrain.dem_loader import _flat_dem_stub

        bbox = (-121.0, 39.0, -120.0, 40.0)
        da = _flat_dem_stub(bbox, cell_size_deg=0.5)

        # 2 lats (39.25, 39.75), 2 lons (-120.75, -120.25)
        assert da.dims == ("lat", "lon")
        assert da.shape == (2, 2)
        assert (da.values == 0.0).all()
        assert da.attrs["terrain_fallback_used"] is True

    def test_flat_stub_no_terrain_fallback_attr_on_real_data(self):
        """Real DataArrays should NOT have terrain_fallback_used=True."""
        import xarray as xr

        real = xr.DataArray(
            np.array([[100.0, 200.0]]),
            dims=("lat", "lon"),
            coords={"lat": [39.5], "lon": [-120.5, -120.0]},
        )
        assert real.attrs.get("terrain_fallback_used") is not True
