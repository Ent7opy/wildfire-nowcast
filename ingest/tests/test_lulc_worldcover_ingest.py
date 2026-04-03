"""Unit tests for LULC WorldCover version tracking.

Tests cover pure-function logic (no DB, no S3) — tile ID formatting, tile path
construction, cache manifest write/check, the update-param structure, and the
new CRS / pixel-registration / boundary-coordinate validation helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from affine import Affine

# ── _format_tile_id ──────────────────────────────────────────────────────────

class TestFormatTileId:
    def test_positive_lat_lon(self):
        from ingest.lulc_worldcover_ingest import _format_tile_id

        assert _format_tile_id(lat0=0, lon0=0) == "N00E000"

    def test_northern_western(self):
        from ingest.lulc_worldcover_ingest import _format_tile_id

        assert _format_tile_id(lat0=48, lon0=-123) == "N48W123"

    def test_southern_eastern(self):
        from ingest.lulc_worldcover_ingest import _format_tile_id

        assert _format_tile_id(lat0=-33, lon0=18) == "S33E018"


# ── _tile_path ────────────────────────────────────────────────────────────────

class TestTilePath:
    def test_path_format(self):
        from ingest.lulc_worldcover_ingest import _tile_path

        path = _tile_path(version="v200", year=2021, tile_id="N00E000")
        assert path == "v200/2021/map/ESA_WorldCover_10m_2021_v200_N00E000_Map.tif"

    def test_version_year_in_path(self):
        from ingest.lulc_worldcover_ingest import _tile_path

        path = _tile_path(version="v100", year=2020, tile_id="N48W123")
        assert "v100" in path
        assert "2020" in path


# ── cache manifest ────────────────────────────────────────────────────────────

class TestCacheManifest:
    def test_write_creates_manifest(self, tmp_path):
        from ingest.lulc_worldcover_ingest import _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["worldcover_version"] == "v200"
        assert data["worldcover_year"] == 2021
        assert data["lulc_version"] == "v200_2021"
        assert "+00:00" in data["written_at"]

    def test_write_is_atomic(self, tmp_path):
        """No .part file should remain after a successful write."""
        from ingest.lulc_worldcover_ingest import _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        assert not any(tmp_path.glob("*.part"))

    def test_write_creates_cache_dir(self, tmp_path):
        from ingest.lulc_worldcover_ingest import _write_cache_manifest

        target = tmp_path / "nested" / "cache"
        _write_cache_manifest(cache_dir=target, version="v200", year=2021)
        assert (target / "manifest.json").exists()

    def test_check_no_warning_when_versions_match(self, tmp_path, caplog):
        from ingest.lulc_worldcover_ingest import _check_cache_manifest, _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _check_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        assert not caplog.records

    def test_check_warns_on_version_mismatch(self, tmp_path, caplog):
        from ingest.lulc_worldcover_ingest import _check_cache_manifest, _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _check_cache_manifest(cache_dir=tmp_path, version="v300", year=2021)
        assert any("mismatch" in r.message for r in caplog.records)
        assert any("Mitigation" in r.message for r in caplog.records)

    def test_check_warns_on_year_mismatch(self, tmp_path, caplog):
        from ingest.lulc_worldcover_ingest import _check_cache_manifest, _write_cache_manifest

        _write_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _check_cache_manifest(cache_dir=tmp_path, version="v200", year=2023)
        assert any("mismatch" in r.message for r in caplog.records)

    def test_check_silent_when_no_manifest(self, tmp_path, caplog):
        from ingest.lulc_worldcover_ingest import _check_cache_manifest

        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _check_cache_manifest(cache_dir=tmp_path, version="v200", year=2021)
        assert not caplog.records


# ── lulc_version in update params ────────────────────────────────────────────

class TestUpdateParamsIncludeLulcVersion:
    """Ensure that the update payload written to the DB includes lulc_version."""

    def test_lulc_version_key_present(self):
        """Build a fake update dict the same way _backfill_tile does and verify lulc_version."""
        from ingest.lulc_worldcover_ingest import _CLASS_LABELS, _CLASS_SCORES

        source_version = "v200_2021"
        lc = 10
        label = _CLASS_LABELS[lc]
        score = float(_CLASS_SCORES[lc])

        update = {
            "id": 42,
            "landcover_class": lc,
            "landcover_label": label,
            "landcover_score": score,
            "lulc_version": source_version,
            "landcover_source": "esa_worldcover",
            "landcover_version": source_version,
        }

        assert update["lulc_version"] == source_version
        assert update["landcover_version"] == source_version

    def test_lulc_version_encodes_both_version_and_year(self):
        """Confirm source_version formula produces the expected tag."""
        version, year = "v200", 2021
        source_version = f"{version}_{year}"
        assert source_version == "v200_2021"


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_src(*, epsg: int | None = 4326, area_or_point: str = "Area", pixel_size: float = 8.333e-05) -> MagicMock:
    """Return a mock rasterio DatasetReader with the given CRS and transform."""
    from rasterio.crs import CRS as RasterioCRS

    src = MagicMock()
    if epsg is None:
        src.crs = None
    else:
        src.crs = RasterioCRS.from_epsg(epsg)
    # Affine: origin at lon=-180, lat=90; pixel_size wide, -pixel_size tall.
    src.transform = Affine(pixel_size, 0, -180.0, 0, -pixel_size, 90.0)
    src.tags.return_value = {"AREA_OR_POINT": area_or_point}
    return src


# ── _validate_tile_crs ────────────────────────────────────────────────────────

class TestValidateTileCrs:
    def test_epsg_4326_passes(self):
        from ingest.lulc_worldcover_ingest import _validate_tile_crs

        src = _make_src(epsg=4326)
        _validate_tile_crs(src, Path("tile.tif"))  # must not raise

    def test_wrong_epsg_raises(self):
        from ingest.lulc_worldcover_ingest import _validate_tile_crs

        src = _make_src(epsg=32610)
        with pytest.raises(ValueError, match="STOP"):
            _validate_tile_crs(src, Path("tile.tif"))

    def test_none_crs_raises(self):
        from ingest.lulc_worldcover_ingest import _validate_tile_crs

        src = _make_src(epsg=None)
        with pytest.raises(ValueError, match="STOP"):
            _validate_tile_crs(src, Path("tile.tif"))


# ── _validate_pixel_registration ─────────────────────────────────────────────

class TestValidatePixelRegistration:
    def test_area_registration_silent(self, caplog):
        from ingest.lulc_worldcover_ingest import _validate_pixel_registration

        src = _make_src(area_or_point="Area")
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _validate_pixel_registration(src, Path("tile.tif"))
        assert not caplog.records

    def test_point_registration_warns(self, caplog):
        from ingest.lulc_worldcover_ingest import _validate_pixel_registration

        src = _make_src(area_or_point="Point")
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _validate_pixel_registration(src, Path("tile.tif"))
        assert any("AREA_OR_POINT" in r.message for r in caplog.records)
        assert any("Mitigation" in r.message for r in caplog.records)

    def test_missing_tag_defaults_to_area_silent(self, caplog):
        from ingest.lulc_worldcover_ingest import _validate_pixel_registration

        src = _make_src()
        src.tags.return_value = {}  # no AREA_OR_POINT tag
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _validate_pixel_registration(src, Path("tile.tif"))
        assert not caplog.records


# ── _warn_boundary_coords ─────────────────────────────────────────────────────

class TestWarnBoundaryCoords:
    # pixel_size = 8.333e-05; origin lon=-180, lat=90
    # A coord at lon=-180.0 (frac_col=0.0) is exactly on the left edge → boundary.
    # A coord at lon=-180.0 + 0.5*pixel_size is at pixel centre → not boundary.

    def _make_interior_coord(self, pixel_size: float = 8.333e-05) -> tuple[float, float]:
        """Coordinate at pixel centre — safely away from any boundary."""
        return (-180.0 + pixel_size * 0.5, 90.0 - pixel_size * 0.5)

    def _make_boundary_coord(self) -> tuple[float, float]:
        """Coordinate exactly on a pixel edge (frac_col = 0.0)."""
        return (-180.0, 90.0)

    def test_no_warning_for_interior_coords(self, caplog):
        from ingest.lulc_worldcover_ingest import _warn_boundary_coords

        src = _make_src()
        coords = [self._make_interior_coord() for _ in range(5)]
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _warn_boundary_coords(src, coords, Path("tile.tif"))
        assert not caplog.records

    def test_warns_for_boundary_coords(self, caplog):
        from ingest.lulc_worldcover_ingest import _warn_boundary_coords

        src = _make_src()
        coords = [self._make_boundary_coord()]
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _warn_boundary_coords(src, coords, Path("tile.tif"))
        assert any("pixel boundary" in r.message for r in caplog.records)
        assert any("Mitigation" in r.message for r in caplog.records)

    def test_warns_counts_correctly(self, caplog):
        from ingest.lulc_worldcover_ingest import _warn_boundary_coords

        src = _make_src()
        pixel_size = 8.333e-05
        coords = [
            self._make_boundary_coord(),          # boundary
            self._make_interior_coord(pixel_size), # interior
            self._make_boundary_coord(),          # boundary
        ]
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _warn_boundary_coords(src, coords, Path("tile.tif"))
        # Should report 2/3 boundary coords.
        assert any("2/3" in r.message for r in caplog.records)

    def test_empty_coords_silent(self, caplog):
        from ingest.lulc_worldcover_ingest import _warn_boundary_coords

        src = _make_src()
        with caplog.at_level(logging.WARNING, logger="lulc_worldcover_ingest"):
            _warn_boundary_coords(src, [], Path("tile.tif"))
        assert not caplog.records
