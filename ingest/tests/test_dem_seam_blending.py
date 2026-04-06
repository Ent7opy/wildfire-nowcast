"""Unit tests for DEM tile-seam blending and QA in dem_preprocess.py."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_profile(
    west: float,
    north: float,
    xsize: float,
    ysize: float,
    width: int,
    height: int,
) -> dict:
    """Build a minimal rasterio-style profile."""
    return {
        "transform": from_origin(west, north, xsize, ysize),
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
    }


def _seam_dem(
    *,
    west: float = -121.0,
    north: float = 41.0,
    cell: float = 0.01,
    width: int = 200,
    height: int = 200,
    jump: float = 50.0,
) -> tuple[np.ndarray, dict]:
    """Create a synthetic DEM with a sharp jump at the integer-degree seam.

    The raster spans from *west* eastward and *north* southward.
    A vertical seam at lon=-120.0 gets a +jump offset on the east side.
    """
    profile = _make_profile(west, north, cell, cell, width, height)
    data = np.full((height, width), 500.0, dtype=np.float64)

    # Integer-degree boundary at lon = -120.0 falls at pixel col =
    # (-120.0 - west) / cell = 100.
    seam_col = int(round((-120.0 - west) / cell))
    data[:, seam_col:] += jump
    return data, profile


# ---------------------------------------------------------------------------
# _integer_degree_pixel_indices
# ---------------------------------------------------------------------------


class TestIntegerDegreePixelIndices:
    def test_detects_interior_boundary(self):
        from ingest.dem_preprocess import _integer_degree_pixel_indices

        # west=-121.0, cell=0.01, 200 cols => lon range [-121, -119]
        # Integer boundaries at -120 => pixel ~100
        indices = _integer_degree_pixel_indices(-121.0, 0.01, 200)
        assert len(indices) >= 1
        # The boundary at -120 should be near pixel 100.
        assert any(abs(i - 100) <= 1 for i in indices)

    def test_excludes_raster_edges(self):
        from ingest.dem_preprocess import _integer_degree_pixel_indices

        # Small raster where the only boundary falls near the edge.
        indices = _integer_degree_pixel_indices(-120.02, 0.01, 5)
        assert len(indices) == 0, "boundaries at raster edge should be excluded"

    def test_no_seams_within_single_tile(self):
        from ingest.dem_preprocess import _integer_degree_pixel_indices

        # Raster entirely within [-120.5, -120.0) — no integer boundary
        indices = _integer_degree_pixel_indices(-120.5, 0.01, 40)
        assert len(indices) == 0


# ---------------------------------------------------------------------------
# blend_tile_seams
# ---------------------------------------------------------------------------


class TestBlendTileSeams:
    def test_reduces_seam_jump(self):
        from ingest.dem_preprocess import blend_tile_seams

        data, profile = _seam_dem(jump=50.0)
        blended = blend_tile_seams(data, profile)

        # The maximum single-pixel elevation difference across the seam
        # should be much smaller than the original 50 m jump.
        seam_col = 100  # approximate location of -120.0 boundary
        orig_diff = np.max(np.abs(np.diff(data[:, seam_col - 1 : seam_col + 2], axis=1)))
        blend_diff = np.max(np.abs(np.diff(blended[:, seam_col - 1 : seam_col + 2], axis=1)))
        assert blend_diff < orig_diff, "blending should reduce the jump"

    def test_interior_unchanged(self):
        from ingest.dem_preprocess import blend_tile_seams

        data, profile = _seam_dem(jump=50.0)
        blended = blend_tile_seams(data, profile)

        # Pixels far from any seam should be identical.
        assert np.array_equal(blended[:, :90], data[:, :90])
        assert np.array_equal(blended[:, 110:], data[:, 110:])

    def test_preserves_dtype(self):
        from ingest.dem_preprocess import blend_tile_seams

        data, profile = _seam_dem()
        data_f32 = data.astype(np.float32)
        blended = blend_tile_seams(data_f32, profile)
        assert blended.dtype == np.float32

    def test_noop_when_no_seams(self):
        from ingest.dem_preprocess import blend_tile_seams

        # Raster entirely within one tile — no integer-degree seams.
        profile = _make_profile(-120.5, 40.5, 0.01, 0.01, 40, 40)
        data = np.full((40, 40), 300.0)
        blended = blend_tile_seams(data, profile)
        np.testing.assert_array_equal(blended, data)

    def test_nan_pixels_preserved(self):
        from ingest.dem_preprocess import blend_tile_seams

        data, profile = _seam_dem(jump=50.0)
        data[10, 99:102] = np.nan  # NaN near the seam
        blended = blend_tile_seams(data, profile)
        assert np.isnan(blended[10, 99:102]).all(), "NaN pixels must remain NaN"


# ---------------------------------------------------------------------------
# check_seam_quality
# ---------------------------------------------------------------------------


class TestCheckSeamQuality:
    def test_warns_on_large_discontinuity(self):
        from ingest.dem_preprocess import check_seam_quality

        data, profile = _seam_dem(jump=500.0)  # huge jump
        warnings = check_seam_quality(data, profile, slope_threshold_deg=5.0)
        assert len(warnings) >= 1
        assert warnings[0]["axis"] == "longitude"

    def test_clean_dem_passes(self):
        from ingest.dem_preprocess import check_seam_quality

        # Smooth DEM with no seam artifacts.
        profile = _make_profile(-121.0, 41.0, 0.01, 0.01, 200, 200)
        data = np.full((200, 200), 500.0)
        warnings = check_seam_quality(data, profile)
        assert warnings == []

    def test_blended_dem_passes(self):
        from ingest.dem_preprocess import blend_tile_seams, check_seam_quality

        data, profile = _seam_dem(jump=20.0)
        blended = blend_tile_seams(data, profile)
        # After blending a modest jump, the QA should pass.
        warnings = check_seam_quality(blended, profile)
        assert warnings == [], f"Expected clean QA after blending, got: {warnings}"
