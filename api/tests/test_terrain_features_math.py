import numpy as np
import pytest

from api.terrain.features_math import compute_slope_aspect


def test_aspect_convention_downslope_east_gradient_points_west():
    # Elevation increases to the east (positive x), so downslope points west (270°).
    h, w = 20, 30
    z = np.tile(np.arange(w, dtype=float)[None, :], (h, 1))
    lat_centers = np.linspace(10.0, 9.0, h)  # north-up raster (rows go south)
    slope, aspect = compute_slope_aspect(z, cell_size_deg=0.01, lat_centers_deg=lat_centers)

    # Ignore edges where gradient uses one-sided differences.
    core = (slice(2, -2), slice(2, -2))
    aspect_core = aspect[core]
    assert np.isfinite(aspect_core).all()
    assert float(np.nanmedian(aspect_core)) == pytest.approx(270.0, abs=1.0)


def test_aspect_convention_downslope_north_gradient_points_south():
    # Elevation increases to the north, so downslope points south (180°).
    h, w = 20, 20
    lat_centers = np.linspace(10.0, 9.0, h)  # decreasing by row
    z = lat_centers[:, None] * np.ones((h, w), dtype=float)
    slope, aspect = compute_slope_aspect(z, cell_size_deg=0.01, lat_centers_deg=lat_centers)

    core = (slice(2, -2), slice(2, -2))
    aspect_core = aspect[core]
    assert np.isfinite(aspect_core).all()
    assert float(np.nanmedian(aspect_core)) == pytest.approx(180.0, abs=1.0)


def test_flat_surface_has_nan_aspect():
    h, w = 10, 10
    z = np.zeros((h, w), dtype=float)
    lat_centers = np.linspace(10.0, 9.0, h)
    slope, aspect = compute_slope_aspect(z, cell_size_deg=0.01, lat_centers_deg=lat_centers)
    assert np.nanmax(slope) == pytest.approx(0.0)
    assert np.isnan(aspect).all()

