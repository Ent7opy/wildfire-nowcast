import numpy as np

from api.core.grid import (
    DEFAULT_CELL_SIZE_DEG,
    DEFAULT_CRS,
    GridSpec,
    bbox_to_window,
    grid_bounds,
    grid_coords,
    index_to_latlon,
    latlon_to_index,
    window_coords,
)


def test_grid_defaults_match_docs():
    assert DEFAULT_CRS == "EPSG:4326"
    assert DEFAULT_CELL_SIZE_DEG == 0.01
    grid = GridSpec()
    assert grid.crs == "EPSG:4326"
    assert grid.cell_size_deg == 0.01


def test_roundtrip_index_latlon_index():
    grid = GridSpec.from_bbox(lat_min=10.0, lat_max=11.0, lon_min=20.0, lon_max=21.0)

    # A few representative indices
    indices = [(0, 0), (1, 2), (50, 75), (grid.n_lat - 1, grid.n_lon - 1)]
    for i, j in indices:
        lat, lon = index_to_latlon(grid, i, j)
        ii, jj = latlon_to_index(grid, float(lat), float(lon))
        assert int(ii) == i
        assert int(jj) == j


def test_bbox_to_window_and_window_coords_shapes_and_monotonic():
    grid = GridSpec.from_bbox(lat_min=10.0, lat_max=11.0, lon_min=20.0, lon_max=21.0)

    # bbox fully inside grid
    min_lon, min_lat, max_lon, max_lat = (20.1, 10.2, 20.5, 10.7)
    i0, i1, j0, j1 = bbox_to_window(grid, min_lon, min_lat, max_lon, max_lat)

    assert 0 <= i0 < i1 <= grid.n_lat
    assert 0 <= j0 < j1 <= grid.n_lon

    lat_w, lon_w = window_coords(grid, i0, i1, j0, j1)
    assert len(lat_w) == (i1 - i0)
    assert len(lon_w) == (j1 - j0)

    assert np.all(np.diff(lat_w) > 0)
    assert np.all(np.diff(lon_w) > 0)


def test_bbox_to_window_clip_outside_grid():
    grid = GridSpec.from_bbox(lat_min=10.0, lat_max=11.0, lon_min=20.0, lon_max=21.0)

    # bbox extends beyond grid; clip should clamp indices to [0, n]
    i0, i1, j0, j1 = bbox_to_window(grid, 19.0, 9.0, 22.0, 12.0, clip=True)
    assert i0 == 0
    assert j0 == 0
    assert i1 == grid.n_lat
    assert j1 == grid.n_lon


def test_grid_coords_match_window_coords_full_extent():
    grid = GridSpec.from_bbox(lat_min=10.0, lat_max=11.0, lon_min=20.0, lon_max=21.0)
    lat, lon = grid_coords(grid)
    lat2, lon2 = window_coords(grid, 0, grid.n_lat, 0, grid.n_lon)
    assert np.allclose(lat, lat2)
    assert np.allclose(lon, lon2)


def test_boundary_behavior_max_edges_are_out_of_bounds_for_points():
    grid = GridSpec.from_bbox(lat_min=10.0, lat_max=11.0, lon_min=20.0, lon_max=21.0)
    min_lon, min_lat, max_lon, max_lat = grid_bounds(grid)

    # Min edges: inclusive (floor -> 0)
    i0, j0 = latlon_to_index(grid, min_lat, min_lon)
    assert int(i0) == 0
    assert int(j0) == 0

    # Max edges: out-of-bounds (half-open extent)
    i1, j1 = latlon_to_index(grid, max_lat, max_lon)
    assert int(i1) == grid.n_lat
    assert int(j1) == grid.n_lon

    # Mixed edges behave deterministically
    i2, j2 = latlon_to_index(grid, max_lat, min_lon)
    assert int(i2) == grid.n_lat
    assert int(j2) == 0


def test_bbox_to_window_max_edge_is_full_extent_window():
    grid = GridSpec.from_bbox(lat_min=10.0, lat_max=11.0, lon_min=20.0, lon_max=21.0)
    min_lon, min_lat, max_lon, max_lat = grid_bounds(grid)

    # Full extent window should return [0:n] exactly even when bbox equals max edges.
    i0, i1, j0, j1 = bbox_to_window(grid, min_lon, min_lat, max_lon, max_lat, clip=True)
    assert (i0, i1, j0, j1) == (0, grid.n_lat, 0, grid.n_lon)


