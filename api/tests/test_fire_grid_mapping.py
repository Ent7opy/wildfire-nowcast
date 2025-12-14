import numpy as np

from api.core.grid import GridSpec
from api.fires.grid_mapping import aggregate_to_grid, fires_to_indices, normalize_lon


def test_fires_to_indices_known_points():
    grid = GridSpec.from_bbox(lat_min=0.0, lat_max=0.03, lon_min=0.0, lon_max=0.03, cell_size_deg=0.01)

    fires = [
        {"lat": 0.005, "lon": 0.005},
        {"lat": 0.015, "lon": 0.025},
        {"lat": 0.029, "lon": 0.001},
    ]
    mapped = fires_to_indices(fires, grid)

    assert [(r["i"], r["j"]) for r in mapped] == [(0, 0), (1, 2), (2, 0)]

    # stability across runs
    mapped2 = fires_to_indices(fires, grid)
    assert [(r["i"], r["j"]) for r in mapped2] == [(0, 0), (1, 2), (2, 0)]


def test_boundary_behavior_upper_edge_is_out_of_bounds():
    grid = GridSpec.from_bbox(lat_min=0.0, lat_max=0.03, lon_min=0.0, lon_max=0.03, cell_size_deg=0.01)

    # upper boundary: origin + n * cell => out-of-bounds (half-open)
    fires = [{"lat": 0.03, "lon": 0.005}]
    mapped_all = fires_to_indices(fires, grid, drop_outside=False)
    assert mapped_all[0]["in_bounds"] is False

    mapped_dropped = fires_to_indices(fires, grid, drop_outside=True)
    assert mapped_dropped == []

    # exact cell boundary inside the grid is assigned by floor
    fires2 = [{"lat": 0.01, "lon": 0.02}]
    mapped2 = fires_to_indices(fires2, grid)
    assert mapped2[0]["i"] == 1
    assert mapped2[0]["j"] == 2


def test_aggregation_count_presence_sum_and_max():
    grid = GridSpec.from_bbox(lat_min=0.0, lat_max=0.03, lon_min=0.0, lon_max=0.03, cell_size_deg=0.01)

    fires = [
        {"lat": 0.005, "lon": 0.005, "frp": 10.0},
        {"lat": 0.006, "lon": 0.006, "frp": 2.5},  # same cell as first
        {"lat": 0.015, "lon": 0.025, "frp": 7.0},
    ]
    mapped = fires_to_indices(fires, grid)

    count = aggregate_to_grid(mapped, grid, mode="count")
    assert count.shape == (grid.n_lat, grid.n_lon)
    assert int(count[0, 0]) == 2
    assert int(count[1, 2]) == 1

    presence = aggregate_to_grid(mapped, grid, mode="presence")
    assert int(presence[0, 0]) == 1
    assert int(presence[1, 2]) == 1

    summed = aggregate_to_grid(mapped, grid, mode="sum", value_col="frp")
    assert np.isclose(float(summed[0, 0]), 12.5)
    assert np.isclose(float(summed[1, 2]), 7.0)

    maxed = aggregate_to_grid(mapped, grid, mode="max", value_col="frp")
    assert np.isclose(float(maxed[0, 0]), 10.0)
    assert np.isclose(float(maxed[1, 2]), 7.0)


def test_lon_normalization_wraps_to_minus_180_180():
    assert normalize_lon(190.0) == -170.0

    grid = GridSpec.from_bbox(
        lat_min=0.0,
        lat_max=10.0,
        lon_min=-180.0,
        lon_max=-160.0,
        cell_size_deg=1.0,
    )
    fires = [{"lat": 5.0, "lon": 190.0}]
    mapped = fires_to_indices(fires, grid, drop_outside=False, normalize_lons=True)
    assert mapped[0]["in_bounds"] is True
    assert mapped[0]["i"] == 5
    assert mapped[0]["j"] == 10

