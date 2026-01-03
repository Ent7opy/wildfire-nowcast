import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timezone
from dataclasses import dataclass

from ml.spread.heuristic_v0 import HeuristicSpreadModelV0, HeuristicSpreadV0Config
from ml.spread.contract import SpreadModelInput

@dataclass(frozen=True)
class MockGrid:
    crs: str
    cell_size_deg: float
    origin_lat: float
    origin_lon: float
    n_lat: int
    n_lon: int

@dataclass(frozen=True)
class MockWindow:
    lat: np.ndarray
    lon: np.ndarray
    i0: int = 0
    i1: int = 10
    j0: int = 0
    j1: int = 10

@dataclass(frozen=True)
class MockFireHeatmap:
    heatmap: np.ndarray
    grid: object = None
    window: object = None

@dataclass(frozen=True)
class MockTerrain:
    valid_data_mask: np.ndarray | None = None
    aoi_mask: np.ndarray | None = None
    slope: np.ndarray | None = None
    aspect: np.ndarray | None = None

def test_heuristic_v0_basic_contract():
    """Verify the model returns a valid SpreadForecast with correct shapes."""
    ny, nx = 20, 30
    lat = np.linspace(35.0, 35.2, ny)
    lon = np.linspace(5.0, 5.3, nx)
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)
    
    # One fire in the middle
    heatmap = np.zeros((ny, nx))
    heatmap[ny//2, nx//2] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)
    
    # Constant wind east
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.ones((ny, nx)) * 5.0),
            "v10": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"lat": lat, "lon": lon}
    )
    
    terrain = MockTerrain()
    
    grid = MockGrid(
        crs="EPSG:4326", 
        cell_size_deg=0.01, 
        origin_lat=35.0, 
        origin_lon=5.0, 
        n_lat=100, 
        n_lon=100
    )
    
    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=terrain,
        forecast_reference_time=datetime(2025, 12, 26, tzinfo=timezone.utc),
        horizons_hours=[24, 48]
    )
    
    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)
    
    # 1. Check dimensions
    assert forecast.probabilities.dims == ("time", "lat", "lon")
    assert forecast.probabilities.shape == (2, ny, nx)
    
    # 2. Check coordinates
    assert "time" in forecast.probabilities.coords
    assert "lat" in forecast.probabilities.coords
    assert "lon" in forecast.probabilities.coords
    assert "lead_time_hours" in forecast.probabilities.coords
    
    # 3. Validation from contract
    forecast.validate()
    
    # 4. Values in [0, 1]
    assert float(forecast.probabilities.min()) >= 0.0
    assert float(forecast.probabilities.max()) <= 1.0

def test_heuristic_v0_downwind_bias():
    """Verify that spread is biased towards the wind direction."""
    ny, nx = 21, 21
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)
    
    # Fire in center
    heatmap = np.zeros((ny, nx))
    center_i, center_j = ny//2, nx//2
    heatmap[center_i, center_j] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)
    
    # Strong wind to the East (u > 0)
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.ones((ny, nx)) * 10.0),
            "v10": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"lat": lat, "lon": lon}
    )
    
    grid = MockGrid(
        crs="EPSG:4326", cell_size_deg=0.01, origin_lat=35.0, origin_lon=5.0, n_lat=100, n_lon=100
    )
    
    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=MockTerrain(),
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24]
    )
    
    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)
    probs = forecast.probabilities.isel(time=0).values
    
    # Probability at center should be max (1.0)
    assert probs[center_i, center_j] == pytest.approx(1.0)
    
    # Compare point to the East vs point to the West at same distance
    east_val = probs[center_i, center_j + 5]
    west_val = probs[center_i, center_j - 5]
    
    assert east_val > west_val, f"East ({east_val}) should be > West ({west_val}) for eastward wind"
    
    # Compare North vs South (should be roughly equal for East wind)
    north_val = probs[center_i + 5, center_j]
    south_val = probs[center_i - 5, center_j]
    assert north_val == pytest.approx(south_val, rel=1e-3)

def test_heuristic_v0_horizon_scaling():
    """Verify that longer horizons produce larger footprints."""
    ny, nx = 41, 41
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)
    
    heatmap = np.zeros((ny, nx))
    heatmap[ny//2, nx//2] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)
    
    # Moderate wind
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.ones((ny, nx)) * 5.0),
            "v10": (("lat", "lon"), np.ones((ny, nx)) * 5.0),
        },
        coords={"lat": lat, "lon": lon}
    )
    
    grid = MockGrid(
        crs="EPSG:4326", cell_size_deg=0.01, origin_lat=35.0, origin_lon=5.0, n_lat=100, n_lon=100
    )
    
    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=MockTerrain(),
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[12, 24, 48]
    )
    
    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)
    
    # Sum of probabilities as a proxy for footprint area/intensity
    sum_12 = float(forecast.probabilities.isel(time=0).sum())
    sum_24 = float(forecast.probabilities.isel(time=1).sum())
    sum_48 = float(forecast.probabilities.isel(time=2).sum())
    
    assert sum_12 < sum_24 < sum_48

def test_heuristic_v0_empty_fires():
    """Verify behavior when no fires are present."""
    ny, nx = 10, 10
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)
    
    fires = MockFireHeatmap(heatmap=np.zeros((ny, nx)))
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.zeros((ny, nx))),
            "v10": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"lat": lat, "lon": lon}
    )
    grid = MockGrid(
        crs="EPSG:4326", cell_size_deg=0.01, origin_lat=35.0, origin_lon=5.0, n_lat=100, n_lon=100
    )
    
    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=MockTerrain(),
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24]
    )
    
    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)
    
    assert (forecast.probabilities.values == 0).all()

def test_heuristic_v0_masks():
    """Verify that terrain masks are respected."""
    ny, nx = 20, 20
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)
    
    heatmap = np.zeros((ny, nx))
    heatmap[ny//2, nx//2] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)
    
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.zeros((ny, nx))),
            "v10": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"lat": lat, "lon": lon}
    )
    
    # Mask out the northern half
    valid_mask = np.ones((ny, nx), dtype=bool)
    valid_mask[ny//2:, :] = False
    
    terrain = MockTerrain(valid_data_mask=valid_mask)
    
    grid = MockGrid(
        crs="EPSG:4326", cell_size_deg=0.01, origin_lat=35.0, origin_lon=5.0, n_lat=100, n_lon=100
    )
    
    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=terrain,
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24]
    )
    
    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)
    
    # Northern half should be 0
    assert (forecast.probabilities.values[0, ny//2:, :] == 0).all()
    # Southern half should have some non-zero
    assert (forecast.probabilities.values[0, :ny//2, :] > 0).any()


def test_heuristic_v0_weather_time_tz_aware_reference_time():
    """Model should be able to select weather by time even if forecast_reference_time is tz-aware.

    Weather ingest normalizes its `time` coord to tz-naive UTC `datetime64[ns]`. This test
    guards against tz-aware datetime selection issues in xarray/pandas.
    """
    ny, nx = 15, 15
    lat = np.linspace(35.0, 35.14, ny)
    lon = np.linspace(5.0, 5.14, nx)
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)

    heatmap = np.zeros((ny, nx))
    heatmap[ny // 2, nx // 2] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)

    # tz-aware reference time (UTC)
    ref_time = datetime(2025, 12, 26, 0, 0, 0, tzinfo=timezone.utc)

    # tz-naive datetime64 time coordinate (mirrors ingest.weather_ingest normalization)
    weather_times = np.array([np.datetime64("2025-12-27T00:00:00", "ns")])  # T+24h
    weather = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), np.ones((1, ny, nx)) * 8.0),
            "v10": (("time", "lat", "lon"), np.ones((1, ny, nx)) * 1.0),
        },
        coords={"time": weather_times, "lat": lat, "lon": lon},
    )

    grid = MockGrid(
        crs="EPSG:4326",
        cell_size_deg=0.01,
        origin_lat=35.0,
        origin_lon=5.0,
        n_lat=100,
        n_lon=100,
    )

    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=MockTerrain(),
        forecast_reference_time=ref_time,
        horizons_hours=[24],
    )

    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)
    forecast.validate()


def test_heuristic_v0_upslope_bias_no_wind():
    """With slope bias enabled and no wind, spread should be biased upslope.

    Terrain convention: aspect is downslope azimuth, clockwise from north.
    We set downslope = East (90°), so upslope = West.
    """
    ny, nx = 41, 41
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)

    heatmap = np.zeros((ny, nx))
    ci, cj = ny // 2, nx // 2
    heatmap[ci, cj] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)

    # No wind
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.zeros((ny, nx))),
            "v10": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"lat": lat, "lon": lon},
    )

    # Uniform moderate slope, downslope to East => upslope to West
    slope = np.ones((ny, nx), dtype=float) * 30.0
    aspect = np.ones((ny, nx), dtype=float) * 90.0
    terrain = MockTerrain(slope=slope, aspect=aspect)

    grid = MockGrid(
        crs="EPSG:4326",
        cell_size_deg=0.01,
        origin_lat=35.0,
        origin_lon=5.0,
        n_lat=100,
        n_lon=100,
    )

    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=terrain,
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24],
    )

    model = HeuristicSpreadModelV0(
        HeuristicSpreadV0Config(enable_slope_bias=True, slope_influence=0.5, slope_reference_deg=30.0)
    )
    forecast = model.predict(inputs)
    probs = forecast.probabilities.isel(time=0).values

    # Compare integrated probability mass West vs East (more robust than single-pixel
    # comparisons, since FFT convolution can introduce tiny signed numerical noise).
    west_mass = float(probs[:, :cj].sum())
    east_mass = float(probs[:, (cj + 1) :].sum())
    assert west_mass > east_mass, f"Expected upslope mass to dominate: west={west_mass} east={east_mass}"


def test_heuristic_v0_upslope_bias_aspect_wraparound_uses_circular_mean():
    """Aspect is circular; window mean must handle 0/360 wrap.

    We create a window with aspects split between 350° and 10° (both ~North downslope).
    The correct circular mean downslope direction is ~0° (North), so upslope should be ~180°
    (South). A naive arithmetic mean would give 180° (South) and flip upslope to North.
    """
    ny, nx = 41, 41
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)

    heatmap = np.zeros((ny, nx))
    ci, cj = ny // 2, nx // 2
    heatmap[ci, cj] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)

    # No wind; isolate slope/aspect effect.
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.zeros((ny, nx))),
            "v10": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"lat": lat, "lon": lon},
    )

    slope = np.ones((ny, nx), dtype=float) * 30.0
    aspect = np.ones((ny, nx), dtype=float) * 10.0
    aspect[:, ::2] = 350.0  # mix values across the 0/360 boundary
    terrain = MockTerrain(slope=slope, aspect=aspect)

    grid = MockGrid(
        crs="EPSG:4326",
        cell_size_deg=0.01,
        origin_lat=35.0,
        origin_lon=5.0,
        n_lat=100,
        n_lon=100,
    )

    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=terrain,
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24],
    )

    model = HeuristicSpreadModelV0(
        HeuristicSpreadV0Config(enable_slope_bias=True, slope_influence=0.6, slope_reference_deg=30.0)
    )
    forecast = model.predict(inputs)
    probs = forecast.probabilities.isel(time=0).values

    # With increasing i being "north" in these tests, upslope ~= South means the
    # southern half (lower i) should get more mass.
    north_mass = float(probs[(ci + 1) :, :].sum())
    south_mass = float(probs[:ci, :].sum())
    assert south_mass > north_mass, f"Expected upslope mass to dominate south: south={south_mass} north={north_mass}"


def test_heuristic_v0_coordinate_consistency():
    """Verify that output coordinates match input window exactly."""
    ny, nx = 15, 15
    lat = np.linspace(35.0, 35.14, ny)
    lon = np.linspace(5.0, 5.14, nx)
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)

    fires = MockFireHeatmap(heatmap=np.zeros((ny, nx)))
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.zeros((ny, nx))),
            "v10": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"lat": lat, "lon": lon},
    )
    grid = MockGrid(
        crs="EPSG:4326", cell_size_deg=0.01, origin_lat=35.0, origin_lon=5.0, n_lat=100, n_lon=100
    )
    
    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=MockTerrain(),
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24]
    )
    
    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)
    
    np.testing.assert_allclose(forecast.probabilities.coords["lat"].values, lat)
    np.testing.assert_allclose(forecast.probabilities.coords["lon"].values, lon)


def test_heuristic_v0_monotonic_footprint_behavior():
    """Verify monotonic footprint via the sanity check helper."""
    from ml.spread.sanity_checks import check_monotonic_footprint

    ny, nx = 51, 51
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)

    heatmap = np.zeros((ny, nx))
    heatmap[ny // 2, nx // 2] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)

    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.ones((ny, nx)) * 2.0),
            "v10": (("lat", "lon"), np.ones((ny, nx)) * 2.0),
        },
        coords={"lat": lat, "lon": lon},
    )
    grid = MockGrid(
        crs="EPSG:4326", cell_size_deg=0.01, origin_lat=35.0, origin_lon=5.0, n_lat=100, n_lon=100
    )
    
    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=MockTerrain(),
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24, 48, 72]
    )
    
    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)
    
    # Should not raise
    check_monotonic_footprint(forecast)


def test_heuristic_v0_wind_behavior():
    """Verify wind-driven behavior (displacement) via the sanity check helper."""
    from ml.spread.sanity_checks import check_wind_displacement

    ny, nx = 81, 81
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)

    heatmap = np.zeros((ny, nx))
    heatmap[ny // 2, nx // 2] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)

    # Strong North wind (v > 0, u = 0)
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.zeros((ny, nx))),
            "v10": (("lat", "lon"), np.ones((ny, nx)) * 12.0),
        },
        coords={"lat": lat, "lon": lon},
    )
    grid = MockGrid(
        crs="EPSG:4326", cell_size_deg=0.01, origin_lat=35.0, origin_lon=5.0, n_lat=100, n_lon=100
    )
    
    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=MockTerrain(),
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[6]
    )
    
    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)
    
    # Should not raise for North wind
    check_wind_displacement(inputs, forecast, min_wind_speed_ms=10.0, min_displacement_px=0.5)


def test_heuristic_v0_nonempty_check():
    """Verify check_nonempty_when_fires_present."""
    from ml.spread.sanity_checks import check_nonempty_when_fires_present

    ny, nx = 20, 20
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)

    # Fires present
    heatmap = np.zeros((ny, nx))
    heatmap[ny // 2, nx // 2] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)

    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.zeros((ny, nx))),
            "v10": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"lat": lat, "lon": lon},
    )
    grid = MockGrid(
        crs="EPSG:4326", cell_size_deg=0.01, origin_lat=35.0, origin_lon=5.0, n_lat=100, n_lon=100
    )

    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=MockTerrain(),
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24],
    )

    model = HeuristicSpreadModelV0()
    forecast = model.predict(inputs)

    # Should not raise
    check_nonempty_when_fires_present(inputs, forecast)


def test_heuristic_v0_wind_elongation():
    """Verify check_wind_elongation."""
    from ml.spread.sanity_checks import check_wind_elongation

    # Need a larger grid for PCA to be stable and clear
    ny, nx = 101, 101
    lat = np.arange(ny) * 0.01 + 35.0
    lon = np.arange(nx) * 0.01 + 5.0
    window = MockWindow(lat=lat, lon=lon, i1=ny, j1=nx)

    heatmap = np.zeros((ny, nx))
    heatmap[ny // 2, nx // 2] = 1.0
    fires = MockFireHeatmap(heatmap=heatmap)

    # Strong East wind
    weather = xr.Dataset(
        data_vars={
            "u10": (("lat", "lon"), np.ones((ny, nx)) * 15.0),
            "v10": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"lat": lat, "lon": lon},
    )
    grid = MockGrid(
        crs="EPSG:4326", cell_size_deg=0.01, origin_lat=35.0, origin_lon=5.0, n_lat=100, n_lon=100
    )

    inputs = SpreadModelInput(
        grid=grid,
        window=window,
        active_fires=fires,
        weather_cube=weather,
        terrain=MockTerrain(),
        forecast_reference_time=datetime.now(timezone.utc),
        horizons_hours=[24],
    )

    model = HeuristicSpreadModelV0(
        HeuristicSpreadV0Config(wind_elongation_factor=2.0, wind_influence_km_h_per_ms=0.2)
    )
    forecast = model.predict(inputs)

    # Should not raise
    check_wind_elongation(inputs, forecast, min_wind_speed_ms=10.0, min_anisotropy=1.2)