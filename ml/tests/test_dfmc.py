"""Tests for equilibrium DFMC computation in spread_features."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ml.spread_features import _compute_dfmc, _add_dfmc_to_weather


# ---------------------------------------------------------------------------
# _compute_dfmc
# ---------------------------------------------------------------------------

def test_dfmc_returns_float32():
    t = np.array([293.15], dtype=np.float32)  # 20°C
    rh = np.array([50.0], dtype=np.float32)
    result = _compute_dfmc(t, rh)
    assert result.dtype == np.float32


def test_dfmc_high_rh_suppresses():
    """High humidity → high DFMC → fire suppression range."""
    t = np.array([293.15])  # 20°C
    rh_dry = np.array([20.0])
    rh_wet = np.array([90.0])
    dfmc_dry = _compute_dfmc(t, rh_dry)
    dfmc_wet = _compute_dfmc(t, rh_wet)
    assert dfmc_wet > dfmc_dry, "Wet conditions should yield higher DFMC"


def test_dfmc_high_temp_dries():
    """High temperature reduces EMC."""
    rh = np.array([50.0])
    t_cool = np.array([283.15])   # 10°C
    t_hot  = np.array([313.15])   # 40°C
    dfmc_cool = _compute_dfmc(t_cool, rh)
    dfmc_hot  = _compute_dfmc(t_hot,  rh)
    assert dfmc_cool > dfmc_hot, "Hot conditions should yield lower DFMC"


def test_dfmc_range():
    """Output must be in [0, 0.40] for all realistic inputs."""
    t  = np.linspace(253.15, 333.15, 50)   # -20°C to 60°C
    rh = np.linspace(0.0, 100.0, 50)
    T, RH = np.meshgrid(t, rh)
    result = _compute_dfmc(T, RH)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 0.40


def test_dfmc_known_value():
    """Check against a hand-calculated reference.

    At T=20°C (68°F), RH=50%: mid-range formula gives
    EMC = 2.22749 + 0.160107*50 - 0.014784*68 ≈ 9.228% → 0.09228 as fraction.
    """
    t = np.array([293.15])   # 20°C → 68°F
    rh = np.array([50.0])
    result = float(_compute_dfmc(t, rh).item())
    expected = (2.22749 + 0.160107 * 50.0 - 0.014784 * 68.0) / 100.0
    assert result == pytest.approx(expected, abs=0.005)


# ---------------------------------------------------------------------------
# _add_dfmc_to_weather
# ---------------------------------------------------------------------------

def _make_weather_cube(t2m_k: float = 293.15, rh2m_pct: float = 50.0) -> xr.Dataset:
    shape = (2, 5, 5)
    return xr.Dataset(
        {
            "u10":  (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
            "v10":  (("time", "lat", "lon"), np.zeros(shape, dtype=np.float32)),
            "t2m":  (("time", "lat", "lon"), np.full(shape, t2m_k,      dtype=np.float32)),
            "rh2m": (("time", "lat", "lon"), np.full(shape, rh2m_pct,   dtype=np.float32)),
        }
    )


def test_add_dfmc_attaches_variable():
    ds = _make_weather_cube()
    result = _add_dfmc_to_weather(ds)
    assert "dfmc" in result.data_vars


def test_add_dfmc_shape_preserved():
    ds = _make_weather_cube()
    result = _add_dfmc_to_weather(ds)
    assert result["dfmc"].shape == ds["t2m"].shape


def test_add_dfmc_values_in_range():
    ds = _make_weather_cube(t2m_k=300.0, rh2m_pct=60.0)
    result = _add_dfmc_to_weather(ds)
    arr = result["dfmc"].values
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) <= 0.40


def test_add_dfmc_nan_when_t2m_all_nan():
    ds = _make_weather_cube()
    ds["t2m"].values[:] = np.nan
    result = _add_dfmc_to_weather(ds)
    assert np.all(np.isnan(result["dfmc"].values))


def test_add_dfmc_missing_t2m():
    ds = xr.Dataset(
        {
            "u10":  (("time", "lat", "lon"), np.zeros((1, 3, 3), dtype=np.float32)),
            "v10":  (("time", "lat", "lon"), np.zeros((1, 3, 3), dtype=np.float32)),
            # t2m and rh2m deliberately absent
        }
    )
    result = _add_dfmc_to_weather(ds)
    assert "dfmc" in result.data_vars
    assert np.all(np.isnan(result["dfmc"].values))
