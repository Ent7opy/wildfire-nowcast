import numpy as np
import pytest
import xarray as xr

from ml.weather_bias_analysis import compute_metrics
from ml.weather_bias_correction import WeatherBiasCorrector


def _make_synthetic_forecast_truth(*, seed: int = 0) -> tuple[xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(seed)

    times = np.array(
        [
            np.datetime64("2025-01-01T00:00:00", "ns"),
            np.datetime64("2025-01-01T01:00:00", "ns"),
            np.datetime64("2025-01-01T02:00:00", "ns"),
            np.datetime64("2025-01-01T03:00:00", "ns"),
            np.datetime64("2025-01-01T04:00:00", "ns"),
            np.datetime64("2025-01-01T05:00:00", "ns"),
        ]
    )
    lat = np.linspace(35.0, 35.09, 10)
    lon = np.linspace(5.0, 5.09, 10)

    shape = (times.size, lat.size, lon.size)
    base = rng.normal(loc=0.0, scale=3.0, size=shape).astype(np.float32)

    # Forecast fields
    fc = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), base + 5.0),
            "v10": (("time", "lat", "lon"), base - 2.0),
            "t2m": (("time", "lat", "lon"), base * 2.0 + 290.0),
            "rh2m": (("time", "lat", "lon"), base * 5.0 + 60.0),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )

    # Truth is an affine transform of forecast + small noise.
    noise = rng.normal(loc=0.0, scale=0.2, size=shape).astype(np.float32)
    tr = xr.Dataset(
        data_vars={
            # Positive bias + mild scaling error
            "u10": (("time", "lat", "lon"), 1.25 + 0.85 * fc["u10"].values + noise),
            "v10": (("time", "lat", "lon"), -0.75 + 1.10 * fc["v10"].values + noise),
            "t2m": (("time", "lat", "lon"), 2.0 + 0.98 * fc["t2m"].values + noise),
            # RH is bounded in [0, 100]; include values that would overshoot without clamping.
            "rh2m": (("time", "lat", "lon"), 5.0 + 1.05 * fc["rh2m"].values + noise * 2.0),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    return fc, tr


def test_weather_bias_corrector_reduces_validation_error():
    fc, tr = _make_synthetic_forecast_truth(seed=123)

    # Time-based split: fit on first 4, validate on last 2.
    fc_train = fc.isel(time=slice(0, 4))
    tr_train = tr.isel(time=slice(0, 4))
    fc_val = fc.isel(time=slice(4, None))
    tr_val = tr.isel(time=slice(4, None))

    corrector = WeatherBiasCorrector.fit(
        forecast=fc_train,
        truth=tr_train,
        variables=["u10", "v10", "t2m", "rh2m"],
    )

    fc_val_corr = corrector.apply(fc_val)

    for v in ["u10", "v10", "t2m"]:
        m_uncorr = compute_metrics(fc_val[v].values, tr_val[v].values)
        m_corr = compute_metrics(fc_val_corr[v].values, tr_val[v].values)

        # Should improve bias and overall error on held-out data.
        assert abs(m_corr["bias_mean"]) < abs(m_uncorr["bias_mean"])
        assert m_corr["rmse"] < m_uncorr["rmse"]
        assert m_corr["mae"] < m_uncorr["mae"]

    # RH should be clamped to a physical range.
    rh = fc_val_corr["rh2m"].values
    assert np.isfinite(rh).any()
    assert float(np.nanmin(rh)) >= 0.0
    assert float(np.nanmax(rh)) <= 100.0


def test_weather_bias_corrector_save_load_roundtrip(tmp_path):
    fc, tr = _make_synthetic_forecast_truth(seed=7)
    corrector = WeatherBiasCorrector.fit(forecast=fc, truth=tr, variables=["u10", "rh2m"])

    out_path = tmp_path / "weather_bias_corrector.json"
    corrector.save_json(out_path)
    loaded = WeatherBiasCorrector.load_json(out_path)

    applied_1 = corrector.apply(fc)
    applied_2 = loaded.apply(fc)

    assert np.allclose(applied_1["u10"].values, applied_2["u10"].values, rtol=0.0, atol=0.0)
    assert np.allclose(applied_1["rh2m"].values, applied_2["rh2m"].values, rtol=0.0, atol=0.0)

