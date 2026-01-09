import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from ml.calibration import fit_from_hindcast_run
from ml.eval_spread_calibration import run_eval as run_cal_eval
from ml.eval_weather_bias_correction import run_eval as run_bias_eval
from ml.weather_bias_correction import WeatherBiasCorrector


def test_eval_spread_calibration_produces_tables_and_plots():
    tmp_root = Path(tempfile.mkdtemp())
    try:
        hindcast_dir = tmp_root / "hindcast"
        hindcast_dir.mkdir()

        # Build a synthetic hindcast where y_obs ~ Bernoulli(y_pred^2) => raw is overconfident.
        rng = np.random.default_rng(0)
        horizons = [24]

        cases = []
        for day in range(4):
            # One horizon, large grid for stable calibration.
            y_pred = rng.uniform(0.0, 1.0, size=(1, 30, 30)).astype(np.float32)
            p_true = np.clip(y_pred**2, 0.0, 1.0)
            y_obs = (rng.uniform(0.0, 1.0, size=p_true.shape) < p_true).astype(np.float32)
            fire_t0 = (rng.uniform(0.0, 1.0, size=(30, 30)) < 0.05).astype(np.float32)

            ref = datetime(2025, 1, 1 + day)
            ds = xr.Dataset(
                data_vars={
                    "y_pred": (["time", "lat", "lon"], y_pred),
                    "y_obs": (["time", "lat", "lon"], y_obs),
                    "fire_t0": (["lat", "lon"], fire_t0),
                },
                coords={
                    "time": [ref],
                    "lat": np.arange(30, dtype=float),
                    "lon": np.arange(30, dtype=float),
                    "lead_time_hours": ("time", horizons),
                },
                attrs={"ref_time": ref.isoformat()},
            )

            path = hindcast_dir / f"case_{day}.nc"
            ds.to_netcdf(path)
            cases.append({"path": str(path)})

        (hindcast_dir / "index.json").write_text(
            json.dumps({"run_id": "synthetic", "cases": cases}, indent=2), encoding="utf-8"
        )

        # Train calibrator (train on first half, eval on second half in fit routine).
        cal_root = tmp_root / "cal"
        calibrator = fit_from_hindcast_run(
            hindcast_run_dir=hindcast_dir,
            method="isotonic",
            split_percentile=0.5,
            out_root=str(cal_root),
        )

        run_dirs = list(cal_root.glob("*"))
        assert len(run_dirs) == 1
        cal_run_dir = run_dirs[0]

        # Evaluate (raw vs calibrated) and assert artifacts exist.
        args = type(
            "Args",
            (),
            {
                "hindcast_run_dir": str(hindcast_dir),
                "calibrator_run_dir": str(cal_run_dir),
                "p_min": calibrator.p_min,
                "n_bins": 10,
                "out_dir": str(tmp_root / "reports"),
            },
        )()
        out_dir = run_cal_eval(args)

        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "plots" / "reliability_h024.png").exists()

        payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        # At least one horizon summary row, and it should report Brier/ECE.
        assert payload["summary"]
        row = payload["summary"][0]
        assert "brier_raw" in row and "brier_cal" in row
        assert "ece_raw" in row and "ece_cal" in row

    finally:
        shutil.rmtree(tmp_root)


def test_eval_weather_bias_correction_reports_improvement():
    tmp_root = Path(tempfile.mkdtemp())
    try:
        # Synthetic forecast/truth datasets (aligned).
        times = np.array([np.datetime64("2025-01-01T00:00:00", "ns"), np.datetime64("2025-01-01T01:00:00", "ns")])
        lat = np.linspace(35.0, 35.09, 10)
        lon = np.linspace(5.0, 5.09, 10)

        rng = np.random.default_rng(1)
        base = rng.normal(size=(2, 10, 10)).astype(np.float32)
        fc = xr.Dataset(
            data_vars={
                "u10": (("time", "lat", "lon"), base + 5.0),
                "v10": (("time", "lat", "lon"), base - 2.0),
            },
            coords={"time": times, "lat": lat, "lon": lon},
        )
        tr = xr.Dataset(
            data_vars={
                "u10": (("time", "lat", "lon"), 1.0 + 0.9 * fc["u10"].values),
                "v10": (("time", "lat", "lon"), -0.5 + 1.1 * fc["v10"].values),
            },
            coords={"time": times, "lat": lat, "lon": lon},
        )

        # Fit corrector on the same data (sufficient for an evaluation smoke test).
        corrector = WeatherBiasCorrector.fit(forecast=fc, truth=tr, variables=["u10", "v10"])

        forecast_nc = tmp_root / "forecast.nc"
        truth_nc = tmp_root / "truth.nc"
        corrector_json = tmp_root / "weather_bias_corrector.json"
        fc.to_netcdf(forecast_nc)
        tr.to_netcdf(truth_nc)
        corrector.save_json(corrector_json)

        args = type(
            "Args",
            (),
            {
                "forecast_nc": forecast_nc,
                "truth_nc": str(truth_nc),
                "corrector_json": corrector_json,
                "variables": None,
                "vars": ["u10", "v10"],
                "out_dir": tmp_root / "reports_bias",
            },
        )()
        out_dir = run_bias_eval(args)

        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "plots" / "bias_map_reduction_u10.png").exists()

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        rows = summary["summary"]
        assert rows
        # Bias and RMSE should improve for at least one variable in this construction.
        assert any(r["rmse_reduction"] > 0 for r in rows)

    finally:
        shutil.rmtree(tmp_root)

