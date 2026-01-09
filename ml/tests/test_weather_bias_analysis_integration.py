
import pytest
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta
import logging
from argparse import Namespace

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.weather_bias_analysis import run_analysis  # noqa: E402

def create_dummy_nc(path: Path, time_start, steps=5):
    lats = np.linspace(30, 40, 10)
    lons = np.linspace(10, 20, 10)
    times = [time_start + timedelta(hours=i) for i in range(steps)]
    
    # Use deterministic data
    np.random.seed(42)
    data = np.random.rand(len(times), len(lats), len(lons))
    
    ds = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), data + 5),
            "v10": (("time", "lat", "lon"), data - 2),
            "t2m": (("time", "lat", "lon"), data * 10 + 290),
            "rh2m": (("time", "lat", "lon"), data * 50 + 20),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        }
    )
    ds.to_netcdf(path)
    return ds

@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "weather_bias_test"
    d.mkdir()
    yield d
    # Teardown logging to release file locks
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)

def test_run_analysis_end_to_end(run_dir):
    forecast_path = run_dir / "forecast.nc"
    truth_path = run_dir / "truth.nc"
    out_dir = run_dir / "output"
    
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    
    # Create forecast
    create_dummy_nc(forecast_path, start_time)
    
    # Create truth (same grid, slightly different values for bias)
    # Truth = Forecast - 1.0 => Bias = +1.0
    ds_truth = create_dummy_nc(truth_path, start_time)
    ds_truth["u10"] = ds_truth["u10"] - 1.0
    ds_truth["v10"] = ds_truth["v10"] - 0.5
    ds_truth.to_netcdf(truth_path)
    
    # Setup args
    args = Namespace(
        forecast_nc=forecast_path,
        truth_nc=str(truth_path),
        variables="u10=u10,v10=v10", # explicit mapping test
        out_dir=out_dir,
        dem_path=None,
        bbox=None
    )
    
    # Run analysis
    run_analysis(args)
    
    # Check outputs
    runs = list(out_dir.glob("*"))
    assert len(runs) == 1
    this_run = runs[0]
    
    assert (this_run / "summary.csv").exists()
    assert (this_run / "summary.json").exists()
    assert (this_run / "plots").exists()
    
    # Check bias correctness
    df = pd.read_csv(this_run / "summary.csv")
    
    # Check u10 bias (should be approx 1.0)
    u10_row = df[df["variable"] == "u10"].iloc[0]
    assert u10_row["bias_mean"] == pytest.approx(1.0, abs=1e-5)
    
    # Check v10 bias (should be approx 0.5)
    v10_row = df[df["variable"] == "v10"].iloc[0]
    assert v10_row["bias_mean"] == pytest.approx(0.5, abs=1e-5)

    # Check that wind_speed was derived and computed
    assert "wind_speed" in df["variable"].values

