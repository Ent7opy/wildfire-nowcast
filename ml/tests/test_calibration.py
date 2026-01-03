import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.isotonic import IsotonicRegression

from ml.calibration import SpreadProbabilityCalibrator, fit_from_hindcast_run


def test_calibrator_monotonicity_and_range():
    """Verify that calibration preserves ordering and stays in [0, 1]."""
    # Create a simple monotone mapping: y = x^2 (miscalibrated)
    x = np.linspace(0, 1, 100)
    y = x**2
    
    # Fit isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    
    calibrator = SpreadProbabilityCalibrator(
        method="isotonic",
        per_horizon_models={24: iso}
    )
    
    # Test random inputs
    raw = np.array([0.1, 0.5, 0.9, -0.1, 1.1, 0.5])
    calibrated = calibrator.calibrate_probs(raw, 24)
    
    # Range check
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    
    # Monotonicity check (excluding clamped values)
    valid_mask = (raw >= 0.0) & (raw <= 1.0)
    valid_raw = raw[valid_mask]
    valid_cal = calibrated[valid_mask]
    
    # Sort by raw to check monotone non-decreasing
    idx = np.argsort(valid_raw)
    assert np.all(np.diff(valid_cal[idx]) >= -1e-12)


def test_calibrator_save_load_roundtrip():
    """Verify that we can save and load a calibrator correctly."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Create dummy models
        iso24 = IsotonicRegression().fit([0, 1], [0.1, 0.9])
        iso48 = IsotonicRegression().fit([0, 1], [0.2, 0.8])
        
        metadata = {"test": "value", "run_id": "test_run"}
        calibrator = SpreadProbabilityCalibrator(
            method="isotonic",
            p_min=0.01,
            per_horizon_models={24: iso24, 48: iso48},
            metadata=metadata
        )
        
        calibrator.save(tmp_dir)
        
        # Check files exist
        assert (tmp_dir / "calibrator.pkl").exists()
        assert (tmp_dir / "metadata.json").exists()
        assert (tmp_dir / "config_resolved.yaml").exists()
        
        # Load back
        loaded = SpreadProbabilityCalibrator.load(tmp_dir)
        
        assert loaded.method == "isotonic"
        assert loaded.p_min == 0.01
        assert loaded.metadata["run_id"] == "test_run"
        assert 24 in loaded.per_horizon_models
        assert 48 in loaded.per_horizon_models
        
        # Verify predictions match
        raw = np.array([0.5])
        assert np.allclose(calibrator.calibrate_probs(raw, 24), loaded.calibrate_probs(raw, 24))
        
    finally:
        shutil.rmtree(tmp_dir)


def test_fit_from_hindcast_run_synthetic():
    """Test the full fit pipeline with a synthetic hindcast run."""
    tmp_root = Path(tempfile.mkdtemp())
    try:
        hindcast_dir = tmp_root / "hindcast_run"
        hindcast_dir.mkdir()
        
        # Create synthetic cases
        # Case 1 (T=0)
        ds1 = xr.Dataset(
            data_vars={
                "y_pred": (["time", "lat", "lon"], np.array([[[0.1, 0.6], [0.2, 0.7]]], dtype=np.float32)),
                "y_obs": (["time", "lat", "lon"], np.array([[[0, 1], [0, 1]]], dtype=np.float32)),
                "fire_t0": (["lat", "lon"], np.array([[1, 0], [0, 1]], dtype=np.float32)),
            },
            coords={
                "time": [datetime(2025, 1, 1)],
                "lat": [0, 1],
                "lon": [0, 1],
                "lead_time_hours": ("time", [24]),
            },
            attrs={"ref_time": "2025-01-01T00:00:00Z"}
        )
        ds1.to_netcdf(hindcast_dir / "case1.nc")
        
        # Case 2 (T=1)
        ds2 = xr.Dataset(
            data_vars={
                "y_pred": (["time", "lat", "lon"], np.array([[[0.2, 0.8], [0.3, 0.9]]], dtype=np.float32)),
                "y_obs": (["time", "lat", "lon"], np.array([[[0, 1], [1, 1]]], dtype=np.float32)),
                "fire_t0": (["lat", "lon"], np.array([[0, 1], [1, 0]], dtype=np.float32)),
            },
            coords={
                "time": [datetime(2025, 1, 2)],
                "lat": [0, 1],
                "lon": [0, 1],
                "lead_time_hours": ("time", [24]),
            },
            attrs={"ref_time": "2025-01-02T00:00:00Z"}
        )
        ds2.to_netcdf(hindcast_dir / "case2.nc")
        
        # Create index.json
        manifest = {
            "run_id": "test_hindcast",
            "cases": [
                {"path": str(hindcast_dir / "case1.nc")},
                {"path": str(hindcast_dir / "case2.nc")},
            ]
        }
        with open(hindcast_dir / "index.json", "w") as f:
            json.dump(manifest, f)
            
        # Run fit
        cal_root = tmp_root / "calibration"
        calibrator = fit_from_hindcast_run(
            hindcast_run_dir=hindcast_dir,
            method="isotonic",
            split_percentile=0.5, # Split such that Case 1 is train, Case 2 is eval
            out_root=str(cal_root)
        )
        
        assert 24 in calibrator.per_horizon_models
        
        # Check that calibration directory was created
        runs = list(cal_root.glob("*"))
        assert len(runs) == 1
        run_dir = runs[0]
        assert (run_dir / "calibrator.pkl").exists()
        assert (run_dir / "metrics.json").exists()
        
    finally:
        shutil.rmtree(tmp_root)

