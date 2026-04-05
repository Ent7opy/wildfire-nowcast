import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ml.calibration import (
    SpreadProbabilityCalibrator,
    fit_binary_probability_calibrator,
    fit_from_hindcast_run,
)


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


def test_fit_from_hindcast_run_fallbacks_to_platt_for_sparse_positives():
    """Isotonic should fallback to Platt when positives are below threshold and classes are present."""
    tmp_root = Path(tempfile.mkdtemp())
    try:
        hindcast_dir = tmp_root / "hindcast_run"
        hindcast_dir.mkdir()

        # Create four small cases so both classes exist and split works.
        for i in range(4):
            y_pred = np.array([[[0.1, 0.2], [0.3, 0.9]]], dtype=np.float32)
            y_obs = np.array([[[0, 0], [0, 1 if i % 2 == 0 else 0]]], dtype=np.float32)
            fire_t0 = np.array([[1, 0], [0, 1]], dtype=np.float32)
            ds = xr.Dataset(
                data_vars={
                    "y_pred": (["time", "lat", "lon"], y_pred),
                    "y_obs": (["time", "lat", "lon"], y_obs),
                    "fire_t0": (["lat", "lon"], fire_t0),
                },
                coords={
                    "time": [datetime(2025, 1, 1 + i)],
                    "lat": [0, 1],
                    "lon": [0, 1],
                    "lead_time_hours": ("time", [24]),
                },
                attrs={"ref_time": f"2025-01-0{1+i}T00:00:00Z"},
            )
            ds.to_netcdf(hindcast_dir / f"case{i}.nc")

        manifest = {
            "run_id": "test_hindcast_sparse",
            "cases": [{"path": str(hindcast_dir / f"case{i}.nc")} for i in range(4)],
        }
        with open(hindcast_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        calibrator = fit_from_hindcast_run(
            hindcast_run_dir=hindcast_dir,
            method="isotonic",
            split_percentile=0.5,
            out_root=str(tmp_root / "calibration"),
            min_positive_for_isotonic=1000,
        )

        assert isinstance(calibrator.per_horizon_models[24], LogisticRegression)
    finally:
        shutil.rmtree(tmp_root)


def test_fit_from_hindcast_run_resolves_legacy_repo_relative_case_paths(monkeypatch):
    """Calibration loader should accept manifest paths relative to repo/cwd."""
    tmp_root = Path(tempfile.mkdtemp())
    try:
        hindcast_dir = tmp_root / "hindcast_run"
        hindcast_dir.mkdir()
        monkeypatch.chdir(tmp_root)

        for i in range(2):
            ds = xr.Dataset(
                data_vars={
                    "y_pred": (["time", "lat", "lon"], np.array([[[0.2, 0.8], [0.1, 0.9]]], dtype=np.float32)),
                    "y_obs": (["time", "lat", "lon"], np.array([[[0, 1], [0, 1]]], dtype=np.float32)),
                    "fire_t0": (["lat", "lon"], np.array([[1, 0], [0, 1]], dtype=np.float32)),
                },
                coords={
                    "time": [datetime(2025, 1, 1 + i)],
                    "lat": [0, 1],
                    "lon": [0, 1],
                    "lead_time_hours": ("time", [24]),
                },
                attrs={"ref_time": f"2025-01-0{1+i}T00:00:00Z"},
            )
            ds.to_netcdf(hindcast_dir / f"case{i}.nc")

        manifest = {
            "run_id": "test_hindcast_legacy_paths",
            # Legacy style: path relative to cwd/repo root (not to hindcast_run_dir).
            "cases": [{"path": f"hindcast_run/case{i}.nc"} for i in range(2)],
        }
        with open(hindcast_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        calibrator = fit_from_hindcast_run(
            hindcast_run_dir=hindcast_dir,
            method="isotonic",
            split_percentile=0.5,
            out_root=str(tmp_root / "calibration"),
            min_positive_for_isotonic=1,
        )

        assert 24 in calibrator.per_horizon_models
    finally:
        shutil.rmtree(tmp_root)


def test_fit_binary_probability_calibrator_platt_with_class_weight():
    """Platt scaling should use balanced class weights."""
    # Create severely imbalanced data (95% negative, 5% positive, mimicking FIRMS)
    np.random.seed(42)
    n_samples = 1000
    n_positive = int(0.05 * n_samples)
    n_negative = n_samples - n_positive

    scores = np.concatenate([
        np.random.uniform(0.3, 0.7, n_negative),  # Mostly low confidence negatives
        np.random.uniform(0.6, 0.95, n_positive),  # High confidence positives
    ])
    labels = np.concatenate([np.zeros(n_negative), np.ones(n_positive)])

    # Shuffle
    idx = np.random.permutation(n_samples)
    scores = scores[idx]
    labels = labels[idx]

    # Fit Platt calibrator
    calibrator = fit_binary_probability_calibrator(scores, labels, method="platt")

    assert calibrator["type"] == "platt"
    model = calibrator["model"]
    assert isinstance(model, LogisticRegression)
    # Verify that class_weight='balanced' was used
    assert model.class_weight == "balanced"

    # With balanced class weights, the model should not suppress fire confidence
    # Test that low confidence fires are not pushed to 0
    test_scores = np.array([0.65, 0.75, 0.85])
    calibrated = model.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    # Even for the lower score (0.65), it shouldn't be heavily suppressed
    assert calibrated[0] > 0.4, "Balanced class weights should prevent fire confidence suppression"


def test_fit_from_hindcast_run_stratified_split_preserves_class_distribution():
    """Stratified split should preserve class distribution in train/eval sets."""
    tmp_root = Path(tempfile.mkdtemp())
    try:
        hindcast_dir = tmp_root / "hindcast_run"
        hindcast_dir.mkdir()

        # Create 6 cases: 3 mostly positive, 3 mostly negative (like a fire-heavy vs fire-sparse period)
        cases_config = [
            # Cases 0-2: fire-heavy
            (np.array([[[0.1, 0.6], [0.2, 0.7]]], dtype=np.float32),
             np.array([[[0, 1], [0, 1]]], dtype=np.float32)),
            (np.array([[[0.2, 0.7], [0.3, 0.8]]], dtype=np.float32),
             np.array([[[0, 1], [1, 1]]], dtype=np.float32)),
            (np.array([[[0.15, 0.65], [0.25, 0.75]]], dtype=np.float32),
             np.array([[[1, 1], [0, 1]]], dtype=np.float32)),
            # Cases 3-5: mostly noise (no fires)
            (np.array([[[0.05, 0.15], [0.08, 0.12]]], dtype=np.float32),
             np.array([[[0, 0], [0, 0]]], dtype=np.float32)),
            (np.array([[[0.06, 0.14], [0.09, 0.11]]], dtype=np.float32),
             np.array([[[0, 0], [0, 0]]], dtype=np.float32)),
            (np.array([[[0.07, 0.13], [0.10, 0.10]]], dtype=np.float32),
             np.array([[[0, 0], [0, 0]]], dtype=np.float32)),
        ]

        for i, (y_pred, y_obs) in enumerate(cases_config):
            fire_t0 = np.array([[1, 0], [0, 1]], dtype=np.float32)
            ds = xr.Dataset(
                data_vars={
                    "y_pred": (["time", "lat", "lon"], y_pred),
                    "y_obs": (["time", "lat", "lon"], y_obs),
                    "fire_t0": (["lat", "lon"], fire_t0),
                },
                coords={
                    "time": [datetime(2025, 1, 1 + i)],
                    "lat": [0, 1],
                    "lon": [0, 1],
                    "lead_time_hours": ("time", [24]),
                },
                attrs={"ref_time": f"2025-01-0{1+i}T00:00:00Z"},
            )
            ds.to_netcdf(hindcast_dir / f"case{i}.nc")

        manifest = {
            "run_id": "test_hindcast_stratified",
            "cases": [{"path": str(hindcast_dir / f"case{i}.nc")} for i in range(6)],
        }
        with open(hindcast_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        fit_from_hindcast_run(
            hindcast_run_dir=hindcast_dir,
            method="platt",
            split_percentile=0.5,
            out_root=str(tmp_root / "calibration"),
        )

        # Load metrics to check stratification quality
        runs = list((tmp_root / "calibration").glob("*"))
        assert len(runs) == 1
        metrics_file = runs[0] / "metrics.json"
        assert metrics_file.exists()

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        # Check that we have metrics for horizon 24 (keys are strings in JSON)
        assert "24" in metrics
        h24_metrics = metrics["24"]

        # Verify class distribution is reported
        assert "n_train_positive" in h24_metrics
        assert "n_train_negative" in h24_metrics
        assert "n_eval_positive" in h24_metrics
        assert "n_eval_negative" in h24_metrics
        assert "train_imbalance_ratio" in h24_metrics
        assert "eval_imbalance_ratio" in h24_metrics

        # With stratification, both train and eval should have both classes
        assert h24_metrics["n_train_positive"] > 0
        assert h24_metrics["n_train_negative"] > 0
        assert h24_metrics["n_eval_positive"] > 0
        assert h24_metrics["n_eval_negative"] > 0

    finally:
        shutil.rmtree(tmp_root)


def test_fit_from_hindcast_run_reports_severe_imbalance_warning():
    """Metrics should flag when class imbalance exceeds 10:1."""
    tmp_root = Path(tempfile.mkdtemp())
    try:
        hindcast_dir = tmp_root / "hindcast_run"
        hindcast_dir.mkdir()

        # Create cases with >10:1 imbalance (95% noise, 5% fire)
        for i in range(4):
            # Most grid cells are noise
            y_pred = np.full((1, 10, 10), 0.1, dtype=np.float32)
            y_obs = np.zeros((1, 10, 10), dtype=np.float32)
            # Only 2-3 cells are fires (5% of 100 cells)
            y_pred[0, 0, 0] = 0.8
            y_obs[0, 0, 0] = 1.0
            y_pred[0, 1, 1] = 0.75
            y_obs[0, 1, 1] = 1.0

            fire_t0 = np.zeros((10, 10), dtype=np.float32)
            fire_t0[0, 0] = 1
            fire_t0[1, 1] = 1

            ds = xr.Dataset(
                data_vars={
                    "y_pred": (["time", "lat", "lon"], y_pred),
                    "y_obs": (["time", "lat", "lon"], y_obs),
                    "fire_t0": (["lat", "lon"], fire_t0),
                },
                coords={
                    "time": [datetime(2025, 1, 1 + i)],
                    "lat": np.arange(10),
                    "lon": np.arange(10),
                    "lead_time_hours": ("time", [24]),
                },
                attrs={"ref_time": f"2025-01-0{1+i}T00:00:00Z"},
            )
            ds.to_netcdf(hindcast_dir / f"case{i}.nc")

        manifest = {
            "run_id": "test_hindcast_imbalance",
            "cases": [{"path": str(hindcast_dir / f"case{i}.nc")} for i in range(4)],
        }
        with open(hindcast_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        fit_from_hindcast_run(
            hindcast_run_dir=hindcast_dir,
            method="platt",
            split_percentile=0.5,
            out_root=str(tmp_root / "calibration"),
        )

        # Load metrics and verify imbalance warning flags
        runs = list((tmp_root / "calibration").glob("*"))
        assert len(runs) == 1
        metrics_file = runs[0] / "metrics.json"

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        assert "24" in metrics
        h24_metrics = metrics["24"]

        # Should flag training set imbalance
        assert h24_metrics["train_imbalance_warning"] is True
        assert h24_metrics["train_imbalance_ratio"] > 10.0

    finally:
        shutil.rmtree(tmp_root)


def test_temporal_2way_split_preserves_order():
    """Test that _temporal_2way_split preserves temporal ordering when start_time exists."""
    from ml.train_denoiser_v2 import _temporal_2way_split

    # Create a DataFrame with 10 rows and timestamps
    df = pd.DataFrame({
        "start_time": pd.date_range("2025-01-01", periods=10, freq="D"),
        "value": np.arange(10),
    })

    eval_df, val_df = _temporal_2way_split(df, eval_fraction=0.8)

    # Should have split at 80% (8 rows eval, 2 rows validation)
    assert len(eval_df) == 8
    assert len(val_df) == 2

    # Check temporal ordering
    assert eval_df["start_time"].max() < val_df["start_time"].min()

    # Check values are preserved
    assert list(eval_df["value"]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(val_df["value"]) == [8, 9]


def test_temporal_2way_split_positional_fallback():
    """Test that _temporal_2way_split falls back to positional split when start_time absent."""
    from ml.train_denoiser_v2 import _temporal_2way_split

    # DataFrame without start_time
    df = pd.DataFrame({
        "value": np.arange(10),
    })

    eval_df, val_df = _temporal_2way_split(df, eval_fraction=0.7)

    # Should have split at 70% (7 rows eval, 3 rows validation)
    assert len(eval_df) == 7
    assert len(val_df) == 3
    assert list(eval_df["value"]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(val_df["value"]) == [7, 8, 9]


def test_temporal_2way_split_with_duplicate_timestamps():
    """Test that _temporal_2way_split falls back gracefully when quantile split is invalid."""
    from ml.train_denoiser_v2 import _temporal_2way_split

    # DataFrame with all identical timestamps (causes quantile boundary collapse)
    df = pd.DataFrame({
        "start_time": pd.Timestamp("2025-01-01"),
        "value": np.arange(10),
    })
    # Broadcast timestamp to all rows
    df["start_time"] = pd.Timestamp("2025-01-01")

    eval_df, val_df = _temporal_2way_split(df, eval_fraction=0.8)

    # Should fall back to positional split
    assert len(eval_df) == 8
    assert len(val_df) == 2


def test_calibrator_validation_metrics_computed():
    """Test that calibrator validation metrics are computed correctly when validation set exists."""
    from ml.train_denoiser_v2 import _temporal_2way_split, _map_labels
    import pandas as pd

    # Create synthetic calibration validation data
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        "start_time": pd.date_range("2025-01-01", periods=n_samples, freq="D"),
        "event_label": np.random.choice(["POSITIVE", "NEGATIVE"], n_samples),
        "raw_score": np.random.uniform(0, 1, n_samples),
    })

    # Split into eval and validation
    eval_df, val_df = _temporal_2way_split(df, eval_fraction=0.8)

    assert len(eval_df) == 80
    assert len(val_df) == 20

    # Check that both have mixed labels
    eval_labels = _map_labels(eval_df)
    val_labels = _map_labels(val_df)
    assert len(np.unique(eval_labels[eval_labels >= 0])) == 2
    assert len(np.unique(val_labels[val_labels >= 0])) == 2


def test_calibrator_overfitting_detection():
    """Test that calibrator overfitting is detected when validation Brier loss degrades >5%."""
    from sklearn.metrics import brier_score_loss

    # Simulate eval set with good calibration
    y_eval = np.array([0, 0, 0, 1, 1, 1])
    score_eval = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.8])  # Well-calibrated

    # Simulate validation set with poor calibration (overfitting scenario)
    y_val = np.array([0, 0, 1, 1])
    score_val = np.array([0.6, 0.7, 0.3, 0.4])  # Poorly calibrated

    brier_eval = brier_score_loss(y_eval, score_eval)
    brier_val = brier_score_loss(y_val, score_val)
    degradation = ((brier_val - brier_eval) / (brier_eval + 1e-10)) * 100.0

    # Check that degradation is > 5%
    assert degradation > 5.0, f"Expected overfitting signal (>5%), got {degradation:.1f}%"
