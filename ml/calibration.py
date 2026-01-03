"""Probability calibration for spread model outputs.

This module provides tools to calibrate raw spread probabilities [0, 1] using
historical hindcast data. It supports per-horizon calibration using
Isotonic Regression (default) or Platt scaling (logistic regression).
"""

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# Add project root to sys.path if needed
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

LOGGER = logging.getLogger(__name__)


class SpreadProbabilityCalibrator:
    """Calibrator for spread probabilities, mapping raw scores to observed frequencies."""

    def __init__(
        self,
        method: str = "isotonic",
        p_min: float = 1e-4,
        per_horizon_models: Optional[Dict[int, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.method = method
        self.p_min = p_min
        self.per_horizon_models = per_horizon_models or {}
        self.metadata = metadata or {}

    def calibrate_probs(self, raw_probs: np.ndarray, horizon_hours: int) -> np.ndarray:
        """Apply calibration mapping for a specific horizon.

        Parameters
        ----------
        raw_probs : np.ndarray
            The raw probabilities to calibrate.
        horizon_hours : int
            The forecast horizon in hours.

        Returns
        -------
        np.ndarray
            The calibrated probabilities, clamped to [0, 1].
        """
        if horizon_hours not in self.per_horizon_models:
            LOGGER.warning(
                f"No calibration model found for horizon {horizon_hours}h; returning raw probabilities."
            )
            return np.clip(raw_probs, 0.0, 1.0)

        model = self.per_horizon_models[horizon_hours]
        orig_shape = raw_probs.shape
        flat_probs = raw_probs.ravel()

        if isinstance(model, IsotonicRegression):
            # IsotonicRegression.predict supports out-of-bounds by clipping to boundary values
            calibrated = model.predict(flat_probs)
        elif hasattr(model, "predict_proba"):
            # LogisticRegression expects 2D input (n_samples, 1)
            # and returns [p(0), p(1)]
            calibrated = model.predict_proba(flat_probs.reshape(-1, 1))[:, 1]
        else:
            raise ValueError(f"Unsupported calibration model type for horizon {horizon_hours}h: {type(model)}")

        return np.clip(calibrated.reshape(orig_shape), 0.0, 1.0)

    def save(self, run_dir: Union[str, Path]):
        """Save calibrator artifacts to a run directory."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save models
        joblib.dump(self.per_horizon_models, run_dir / "calibrator.pkl")

        # 2. Save metadata
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        # 3. Save config
        config = {
            "method": self.method,
            "p_min": self.p_min,
        }
        with open(run_dir / "config_resolved.yaml", "w") as f:
            yaml.safe_dump(config, f)

        LOGGER.info(f"Calibrator artifacts saved to {run_dir}")

    @classmethod
    def load(cls, run_dir: Union[str, Path]) -> "SpreadProbabilityCalibrator":
        """Load calibrator from a run directory."""
        run_dir = Path(run_dir)
        model_path = run_dir / "calibrator.pkl"
        meta_path = run_dir / "metadata.json"
        cfg_path = run_dir / "config_resolved.yaml"

        if not model_path.exists():
            raise FileNotFoundError(f"Calibrator model not found: {model_path}")

        per_horizon_models = joblib.load(model_path)
        
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        method = "isotonic"
        p_min = 1e-4
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    method = cfg.get("method", method)
                    p_min = cfg.get("p_min", p_min)

        return cls(
            method=method,
            p_min=p_min,
            per_horizon_models=per_horizon_models,
            metadata=metadata,
        )


def _maybe_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        sha = out.strip()
        return sha if sha else None
    except Exception:
        return None


def fit_from_hindcast_run(
    hindcast_run_dir: Union[str, Path],
    method: str = "isotonic",
    p_min: float = 1e-4,
    split_percentile: float = 0.8,
    out_root: str = "models/spread_calibration",
) -> SpreadProbabilityCalibrator:
    """Train a calibrator from a hindcast run directory.

    Parameters
    ----------
    hindcast_run_dir : Union[str, Path]
        The directory containing the hindcast index.json and cases.
    method : str
        The calibration method ('isotonic' or 'platt').
    p_min : float
        Minimum probability threshold for the candidate cell mask.
    split_percentile : float
        Proportion of cases to use for training vs evaluation.
    out_root : str
        Root directory to save calibration artifacts.

    Returns
    -------
    SpreadProbabilityCalibrator
        The fitted calibrator.
    """
    hindcast_run_dir = Path(hindcast_run_dir)
    index_path = hindcast_run_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Hindcast index not found: {index_path}")

    with open(index_path, "r") as f:
        manifest = json.load(f)

    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError("No cases found in hindcast manifest.")

    # 1. Group cases by horizon
    # Each case can have multiple horizons.
    # We load each case once and split data by horizon.
    horizon_data: Dict[int, List[Dict[str, np.ndarray]]] = {}

    LOGGER.info(f"Loading {len(cases)} cases from {hindcast_run_dir}...")
    for case_meta in cases:
        case_path = Path(case_meta["path"])
        # Handle relative paths in manifest if needed
        if not case_path.is_absolute():
            case_path = hindcast_run_dir / case_path

        if not case_path.exists():
            LOGGER.warning(f"Case file not found: {case_path}; skipping.")
            continue

        with xr.open_dataset(case_path) as ds:
            # ref_time is in attrs as ISO string
            ref_time = pd.to_datetime(ds.attrs["ref_time"])
            
            # y_pred: (time, lat, lon), y_obs: (time, lat, lon)
            # lead_time_hours: (time,)
            horizons = ds.lead_time_hours.values
            for i, h in enumerate(horizons):
                h = int(h)
                if h not in horizon_data:
                    horizon_data[h] = []
                
                # candidate mask: (fire_t0 > 0) | (y_pred >= p_min)
                fire_t0 = ds.fire_t0.values
                y_pred = ds.y_pred.isel(time=i).values
                y_obs = ds.y_obs.isel(time=i).values
                
                mask = (fire_t0 > 0) | (y_pred >= p_min)
                
                horizon_data[h].append({
                    "ref_time": ref_time,
                    "y_pred": y_pred[mask],
                    "y_obs": y_obs[mask],
                })

    # 2. Fit and evaluate per horizon
    per_horizon_models = {}
    metrics = {}
    
    # Setup run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = _maybe_git_sha() or "unknown"
    run_name = f"{timestamp}_{git_sha}"
    run_dir = Path(out_root) / run_name

    for h, data_list in horizon_data.items():
        if not data_list:
            continue
            
        # Convert to DataFrame for easier splitting
        df = pd.DataFrame(data_list)
        df = df.sort_values("ref_time")
        
        split_idx = int(len(df) * split_percentile)
        train_cases = df.iloc[:split_idx]
        eval_cases = df.iloc[split_idx:]
        
        if train_cases.empty:
            LOGGER.warning(f"No training data for horizon {h}h; skipping.")
            continue
            
        X_train = np.concatenate(train_cases["y_pred"].values)
        y_train = np.concatenate(train_cases["y_obs"].values)
        
        # 2a. Fit
        # Check if we have both classes
        effective_method = method
        if len(np.unique(y_train)) < 2:
            if method == "platt":
                LOGGER.warning(
                    f"Only one class in training data for T+{h}h; "
                    "falling back to Isotonic (constant) predictor as Platt scaling requires 2 classes."
                )
                effective_method = "isotonic"
            else:
                LOGGER.info(f"Only one class in training data for T+{h}h; Isotonic will predict constant value.")

        LOGGER.info(f"Fitting {effective_method} calibrator for T+{h}h on {len(X_train)} samples...")
        
        if effective_method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(X_train, y_train)
        elif effective_method == "platt":
            model = LogisticRegression(solver="lbfgs")
            # X needs to be (n_samples, 1)
            model.fit(X_train.reshape(-1, 1), y_train)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        per_horizon_models[h] = model
        
        # 2b. Evaluate
        if not eval_cases.empty:
            X_eval = np.concatenate(eval_cases["y_pred"].values)
            y_eval = np.concatenate(eval_cases["y_obs"].values)
            
            # Predict
            if isinstance(model, IsotonicRegression):
                y_cal = model.predict(X_eval)
            else:
                y_cal = model.predict_proba(X_eval.reshape(-1, 1))[:, 1]
            
            # Metrics
            brier_raw = brier_score_loss(y_eval, X_eval)
            brier_cal = brier_score_loss(y_eval, y_cal)
            
            # Reliability curve
            prob_true_raw, prob_pred_raw = calibration_curve(y_eval, X_eval, n_bins=10)
            prob_true_cal, prob_pred_cal = calibration_curve(y_eval, y_cal, n_bins=10)
            
            metrics[h] = {
                "brier_raw": float(brier_raw),
                "brier_cal": float(brier_cal),
                "improvement": float(brier_raw - brier_cal),
                "reliability_raw": {
                    "prob_true": prob_true_raw.tolist(),
                    "prob_pred": prob_pred_raw.tolist(),
                },
                "reliability_cal": {
                    "prob_true": prob_true_cal.tolist(),
                    "prob_pred": prob_pred_cal.tolist(),
                }
            }
            LOGGER.info(
                f"T+{h}h calibration: Brier {brier_raw:.4f} -> {brier_cal:.4f} "
                f"(diff={brier_raw-brier_cal:.4f})"
            )

    # 3. Create calibrator and save
    metadata = {
        "run_id": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "method": method,
        "p_min": p_min,
        "hindcast_run_dir": str(hindcast_run_dir),
        "split_percentile": split_percentile,
        "horizons": list(per_horizon_models.keys()),
        "package_versions": {
            "sklearn": sys.modules["sklearn"].__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }

    calibrator = SpreadProbabilityCalibrator(
        method=method,
        p_min=p_min,
        per_horizon_models=per_horizon_models,
        metadata=metadata,
    )
    
    run_dir.mkdir(parents=True, exist_ok=True)
    calibrator.save(run_dir)
    
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    return calibrator


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Train probability calibration for spread models.")
    parser.add_argument("--hindcast_run", type=str, required=True, help="Path to hindcast run directory.")
    parser.add_argument("--method", type=str, choices=["isotonic", "platt"], default="isotonic", help="Calibration method.")
    parser.add_argument("--p_min", type=float, default=1e-4, help="Min probability for candidate mask.")
    parser.add_argument("--split", type=float, default=0.8, help="Train/eval split percentile.")
    parser.add_argument("--out_root", type=str, default="models/spread_calibration", help="Root for artifacts.")
    
    args = parser.parse_args()
    
    try:
        fit_from_hindcast_run(
            hindcast_run_dir=args.hindcast_run,
            method=args.method,
            p_min=args.p_min,
            split_percentile=args.split,
            out_root=args.out_root,
        )
    except Exception as e:
        LOGGER.exception(f"Calibration training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

