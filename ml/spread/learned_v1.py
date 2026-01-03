"""Learned spread model (v1) inference implementation.

See `docs/spread_model_design.md` for training/inference details and evaluation notes.
"""

import logging
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from ml.calibration import SpreadProbabilityCalibrator
from ml.spread.contract import SpreadForecast, SpreadModel, SpreadModelInput

LOGGER = logging.getLogger(__name__)


class LearnedSpreadModelV1(SpreadModel):
    """Learned spread model using an ensemble of per-horizon classifiers."""

    def __init__(self, model_run_dir: str, calibrator_run_dir: str | None = None):
        self.model_run_dir = model_run_dir
        self.calibrator_run_dir = calibrator_run_dir
        self.models: Dict[int, Any] = {}
        self.feature_list: List[str] = []
        self.calibrator: SpreadProbabilityCalibrator | None = None
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.model_run_dir, "model.pkl")
        features_path = os.path.join(self.model_run_dir, "feature_list.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature list file not found: {features_path}")

        LOGGER.info(f"Loading learned spread v1 models from {model_path}")
        self.models = joblib.load(model_path)
        
        import json
        with open(features_path, "r") as f:
            self.feature_list = json.load(f)

        if self.calibrator_run_dir:
            LOGGER.info(f"Loading calibrator from {self.calibrator_run_dir}")
            try:
                self.calibrator = SpreadProbabilityCalibrator.load(self.calibrator_run_dir)
            except Exception as e:
                LOGGER.error(f"Failed to load calibrator: {e}")
                # We don't raise here to allow model to run without calibration if it fails?
                # Actually, if config requests it, we should probably warn or raise.
                # Let's raise to be safe, or just log error.
                raise

    def _build_tabular_features(self, inputs: SpreadModelInput, horizon_idx: int) -> pd.DataFrame:
        """Extract features from SpreadModelInput for a specific horizon."""
        ny, nx = inputs.active_fires.heatmap.shape
        
        # Static features
        slope = inputs.terrain.slope
        aspect = inputs.terrain.aspect
        elevation = inputs.terrain.elevation
        
        aspect_rad = np.radians(aspect)
        aspect_sin = np.sin(aspect_rad)
        aspect_cos = np.cos(aspect_rad)
        
        fire_t0 = inputs.active_fires.heatmap
        
        # Weather features at this horizon
        weather_h = inputs.weather_cube.isel(time=horizon_idx)
        u10 = weather_h["u10"].values
        v10 = weather_h["v10"].values
        t2m = weather_h.get("t2m")
        rh2m = weather_h.get("rh2m")
        
        # Tabularize
        data = {
            "fire_t0": fire_t0.ravel(),
            "slope_deg": slope.ravel(),
            "aspect_sin": aspect_sin.ravel(),
            "aspect_cos": aspect_cos.ravel(),
            "u10": u10.ravel(),
            "v10": v10.ravel(),
        }
        
        if elevation is not None:
            data["elevation_m"] = elevation.ravel()
        if t2m is not None:
            data["t2m"] = t2m.values.ravel()
        if rh2m is not None:
            data["rh2m"] = rh2m.values.ravel()
            
        df = pd.DataFrame(data)
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        
        # Align with feature_list (fill missing with NaN)
        for col in self.feature_list:
            if col not in df.columns:
                df[col] = np.nan
                
        return df[self.feature_list]

    def predict(self, inputs: SpreadModelInput) -> SpreadForecast:
        """Predict fire spread probability using learned classifiers."""
        horizons = list(inputs.horizons_hours)
        ny, nx = inputs.active_fires.heatmap.shape
        
        forecast_grids = []
        
        for h_idx, h in enumerate(horizons):
            if h not in self.models:
                LOGGER.warning(f"No model found for horizon {h}h; returning zero grid.")
                forecast_grids.append(np.zeros((ny, nx), dtype=np.float32))
                continue
                
            clf = self.models[h]
            X = self._build_tabular_features(inputs, h_idx)
            
            # Predict probabilities
            probs = clf.predict_proba(X)[:, 1]

            # Apply calibration if available
            if self.calibrator:
                probs = self.calibrator.calibrate_probs(probs, h)

            prob_grid = probs.reshape((ny, nx)).astype(np.float32)
            
            # Apply terrain masks if present (mirroring heuristic_v0)
            if inputs.terrain.valid_data_mask is not None:
                prob_grid = prob_grid * inputs.terrain.valid_data_mask
            if inputs.terrain.aoi_mask is not None:
                prob_grid = prob_grid * inputs.terrain.aoi_mask
                
            forecast_grids.append(prob_grid)
            
        # Package into SpreadForecast
        from datetime import timedelta
        times = [inputs.forecast_reference_time + timedelta(hours=h) for h in horizons]
        
        da = xr.DataArray(
            np.stack(forecast_grids),
            coords={
                "time": times,
                "lat": inputs.window.lat,
                "lon": inputs.window.lon,
                "lead_time_hours": ("time", horizons),
            },
            dims=("time", "lat", "lon"),
            name="spread_probability",
        )
        
        forecast = SpreadForecast(
            probabilities=da,
            forecast_reference_time=inputs.forecast_reference_time,
            horizons_hours=horizons,
        )
        forecast.validate()
        return forecast

