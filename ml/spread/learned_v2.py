"""Learned spread model v2: spatial U-Net with ONNX Runtime inference."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ml.calibration import SpreadProbabilityCalibrator
from ml.spread.contract import SpreadForecast, SpreadModel, SpreadModelInput
from ml.spread.hindcast_dataset import V2_TENSOR_CHANNELS

LOGGER = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_weather_slice_mean(ds: xr.Dataset, var: str) -> np.ndarray | None:
    if var not in ds:
        return None
    arr = ds[var]
    if "time" in arr.dims:
        arr = arr.mean(dim="time")
    return np.asarray(arr.values, dtype=np.float32)


class LearnedSpreadModelV2(SpreadModel):
    """CPU-first spread model using ONNX Runtime artifacts produced by train_spread_v2."""

    def __init__(
        self,
        model_run_dir: str,
        calibrator_run_dir: str | None = None,
        onnx_filename: str = "model.int8.onnx",
    ):
        self.model_run_dir = model_run_dir
        self.calibrator_run_dir = calibrator_run_dir
        self.onnx_filename = onnx_filename
        self.model_name = "LearnedSpreadModelV2"
        self.model_version = "v2"
        self.channel_names = list(V2_TENSOR_CHANNELS)
        self.calibrator: SpreadProbabilityCalibrator | None = None
        self._session: Any | None = None
        self._input_name = "x"
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        run_dir = Path(self.model_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Model run directory not found: {run_dir}")

        schema_path = run_dir / "feature_schema.json"
        if schema_path.exists():
            with schema_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            channels = payload.get("channels")
            if isinstance(channels, list) and channels:
                self.channel_names = [str(c) for c in channels]

        onnx_path = run_dir / self.onnx_filename
        if not onnx_path.exists():
            fallback = run_dir / "model.onnx"
            if fallback.exists():
                onnx_path = fallback
            else:
                raise FileNotFoundError(
                    f"Neither {self.onnx_filename} nor model.onnx exists in {run_dir}"
                )

        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "onnxruntime is required for LearnedSpreadModelV2 inference."
            ) from e

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, int(os.getenv("SPREAD_ONNX_THREADS", "1")))
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        if self.calibrator_run_dir:
            try:
                self.calibrator = SpreadProbabilityCalibrator.load(self.calibrator_run_dir)
            except Exception:
                LOGGER.exception("Failed to load v2 calibrator. Continuing with raw probabilities.")
                self.calibrator = None

    def _build_feature_tensor(self, inputs: SpreadModelInput) -> np.ndarray:
        ny, nx = inputs.active_fires.heatmap.shape
        fire_t0 = np.asarray(inputs.active_fires.heatmap, dtype=np.float32)
        fire_t_minus_6 = fire_t0.copy()
        fire_t_minus_12 = fire_t0.copy()

        slope = np.asarray(inputs.terrain.slope, dtype=np.float32)
        aspect = np.asarray(inputs.terrain.aspect, dtype=np.float32)
        aspect_rad = np.radians(aspect)
        aspect_sin = np.sin(aspect_rad).astype(np.float32, copy=False)
        aspect_cos = np.cos(aspect_rad).astype(np.float32, copy=False)

        elevation = (
            np.asarray(inputs.terrain.elevation, dtype=np.float32)
            if inputs.terrain.elevation is not None
            else np.zeros((ny, nx), dtype=np.float32)
        )
        grad_y, grad_x = np.gradient(elevation)
        ruggedness = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32, copy=False)
        tpi = (elevation - np.float32(np.nanmean(elevation))).astype(np.float32, copy=False)

        u10 = _safe_weather_slice_mean(inputs.weather_cube, "u10")
        v10 = _safe_weather_slice_mean(inputs.weather_cube, "v10")
        t2m = _safe_weather_slice_mean(inputs.weather_cube, "t2m")
        rh2m = _safe_weather_slice_mean(inputs.weather_cube, "rh2m")
        precip_24h = _safe_weather_slice_mean(inputs.weather_cube, "precip_24h")
        ndvi = _safe_weather_slice_mean(inputs.weather_cube, "ndvi")
        lfmc = _safe_weather_slice_mean(inputs.weather_cube, "lfmc")
        dfmc = _safe_weather_slice_mean(inputs.weather_cube, "dfmc")

        def _fill(arr: np.ndarray | None) -> np.ndarray:
            if arr is None:
                return np.zeros((ny, nx), dtype=np.float32)
            return np.asarray(arr, dtype=np.float32)

        region_bucket = np.zeros((ny, nx), dtype=np.float32)

        channel_data = {
            "fire_t0": fire_t0,
            "fire_t-6h": fire_t_minus_6,
            "fire_t-12h": fire_t_minus_12,
            "u10": _fill(u10),
            "v10": _fill(v10),
            "t2m": _fill(t2m),
            "rh2m": _fill(rh2m),
            "precip_24h": _fill(precip_24h),
            "slope_deg": slope,
            "aspect_sin": aspect_sin,
            "aspect_cos": aspect_cos,
            "elevation_m": elevation,
            "ruggedness": ruggedness,
            "tpi": tpi,
            "ndvi": _fill(ndvi),
            "lfmc": _fill(lfmc),
            "dfmc": _fill(dfmc),
            "region_id_embedding_input": region_bucket,
        }
        x = np.stack([channel_data[c] for c in self.channel_names], axis=0).astype(np.float32, copy=False)
        return x[None, ...]  # (1, C, H, W)

    def predict(self, inputs: SpreadModelInput) -> SpreadForecast:
        if self._session is None:
            raise RuntimeError("ONNX Runtime session is not initialized.")

        x = self._build_feature_tensor(inputs)
        raw_out = self._session.run(None, {self._input_name: x})[0]
        out = np.asarray(raw_out, dtype=np.float32)
        if out.ndim != 4:
            raise ValueError(f"Expected ONNX output with shape (N,T,H,W), got {out.shape!r}")

        logits_or_probs = out[0]
        if float(np.nanmin(logits_or_probs)) < 0.0 or float(np.nanmax(logits_or_probs)) > 1.0:
            probs = _sigmoid(logits_or_probs).astype(np.float32, copy=False)
        else:
            probs = np.clip(logits_or_probs, 0.0, 1.0).astype(np.float32, copy=False)

        horizons = list(inputs.horizons_hours)
        if probs.shape[0] < len(horizons):
            raise ValueError(
                f"Model output has {probs.shape[0]} horizons, requested {len(horizons)}."
            )
        probs = probs[: len(horizons)]

        if self.calibrator is not None:
            calibrated = []
            for i, h in enumerate(horizons):
                calibrated.append(
                    np.asarray(self.calibrator.calibrate_probs(probs[i], int(h)), dtype=np.float32)
                )
            probs = np.stack(calibrated, axis=0)

        from datetime import timedelta

        times = [inputs.forecast_reference_time + timedelta(hours=h) for h in horizons]
        da = xr.DataArray(
            probs,
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
            model_name=self.model_name,
            model_version=self.model_version,
        )
        forecast.validate()
        return forecast
