"""Learned spread model v3: hardened v2 successor with train/infer parity fixes."""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from api.fires.service import get_fire_cells_heatmap
from ml.calibration import SpreadProbabilityCalibrator
from ml.spread.contract import SpreadForecast, SpreadModel, SpreadModelInput
from ml.spread.hindcast_dataset import V3_TENSOR_CHANNELS
from ml.spread.region_key import deterministic_region_bucket

LOGGER = logging.getLogger(__name__)
_WEATHER_AGGREGATION = "horizon_weighted_mean"
_DEFAULT_SPATIAL_MULTIPLE = 16


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _resolve_spatial_multiple() -> int:
    raw = os.getenv("SPREAD_ONNX_SPATIAL_MULTIPLE", str(_DEFAULT_SPATIAL_MULTIPLE))
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"Invalid SPREAD_ONNX_SPATIAL_MULTIPLE={raw!r}; expected integer >= 1.") from exc
    if value < 1:
        raise ValueError(f"SPREAD_ONNX_SPATIAL_MULTIPLE must be >= 1, got {value}.")
    return value


def _pad_to_multiple(x: np.ndarray, multiple: int) -> tuple[np.ndarray, int, int]:
    """Pad BCHW tensor on spatial dims to a given multiple using edge replication."""
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={x.shape!r}")
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    x_pad = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return x_pad, h, w


def _crop_to_shape(y: np.ndarray, h: int, w: int) -> np.ndarray:
    """Crop NT HW output tensor back to original spatial shape."""
    if y.ndim != 4:
        raise ValueError(f"Expected model output shape (N,T,H,W), got {y.shape!r}")
    if y.shape[-2] < h or y.shape[-1] < w:
        raise ValueError(
            "Model output spatial shape is smaller than requested crop: "
            f"output={y.shape[-2:]} requested={(h, w)}"
        )
    return y[:, :, :h, :w]


def _window_bbox_with_edges(inputs: SpreadModelInput) -> tuple[float, float, float, float]:
    lat = np.asarray(inputs.window.lat, dtype=np.float64)
    lon = np.asarray(inputs.window.lon, dtype=np.float64)
    if lat.size == 0 or lon.size == 0:
        raise ValueError("Cannot derive bbox from empty window coordinates.")

    cell = float(getattr(inputs.grid, "cell_size_deg", 0.0) or 0.0)
    if cell <= 0.0:
        cell_lon = float(abs(lon[1] - lon[0])) if lon.size > 1 else 0.01
        cell_lat = float(abs(lat[1] - lat[0])) if lat.size > 1 else 0.01
        cell = max(cell_lon, cell_lat)
    half = cell / 2.0
    return (
        float(lon.min() - half),
        float(lat.min() - half),
        float(lon.max() + half),
        float(lat.max() + half),
    )


def _load_fire_lag(inputs: SpreadModelInput, lookback_hours: int) -> np.ndarray:
    bbox = _window_bbox_with_edges(inputs)
    end_time = inputs.forecast_reference_time
    start_time = end_time - timedelta(hours=int(lookback_hours))
    lag = get_fire_cells_heatmap(
        region_name=None,
        grid=inputs.grid,
        bbox=bbox,
        start_time=start_time,
        end_time=end_time,
        mode="presence",
        clip=True,
    ).heatmap.astype(np.float32, copy=False)
    expected_shape = inputs.active_fires.heatmap.shape
    if lag.shape != expected_shape:
        raise ValueError(
            "Lag fire heatmap shape mismatch: "
            f"lag={lag.shape} expected={expected_shape} lookback_hours={lookback_hours}"
        )
    return lag


def _collapse_weather_over_horizons(
    ds: xr.Dataset,
    var: str,
    horizons_hours: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    if var not in ds:
        return np.zeros(shape, dtype=np.float32)

    arr = np.asarray(ds[var].values, dtype=np.float32)
    if arr.ndim == 2:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 3:
        raise ValueError(f"Weather variable {var!r} must be 2D or 3D, got shape={arr.shape!r}.")

    n_t = int(arr.shape[0])
    if n_t <= 0:
        return np.zeros(shape, dtype=np.float32)
    if n_t != len(horizons_hours):
        raise ValueError(
            f"Weather variable {var!r} time dimension mismatch: n_time={n_t} "
            f"len(horizons_hours)={len(horizons_hours)}."
        )

    weights = np.asarray([max(1.0, float(h)) for h in horizons_hours], dtype=np.float32)
    weights = weights / np.maximum(np.sum(weights), 1e-6)
    out = np.tensordot(weights, arr, axes=(0, 0)).astype(np.float32, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class LearnedSpreadModelV3(SpreadModel):
    """CPU-first spread model using ONNX Runtime artifacts with hardened feature parity."""

    def __init__(
        self,
        model_run_dir: str,
        calibrator_run_dir: str | None = None,
        onnx_filename: str = "model.int8.onnx",
    ):
        self.model_run_dir = model_run_dir
        self.calibrator_run_dir = calibrator_run_dir
        self.onnx_filename = onnx_filename
        self.model_name = "LearnedSpreadModelV3"
        self.model_version = "v3"
        self.channel_names = list(V3_TENSOR_CHANNELS)
        self.weather_aggregation = _WEATHER_AGGREGATION
        self.spatial_multiple = _resolve_spatial_multiple()
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
            agg = payload.get("weather_aggregation")
            if agg is not None and str(agg) != _WEATHER_AGGREGATION:
                raise ValueError(
                    f"Unsupported weather_aggregation={agg!r}; expected {_WEATHER_AGGREGATION!r}."
                )

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
                "onnxruntime is required for LearnedSpreadModelV3 inference."
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
                LOGGER.exception("Failed to load v3 calibrator. Continuing with raw probabilities.")
                self.calibrator = None

    def _build_feature_tensor(self, inputs: SpreadModelInput) -> np.ndarray:
        ny, nx = inputs.active_fires.heatmap.shape
        horizons = [int(h) for h in inputs.horizons_hours]
        fire_t0 = np.nan_to_num(np.asarray(inputs.active_fires.heatmap, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        fire_t_minus_6 = _load_fire_lag(inputs, lookback_hours=6)
        fire_t_minus_12 = _load_fire_lag(inputs, lookback_hours=12)

        slope = np.nan_to_num(np.asarray(inputs.terrain.slope, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        aspect = np.nan_to_num(np.asarray(inputs.terrain.aspect, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        aspect_rad = np.radians(aspect)
        aspect_sin = np.sin(aspect_rad).astype(np.float32, copy=False)
        aspect_cos = np.cos(aspect_rad).astype(np.float32, copy=False)

        elevation = (
            np.asarray(inputs.terrain.elevation, dtype=np.float32)
            if inputs.terrain.elevation is not None
            else np.zeros((ny, nx), dtype=np.float32)
        )
        elevation = np.nan_to_num(elevation, nan=0.0, posinf=0.0, neginf=0.0)
        grad_y, grad_x = np.gradient(elevation)
        ruggedness = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32, copy=False)
        tpi = (elevation - np.float32(np.nanmean(elevation))).astype(np.float32, copy=False)

        u10 = _collapse_weather_over_horizons(inputs.weather_cube, "u10", horizons, (ny, nx))
        v10 = _collapse_weather_over_horizons(inputs.weather_cube, "v10", horizons, (ny, nx))
        t2m = _collapse_weather_over_horizons(inputs.weather_cube, "t2m", horizons, (ny, nx))
        rh2m = _collapse_weather_over_horizons(inputs.weather_cube, "rh2m", horizons, (ny, nx))
        precip_24h = _collapse_weather_over_horizons(inputs.weather_cube, "precip_24h", horizons, (ny, nx))
        ndvi = _collapse_weather_over_horizons(inputs.weather_cube, "ndvi", horizons, (ny, nx))
        lfmc = _collapse_weather_over_horizons(inputs.weather_cube, "lfmc", horizons, (ny, nx))
        dfmc = _collapse_weather_over_horizons(inputs.weather_cube, "dfmc", horizons, (ny, nx))

        bbox = _window_bbox_with_edges(inputs)
        region_bucket = deterministic_region_bucket(bbox=bbox, n_buckets=1024)
        region_bucket_arr = np.full((ny, nx), float(region_bucket), dtype=np.float32)

        channel_data = {
            "fire_t0": fire_t0,
            "fire_t-6h": fire_t_minus_6,
            "fire_t-12h": fire_t_minus_12,
            "u10": u10,
            "v10": v10,
            "t2m": t2m,
            "rh2m": rh2m,
            "precip_24h": precip_24h,
            "slope_deg": slope,
            "aspect_sin": aspect_sin,
            "aspect_cos": aspect_cos,
            "elevation_m": elevation,
            "ruggedness": ruggedness,
            "tpi": tpi,
            "ndvi": ndvi,
            "lfmc": lfmc,
            "dfmc": dfmc,
            "region_id_embedding_input": region_bucket_arr,
        }
        missing_channels = [c for c in self.channel_names if c not in channel_data]
        if missing_channels:
            raise ValueError(f"Missing v3 feature channels: {missing_channels}")
        x = np.stack([channel_data[c] for c in self.channel_names], axis=0).astype(np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x[None, ...]  # (1, C, H, W)

    def predict(self, inputs: SpreadModelInput) -> SpreadForecast:
        if self._session is None:
            raise RuntimeError("ONNX Runtime session is not initialized.")

        x_raw = self._build_feature_tensor(inputs)
        x, orig_h, orig_w = _pad_to_multiple(x_raw, self.spatial_multiple)
        if x.shape != x_raw.shape:
            LOGGER.debug(
                "Padded v3 ONNX input from %s to %s (multiple=%d).",
                x_raw.shape,
                x.shape,
                self.spatial_multiple,
            )
        raw_out = self._session.run(None, {self._input_name: x})[0]
        out = _crop_to_shape(np.asarray(raw_out, dtype=np.float32), orig_h, orig_w)

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
