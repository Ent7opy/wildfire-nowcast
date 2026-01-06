"""Spread forecast service for orchestrating model execution.

See `docs/spread_model_design.md` for model behavior, assumptions, and limitations.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from ml.calibration import SpreadProbabilityCalibrator
from ml.spread.contract import DEFAULT_HORIZONS_HOURS, SpreadForecast, SpreadModel
from ml.spread.heuristic_v0 import HeuristicSpreadModelV0

# Lazily imported to avoid pulling heavy DB/raster deps at module import time.
# Kept as a module attribute so tests can patch `ml.spread.service.build_spread_inputs`.
build_spread_inputs = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

# Env/config knobs for operational inference.
SPREAD_CALIBRATOR_RUN_DIR_ENV = "SPREAD_CALIBRATOR_RUN_DIR"
SPREAD_CALIBRATOR_ROOT_ENV = "SPREAD_CALIBRATOR_ROOT"
WEATHER_BIAS_CORRECTOR_PATH_ENV = "WEATHER_BIAS_CORRECTOR_PATH"
WEATHER_BIAS_CORRECTOR_ROOT_ENV = "WEATHER_BIAS_CORRECTOR_ROOT"

# Performance limit: avoid OOM/high latency for very large areas in synchronous calls.
# 200x200 = 40,000 cells. At 0.01 degree, this is roughly 220km x 220km.
MAX_AOI_CELLS = 40000


@dataclass(frozen=True, slots=True)
class SpreadForecastRequest:
    """Request parameters for a spread forecast."""

    region_name: str
    bbox: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    forecast_reference_time: datetime
    horizons_hours: Sequence[int] = DEFAULT_HORIZONS_HOURS
    fire_lookback_hours: int = 24
    fire_cluster_id: str | None = None
    # Model configuration overrides can be added here if needed.


def run_spread_forecast(
    request: SpreadForecastRequest,
    model: SpreadModel | None = None,
) -> SpreadForecast:
    """Run an end-to-end spread forecast for an AOI or fire cluster.

    This function orchestrates:
    1. Input resolution (grid, fires, weather, terrain).
    2. Model execution.
    3. Packaging and logging.

    Parameters
    ----------
    request : SpreadForecastRequest
        The forecast request details.
    model : SpreadModel, optional
        The model implementation to use. Defaults to HeuristicSpreadModelV0.

    Returns
    -------
    SpreadForecast
        The resulting probability grids and metadata.
    """
    if request.fire_cluster_id is not None:
        # TODO: Implement cluster-to-bbox resolution once clustering logic exists.
        raise NotImplementedError("Referencing fire_cluster_id is not yet supported.")

    start_time = time.perf_counter()
    LOGGER.info(
        "Starting spread forecast",
        extra={
            "region": request.region_name,
            "bbox": request.bbox,
            "ref_time": request.forecast_reference_time.isoformat(),
            "horizons": list(request.horizons_hours),
            "fire_lookback": request.fire_lookback_hours,
        },
    )

    # 1. Resolve inputs
    # This involves DB queries and raster I/O.
    global build_spread_inputs
    if build_spread_inputs is None:
        # Import lazily to avoid pulling heavy optional dependencies during module import
        # (and to make unit tests easier to run with mocks).
        from ml.spread_features import build_spread_inputs as _build_spread_inputs

        build_spread_inputs = _build_spread_inputs

    # Resolve operational bias-correction + calibration artifacts.
    weather_bias_corrector_path = _resolve_weather_bias_corrector_path(request.region_name)
    if weather_bias_corrector_path is not None:
        LOGGER.info(
            "Using weather bias corrector",
            extra={"region": request.region_name, "path": str(weather_bias_corrector_path)},
        )
    else:
        LOGGER.warning(
            "No weather bias corrector configured; using uncorrected weather inputs.",
            extra={"region": request.region_name, "env": WEATHER_BIAS_CORRECTOR_PATH_ENV},
        )

    inputs_package = build_spread_inputs(
        region_name=request.region_name,
        bbox=request.bbox,
        forecast_reference_time=request.forecast_reference_time,
        horizons_hours=request.horizons_hours,
        fire_lookback_hours=request.fire_lookback_hours,
        weather_bias_corrector_path=weather_bias_corrector_path,
    )
    
    # Check AOI size limit
    n_cells = inputs_package.window.lat.size * inputs_package.window.lon.size
    LOGGER.info(
        "Inputs resolved",
        extra={
            "grid_n_cells": int(n_cells),
            "window_shape": (inputs_package.window.lat.size, inputs_package.window.lon.size),
            "active_fires_count": float(inputs_package.active_fires.heatmap.sum()),
        }
    )

    if n_cells == 0:
        raise ValueError(
            f"AOI produces an empty window for region {request.region_name!r} and bbox {request.bbox}. "
            "Ensure the bbox overlaps with the region's extent."
        )

    if n_cells > MAX_AOI_CELLS:
        raise ValueError(
            f"AOI too large: {n_cells} cells (max {MAX_AOI_CELLS}). "
            f"Window: {inputs_package.window.lat.size}x{inputs_package.window.lon.size}"
        )

    # 2. Select and run model
    if model is None:
        # Default to baseline heuristic
        model = HeuristicSpreadModelV0()
    
    model_name = model.__class__.__name__
    LOGGER.info(f"Using spread model: {model_name}")

    # 3. Predict
    forecast = model.predict(inputs_package.to_model_input())

    # 4. Validate output contract
    forecast.validate()
    forecast = _annotate_weather_bias(
        forecast,
        weather_bias_corrected=bool(getattr(inputs_package.weather_cube, "attrs", {}).get("weather_bias_corrected", False)),
        weather_bias_corrector_path=(
            getattr(inputs_package.weather_cube, "attrs", {}).get("weather_bias_corrector_path")
            or (str(weather_bias_corrector_path) if weather_bias_corrector_path is not None else None)
        ),
    )

    # 4b. Calibrate probabilities (default behavior).
    # - If the model already has an embedded calibrator, we treat it as authoritative.
    # - Otherwise, we try to load an operational calibrator and apply it here.
    embedded_calibrator = getattr(model, "calibrator", None)
    if isinstance(embedded_calibrator, SpreadProbabilityCalibrator):
        meta = getattr(embedded_calibrator, "metadata", {}) or {}
        LOGGER.info(
            "Using embedded probability calibration from model",
            extra={
                "region": request.region_name,
                "calibrator_run_id": meta.get("run_id"),
                "calibrator_method": meta.get("method"),
                "calibrator_horizons": meta.get("horizons"),
            },
        )
        forecast = _annotate_forecast(
            forecast,
            calibration_applied=True,
            calibration_source="embedded",
            calibration_run_id=meta.get("run_id"),
            calibration_run_dir=None,
        )
    else:
        calibrator_run_dir = _resolve_spread_calibrator_run_dir(request.region_name)
        if calibrator_run_dir is None:
            LOGGER.warning(
                "No spread calibrator configured; returning uncalibrated probabilities.",
                extra={"region": request.region_name, "env": SPREAD_CALIBRATOR_RUN_DIR_ENV},
            )
            forecast = _annotate_forecast(
                forecast,
                calibration_applied=False,
                calibration_source="missing",
                calibration_run_id=None,
                calibration_run_dir=None,
            )
        else:
            try:
                calibrator = SpreadProbabilityCalibrator.load(calibrator_run_dir)
                forecast = _apply_spread_calibration(
                    forecast=forecast,
                    calibrator=calibrator,
                    calibrator_run_dir=calibrator_run_dir,
                    region_name=request.region_name,
                )
            except Exception:
                LOGGER.exception(
                    "Failed to load/apply spread calibrator; returning uncalibrated probabilities.",
                    extra={"region": request.region_name, "calibrator_run_dir": str(calibrator_run_dir)},
                )
                forecast = _annotate_forecast(
                    forecast,
                    calibration_applied=False,
                    calibration_source="error",
                    calibration_run_id=None,
                    calibration_run_dir=str(calibrator_run_dir),
                )
    
    # 5. Finalize and log
    duration = time.perf_counter() - start_time
    LOGGER.info(
        "Spread forecast completed",
        extra={
            "duration_s": round(duration, 3),
            "model": model_name,
            "n_cells": int(n_cells),
            "output_min": float(forecast.probabilities.min()),
            "output_max": float(forecast.probabilities.max()),
        },
    )

    return forecast


def _resolve_latest_run_dir(root: Path) -> Path | None:
    """Return the latest run directory under root (by mtime), if any."""
    if not root.exists() or not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_weather_bias_corrector_path(region_name: str) -> Path | None:
    # 1) Explicit file env var wins.
    if (p := os.environ.get(WEATHER_BIAS_CORRECTOR_PATH_ENV)):
        return Path(p)

    # 2) Region-aware root, else global root.
    root_env = os.environ.get(WEATHER_BIAS_CORRECTOR_ROOT_ENV)
    roots: list[Path] = []
    if root_env:
        roots.append(Path(root_env) / region_name)
        roots.append(Path(root_env))

    # 3) Conventional default under repo: models/weather_bias_corrector
    repo_root = Path(__file__).resolve().parents[2]
    roots.append(repo_root / "models" / "weather_bias_corrector" / region_name)
    roots.append(repo_root / "models" / "weather_bias_corrector")

    for root in roots:
        latest = _resolve_latest_run_dir(root)
        if latest is None:
            # Also allow the root itself to be a run directory.
            latest = root if root.is_dir() else None
        if latest is None:
            continue
        candidate = latest / "weather_bias_corrector.json"
        if candidate.exists():
            return candidate
        if latest.is_file() and latest.name.endswith(".json"):
            return latest
    return None


def _resolve_spread_calibrator_run_dir(region_name: str) -> Path | None:
    # 1) Explicit run dir env var wins.
    if (p := os.environ.get(SPREAD_CALIBRATOR_RUN_DIR_ENV)):
        return Path(p)

    # 2) Region-aware root, else global root.
    root_env = os.environ.get(SPREAD_CALIBRATOR_ROOT_ENV)
    roots: list[Path] = []
    if root_env:
        roots.append(Path(root_env) / region_name)
        roots.append(Path(root_env))

    # 3) Conventional default under repo: models/spread_calibration
    repo_root = Path(__file__).resolve().parents[2]
    roots.append(repo_root / "models" / "spread_calibration" / region_name)
    roots.append(repo_root / "models" / "spread_calibration")

    for root in roots:
        latest = _resolve_latest_run_dir(root)
        if latest is None:
            latest = root if root.is_dir() else None
        if latest is None:
            continue
        # Valid calibrator run dir must include calibrator.pkl
        if (latest / "calibrator.pkl").exists():
            return latest
    return None


def _annotate_forecast(
    forecast: SpreadForecast,
    *,
    calibration_applied: bool,
    calibration_source: str,
    calibration_run_id: str | None,
    calibration_run_dir: str | None,
) -> SpreadForecast:
    # Store operational details on the output array for downstream persistence/debugging.
    try:
        attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
        attrs.update(
            {
                "calibration_applied": bool(calibration_applied),
                "calibration_source": str(calibration_source),
                "calibration_run_id": calibration_run_id,
                "calibration_run_dir": calibration_run_dir,
            }
        )
        forecast.probabilities.attrs = attrs
    except Exception:  # pragma: no cover
        pass
    return forecast


def _annotate_weather_bias(
    forecast: SpreadForecast,
    *,
    weather_bias_corrected: bool,
    weather_bias_corrector_path: str | None,
) -> SpreadForecast:
    try:
        attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
        attrs.update(
            {
                "weather_bias_corrected": bool(weather_bias_corrected),
                "weather_bias_corrector_path": weather_bias_corrector_path,
            }
        )
        forecast.probabilities.attrs = attrs
    except Exception:  # pragma: no cover
        pass

    LOGGER.info(
        "Weather inputs prepared for spread model",
        extra={
            "weather_bias_corrected": bool(weather_bias_corrected),
            "weather_bias_corrector_path": weather_bias_corrector_path,
        },
    )
    return forecast


def _apply_spread_calibration(
    *,
    forecast: SpreadForecast,
    calibrator: SpreadProbabilityCalibrator,
    calibrator_run_dir: Path,
    region_name: str,
) -> SpreadForecast:
    # Apply per-horizon calibration to preserve the (time,lat,lon) contract.
    horizons = list(forecast.horizons_hours)
    missing_h = [int(h) for h in horizons if int(h) not in calibrator.per_horizon_models]
    if missing_h:
        LOGGER.warning(
            "Calibration missing for some horizons; returning raw probabilities for those horizons.",
            extra={"region": region_name, "missing_horizons_hours": missing_h},
        )

    p = forecast.probabilities
    calibrated_slices = []
    for i, h in enumerate(horizons):
        raw = np.asarray(p.isel(time=i).values)
        calibrated = calibrator.calibrate_probs(raw, int(h))
        calibrated_slices.append(np.asarray(calibrated, dtype=np.float32))

    calibrated_stack = np.stack(calibrated_slices, axis=0).astype(np.float32, copy=False)
    out = p.copy(deep=False)
    out.values = calibrated_stack

    # Add lightweight provenance to attrs.
    meta = getattr(calibrator, "metadata", {}) or {}
    LOGGER.info(
        "Applied probability calibration",
        extra={
            "region": region_name,
            "calibrator_run_dir": str(calibrator_run_dir),
            "calibrator_run_id": meta.get("run_id"),
            "calibrator_method": meta.get("method"),
        },
    )

    forecast = SpreadForecast(
        probabilities=out,
        forecast_reference_time=forecast.forecast_reference_time,
        horizons_hours=forecast.horizons_hours,
    )
    forecast.validate()

    return _annotate_forecast(
        forecast,
        calibration_applied=True,
        calibration_source="service",
        calibration_run_id=meta.get("run_id"),
        calibration_run_dir=str(calibrator_run_dir),
    )

