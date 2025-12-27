"""Spread forecast service for orchestrating model execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from ml.spread.contract import DEFAULT_HORIZONS_HOURS, SpreadForecast, SpreadModel
from ml.spread.heuristic_v0 import HeuristicSpreadModelV0

# Lazily imported to avoid pulling heavy DB/raster deps at module import time.
# Kept as a module attribute so tests can patch `ml.spread.service.build_spread_inputs`.
build_spread_inputs = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

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

    inputs_package = build_spread_inputs(
        region_name=request.region_name,
        bbox=request.bbox,
        forecast_reference_time=request.forecast_reference_time,
        horizons_hours=request.horizons_hours,
        fire_lookback_hours=request.fire_lookback_hours,
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
    
    # 4. Finalize and log
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

