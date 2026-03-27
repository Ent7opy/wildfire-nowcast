"""Spread forecast service for orchestrating model execution.

See `docs/spread_model_design.md` for model behavior, assumptions, and limitations.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ml.calibration import SpreadProbabilityCalibrator
from ml.spread.contract import DEFAULT_HORIZONS_HOURS, SpreadForecast, SpreadModel, SpreadModelInput
from ml.spread.factory import get_model_version_hint
from ml.spread.heuristic_v0 import HeuristicSpreadModelV0
from ml.weather_bias_correction import resolve_weather_bias_corrector_path_full

# Lazily imported to avoid pulling heavy DB/raster deps at module import time.
# Kept as a module attribute so tests can patch `ml.spread.service.build_spread_inputs`.
build_spread_inputs = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

# Env/config knobs for operational inference.
SPREAD_CALIBRATOR_RUN_DIR_ENV = "SPREAD_CALIBRATOR_RUN_DIR"
SPREAD_CALIBRATOR_ROOT_ENV = "SPREAD_CALIBRATOR_ROOT"
WEATHER_BIAS_CORRECTOR_PATH_ENV = "WEATHER_BIAS_CORRECTOR_PATH"
WEATHER_BIAS_CORRECTOR_ROOT_ENV = "WEATHER_BIAS_CORRECTOR_ROOT"
STRICT_FORECAST_INPUTS_ENV = "STRICT_FORECAST_INPUTS"
SPREAD_STALE_WARN_HOURS_ENV = "SPREAD_STALE_WARN_HOURS"
SPREAD_SERVE_STALE_ENV = "SPREAD_SERVE_STALE"
SPREAD_SHADOW_ENABLED_ENV = "SPREAD_SHADOW_ENABLED"
SPREAD_SANITY_ENABLED_ENV = "SPREAD_SANITY_ENABLED"
SPREAD_SANITY_HIGH_PROB_THRESHOLD_ENV = "SPREAD_SANITY_HIGH_PROB_THRESHOLD"
SPREAD_SANITY_SEED_MIN_PROB_ENV = "SPREAD_SANITY_SEED_MIN_PROB"
SPREAD_SANITY_NEAR_SEED_PX_ENV = "SPREAD_SANITY_NEAR_SEED_PX"
SPREAD_MVP_GUARD_ENABLED_ENV = "SPREAD_MVP_GUARD_ENABLED"
SPREAD_MVP_GUARD_HORIZON_HOURS_ENV = "SPREAD_MVP_GUARD_HORIZON_HOURS"
SPREAD_MVP_GUARD_PROB_THRESHOLD_ENV = "SPREAD_MVP_GUARD_PROB_THRESHOLD"
SPREAD_MVP_GUARD_MAX_COVERAGE_ENV = "SPREAD_MVP_GUARD_MAX_COVERAGE"
SPREAD_MVP_GUARD_MAX_SEED_CELLS_ENV = "SPREAD_MVP_GUARD_MAX_SEED_CELLS"
# Science-grade weather strictness: hard-stop on zero-wind fallback when true.
# Set to true for science_grade maturity deployments; default false (mvp_operational).
SPREAD_STRICT_WEATHER_ENV = "SPREAD_STRICT_WEATHER"

# Performance limit: avoid OOM/high latency for very large areas in synchronous calls.
# 200x200 = 40,000 cells. At 0.01 degree, this is roughly 220km x 220km.
MAX_AOI_CELLS = 40000


class ForecastInputFallbackError(RuntimeError):
    """Raised when strict mode forbids fallback input data."""


class WeatherFallbackBlockedError(RuntimeError):
    """Raised when SPREAD_STRICT_WEATHER=true and zero-wind weather fallback was used.

    This is the science_grade hard-stop: do not serve a forecast built on fabricated
    wind inputs.  Set SPREAD_STRICT_WEATHER=false (default) for mvp_operational deployments
    where the warning path is acceptable.
    """


def _resolve_cluster_to_bbox(
    cluster_id: str,
) -> tuple[float, float, float, float]:
    """Decode a client-side cluster ID into a geographic bounding box.

    The UI encodes cluster IDs as ``cluster_z{zoom}_{row}_{col}`` where:
    - ``zoom``  — integer zoom level (1–10), clamped to [1, 10]
    - ``row``   — floor(lat / cellDeg), may be negative
    - ``col``   — floor(lon / cellDeg), may be negative
    - ``cellDeg = max(0.08, 8 / 2**zoom)`` — mirrors the JS formula in layerUtils.ts

    Returns
    -------
    tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat)

    Raises
    ------
    ValueError
        If the cluster_id does not match the expected format.
    """
    m = re.fullmatch(r"cluster_z(\d+)_(-?\d+)_(-?\d+)", cluster_id)
    if not m:
        raise ValueError(
            f"Cannot resolve cluster bbox: unrecognised cluster_id format {cluster_id!r}. "
            "Expected 'cluster_z<zoom>_<row>_<col>'."
        )

    zoom = max(1, min(int(m.group(1)), 10))
    row = int(m.group(2))
    col = int(m.group(3))
    cell_deg = max(0.08, 8.0 / (2 ** zoom))

    min_lat = row * cell_deg
    max_lat = min_lat + cell_deg
    min_lon = col * cell_deg
    max_lon = min_lon + cell_deg

    return (min_lon, min_lat, max_lon, max_lat)


@dataclass(frozen=True, slots=True)
class SpreadForecastRequest:
    """Request parameters for a spread forecast."""

    region_name: str | None  # None for location-based forecasting (grid derived from bbox)
    bbox: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    forecast_reference_time: datetime
    horizons_hours: Sequence[int] = DEFAULT_HORIZONS_HOURS
    fire_lookback_hours: int = 24
    fire_cluster_id: str | None = None
    strict_inputs: bool | None = None
    model_name: str | None = None
    model_params: dict[str, Any] | None = None
    shadow_model_name: str | None = None
    shadow_model_params: dict[str, Any] | None = None


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
        resolved_bbox = _resolve_cluster_to_bbox(request.fire_cluster_id)
        LOGGER.info(
            "Resolved fire_cluster_id %r to bbox %s",
            request.fire_cluster_id,
            resolved_bbox,
        )
        request = replace(request, bbox=resolved_bbox, fire_cluster_id=None)

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
    # Only resolve if region_name is provided (location-based forecasts skip bias correction).
    # SCIENCE_DEBT SD-02: snap location-based queries to nearest region to avoid bypassing
    # bias correction; see SCIENCE_DEBT.md for mitigation plan.
    weather_bias_corrector_path = None
    if request.region_name is not None:
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

    weather_fallback_reason = getattr(inputs_package.weather_cube, "attrs", {}).get("weather_fallback_reason")
    _enforce_strict_weather(
        weather_fallback_used=inputs_package.weather_fallback_used,
        weather_fallback_reason=weather_fallback_reason,
    )
    _enforce_no_fallback_if_strict(
        request=request,
        weather_fallback_used=inputs_package.weather_fallback_used,
        weather_fallback_reason=weather_fallback_reason,
        terrain_fallback_used=inputs_package.terrain_fallback_used,
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
        region_msg = f"region {request.region_name!r}" if request.region_name else "bbox"
        raise ValueError(
            f"AOI produces an empty window for {region_msg} and bbox {request.bbox}. "
            "Ensure the bbox is valid."
        )

    if n_cells > MAX_AOI_CELLS:
        raise ValueError(
            f"AOI too large: {n_cells} cells (max {MAX_AOI_CELLS}). "
            f"Window: {inputs_package.window.lat.size}x{inputs_package.window.lon.size}"
        )

    # Check industrial source coverage — warn if forecasting in a blind spot.
    # Lazy import mirrors the pattern used for build_spread_inputs.
    try:
        from api.industrial_coverage import query_industrial_coverage as _query_industrial_coverage

        _ind_cov = _query_industrial_coverage(request.bbox)
        if _ind_cov["source_count"] == 0:
            LOGGER.warning(
                "No industrial sources in forecast bbox — industrial noise filtering blind spot; "
                "fire detections in this region are not masked against false industrial alarms. "
                "Mitigation: ingest industrial sources for this region (science_grade target).",
                extra={"bbox": request.bbox, "region": request.region_name},
            )
        else:
            LOGGER.info(
                "Industrial source coverage ok",
                extra={
                    "bbox": request.bbox,
                    "source_count": _ind_cov["source_count"],
                    "coverage_fraction": _ind_cov["coverage_fraction"],
                },
            )
    except Exception as _exc:
        LOGGER.warning(
            "Could not check industrial source coverage: %s",
            _exc,
            extra={"bbox": request.bbox},
        )

    # 2. Select and run model
    if model is None:
        # Default to baseline heuristic unless request includes explicit selection.
        if request.model_name:
            from ml.spread.factory import get_spread_model

            model = get_spread_model(request.model_name, request.model_params)
        else:
            model = HeuristicSpreadModelV0()
    
    model_name = model.__class__.__name__
    LOGGER.info("Using spread model", extra={"model_name": model_name})

    # 3. Predict (champion)
    model_input = inputs_package.to_model_input()
    champion_start = time.perf_counter()
    forecast = model.predict(model_input)
    champion_latency_ms = (time.perf_counter() - champion_start) * 1000.0

    # Optional shadow inference: run challenger in parallel path but never serve it.
    shadow_summary: dict[str, Any] | None = None
    shadow_enabled = _env_bool(SPREAD_SHADOW_ENABLED_ENV, default=False)
    if shadow_enabled and request.shadow_model_name:
        try:
            from ml.spread.factory import get_spread_model

            challenger = get_spread_model(
                request.shadow_model_name,
                request.shadow_model_params,
            )
            shadow_start = time.perf_counter()
            challenger_fc = challenger.predict(model_input)
            shadow_latency_ms = (time.perf_counter() - shadow_start) * 1000.0

            champion_probs = np.asarray(forecast.probabilities.values, dtype=np.float32)
            challenger_probs = np.asarray(challenger_fc.probabilities.values, dtype=np.float32)
            if challenger_probs.shape[:1] != champion_probs.shape[:1]:
                n_t = min(challenger_probs.shape[0], champion_probs.shape[0])
                challenger_probs = challenger_probs[:n_t]
                champion_probs = champion_probs[:n_t]
            abs_delta = np.abs(challenger_probs - champion_probs)
            shadow_summary = {
                "shadow_model_name": request.shadow_model_name,
                "shadow_model_version": getattr(challenger_fc, "model_version", "") or "",
                "mean_abs_probability_delta": float(np.mean(abs_delta)),
                "max_abs_probability_delta": float(np.max(abs_delta)),
                "champion_latency_ms": float(champion_latency_ms),
                "shadow_latency_ms": float(shadow_latency_ms),
                "latency_delta_ms": float(shadow_latency_ms - champion_latency_ms),
                # These require observed outcomes and are computed offline by gate jobs.
                "brier_delta": None,
                "ece_delta": None,
            }
            LOGGER.info(
                "Shadow spread inference completed",
                extra={
                    "shadow_model_name": request.shadow_model_name,
                    "latency_delta_ms": shadow_summary["latency_delta_ms"],
                    "mean_abs_probability_delta": shadow_summary["mean_abs_probability_delta"],
                },
            )
        except Exception:
            LOGGER.exception(
                "Shadow spread inference failed; serving champion output only.",
                extra={"shadow_model_name": request.shadow_model_name},
            )
    if forecast.model_name == "unknown" or not forecast.model_name:
        forecast = SpreadForecast(
            probabilities=forecast.probabilities,
            forecast_reference_time=forecast.forecast_reference_time,
            horizons_hours=forecast.horizons_hours,
            model_name=model_name,
            model_version=forecast.model_version,
        )
    if not forecast.model_version:
        forecast = SpreadForecast(
            probabilities=forecast.probabilities,
            forecast_reference_time=forecast.forecast_reference_time,
            horizons_hours=forecast.horizons_hours,
            model_name=forecast.model_name,
            model_version=get_model_version_hint(forecast.model_name),
        )

    # 4. Validate output contract
    forecast.validate()
    forecast = _apply_spatial_sanity_guard(
        forecast=forecast,
        model=model,
        model_input=model_input,
        active_fire_heatmap=np.asarray(inputs_package.active_fires.heatmap),
    )
    forecast = _annotate_weather_bias(
        forecast,
        weather_bias_corrected=bool(getattr(inputs_package.weather_cube, "attrs", {}).get("weather_bias_corrected", False)),
        weather_bias_corrector_path=(
            getattr(inputs_package.weather_cube, "attrs", {}).get("weather_bias_corrector_path")
            or (str(weather_bias_corrector_path) if weather_bias_corrector_path is not None else None)
        ),
    )

    # 4a. Annotate fallback information for observability
    forecast = _annotate_fallback_info(
        forecast,
        weather_fallback_used=inputs_package.weather_fallback_used,
        weather_fallback_reason=getattr(inputs_package.weather_cube, "attrs", {}).get("weather_fallback_reason"),
        terrain_fallback_used=inputs_package.terrain_fallback_used,
    )
    forecast = _annotate_confidence_info(
        forecast,
        weather_cube=inputs_package.weather_cube,
        forecast_reference_time=request.forecast_reference_time,
        weather_fallback_used=inputs_package.weather_fallback_used,
        terrain_fallback_used=inputs_package.terrain_fallback_used,
    )
    forecast = _annotate_shadow_info(
        forecast,
        shadow_summary=shadow_summary,
    )
    forecast = _annotate_lineage_info(
        forecast,
        weather_cube=inputs_package.weather_cube,
        terrain_fallback_used=inputs_package.terrain_fallback_used,
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
        # Only resolve calibrator if region_name is provided
        calibrator_run_dir = None
        if request.region_name is not None:
            calibrator_run_dir = _resolve_spread_calibrator_run_dir(request.region_name)
        if calibrator_run_dir is None:
            region_msg = request.region_name if request.region_name else "location-based"
            LOGGER.warning(
                "No spread calibrator configured; returning uncalibrated probabilities.",
                extra={"region": region_msg, "env": SPREAD_CALIBRATOR_RUN_DIR_ENV},
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

    forecast = _apply_mvp_footprint_guard(
        forecast=forecast,
        request=request,
        active_fire_heatmap=np.asarray(inputs_package.active_fires.heatmap),
    )
    
    # 5. Finalize and log
    duration = time.perf_counter() - start_time
    LOGGER.info(
        "Spread forecast completed",
        extra={
            "duration_s": round(duration, 3),
            "model": forecast.model_name,
            "n_cells": int(n_cells),
            "output_min": float(forecast.probabilities.min()),
            "output_max": float(forecast.probabilities.max()),
        },
    )

    return forecast


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _resolve_strict_inputs(strict_inputs: bool | None) -> bool:
    if strict_inputs is not None:
        return bool(strict_inputs)
    return _env_bool(STRICT_FORECAST_INPUTS_ENV, default=False)


def _staleness_hours_from_weather(
    weather_cube: Any,
    *,
    forecast_reference_time: datetime,
) -> float | None:
    attrs = dict(getattr(weather_cube, "attrs", {}) or {})
    run_time_raw = attrs.get("weather_run_time")
    if not run_time_raw:
        return None
    try:
        run_time = datetime.fromisoformat(str(run_time_raw).replace("Z", "+00:00"))
        if run_time.tzinfo is None:
            run_time = run_time.replace(tzinfo=timezone.utc)
        run_time = run_time.astimezone(timezone.utc)
        ref_time = forecast_reference_time.astimezone(timezone.utc)
        delta = ref_time - run_time
        return max(0.0, delta.total_seconds() / 3600.0)
    except Exception:
        return None


def _enforce_no_fallback_if_strict(
    *,
    request: SpreadForecastRequest,
    weather_fallback_used: bool,
    weather_fallback_reason: str | None,
    terrain_fallback_used: bool,
) -> None:
    if not _resolve_strict_inputs(request.strict_inputs):
        return

    reasons: list[str] = []
    if weather_fallback_used:
        reason = weather_fallback_reason or "unknown"
        reasons.append(f"weather fallback used ({reason})")
    # Location-based forecasts intentionally use terrain fallback when region_name=None.
    if request.region_name is not None and terrain_fallback_used:
        reasons.append("terrain fallback used")
    if reasons:
        raise ForecastInputFallbackError(
            "Strict forecast inputs mode rejected this request: " + "; ".join(reasons)
        )


def _enforce_strict_weather(
    *,
    weather_fallback_used: bool,
    weather_fallback_reason: str | None,
) -> None:
    """Hard-stop if SPREAD_STRICT_WEATHER=true and zero-wind fallback was used.

    This is the science_grade enforcement gate.  Set SPREAD_STRICT_WEATHER=true on
    science_grade deployments; leave false (default) for mvp_operational.

    Raises
    ------
    WeatherFallbackBlockedError
        STOP: zero-wind weather fallback is not permitted in strict-weather mode.
    """
    if not _env_bool(SPREAD_STRICT_WEATHER_ENV, default=False):
        return
    if weather_fallback_used:
        reason = weather_fallback_reason or "unknown"
        raise WeatherFallbackBlockedError(
            f"STOP: SPREAD_STRICT_WEATHER=true rejects zero-wind weather fallback "
            f"(reason: {reason}). "
            "Ensure a valid weather run is available or disable strict-weather mode "
            "for mvp_operational deployments."
        )


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
    """Resolve weather bias corrector path using centralized resolution.
    
    Delegates to ml.weather_bias_correction.resolve_weather_bias_corrector_path_full
    to ensure consistent path resolution across the codebase.
    """
    return resolve_weather_bias_corrector_path_full(
        region_name,
        explicit_path_env=WEATHER_BIAS_CORRECTOR_PATH_ENV,
        root_env=WEATHER_BIAS_CORRECTOR_ROOT_ENV,
    )


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
    has_uncalibrated_horizons: bool = False,
    uncalibrated_horizons: list[int] | None = None,
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
                "has_uncalibrated_horizons": has_uncalibrated_horizons,
                "uncalibrated_horizons": uncalibrated_horizons if uncalibrated_horizons else [],
                "model_name": forecast.model_name,
                "model_version": forecast.model_version,
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


def _annotate_fallback_info(
    forecast: SpreadForecast,
    *,
    weather_fallback_used: bool,
    weather_fallback_reason: str | None,
    terrain_fallback_used: bool,
) -> SpreadForecast:
    """Annotate forecast with fallback information for observability.
    
    This helps users understand if the forecast was generated using fallback
    data (e.g., zero-wind weather or empty terrain), which may affect accuracy.
    """
    try:
        attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
        attrs.update(
            {
                "weather_fallback_used": bool(weather_fallback_used),
                "weather_fallback_reason": weather_fallback_reason,
                "terrain_fallback_used": bool(terrain_fallback_used),
            }
        )
        forecast.probabilities.attrs = attrs
    except Exception:  # pragma: no cover
        pass

    if weather_fallback_used:
        LOGGER.warning(
            "Forecast generated with fallback weather data (zero-wind). "
            "Reason: %s. Forecast accuracy may be reduced.",
            weather_fallback_reason or "unknown",
            extra={
                "weather_fallback_used": True,
                "weather_fallback_reason": weather_fallback_reason,
            },
        )
    if terrain_fallback_used:
        LOGGER.warning(
            "Forecast generated with fallback terrain data (empty). "
            "Terrain-based spread factors are disabled.",
            extra={"terrain_fallback_used": True},
        )

    return forecast


def _annotate_confidence_info(
    forecast: SpreadForecast,
    *,
    weather_cube: Any,
    forecast_reference_time: datetime,
    weather_fallback_used: bool,
    terrain_fallback_used: bool,
) -> SpreadForecast:
    stale_warn_h = _env_float(SPREAD_STALE_WARN_HOURS_ENV, default=12.0)
    staleness_h = _staleness_hours_from_weather(
        weather_cube, forecast_reference_time=forecast_reference_time
    )
    fallback_used = bool(weather_fallback_used or terrain_fallback_used)
    confidence_level = "normal"
    if fallback_used or (staleness_h is not None and staleness_h >= stale_warn_h):
        confidence_level = "low"

    try:
        attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
        attrs.update(
            {
                "confidence_level": confidence_level,
                "staleness_hours": staleness_h,
                "fallback_used": fallback_used,
                "stale_warn_hours": stale_warn_h,
                "serve_stale": _env_bool(SPREAD_SERVE_STALE_ENV, default=True),
            }
        )
        forecast.probabilities.attrs = attrs
    except Exception:  # pragma: no cover
        pass

    return forecast


def _annotate_lineage_info(
    forecast: SpreadForecast,
    *,
    weather_cube: Any,
    terrain_fallback_used: bool,
) -> SpreadForecast:
    """Attach authoritative data-lineage attributes to the forecast output.

    These attrs are the machine-readable counterpart to ``docs/spread_data_sources.md``
    and must be persisted with every forecast run for traceability.
    """
    weather_attrs = dict(getattr(weather_cube, "attrs", {}) or {})
    weather_fallback = bool(weather_attrs.get("weather_fallback_used", False))

    # Derive weather source label from run metadata, falling back to "fallback_zeros".
    if weather_fallback:
        weather_source = "fallback_zeros"
    else:
        # weather_run_id is the DB identifier; model name is not stored separately in attrs yet.
        weather_source = "noaa_gfs_025deg"

    # Detect whether the source doc exists so consumers can flag undeclared lineage.
    try:
        from pathlib import Path
        _doc = Path(__file__).parents[2] / "docs" / "spread_data_sources.md"
        sources_declared = _doc.exists()
    except Exception:
        sources_declared = False

    try:
        attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
        attrs.update(
            {
                "lineage_fires_source": "nasa_firms_viirs_nrt",
                "lineage_weather_source": weather_source,
                "lineage_weather_run_id": weather_attrs.get("weather_run_id"),
                "lineage_terrain_source": "fallback_zeros" if terrain_fallback_used else "dem_derived",
                "lineage_fuels_ndvi_source": "esa_worldcover_10m",
                "lineage_fuels_lfmc_source": "ecmwf_ecland_lfmc",
                "lineage_fuels_dfmc_source": "nfdrs_nelson1984",
                "lineage_data_sources_declared": sources_declared,
            }
        )
        forecast.probabilities.attrs = attrs
    except Exception:  # pragma: no cover
        pass

    LOGGER.info(
        "Spread forecast lineage",
        extra={
            "lineage_fires_source": "nasa_firms_viirs_nrt",
            "lineage_weather_source": weather_source,
            "lineage_weather_run_id": weather_attrs.get("weather_run_id"),
            "lineage_terrain_source": "fallback_zeros" if terrain_fallback_used else "dem_derived",
            "lineage_data_sources_declared": sources_declared,
        },
    )
    return forecast


def _annotate_shadow_info(
    forecast: SpreadForecast,
    *,
    shadow_summary: dict[str, Any] | None,
) -> SpreadForecast:
    try:
        attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
        attrs["shadow_evaluated"] = bool(shadow_summary is not None)
        attrs["shadow_metrics_summary"] = shadow_summary or {}
        forecast.probabilities.attrs = attrs
    except Exception:  # pragma: no cover
        pass
    return forecast


def _spatial_sanity_failure_reason(
    *,
    forecast: SpreadForecast,
    active_fire_heatmap: np.ndarray,
    high_prob_threshold: float,
    seed_min_prob: float,
    near_seed_px: int,
) -> str | None:
    """Return a reason string when spread is spatially implausible, else None.

    Heuristic used for operational safety:
    - We expect at least some high-probability cells to stay near seeded active fire cells.
    - If seed probability is near-zero while all high-probability cells are away from seeds,
      learned output is likely out-of-distribution artifact and should be rejected.
    """
    try:
        seeds = np.asarray(active_fire_heatmap, dtype=np.float32) > 0.0
        if seeds.ndim != 2 or not np.any(seeds):
            return None

        horizons = list(getattr(forecast, "horizons_hours", []) or [])
        if not horizons:
            return None
        # Prefer the longest available horizon for the sanity check; it has the
        # most developed footprint and is most likely to expose out-of-distribution outputs.
        idx = len(horizons) - 1
        horizon_h = int(horizons[idx])
        probs = np.asarray(forecast.probabilities.isel(time=idx).values, dtype=np.float32)
        if probs.shape != seeds.shape:
            return None

        high = probs >= float(high_prob_threshold)
        if not np.any(high):
            return None

        seed_prob_max = float(np.max(probs[seeds]))
        r = max(0, int(near_seed_px))
        seed_ij = np.argwhere(seeds)
        has_high_near_seed = False
        for i, j in seed_ij:
            i0 = max(0, int(i) - r)
            i1 = min(probs.shape[0], int(i) + r + 1)
            j0 = max(0, int(j) - r)
            j1 = min(probs.shape[1], int(j) + r + 1)
            if np.any(high[i0:i1, j0:j1]):
                has_high_near_seed = True
                break

        if seed_prob_max < float(seed_min_prob) and not has_high_near_seed:
            return (
                "spatial_sanity_failed:"
                f"horizon={horizon_h},seed_prob_max={seed_prob_max:.4f},"
                f"high_threshold={float(high_prob_threshold):.3f},near_seed_px={r}"
            )
        return None
    except Exception:
        LOGGER.exception("Spatial sanity check failed unexpectedly; skipping guard.")
        return None


def _annotate_spatial_sanity(
    forecast: SpreadForecast,
    *,
    sanity_fallback_used: bool,
    sanity_fallback_reason: str | None,
    sanity_original_model_name: str | None,
) -> SpreadForecast:
    try:
        attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
        attrs.update(
            {
                "sanity_fallback_used": bool(sanity_fallback_used),
                "sanity_fallback_reason": sanity_fallback_reason,
                "sanity_original_model_name": sanity_original_model_name,
                "sanity_served_model_name": forecast.model_name,
            }
        )
        forecast.probabilities.attrs = attrs
    except Exception:  # pragma: no cover
        pass
    return forecast


def _apply_spatial_sanity_guard(
    *,
    forecast: SpreadForecast,
    model: SpreadModel,
    model_input: SpreadModelInput,
    active_fire_heatmap: np.ndarray,
) -> SpreadForecast:
    if not _env_bool(SPREAD_SANITY_ENABLED_ENV, default=True):
        return forecast
    if (
        getattr(model.__class__, "__name__", "") == "HeuristicSpreadModelV0"
        or getattr(forecast, "model_name", "") == "HeuristicSpreadModelV0"
    ):
        return _annotate_spatial_sanity(
            forecast,
            sanity_fallback_used=False,
            sanity_fallback_reason=None,
            sanity_original_model_name=None,
        )

    high_prob_threshold = _env_float(SPREAD_SANITY_HIGH_PROB_THRESHOLD_ENV, default=0.3)
    seed_min_prob = _env_float(SPREAD_SANITY_SEED_MIN_PROB_ENV, default=0.05)
    near_seed_px = _env_int(SPREAD_SANITY_NEAR_SEED_PX_ENV, default=8)
    reason = _spatial_sanity_failure_reason(
        forecast=forecast,
        active_fire_heatmap=active_fire_heatmap,
        high_prob_threshold=high_prob_threshold,
        seed_min_prob=seed_min_prob,
        near_seed_px=near_seed_px,
    )
    if reason is None:
        return _annotate_spatial_sanity(
            forecast,
            sanity_fallback_used=False,
            sanity_fallback_reason=None,
            sanity_original_model_name=model.__class__.__name__,
        )

    LOGGER.warning(
        "Spatial sanity guard triggered; serving heuristic fallback.",
        extra={
            "reason": reason,
            "original_model": model.__class__.__name__,
            "high_prob_threshold": high_prob_threshold,
            "seed_min_prob": seed_min_prob,
            "near_seed_px": near_seed_px,
        },
    )
    fallback = HeuristicSpreadModelV0()
    out = fallback.predict(model_input)
    out.validate()
    return _annotate_spatial_sanity(
        out,
        sanity_fallback_used=True,
        sanity_fallback_reason=reason,
        sanity_original_model_name=model.__class__.__name__,
    )


def _mvp_footprint_guard_metrics(
    *,
    forecast: SpreadForecast,
    active_fire_heatmap: np.ndarray,
    horizon_hours: int,
    probability_threshold: float,
) -> dict[str, Any]:
    try:
        seeds = np.asarray(active_fire_heatmap, dtype=np.float32) > 0.0
        seed_count = int(np.count_nonzero(seeds))
        horizons = list(getattr(forecast, "horizons_hours", []) or [])
        if not horizons:
            return {
                "seed_cell_count": seed_count,
                "horizon_hours": int(horizon_hours),
                "probability_threshold": float(probability_threshold),
                "coverage_fraction": None,
                "status": "missing_horizons",
            }

        horizons_int = [int(h) for h in horizons]
        if int(horizon_hours) not in horizons_int:
            return {
                "seed_cell_count": seed_count,
                "horizon_hours": int(horizon_hours),
                "probability_threshold": float(probability_threshold),
                "coverage_fraction": None,
                "status": "horizon_not_available",
            }

        idx = horizons_int.index(int(horizon_hours))
        probs = np.asarray(forecast.probabilities.isel(time=idx).values, dtype=np.float32)
        if probs.ndim != 2 or probs.size == 0:
            return {
                "seed_cell_count": seed_count,
                "horizon_hours": int(horizon_hours),
                "probability_threshold": float(probability_threshold),
                "coverage_fraction": None,
                "status": "empty_probability_grid",
            }

        coverage_fraction = float(np.count_nonzero(probs >= float(probability_threshold)) / probs.size)
        return {
            "seed_cell_count": seed_count,
            "horizon_hours": int(horizon_hours),
            "probability_threshold": float(probability_threshold),
            "coverage_fraction": coverage_fraction,
            "status": "ok",
        }
    except Exception:
        return {
            "seed_cell_count": int(np.count_nonzero(np.asarray(active_fire_heatmap, dtype=np.float32) > 0.0)),
            "horizon_hours": int(horizon_hours),
            "probability_threshold": float(probability_threshold),
            "coverage_fraction": None,
            "status": "metrics_error",
        }


def _annotate_mvp_guardrail(
    forecast: SpreadForecast,
    *,
    triggered: bool,
    mode: str,
    reason: str | None,
    metrics: dict[str, Any],
) -> SpreadForecast:
    try:
        attrs = dict(getattr(forecast.probabilities, "attrs", {}) or {})
        attrs.update(
            {
                "mvp_guardrail_triggered": bool(triggered),
                "mvp_guardrail_mode": mode,
                "mvp_guardrail_reason": reason,
                "mvp_guardrail_metrics": metrics,
            }
        )
        if triggered and mode == "warn":
            attrs["confidence_level"] = "low"
        forecast.probabilities.attrs = attrs
    except Exception:  # pragma: no cover
        pass
    return forecast


def _apply_mvp_footprint_guard(
    *,
    forecast: SpreadForecast,
    request: SpreadForecastRequest,
    active_fire_heatmap: np.ndarray,
) -> SpreadForecast:
    if not _env_bool(SPREAD_MVP_GUARD_ENABLED_ENV, default=True):
        return forecast

    horizon_h = _env_int(SPREAD_MVP_GUARD_HORIZON_HOURS_ENV, default=12)
    prob_threshold = _env_float(SPREAD_MVP_GUARD_PROB_THRESHOLD_ENV, default=0.7)
    max_coverage = _env_float(SPREAD_MVP_GUARD_MAX_COVERAGE_ENV, default=0.60)
    max_seed_cells = _env_int(SPREAD_MVP_GUARD_MAX_SEED_CELLS_ENV, default=1)
    strict_mode = _resolve_strict_inputs(request.strict_inputs)

    metrics = _mvp_footprint_guard_metrics(
        forecast=forecast,
        active_fire_heatmap=active_fire_heatmap,
        horizon_hours=horizon_h,
        probability_threshold=prob_threshold,
    )
    seed_count = int(metrics.get("seed_cell_count") or 0)
    coverage_fraction = metrics.get("coverage_fraction")
    status = str(metrics.get("status") or "unknown")

    # Guard applies only to single-seed (or configured max seed count) runs.
    if seed_count > int(max_seed_cells):
        return _annotate_mvp_guardrail(
            forecast,
            triggered=False,
            mode="skip",
            reason=f"seed_count={seed_count} exceeds max_seed_cells={int(max_seed_cells)}",
            metrics=metrics,
        )

    if status != "ok" or coverage_fraction is None:
        return _annotate_mvp_guardrail(
            forecast,
            triggered=False,
            mode="skip",
            reason=f"guard_not_evaluated:{status}",
            metrics=metrics,
        )

    coverage_fraction = float(coverage_fraction)
    if coverage_fraction <= float(max_coverage):
        return _annotate_mvp_guardrail(
            forecast,
            triggered=False,
            mode="pass",
            reason=None,
            metrics=metrics,
        )

    reason = (
        "mvp_guardrail_oversized_footprint:"
        f"seed_cell_count={seed_count},"
        f"horizon_hours={int(horizon_h)},"
        f"probability_threshold={float(prob_threshold):.3f},"
        f"coverage_fraction={coverage_fraction:.4f},"
        f"max_coverage={float(max_coverage):.4f}"
    )
    if strict_mode:
        raise ValueError(f"MVP_GUARD_STOP: {reason}")

    LOGGER.warning(
        "MVP footprint guardrail triggered in warn mode.",
        extra={
            "reason": reason,
            "metrics": metrics,
            "strict_mode": strict_mode,
        },
    )
    return _annotate_mvp_guardrail(
        forecast,
        triggered=True,
        mode="warn",
        reason=reason,
        metrics=metrics,
    )


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
    has_uncalibrated_horizons = len(missing_h) > 0
    
    if has_uncalibrated_horizons:
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
            "uncalibrated_horizons": missing_h if has_uncalibrated_horizons else [],
        },
    )

    forecast = SpreadForecast(
        probabilities=out,
        forecast_reference_time=forecast.forecast_reference_time,
        horizons_hours=forecast.horizons_hours,
        model_name=forecast.model_name,
        model_version=forecast.model_version,
    )
    forecast.validate()

    return _annotate_forecast(
        forecast,
        calibration_applied=True,
        calibration_source="service",
        calibration_run_id=meta.get("run_id"),
        calibration_run_dir=str(calibrator_run_dir),
        has_uncalibrated_horizons=has_uncalibrated_horizons,
        uncalibrated_horizons=missing_h if has_uncalibrated_horizons else [],
    )
