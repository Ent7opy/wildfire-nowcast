"""Ignition probability grid computation."""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.db import get_engine
from api.model_registry import resolve_active_model

LOGGER = logging.getLogger(__name__)

_MAX_CELLS = 500
_DROUGHT_STALE_DAYS = 10


class IgnitionModelUnavailable(Exception):
    pass


class IgnitionInferenceFailed(Exception):
    pass


def _classify_level(probability: float, thresholds: dict[str, float]) -> str:
    low_max = thresholds.get("low_max", 0.2)
    elevated_max = thresholds.get("elevated_max", 0.5)
    high_max = thresholds.get("high_max", 0.8)
    if probability <= low_max:
        return "low"
    elif probability <= elevated_max:
        return "elevated"
    elif probability <= high_max:
        return "high"
    return "critical"


def _build_cell_id(lat: float, lon: float) -> str:
    lat_str = f"{lat:.4f}".replace("-", "S").replace(".", "d")
    lon_str = f"{lon:.4f}".replace("-", "W").replace(".", "d")
    return f"cell_{lat_str}_{lon_str}"


def _compute_grid_dims(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    cell_size_km: float,
) -> tuple[int, int, float, float]:
    center_lat = (min_lat + max_lat) / 2.0
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * math.cos(math.radians(center_lat)))

    cell_size_lat = cell_size_km * lat_per_km
    cell_size_lon = cell_size_km * lon_per_km

    n_lat = max(1, int((max_lat - min_lat) / cell_size_lat))
    n_lon = max(1, int((max_lon - min_lon) / cell_size_lon))

    if n_lat * n_lon > _MAX_CELLS:
        scale_factor = math.sqrt(_MAX_CELLS / (n_lat * n_lon))
        n_lat = max(1, int(n_lat * scale_factor))
        n_lon = max(1, int(n_lon * scale_factor))

    cell_size_lat = (max_lat - min_lat) / n_lat
    cell_size_lon = (max_lon - min_lon) / n_lon

    return n_lat, n_lon, cell_size_lat, cell_size_lon


def _resolve_valid_time(horizon: str, now: datetime) -> tuple[datetime, int]:
    if horizon == "now":
        return now, 0
    elif horizon == "+24h":
        return now + timedelta(hours=24), 24
    elif horizon == "+48h":
        return now + timedelta(hours=48), 48
    raise ValueError(f"Unknown horizon: {horizon!r}")


def _query_weather_for_cells(
    engine: Engine,
    lats: list[float],
    lons: list[float],
    ref_time: datetime,
    forecast_hour: int,
) -> dict[tuple[float, float], dict[str, Any]]:
    if not lats:
        return {}

    gfs_grid_deg = 0.25

    def snap(v: float) -> float:
        return round(round(v / gfs_grid_deg) * gfs_grid_deg, 6)

    snapped_pairs = {(snap(lat), snap(lon)) for lat, lon in zip(lats, lons)}
    min_lat = min(p[0] for p in snapped_pairs) - gfs_grid_deg
    max_lat = max(p[0] for p in snapped_pairs) + gfs_grid_deg
    min_lon = min(p[1] for p in snapped_pairs) - gfs_grid_deg
    max_lon = max(p[1] for p in snapped_pairs) + gfs_grid_deg

    stmt = text(
        """
        SELECT DISTINCT ON (wpc.lat_grid, wpc.lon_grid)
            wpc.lat_grid,
            wpc.lon_grid,
            wpc.t2m - 273.15                                   AS temperature_c,
            wpc.rh2m                                           AS relative_humidity,
            SQRT(wpc.u10 * wpc.u10 + wpc.v10 * wpc.v10) * 3.6 AS wind_speed_kmh,
            wpc.tp                                             AS tp
        FROM weather_point_cache wpc
        JOIN weather_runs wr ON wpc.run_id = wr.id
        WHERE wpc.lat_grid BETWEEN :min_lat AND :max_lat
          AND wpc.lon_grid BETWEEN :min_lon AND :max_lon
          AND wpc.forecast_hour = :forecast_hour
          AND wr.run_time <= :ref_time
          AND wr.status = 'complete'
        ORDER BY wpc.lat_grid, wpc.lon_grid, wr.run_time DESC
        """
    )

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                stmt,
                {
                    "min_lat": min_lat,
                    "max_lat": max_lat,
                    "min_lon": min_lon,
                    "max_lon": max_lon,
                    "forecast_hour": forecast_hour,
                    "ref_time": ref_time,
                },
            ).mappings().fetchall()
    except Exception as exc:
        LOGGER.warning("Weather query failed for ignition grid: %s", exc)
        return {}

    result: dict[tuple[float, float], dict[str, Any]] = {}
    for row in rows:
        key = (float(row["lat_grid"]), float(row["lon_grid"]))
        result[key] = dict(row)
    return result


def _query_latest_weather_run_time(engine: Engine, forecast_hour: int) -> datetime | None:
    stmt = text(
        """
        SELECT MAX(wr.run_time) AS latest
        FROM weather_runs wr
        WHERE wr.status = 'complete'
          AND EXISTS (
              SELECT 1 FROM weather_point_cache wpc
              WHERE wpc.run_id = wr.id AND wpc.forecast_hour = :forecast_hour
              LIMIT 1
          )
        """
    )
    try:
        with engine.connect() as conn:
            row = conn.execute(stmt, {"forecast_hour": forecast_hour}).mappings().first()
        if row and row["latest"] is not None:
            val = row["latest"]
            if val.tzinfo is None:
                return val.replace(tzinfo=timezone.utc)
            return val.astimezone(timezone.utc)
    except Exception as exc:
        LOGGER.warning("Failed to query latest weather run time: %s", exc)
    return None


def _query_drought_index_freshness(
    engine: Engine,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    now: datetime,
) -> datetime | None:
    stmt = text(
        """
        SELECT MAX(valid_time) AS latest
        FROM drought_index_runs
        WHERE status = 'complete'
          AND bbox_min_lon <= :max_lon AND bbox_max_lon >= :min_lon
          AND bbox_min_lat <= :max_lat AND bbox_max_lat >= :min_lat
        """
    )
    try:
        with engine.connect() as conn:
            row = conn.execute(
                stmt,
                {
                    "min_lon": min_lon, "max_lon": max_lon,
                    "min_lat": min_lat, "max_lat": max_lat,
                },
            ).mappings().first()
        if row and row["latest"] is not None:
            val = row["latest"]
            if val.tzinfo is None:
                return val.replace(tzinfo=timezone.utc)
            return val.astimezone(timezone.utc)
    except Exception as exc:
        LOGGER.warning("drought_index_freshness query failed: %s", exc)
    return None


def _query_thunderstorm_present(
    engine: Engine,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    ref_time: datetime,
    tolerance_hours: float = 6.0,
) -> bool:
    stmt = text(
        """
        SELECT EXISTS (
            SELECT 1 FROM ignition_lightning_proxy
            WHERE grid_lon BETWEEN :min_lon AND :max_lon
              AND grid_lat BETWEEN :min_lat AND :max_lat
              AND valid_time BETWEEN :t_start AND :t_end
        ) AS present
        """
    )
    t_start = ref_time - timedelta(hours=tolerance_hours)
    t_end = ref_time + timedelta(hours=tolerance_hours)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                stmt,
                {
                    "min_lon": min_lon, "max_lon": max_lon,
                    "min_lat": min_lat, "max_lat": max_lat,
                    "t_start": t_start, "t_end": t_end,
                },
            ).mappings().first()
        if row:
            return bool(row["present"])
    except Exception as exc:
        LOGGER.warning("thunderstorm_present query failed: %s", exc)
    return False


def _check_gfs_48h_available(
    engine: Engine,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    ref_time: datetime,
) -> bool:
    gfs_grid_deg = 0.25
    stmt = text(
        """
        SELECT COUNT(*) AS cnt
        FROM weather_point_cache wpc
        JOIN weather_runs wr ON wpc.run_id = wr.id
        WHERE wpc.lat_grid BETWEEN :min_lat AND :max_lat
          AND wpc.lon_grid BETWEEN :min_lon AND :max_lon
          AND wpc.forecast_hour = 48
          AND wr.run_time <= :ref_time
          AND wr.status = 'complete'
        LIMIT 1
        """
    )
    try:
        with engine.connect() as conn:
            row = conn.execute(
                stmt,
                {
                    "min_lat": min_lat - gfs_grid_deg,
                    "max_lat": max_lat + gfs_grid_deg,
                    "min_lon": min_lon - gfs_grid_deg,
                    "max_lon": max_lon + gfs_grid_deg,
                    "ref_time": ref_time,
                },
            ).mappings().first()
        return bool(row and row["cnt"] > 0)
    except Exception:
        return False


def _get_weather_for_cell(
    weather_map: dict[tuple[float, float], dict[str, Any]],
    lat: float,
    lon: float,
    gfs_grid_deg: float = 0.25,
) -> dict[str, Any]:
    def snap(v: float) -> float:
        return round(round(v / gfs_grid_deg) * gfs_grid_deg, 6)

    key = (snap(lat), snap(lon))
    return weather_map.get(key, {})


def _run_onnx_inference(
    artifact_uri: str,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    import onnxruntime as ort  # noqa: PLC0415

    sess = ort.InferenceSession(artifact_uri, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: feature_matrix.astype(np.float32)})
    proba = output[1] if len(output) > 1 else output[0]
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1].astype(float)
    return proba.ravel().astype(float)


def compute_ignition_grid(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    *,
    cell_size_km: float = 10.0,
    horizon: str = "now",
    engine: Engine | None = None,
) -> dict:
    db = engine or get_engine()
    now = datetime.now(timezone.utc)

    active = resolve_active_model("ignition", engine=db)
    if active is None:
        ignition_required = os.environ.get("IGNITION_REQUIRED", "true").lower() == "true"
        if ignition_required:
            raise IgnitionModelUnavailable("No promoted ignition model found.")
        LOGGER.warning(
            "WARNING [ignition-grid] No promoted ignition model and IGNITION_REQUIRED=false. "
            "Returning zero probabilities. TARGET_STAGE: science_grade"
        )

    model_id = active["model_id"] if active else "none"
    artifact_uri = active["artifact_uri"] if active else None
    metrics_json = active["metrics_json"] if active else {}

    runtime_contract = metrics_json.get("runtime_contract", {}) if metrics_json else {}
    # top_features is for the `signals` response field only — display purposes
    top_features: list[str] = runtime_contract.get("top_features", [
        "temperature_c", "relative_humidity", "wind_speed_kmh",
        "precip_last_7d_mm", "days_since_last_burn",
    ])[:5]
    # required_features governs the inference input matrix (exact order matters)
    _default_required = [
        "temperature_c", "relative_humidity", "wind_speed_kmh",
        "precip_last_7d_mm", "days_since_last_burn",
    ]
    required_features: list[str] = runtime_contract.get("required_features", _default_required)
    missing_feature_policy: str = runtime_contract.get("missing_feature_policy", "")
    thresholds: dict[str, float] = runtime_contract.get("thresholds", {
        "low_max": 0.2,
        "elevated_max": 0.5,
        "high_max": 0.8,
    })

    valid_time, forecast_hour = _resolve_valid_time(horizon, now)
    low_confidence = horizon == "+48h"

    coverage_warnings: list[str] = []

    if horizon == "now":
        latest_run_time = _query_latest_weather_run_time(db, forecast_hour=0)
        if latest_run_time is not None:
            display_valid_time = latest_run_time
        else:
            display_valid_time = now
            coverage_warnings.append("weather_run_unavailable: valid_time is approximate")
    else:
        display_valid_time = valid_time

    if horizon == "+48h":
        gfs_48h_ok = _check_gfs_48h_available(db, min_lon, min_lat, max_lon, max_lat, now)
        if not gfs_48h_ok:
            coverage_warnings.append("gfs_+48h_unavailable: using latest available step")

    drought_freshness = _query_drought_index_freshness(db, min_lon, min_lat, max_lon, max_lat, now)
    if drought_freshness is None:
        drought_stale_date = "unknown"
        coverage_warnings.append(f"drought_index_stale: last updated {drought_stale_date}")
    else:
        age_days = (now - drought_freshness).total_seconds() / 86400.0
        if age_days > _DROUGHT_STALE_DAYS:
            coverage_warnings.append(
                f"drought_index_stale: last updated {drought_freshness.strftime('%Y-%m-%d')}"
            )

    thunderstorm_present = _query_thunderstorm_present(
        db, min_lon, min_lat, max_lon, max_lat, display_valid_time
    )
    if not thunderstorm_present:
        coverage_warnings.append("thunderstorm_data_missing")

    n_lat, n_lon, cell_size_lat, cell_size_lon = _compute_grid_dims(
        min_lon, min_lat, max_lon, max_lat, cell_size_km
    )

    cell_lats: list[float] = []
    cell_lons: list[float] = []

    for i_lat in range(n_lat):
        for i_lon in range(n_lon):
            cell_center_lat = min_lat + (i_lat + 0.5) * cell_size_lat
            cell_center_lon = min_lon + (i_lon + 0.5) * cell_size_lon
            cell_lats.append(cell_center_lat)
            cell_lons.append(cell_center_lon)

    weather_map = _query_weather_for_cells(
        db, cell_lats, cell_lons, display_valid_time, forecast_hour
    )

    # Map from canonical weather column aliases to feature names used in the contract
    _WEATHER_ALIASES: dict[str, str] = {
        "precip_last_7d_mm": "tp",
        "days_since_last_burn": None,  # not from weather; use static default
    }
    _STATIC_DEFAULTS: dict[str, float] = {
        "temperature_c": 20.0,
        "relative_humidity": 50.0,
        "wind_speed_kmh": 10.0,
        "precip_last_7d_mm": 0.0,
        "days_since_last_burn": 365.0,
        "tp": 0.0,
    }

    feature_rows: list[list[float]] = []
    for lat, lon in zip(cell_lats, cell_lons):
        w = _get_weather_for_cell(weather_map, lat, lon)
        # Build a lookup that maps feature names to their values
        cell_values: dict[str, float] = {}
        # Populate from weather row using both direct names and aliases
        for feat in required_features:
            alias = _WEATHER_ALIASES.get(feat)
            if alias is None and feat in ("days_since_last_burn",):
                # static feature not sourced from weather
                cell_values[feat] = _STATIC_DEFAULTS.get(feat, float("nan"))
            elif alias is not None:
                raw = w.get(alias)
                cell_values[feat] = (
                    float(raw) if raw is not None else _STATIC_DEFAULTS.get(feat, float("nan"))
                )
            else:
                raw = w.get(feat)
                cell_values[feat] = (
                    float(raw) if raw is not None else _STATIC_DEFAULTS.get(feat, float("nan"))
                )

        row: list[float] = []
        for feat in required_features:
            val = cell_values.get(feat, float("nan"))
            if val != val:  # NaN check
                if missing_feature_policy == "BLOCKER":
                    LOGGER.error(
                        "BLOCKER [ignition-grid] Feature %r missing/NaN at (%.4f, %.4f). "
                        "Train/infer contract mismatch. TARGET_STAGE: science_grade",
                        feat, lat, lon,
                    )
                    raise IgnitionInferenceFailed(
                        f"Required feature {feat!r} is missing from assembled cell data."
                    )
            row.append(val)
        feature_rows.append(row)

    feature_matrix = np.array(feature_rows, dtype=np.float32)

    if active is None or not artifact_uri:
        # No model available — raise; caller decides based on IGNITION_REQUIRED
        raise IgnitionModelUnavailable("No promoted ignition model found.")

    try:
        probabilities = _run_onnx_inference(artifact_uri, feature_matrix)
    except IgnitionInferenceFailed:
        raise
    except Exception as exc:
        LOGGER.error(
            "BLOCKER [ignition-grid] ONNX inference failed: %s. Refusing to serve fabricated data.",
            exc,
        )
        raise IgnitionInferenceFailed(f"ONNX inference failed: {exc}") from exc

    cells: list[dict] = []
    for idx, (lat, lon, prob) in enumerate(zip(cell_lats, cell_lons, probabilities)):
        # raw_signals from feature_matrix (the authoritative inference input values)
        raw_signals = {
            feat: float(feature_matrix[idx][i])
            for i, feat in enumerate(required_features)
        }
        signals = {
            fname: raw_signals.get(fname, float("nan"))
            for fname in top_features
            if fname in raw_signals
        }

        cells.append({
            "cell_id": _build_cell_id(lat, lon),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "probability": round(float(prob), 4),
            "level": _classify_level(float(prob), thresholds),
            "signals": signals,
        })

    return {
        "horizon": horizon,
        "valid_time": display_valid_time.isoformat(),
        "model_id": model_id,
        "low_confidence": low_confidence,
        "cells": cells,
        "coverage_warnings": coverage_warnings,
    }
