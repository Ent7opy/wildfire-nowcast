"""Event-level denoiser inference v2 with fail-closed safety rules."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.db import get_engine
from ml.denoiser.eventize import EventizeParams, eventize_detections
from ml.denoiser.weather_context import WeatherContextParams, append_weather_context_features
from ml.denoiser.moisture_context import MoistureContextParams, append_moisture_context_features
from ingest.config import settings as ingest_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("denoiser_inference_v2")

_NEUTRAL_LANDCOVER = (0.5, -1.0)
_NEUTRAL_PERSISTENCE = (0.3, 0.5, -1.0)
_NEUTRAL_WEATHER = (0.5, -1.0)
_SCAN_ANGLE_MAX_DEG = 50.0
_WEATHER_TIME_TOLERANCE_HOURS = float(
    os.getenv("FIRE_SCORING_WEATHER_TIME_TOLERANCE_HOURS", "6")
)
_MOISTURE_TIME_TOLERANCE_HOURS = float(
    os.getenv("DENOISER_MOISTURE_TIME_TOLERANCE_HOURS", "48")
)
_INDUSTRIAL_GOLD_BUFFER_M = 375.0
_INDUSTRIAL_SILVER_BUFFER_M = 750.0
_DEFAULT_INDUSTRIAL_POLICY_VERSION = "global_authoritative_industrial_v1"
_FRP_NOISE_FLOOR_MW = 5.0
_BIOPHYSICAL_ZERO_FUEL_MAX = 0.1
_AGRICULTURE_SCORE = 0.7
_AGRICULTURE_SCORE_TOL = 0.05
_AGRICULTURE_LULC_CODES = {40}
_AGRICULTURE_LABEL_TOKENS = ("crop", "cropland", "agri", "agriculture", "farmland")

# Thermal potential class → estimated max normal FRP output (MW).
# Used to compute industrial_frp_ratio: observed FRP / expected industrial FRP.
# Values based on satellite thermal anomaly literature for industrial source types.
_TPC_TO_ESTIMATED_MAX_FRP_MW: dict[float, float] = {
    1.00: 800.0,   # steel / primary metals
    0.95: 600.0,   # cement / nonmetallic minerals
    0.90: 400.0,   # refinery / coke
    0.85: 300.0,   # oil/gas extraction
    0.80: 250.0,   # chemicals
    0.70: 150.0,   # utilities / power plants
    0.65: 100.0,   # biomass
    0.50: 80.0,    # generic industrial
    0.40: 50.0,    # nuclear
    0.10: 20.0,    # hydro
    0.05: 10.0,    # wind/solar
}


def _estimated_industrial_frp(tpc: float) -> float:
    """Linearly interpolate estimated max FRP for a given thermal potential class."""
    keys = sorted(_TPC_TO_ESTIMATED_MAX_FRP_MW.keys())
    if tpc >= keys[-1]:
        return _TPC_TO_ESTIMATED_MAX_FRP_MW[keys[-1]]
    if tpc <= keys[0]:
        return _TPC_TO_ESTIMATED_MAX_FRP_MW[keys[0]]
    for i in range(len(keys) - 1):
        if keys[i] <= tpc <= keys[i + 1]:
            frac = (tpc - keys[i]) / (keys[i + 1] - keys[i])
            return (
                _TPC_TO_ESTIMATED_MAX_FRP_MW[keys[i]] * (1 - frac)
                + _TPC_TO_ESTIMATED_MAX_FRP_MW[keys[i + 1]] * frac
            )
    return 80.0  # fallback


def _confidence_is_high(
    confidence_series: pd.Series,
    raw_properties_series: pd.Series,
    *,
    fail_closed_confidence: float,
) -> pd.Series:
    conf_num = pd.to_numeric(confidence_series, errors="coerce")
    conf_label = confidence_series.astype(str).str.strip().str.lower()
    from_conf_col = conf_label.isin({"h", "high"})

    def _raw_has_high(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        for key in ("confidence", "confidence_label", "firms_confidence", "viirs_confidence"):
            raw = value.get(key)
            if raw is None:
                continue
            s = str(raw).strip().lower()
            if s in {"h", "high"}:
                return True
        return False

    from_raw = raw_properties_series.apply(_raw_has_high)
    return ((conf_num >= float(fail_closed_confidence)) | from_conf_col | from_raw).fillna(False)


def _build_minimal_event_rollup(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame(columns=["event_id", "frp_max", "confidence_max", "landcover_mean"])
    det = detections.copy()
    det["event_id"] = det["event_id"].fillna(det["id"].map(lambda x: f"det_{int(x)}"))
    det["frp"] = pd.to_numeric(det["frp"], errors="coerce")
    det["confidence"] = pd.to_numeric(det["confidence"], errors="coerce")
    _, det["landcover_clean"] = _normalize_static_score(det["landcover_score"], _NEUTRAL_LANDCOVER)
    det["agriculture_lulc"] = det.apply(
        lambda row: _is_agriculture_lulc(
            landcover_score=row.get("landcover_score"),
            landcover_class=row.get("landcover_class"),
            landcover_label=row.get("landcover_label"),
            raw_properties=row.get("raw_properties"),
        ),
        axis=1,
    )
    out = (
        det.groupby("event_id", dropna=False)
        .agg(
            frp_max=("frp", "max"),
            confidence_max=("confidence", "max"),
            landcover_mean=("landcover_clean", "mean"),
            agriculture_lulc=("agriculture_lulc", "max"),
        )
        .reset_index()
    )
    return out


def _is_agriculture_lulc(
    *,
    landcover_score: object,
    landcover_class: object = None,
    landcover_label: object = None,
    raw_properties: object,
) -> bool:
    score = pd.to_numeric(pd.Series([landcover_score]), errors="coerce").iloc[0]
    if pd.notna(score) and abs(float(score) - _AGRICULTURE_SCORE) <= _AGRICULTURE_SCORE_TOL:
        return True

    if landcover_class is not None:
        try:
            if int(landcover_class) in _AGRICULTURE_LULC_CODES:
                return True
        except (TypeError, ValueError):
            pass

    if landcover_label is not None:
        label = str(landcover_label).strip().lower()
        if any(token in label for token in _AGRICULTURE_LABEL_TOKENS):
            return True

    if not isinstance(raw_properties, dict):
        return False

    for key in ("landcover_class", "landcover_code", "lulc_class", "lc_class"):
        raw_code = raw_properties.get(key)
        if raw_code is None:
            continue
        try:
            if int(raw_code) in _AGRICULTURE_LULC_CODES:
                return True
        except (TypeError, ValueError):
            continue

    for key in ("landcover_label", "landcover_type", "lulc_label", "land_use", "landcover"):
        raw_label = raw_properties.get(key)
        if raw_label is None:
            continue
        label = str(raw_label).strip().lower()
        if any(token in label for token in _AGRICULTURE_LABEL_TOKENS):
            return True

    return False


def _active_industrial_policy(
    engine: Engine,
    *,
    policy_version: str | None = None,
) -> dict[str, Any] | None:
    stmt = text(
        """
        SELECT
            policy_version,
            strict_no_go
        FROM industrial_mask_policies
        WHERE (
                :policy_version IS NOT NULL
                AND policy_version = :policy_version
              )
           OR (
                :policy_version IS NULL
                AND (active_to IS NULL OR active_to > NOW())
              )
        ORDER BY active_from DESC, policy_version DESC
        LIMIT 1
        """
    )
    with engine.begin() as conn:
        row = conn.execute(stmt, {"policy_version": policy_version}).mappings().first()
    return dict(row) if row else None


def _load_industrial_masked_event_ids(
    engine: Engine,
    *,
    batch_id: int,
    policy_version: str | None,
    strict_no_go: bool,
) -> tuple[set[str], dict[str, float]]:
    """Return (masked_event_ids, event_id→max_thermal_potential_class)."""
    meters_to_deg = 1.0 / 111000.0
    stmt = text(
        """
        WITH base AS (
            SELECT
                d.event_id,
                d.geom,
                d.acq_time
            FROM fire_detections d
            WHERE d.ingest_batch_id = :batch_id
              AND d.event_id IS NOT NULL
        )
        SELECT
            b.event_id::text AS event_id,
            MAX(COALESCE(i.thermal_potential_class, 0.5)) AS max_tpc
        FROM base b
        JOIN industrial_sources i
          ON COALESCE(i.is_active, TRUE)
         AND i.authority_tier IN ('gold', 'silver')
         AND (i.valid_from IS NULL OR i.valid_from <= b.acq_time)
         AND (i.valid_to IS NULL OR i.valid_to >= b.acq_time)
         AND i.geom && ST_Expand(
                b.geom,
                CASE
                    WHEN i.authority_tier = 'gold' THEN :gold_buffer_deg
                    ELSE :silver_buffer_deg
                END
            )
         AND ST_DWithin(
                b.geom::geography,
                i.geom::geography,
                CASE
                    WHEN i.authority_tier = 'gold' THEN :gold_buffer_m
                    ELSE :silver_buffer_m
                END
            )
        WHERE NOT (
            :strict_no_go
            AND :policy_version IS NOT NULL
            AND EXISTS (
                SELECT 1
                FROM industrial_no_go_zones z
                WHERE z.is_active
                  AND z.policy_version = :policy_version
                  AND z.geom && b.geom
                  AND ST_Intersects(z.geom, b.geom)
            )
        )
        GROUP BY b.event_id
        """
    )
    params = {
        "batch_id": int(batch_id),
        "policy_version": policy_version,
        "strict_no_go": bool(strict_no_go),
        "gold_buffer_m": float(_INDUSTRIAL_GOLD_BUFFER_M),
        "silver_buffer_m": float(_INDUSTRIAL_SILVER_BUFFER_M),
        "gold_buffer_deg": float(_INDUSTRIAL_GOLD_BUFFER_M) * meters_to_deg,
        "silver_buffer_deg": float(_INDUSTRIAL_SILVER_BUFFER_M) * meters_to_deg,
    }
    with engine.begin() as conn:
        rows = conn.execute(stmt, params).mappings().all()
    event_ids: set[str] = set()
    tpc_map: dict[str, float] = {}
    for row in rows:
        eid = row.get("event_id")
        if eid is not None:
            eid_str = str(eid)
            event_ids.add(eid_str)
            tpc_map[eid_str] = float(row.get("max_tpc") or 0.5)
    return event_ids, tpc_map


def _normalize_static_score(series: pd.Series, neutral_values: tuple[float, ...]) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    available = numeric.notna()
    for neutral in neutral_values:
        available &= ~np.isclose(numeric, neutral, atol=1e-12, rtol=0.0)
    cleaned = numeric.where(available, np.nan)
    return available.astype(bool), cleaned.astype(float)


def _load_bundle(model_run_dir: str) -> dict[str, Any]:
    bundle_path = os.path.join(model_run_dir, "model_bundle.pkl")
    if os.path.exists(bundle_path):
        return joblib.load(bundle_path)

    # Minimal compatibility fallback.
    model = joblib.load(os.path.join(model_run_dir, "model.pkl"))
    with open(os.path.join(model_run_dir, "feature_list.json"), "r", encoding="utf-8") as f:
        features = json.load(f)
    return {
        "model": model,
        "features": features,
        "slice_cols": ["sensor_id", "biome_slice"],
        "global_calibrator": {"type": "identity", "model": None},
        "slice_calibrators": {},
        "thresholds": {
            "decision": 0.5,
            "strong_filter": ingest_settings.denoiser_strong_filter_threshold,
            "downweight": ingest_settings.denoiser_downweight_threshold,
            "uncertainty_band_low": ingest_settings.denoiser_uncertainty_band_low,
            "uncertainty_band_high": ingest_settings.denoiser_uncertainty_band_high,
        },
    }


def _apply_calibrator(cal: dict[str, Any], scores: np.ndarray) -> np.ndarray:
    ctype = cal.get("type")
    model = cal.get("model")
    if ctype == "isotonic" and model is not None:
        return np.asarray(model.predict(scores), dtype=float)
    if ctype == "platt" and model is not None:
        return np.asarray(model.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(scores, dtype=float)


def _slice_key(row: pd.Series, slice_cols: list[str]) -> str:
    return "|".join(f"{col}={row.get(col, 'unknown')}" for col in slice_cols)


def _normalize_sensor_id(*, sensor: object, source: object, satellite_name: object) -> str:
    joined = " ".join(
        str(value).strip().lower()
        for value in (sensor, source, satellite_name)
        if value is not None and str(value).strip()
    )
    if not joined:
        return "unknown"
    if "noaa-21" in joined or "noaa21" in joined or "j02" in joined or "j2" in joined:
        return "NOAA-21"
    if "noaa-20" in joined or "noaa20" in joined or "j01" in joined or "j1" in joined:
        return "NOAA-20"
    if "s-npp" in joined or "snpp" in joined or "suomi" in joined or "npp" in joined:
        return "S-NPP"
    return "unknown"


def _predict_raw(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    pred = np.asarray(model.predict(x), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def _get_batch_detections(engine: Engine, batch_id: int) -> pd.DataFrame:
    query = text(
        """
        SELECT
            id,
            front_id,
            event_id,
            source,
            sensor,
            acq_time,
            lat,
            lon,
            confidence,
            frp,
            brightness,
            bright_t31,
            scan,
            track,
            landcover_class,
            landcover_label,
            raw_properties->>'satellite' AS satellite_name,
            CASE
                WHEN COALESCE(raw_properties->>'scan_angle', '') ~ '^[+-]?[0-9]+(\.[0-9]+)?$'
                    THEN (raw_properties->>'scan_angle')::double precision
                WHEN COALESCE(raw_properties->>'scan_angle_deg', '') ~ '^[+-]?[0-9]+(\.[0-9]+)?$'
                    THEN (raw_properties->>'scan_angle_deg')::double precision
                WHEN COALESCE(raw_properties->>'satellite_zenith_angle', '') ~ '^[+-]?[0-9]+(\.[0-9]+)?$'
                    THEN (raw_properties->>'satellite_zenith_angle')::double precision
                WHEN COALESCE(raw_properties->>'sensor_zenith_angle', '') ~ '^[+-]?[0-9]+(\.[0-9]+)?$'
                    THEN (raw_properties->>'sensor_zenith_angle')::double precision
                ELSE NULL
            END AS scan_angle,
            landcover_score,
            persistence_score,
            weather_score,
            raw_properties,
            denoised_score,
            is_noise
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
        """
    )
    with engine.begin() as conn:
        return pd.read_sql(query, conn, params={"batch_id": int(batch_id)})


def _build_event_features(batch_df: pd.DataFrame, *, engine: Engine) -> pd.DataFrame:
    df = batch_df.copy()
    if df.empty:
        return pd.DataFrame()
    df["acq_time"] = pd.to_datetime(df["acq_time"], utc=True)
    df["event_id"] = df["event_id"].fillna(df["id"].map(lambda x: f"det_{int(x)}"))
    df["sensor_id"] = df.apply(
        lambda row: _normalize_sensor_id(
            sensor=row.get("sensor"),
            source=row.get("source"),
            satellite_name=row.get("satellite_name"),
        ),
        axis=1,
    )
    df["sensor_id_code"] = df["sensor_id"].astype("category").cat.codes.astype(np.int16)
    df["scan_angle"] = pd.to_numeric(df["scan_angle"], errors="coerce")
    df["scan_angle_is_available"] = df["scan_angle"].notna().astype(bool)
    # Match training hard-filter policy for extreme scan distortion.
    df = df[(df["scan_angle"].isna()) | (df["scan_angle"] <= _SCAN_ANGLE_MAX_DEG)].copy()
    if df.empty:
        return pd.DataFrame()
    df = append_weather_context_features(
        df,
        engine=engine,
        params=WeatherContextParams(time_tolerance_hours=_WEATHER_TIME_TOLERANCE_HOURS),
    )
    df = append_moisture_context_features(
        df,
        engine=engine,
        params=MoistureContextParams(time_tolerance_hours=_MOISTURE_TIME_TOLERANCE_HOURS),
    )
    df["is_day"] = df["raw_properties"].apply(
        lambda x: 1 if isinstance(x, dict) and x.get("daynight") == "D" else 0
    )
    df["agriculture_lulc"] = df.apply(
        lambda row: _is_agriculture_lulc(
            landcover_score=row.get("landcover_score"),
            landcover_class=row.get("landcover_class"),
            landcover_label=row.get("landcover_label"),
            raw_properties=row.get("raw_properties"),
        ),
        axis=1,
    )
    # FRP density: FRP normalized by pixel area for sensor-invariant comparison.
    scan_num = pd.to_numeric(df["scan"], errors="coerce")
    track_num = pd.to_numeric(df["track"], errors="coerce")
    pixel_area = scan_num * track_num
    df["frp_density_obs"] = np.where(
        (pixel_area > 0.0) & df["frp"].notna(),
        pd.to_numeric(df["frp"], errors="coerce") / pixel_area,
        np.nan,
    )

    df["landcover_is_available"], df["landcover_score_clean"] = _normalize_static_score(
        df["landcover_score"], _NEUTRAL_LANDCOVER
    )
    df["persistence_is_available"], df["persistence_score_clean"] = _normalize_static_score(
        df["persistence_score"], _NEUTRAL_PERSISTENCE
    )
    df["weather_is_available"], df["weather_score_clean"] = _normalize_static_score(
        df["weather_score"], _NEUTRAL_WEATHER
    )

    out = (
        df.groupby("event_id", dropna=False)
        .agg(
            source=("source", lambda s: str(s.dropna().mode().iloc[0]) if not s.dropna().empty else "unknown"),
            sensor=("sensor", lambda s: str(s.dropna().mode().iloc[0]) if not s.dropna().empty else "unknown"),
            sensor_id=("sensor_id", lambda s: str(s.dropna().mode().iloc[0]) if not s.dropna().empty else "unknown"),
            sensor_id_code=("sensor_id_code", "max"),
            detection_count=("id", "count"),
            start_time=("acq_time", "min"),
            end_time=("acq_time", "max"),
            confidence_mean=("confidence", "mean"),
            confidence_max=("confidence", "max"),
            frp_mean=("frp", "mean"),
            frp_max=("frp", "max"),
            frp_density_mean=("frp_density_obs", "mean"),
            frp_density_max=("frp_density_obs", "max"),
            frp_spatial_std=("frp", "std"),
            brightness_mean=("brightness", "mean"),
            bright_t31_mean=("bright_t31", "mean"),
            scan_mean=("scan", "mean"),
            scan_angle_mean=("scan_angle", "mean"),
            scan_angle_max=("scan_angle", "max"),
            track_mean=("track", "mean"),
            landcover_mean=("landcover_score_clean", "mean"),
            persistence_mean=("persistence_score_clean", "mean"),
            weather_mean=("weather_score_clean", "mean"),
            landcover_is_available=("landcover_is_available", "max"),
            persistence_is_available=("persistence_is_available", "max"),
            weather_is_available=("weather_is_available", "max"),
            scan_angle_is_available=("scan_angle_is_available", "max"),
            agriculture_lulc=("agriculture_lulc", "max"),
            rh2m_mean=("rh2m", "mean"),
            u10_mean=("u10", "mean"),
            v10_mean=("v10", "mean"),
            wind_speed_mean=("wind_speed", "mean"),
            rh2m_is_available=("rh2m_is_available", "max"),
            wind_is_available=("wind_is_available", "max"),
            lfmc_mean=("lfmc", "mean"),
            dfmc_10hr_mean=("dfmc_10hr", "mean"),
            lfmc_is_available=("lfmc_is_available", "max"),
            dfmc_is_available=("dfmc_is_available", "max"),
            is_day_ratio=("is_day", "mean"),
            lon_centroid=("lon", "mean"),
        )
        .reset_index()
    )
    out["landcover_is_available"] = out["landcover_is_available"].astype(np.float32)
    out["persistence_is_available"] = out["persistence_is_available"].astype(np.float32)
    out["weather_is_available"] = out["weather_is_available"].astype(np.float32)
    out["scan_angle_is_available"] = out["scan_angle_is_available"].astype(np.float32)
    out["rh2m_is_available"] = out["rh2m_is_available"].astype(np.float32)
    out["wind_is_available"] = out["wind_is_available"].astype(np.float32)
    out["lfmc_is_available"] = out["lfmc_is_available"].astype(np.float32)
    out["dfmc_is_available"] = out["dfmc_is_available"].astype(np.float32)
    out["duration_hours"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 3600.0

    # FRP spatial std: 0.0 for single-detection events (undefined std).
    out["frp_spatial_std"] = out["frp_spatial_std"].fillna(0.0)

    # FRP growth rate: (frp_last - frp_first) / duration_hours.
    # Fires grow; industrial sources stay flat.
    frp_temporal = (
        df.sort_values("acq_time")
        .groupby("event_id", dropna=False)
        .agg(
            frp_first=("frp", "first"),
            frp_last=("frp", "last"),
        )
        .reset_index()
    )
    out = out.merge(frp_temporal, on="event_id", how="left")
    out["frp_growth_rate"] = np.where(
        out["duration_hours"] > 0.0,
        (pd.to_numeric(out["frp_last"], errors="coerce").fillna(0.0)
         - pd.to_numeric(out["frp_first"], errors="coerce").fillna(0.0))
        / out["duration_hours"],
        0.0,
    )
    out.drop(columns=["frp_first", "frp_last"], inplace=True)

    # FRP peak local hour: local solar time of the detection with max FRP.
    # Wildfires peak in afternoon (1400-1800 local); industrial sources are more uniform.
    peak_frp_rows = (
        df.dropna(subset=["frp"])
        .sort_values("frp", ascending=False)
        .groupby("event_id", dropna=False)
        .first()
        .reset_index()[["event_id", "acq_time"]]
    )
    peak_frp_rows["_peak_utc_hour"] = (
        peak_frp_rows["acq_time"].dt.hour
        + peak_frp_rows["acq_time"].dt.minute / 60.0
    )
    out = out.merge(
        peak_frp_rows[["event_id", "_peak_utc_hour"]],
        on="event_id",
        how="left",
    )
    out["frp_peak_local_hour"] = (
        out["_peak_utc_hour"] + out["lon_centroid"].fillna(0.0) / 15.0
    ) % 24.0
    out["sin_frp_peak_hour"] = np.sin(2 * np.pi * out["frp_peak_local_hour"] / 24.0)
    out["cos_frp_peak_hour"] = np.cos(2 * np.pi * out["frp_peak_local_hour"] / 24.0)
    out.drop(columns=["_peak_utc_hour"], inplace=True)

    out["hour_of_day"] = (
        out["start_time"].dt.hour
        + out["start_time"].dt.minute / 60.0
        + out["start_time"].dt.second / 3600.0
    )
    out["day_of_year"] = out["start_time"].dt.dayofyear.fillna(1).astype(np.int16)
    out["sin_hour"] = np.sin(2 * np.pi * out["hour_of_day"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour_of_day"] / 24.0)
    out["sin_day_of_year"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["cos_day_of_year"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["sin_doy"] = out["sin_day_of_year"]
    out["cos_doy"] = out["cos_day_of_year"]
    out["biome_slice"] = pd.cut(
        out["landcover_mean"].fillna(0.5),
        bins=[-np.inf, 0.25, 0.6, np.inf],
        labels=["low_fuel", "mixed_fuel", "high_fuel"],
    ).astype(str)
    return out


def _decide_event(
    row: pd.Series,
    *,
    strong_filter_threshold: float,
    downweight_threshold: float,
    uncertainty_band_low: float,
    uncertainty_band_high: float,
    fail_closed_frp_mw: float,
    fail_closed_confidence: float,
    high_risk_landcover_min: float,
) -> tuple[str, bool]:
    score = float(row["event_score"])
    frp_max = float(row.get("frp_max") or 0.0)
    conf_max = float(row.get("confidence_max") or 0.0)
    landcover = float(row.get("landcover_mean") or 0.0)

    fail_closed = frp_max > fail_closed_frp_mw or (
        conf_max >= fail_closed_confidence and landcover >= high_risk_landcover_min
    )
    if fail_closed:
        return "review", True
    if uncertainty_band_low <= score <= uncertainty_band_high:
        return "review", True
    if score < strong_filter_threshold:
        return "drop", False
    if score < downweight_threshold:
        return "downweight", False
    return "pass", False


def run_inference_v2(
    *,
    batch_id: int,
    model_run_dir: str,
    shadow_mode: bool,
    strong_filter_threshold: float,
    downweight_threshold: float,
    uncertainty_band_low: float,
    uncertainty_band_high: float,
    event_front_radius_m: float,
    event_front_max_gap_minutes: int,
    event_link_radius_m: float,
    event_link_max_gap_days: int,
    event_static_persistence_threshold: float,
    event_strict_static_split: bool,
    fail_closed_frp_mw: float = 500.0,
    fail_closed_confidence: float = 80.0,
    high_risk_landcover_min: float = 0.6,
    industrial_policy_version: str | None = _DEFAULT_INDUSTRIAL_POLICY_VERSION,
    early_ignition_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engine = get_engine()

    # Ensure front/event IDs exist for this batch.
    eventize_stats = eventize_detections(
        engine,
        batch_id=batch_id,
        params=EventizeParams(
            front_link_radius_m=float(event_front_radius_m),
            front_max_gap_minutes=int(event_front_max_gap_minutes),
            event_link_radius_m=float(event_link_radius_m),
            event_max_gap_days=int(event_link_max_gap_days),
            static_persistence_threshold=float(event_static_persistence_threshold),
            strict_static_split=bool(event_strict_static_split),
        ),
    )
    LOGGER.info("Eventize completed for batch %s: %s", batch_id, eventize_stats)

    detections = _get_batch_detections(engine, batch_id)
    if detections.empty:
        return {"batch_id": batch_id, "count": 0, "events": 0, "shadow_mode": shadow_mode}
    detections["event_id"] = detections["event_id"].fillna(detections["id"].map(lambda x: f"det_{int(x)}"))

    # Industrial masking must execute BEFORE hard bypass so that high-FRP
    # industrial sources (e.g. 600 MW steel mills) are not exempt from suppression.
    industrial_policy = _active_industrial_policy(engine, policy_version=industrial_policy_version)
    industrial_mask_event_ids, industrial_tpc_map = _load_industrial_masked_event_ids(
        engine,
        batch_id=int(batch_id),
        policy_version=(
            str(industrial_policy.get("policy_version"))
            if industrial_policy is not None
            else industrial_policy_version
        ),
        strict_no_go=bool((industrial_policy or {}).get("strict_no_go", False)),
    )

    # Hard fail-closed bypass — but NOT for events near known industrial sources.
    detection_frp = pd.to_numeric(detections["frp"], errors="coerce")
    _, detection_landcover_clean = _normalize_static_score(detections["landcover_score"], _NEUTRAL_LANDCOVER)
    high_risk_detection = detection_landcover_clean >= float(high_risk_landcover_min)
    confidence_high = _confidence_is_high(
        detections["confidence"],
        detections["raw_properties"],
        fail_closed_confidence=float(fail_closed_confidence),
    )
    detection_in_industrial_event = detections["event_id"].astype(str).isin(industrial_mask_event_ids)
    hard_bypass_mask = (
        (detection_frp > float(fail_closed_frp_mw))
        & ~detection_in_industrial_event
    ) | (
        confidence_high & high_risk_detection.fillna(False)
    )
    hard_bypass_event_ids = set(detections.loc[hard_bypass_mask, "event_id"].astype(str).tolist())

    bundle = _load_bundle(model_run_dir)
    model = bundle["model"]
    features = list(bundle["features"])
    slice_cols = list(bundle.get("slice_cols", ["sensor_id", "biome_slice"]))
    global_cal = bundle.get("global_calibrator", {"type": "identity", "model": None})
    slice_cals = dict(bundle.get("slice_calibrators", {}))

    events_df = _build_event_features(detections, engine=engine)
    if events_df.empty and not hard_bypass_event_ids:
        LOGGER.warning(
            "No eligible events for inference after scan_angle <= %.1f filter (batch_id=%s).",
            _SCAN_ANGLE_MAX_DEG,
            batch_id,
        )
        return {"batch_id": batch_id, "count": int(len(detections)), "events": 0, "shadow_mode": shadow_mode}
    minimal_events = _build_minimal_event_rollup(detections)
    if events_df.empty and hard_bypass_event_ids:
        events_df = minimal_events[minimal_events["event_id"].astype(str).isin(hard_bypass_event_ids)].copy()
    elif not events_df.empty and hard_bypass_event_ids:
        missing_bypass = hard_bypass_event_ids - set(events_df["event_id"].astype(str).tolist())
        if missing_bypass:
            add_rows = minimal_events[minimal_events["event_id"].astype(str).isin(missing_bypass)].copy()
            events_df = pd.concat([events_df, add_rows], ignore_index=True, sort=False)

    for col in features:
        if col not in events_df.columns:
            events_df[col] = np.nan

    events_df["event_score"] = np.nan
    events_df["fail_closed_hard_bypass"] = events_df["event_id"].astype(str).isin(hard_bypass_event_ids)
    events_df["industrial_masked"] = events_df["event_id"].astype(str).isin(industrial_mask_event_ids)

    # Industrial FRP ratio: observed frp_max / estimated industrial source FRP.
    # Ratio < 1.0 → FRP consistent with industrial source; ratio >> 1.0 → anomalous.
    event_id_strs = events_df["event_id"].astype(str)
    tpc_series = event_id_strs.map(industrial_tpc_map)
    estimated_frp = tpc_series.map(
        lambda tpc: _estimated_industrial_frp(tpc) if pd.notna(tpc) else np.nan
    )
    frp_max_num = pd.to_numeric(events_df.get("frp_max"), errors="coerce").fillna(0.0)
    events_df["industrial_frp_ratio"] = np.where(
        estimated_frp.notna() & (estimated_frp > 0.0),
        frp_max_num / estimated_frp,
        np.nan,
    )

    events_df["low_frp_noise"] = (
        pd.to_numeric(events_df.get("frp_max"), errors="coerce").fillna(0.0) < _FRP_NOISE_FLOOR_MW
    )
    events_df["biophysical_zero_fuel"] = (
        pd.to_numeric(events_df.get("landcover_mean"), errors="coerce").fillna(1.0)
        <= _BIOPHYSICAL_ZERO_FUEL_MAX
    )
    if "agriculture_lulc" in events_df.columns:
        agriculture_col = events_df["agriculture_lulc"]
    else:
        agriculture_col = pd.Series(False, index=events_df.index, dtype=bool)
    events_df["agriculture_masked"] = agriculture_col.fillna(False).astype(bool)
    events_df["physical_noise_masked"] = (
        events_df["low_frp_noise"].fillna(False).astype(bool)
        | events_df["biophysical_zero_fuel"].fillna(False).astype(bool)
        | events_df["agriculture_masked"].fillna(False).astype(bool)
    )

    # Early ignition candidate flag: young events with few detections.
    ei_cfg = early_ignition_config or {}
    if ei_cfg.get("enabled"):
        ei_max_dur = float(ei_cfg.get("max_duration_hours", 2.0))
        ei_max_det = int(ei_cfg.get("max_detection_count", 3))
        duration_col = pd.to_numeric(events_df.get("duration_hours"), errors="coerce").fillna(0.0)
        det_count_col = pd.to_numeric(events_df.get("detection_count"), errors="coerce").fillna(1)
        events_df["is_early_ignition_candidate"] = (
            (duration_col <= ei_max_dur) & (det_count_col <= ei_max_det)
        )
    else:
        events_df["is_early_ignition_candidate"] = False

    score_mask = ~events_df["fail_closed_hard_bypass"].fillna(False).astype(bool)
    if bool(score_mask.any()):
        score_df = events_df.loc[score_mask].copy().reset_index()
        raw = _predict_raw(model, score_df[features])
        calibrated = np.zeros(len(score_df), dtype=float)
        for i, row in score_df.iterrows():
            key = _slice_key(row, slice_cols)
            cal = slice_cals.get(key, global_cal)
            calibrated[i] = float(_apply_calibrator(cal, np.asarray([raw[i]]))[0])
        events_df.loc[score_df["index"].to_numpy(dtype=int), "event_score"] = calibrated
    events_df.loc[events_df["fail_closed_hard_bypass"], "event_score"] = 1.0
    # Suppress masks apply unconditionally — industrial events are already
    # excluded from hard_bypass_event_ids, so the bypass flag is only set for
    # genuinely non-industrial high-energy detections.
    industrial_suppress_mask = events_df["industrial_masked"].fillna(False).astype(bool)
    agriculture_suppress_mask = events_df["agriculture_masked"].fillna(False).astype(bool)
    physical_suppress_mask = events_df["physical_noise_masked"].fillna(False).astype(bool)
    events_df.loc[
        industrial_suppress_mask | agriculture_suppress_mask | physical_suppress_mask,
        "event_score",
    ] = 0.0

    ei_enabled = bool(ei_cfg.get("enabled"))
    ei_review_threshold = float(ei_cfg.get("review_threshold", 0.35))

    decisions: list[str] = []
    review_flags: list[bool] = []
    for _, row in events_df.iterrows():
        if bool(row.get("industrial_masked")):
            decision, review_required = "drop", False
        elif bool(row.get("agriculture_masked")):
            decision, review_required = "drop", False
        elif bool(row.get("physical_noise_masked")):
            decision, review_required = "drop", False
        elif bool(row.get("fail_closed_hard_bypass")):
            decision, review_required = "review", True
        elif ei_enabled and bool(row.get("is_early_ignition_candidate")):
            # Early ignition safety net: lower review threshold for young events
            # to avoid auto-dropping new fires before evidence accumulates.
            score = float(row.get("event_score") or 0.0)
            if score >= ei_review_threshold:
                decision, review_required = "early_ignition_review", True
            else:
                decision, review_required = _decide_event(
                    row,
                    strong_filter_threshold=strong_filter_threshold,
                    downweight_threshold=downweight_threshold,
                    uncertainty_band_low=uncertainty_band_low,
                    uncertainty_band_high=uncertainty_band_high,
                    fail_closed_frp_mw=fail_closed_frp_mw,
                    fail_closed_confidence=fail_closed_confidence,
                    high_risk_landcover_min=high_risk_landcover_min,
                )
        else:
            decision, review_required = _decide_event(
                row,
                strong_filter_threshold=strong_filter_threshold,
                downweight_threshold=downweight_threshold,
                uncertainty_band_low=uncertainty_band_low,
                uncertainty_band_high=uncertainty_band_high,
                fail_closed_frp_mw=fail_closed_frp_mw,
                fail_closed_confidence=fail_closed_confidence,
                high_risk_landcover_min=high_risk_landcover_min,
            )
        decisions.append(decision)
        review_flags.append(review_required)

    events_df["denoiser_decision"] = decisions
    events_df["review_required"] = review_flags

    # Persist event-level decisions.
    update_event_stmt = text(
        """
        UPDATE fire_events
        SET
            event_score = :event_score,
            denoiser_decision = :denoiser_decision,
            review_required = :review_required,
            updated_at = NOW()
        WHERE event_id = :event_id
        """
    )

    review_upsert_stmt = text(
        """
        INSERT INTO denoiser_review_queue (
            event_id,
            reason,
            severity,
            status,
            payload_json,
            created_at,
            updated_at
        )
        VALUES (
            :event_id,
            :reason,
            :severity,
            'open',
            :payload_json,
            NOW(),
            NOW()
        )
        """
    )

    update_detection_stmt = text(
        """
        UPDATE fire_detections d
        SET
            event_id = s.event_id,
            front_id = COALESCE(d.front_id, s.front_id),
            event_score = s.event_score,
            denoiser_decision = s.denoiser_decision,
            review_required = s.review_required,
            denoiser_model_id = :model_id,
            denoiser_scored_at = NOW(),
            denoised_score = CASE WHEN :shadow_mode THEN d.denoised_score ELSE s.event_score END,
            is_noise = CASE WHEN :shadow_mode THEN d.is_noise ELSE (s.denoiser_decision = 'drop') END
        FROM (
            SELECT
                d0.id,
                d0.event_id,
                d0.front_id,
                e.event_score,
                e.denoiser_decision,
                e.review_required
            FROM fire_detections d0
            JOIN (
                SELECT
                    event_id,
                    event_score,
                    denoiser_decision,
                    review_required
                FROM fire_events
                WHERE event_id = ANY(:event_ids)
            ) e ON e.event_id = d0.event_id
            WHERE d0.ingest_batch_id = :batch_id
        ) s
        WHERE d.id = s.id
        """
    )

    with engine.begin() as conn:
        for row in events_df.itertuples(index=False):
            conn.execute(
                update_event_stmt,
                {
                    "event_id": row.event_id,
                    "event_score": float(row.event_score),
                    "denoiser_decision": row.denoiser_decision,
                    "review_required": bool(row.review_required),
                },
            )
            if bool(row.review_required):
                conn.execute(
                    review_upsert_stmt,
                    {
                        "event_id": row.event_id,
                        "reason": (
                            "fail_closed_hard_bypass"
                            if bool(getattr(row, "fail_closed_hard_bypass", False))
                            else "fail_closed_or_uncertainty"
                        ),
                        "severity": "high",
                        "payload_json": json.dumps(
                            {
                                "event_score": float(row.event_score),
                                "frp_max": float(row.frp_max or 0),
                                "confidence_max": float(row.confidence_max or 0),
                                "fail_closed_hard_bypass": bool(
                                    getattr(row, "fail_closed_hard_bypass", False)
                                ),
                            }
                        ),
                    },
                )

        conn.execute(
            update_detection_stmt,
            {
                "batch_id": int(batch_id),
                "event_ids": list(events_df["event_id"].astype(str).unique()),
                "shadow_mode": bool(shadow_mode),
                "model_id": os.path.basename(model_run_dir.rstrip(os.sep)),
            },
        )

        if shadow_mode:
            # Persist summary for live-vs-shadow monitoring.
            summary_row = {
                "run_id": f"shadow_batch_{batch_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "model_id": os.path.basename(model_run_dir.rstrip(os.sep)),
                "family": "denoiser",
                "status": "shadow",
                "metrics_json": json.dumps({
                    "batch_id": int(batch_id),
                    "event_count": int(len(events_df)),
                    "decision_counts": events_df["denoiser_decision"].value_counts().to_dict(),
                }),
                "gate_report_json": None,
                "slice_metrics_json": None,
                "artifact_uri": model_run_dir,
            }
            conn.execute(
                text(
                    """
                    INSERT INTO denoiser_eval_runs (
                        run_id,
                        model_id,
                        family,
                        status,
                        metrics_json,
                        gate_report_json,
                        slice_metrics_json,
                        artifact_uri,
                        evaluated_at,
                        created_at
                    ) VALUES (
                        :run_id,
                        :model_id,
                        :family,
                        :status,
                        CAST(:metrics_json AS jsonb),
                        CAST(:gate_report_json AS jsonb),
                        CAST(:slice_metrics_json AS jsonb),
                        :artifact_uri,
                        NOW(),
                        NOW()
                    )
                    """
                ),
                summary_row,
            )

    decision_counts = {k: int(v) for k, v in events_df["denoiser_decision"].value_counts().to_dict().items()}
    summary = {
        "batch_id": int(batch_id),
        "count": int(len(detections)),
        "events": int(len(events_df)),
        "mean_event_score": float(events_df["event_score"].mean()),
        "decision_counts": decision_counts,
        "review_count": int(events_df["review_required"].sum()),
        "hard_bypass_event_count": int(events_df["fail_closed_hard_bypass"].sum()),
        "physical_noise_masked_event_count": int(events_df["physical_noise_masked"].sum()),
        "biophysical_zero_fuel_event_count": int(events_df["biophysical_zero_fuel"].sum()),
        "agriculture_masked_event_count": int(events_df["agriculture_masked"].sum()),
        "frp_noise_floor_event_count": int(events_df["low_frp_noise"].sum()),
        "agriculture_suppressed_event_count": int(agriculture_suppress_mask.sum()),
        "physical_suppressed_event_count": int(physical_suppress_mask.sum()),
        "industrial_masked_event_count": int(events_df["industrial_masked"].sum()),
        "industrial_suppressed_event_count": int(industrial_suppress_mask.sum()),
        "shadow_mode": bool(shadow_mode),
        "strong_filter_threshold": float(strong_filter_threshold),
        "downweight_threshold": float(downweight_threshold),
        "uncertainty_band_low": float(uncertainty_band_low),
        "uncertainty_band_high": float(uncertainty_band_high),
    }
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run denoiser v2 inference for a batch.")
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--model-run", type=str, default=ingest_settings.denoiser_model_run_dir)
    parser.add_argument("--shadow-mode", action="store_true")
    parser.add_argument(
        "--strong-filter-threshold",
        type=float,
        default=ingest_settings.denoiser_strong_filter_threshold,
    )
    parser.add_argument(
        "--downweight-threshold",
        type=float,
        default=ingest_settings.denoiser_downweight_threshold,
    )
    parser.add_argument(
        "--uncertainty-band-low",
        type=float,
        default=ingest_settings.denoiser_uncertainty_band_low,
    )
    parser.add_argument(
        "--uncertainty-band-high",
        type=float,
        default=ingest_settings.denoiser_uncertainty_band_high,
    )
    parser.add_argument(
        "--event-front-radius-m",
        type=float,
        default=ingest_settings.denoiser_event_front_radius_m,
    )
    parser.add_argument(
        "--event-front-max-gap-minutes",
        type=int,
        default=ingest_settings.denoiser_event_front_max_gap_minutes,
    )
    parser.add_argument(
        "--event-link-radius-m",
        type=float,
        default=ingest_settings.denoiser_event_link_radius_m,
    )
    parser.add_argument(
        "--event-link-max-gap-days",
        type=int,
        default=ingest_settings.denoiser_event_link_max_gap_days,
    )
    parser.add_argument(
        "--event-static-persistence-threshold",
        type=float,
        default=ingest_settings.denoiser_event_static_persistence_threshold,
    )
    parser.add_argument("--event-strict-static-split", action="store_true")
    parser.add_argument("--no-event-strict-static-split", action="store_true")
    parser.add_argument(
        "--industrial-policy-version",
        type=str,
        default=_DEFAULT_INDUSTRIAL_POLICY_VERSION,
    )
    args = parser.parse_args(argv)

    if not args.model_run:
        raise SystemExit("Missing --model-run / DENOISER_MODEL_RUN_DIR for v2 inference")

    event_strict_static_split = bool(ingest_settings.denoiser_event_strict_static_split)
    if args.no_event_strict_static_split:
        event_strict_static_split = False
    elif args.event_strict_static_split:
        event_strict_static_split = True

    summary = run_inference_v2(
        batch_id=int(args.batch_id),
        model_run_dir=str(args.model_run),
        shadow_mode=bool(args.shadow_mode),
        strong_filter_threshold=float(args.strong_filter_threshold),
        downweight_threshold=float(args.downweight_threshold),
        uncertainty_band_low=float(args.uncertainty_band_low),
        uncertainty_band_high=float(args.uncertainty_band_high),
        event_front_radius_m=float(args.event_front_radius_m),
        event_front_max_gap_minutes=int(args.event_front_max_gap_minutes),
        event_link_radius_m=float(args.event_link_radius_m),
        event_link_max_gap_days=int(args.event_link_max_gap_days),
        event_static_persistence_threshold=float(args.event_static_persistence_threshold),
        event_strict_static_split=event_strict_static_split,
        industrial_policy_version=args.industrial_policy_version,
    )
    LOGGER.info("Denoiser v2 inference summary: %s", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
