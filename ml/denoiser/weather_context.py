"""Authoritative weather context sampling for denoiser features.

This module samples ingested weather runs (NetCDF files referenced by `weather_runs`)
at detection point-times and appends atmospheric variables to a dataframe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class WeatherContextParams:
    time_tolerance_hours: float = 6.0

    def __post_init__(self) -> None:
        if float(self.time_tolerance_hours) <= 0.0:
            raise ValueError("time_tolerance_hours must be > 0")


def _to_numpy_datetime64_utc(dt: pd.Timestamp) -> np.datetime64:
    """Convert pandas timestamp to naive UTC numpy datetime64[ms]."""
    if dt.tzinfo is None:
        dt_utc = dt.tz_localize(timezone.utc)
    else:
        dt_utc = dt.tz_convert(timezone.utc)
    return np.datetime64(dt_utc.tz_localize(None).to_pydatetime(), "ms")


def _resolve_storage_path(raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _load_candidate_weather_runs(
    engine: Engine,
    *,
    min_time: pd.Timestamp,
    max_time: pd.Timestamp,
    bbox: tuple[float, float, float, float],
    tolerance_hours: float,
) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox
    stmt = text(
        """
        SELECT
            id,
            run_time,
            storage_path,
            COALESCE(bbox_min_lon, -180.0) AS bbox_min_lon,
            COALESCE(bbox_min_lat, -90.0) AS bbox_min_lat,
            COALESCE(bbox_max_lon, 180.0) AS bbox_max_lon,
            COALESCE(bbox_max_lat, 90.0) AS bbox_max_lat,
            created_at
        FROM weather_runs
        WHERE status = 'completed'
          AND run_time <= :max_time
          AND run_time >= :min_time - INTERVAL '1 hour' * :tolerance_hours
          AND COALESCE(bbox_min_lon, -180.0) <= :max_lon
          AND COALESCE(bbox_max_lon, 180.0) >= :min_lon
          AND COALESCE(bbox_min_lat, -90.0) <= :max_lat
          AND COALESCE(bbox_max_lat, 90.0) >= :min_lat
        ORDER BY run_time DESC, created_at DESC
        """
    )
    with engine.begin() as conn:
        runs = pd.read_sql(
            stmt,
            conn,
            params={
                "min_time": min_time.to_pydatetime(),
                "max_time": max_time.to_pydatetime(),
                "tolerance_hours": float(tolerance_hours),
                "min_lon": float(min_lon),
                "min_lat": float(min_lat),
                "max_lon": float(max_lon),
                "max_lat": float(max_lat),
            },
        )
    if runs.empty:
        return runs
    runs["run_time"] = pd.to_datetime(runs["run_time"], utc=True)
    runs["created_at"] = pd.to_datetime(runs["created_at"], utc=True)
    return runs


def _assign_weather_run_ids(
    df: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    tolerance_hours: float,
) -> pd.Series:
    if df.empty or runs.empty:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")

    assigned = pd.Series(pd.NA, index=df.index, dtype="Int64")
    tolerance = pd.to_timedelta(float(tolerance_hours), unit="hour")
    run_records = tuple(runs.itertuples(index=False, name="RunRecord"))

    for idx, row in df.iterrows():
        ts = row["acq_time"]
        lat = float(row["lat"])
        lon = float(row["lon"])
        lower = ts - tolerance
        match_id: int | None = None
        for run in run_records:
            run_time = run.run_time
            if run_time > ts or run_time < lower:
                continue
            if not (float(run.bbox_min_lon) <= lon <= float(run.bbox_max_lon)):
                continue
            if not (float(run.bbox_min_lat) <= lat <= float(run.bbox_max_lat)):
                continue
            match_id = int(run.id)
            break
        if match_id is not None:
            assigned.at[idx] = match_id

    return assigned


def append_weather_context_features(
    df: pd.DataFrame,
    *,
    engine: Engine,
    params: WeatherContextParams | None = None,
) -> pd.DataFrame:
    """Append weather point-time variables from authoritative `weather_runs`.

    Added columns:
      - rh2m
      - u10
      - v10
      - wind_speed
      - rh2m_is_available
      - wind_is_available
    """
    params = params or WeatherContextParams()
    if df.empty:
        out = df.copy()
        out["rh2m"] = np.nan
        out["u10"] = np.nan
        out["v10"] = np.nan
        out["wind_speed"] = np.nan
        out["rh2m_is_available"] = False
        out["wind_is_available"] = False
        return out

    out = df.copy()
    out["acq_time"] = pd.to_datetime(out["acq_time"], utc=True)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    min_time = out["acq_time"].min()
    max_time = out["acq_time"].max()
    bbox = (
        float(out["lon"].min()),
        float(out["lat"].min()),
        float(out["lon"].max()),
        float(out["lat"].max()),
    )
    runs = _load_candidate_weather_runs(
        engine,
        min_time=min_time,
        max_time=max_time,
        bbox=bbox,
        tolerance_hours=float(params.time_tolerance_hours),
    )

    out["rh2m"] = np.nan
    out["u10"] = np.nan
    out["v10"] = np.nan
    out["wind_speed"] = np.nan

    if runs.empty:
        out["rh2m_is_available"] = False
        out["wind_is_available"] = False
        return out

    out["_weather_run_id"] = _assign_weather_run_ids(
        out,
        runs,
        tolerance_hours=float(params.time_tolerance_hours),
    )

    run_by_id = {int(row["id"]): row for _, row in runs.iterrows()}
    for run_id in out["_weather_run_id"].dropna().astype(int).unique():
        run_row = run_by_id.get(int(run_id))
        if run_row is None:
            continue
        path = _resolve_storage_path(str(run_row["storage_path"]))
        if not path.exists():
            continue
        idx = out.index[out["_weather_run_id"] == int(run_id)]
        if len(idx) == 0:
            continue

        ds = None
        try:
            ds = xr.open_dataset(path)
            if "lat" not in ds.coords or "lon" not in ds.coords:
                continue

            lat_da = xr.DataArray(out.loc[idx, "lat"].to_numpy(dtype=float), dims="obs")
            lon_da = xr.DataArray(out.loc[idx, "lon"].to_numpy(dtype=float), dims="obs")
            point_ds = ds.sel(lat=lat_da, lon=lon_da, method="nearest")

            if "time" in point_ds.coords:
                time_vals = np.asarray(
                    [_to_numpy_datetime64_utc(ts) for ts in out.loc[idx, "acq_time"]],
                    dtype="datetime64[ms]",
                )
                time_da = xr.DataArray(time_vals, dims="obs")
                point_ds = point_ds.sel(time=time_da, method="nearest")

            n_obs = len(idx)
            if "rh2m" in point_ds.data_vars:
                vals = np.asarray(point_ds["rh2m"].values, dtype=float).reshape(-1)
                if vals.size == n_obs:
                    out.loc[idx, "rh2m"] = vals
            if "u10" in point_ds.data_vars:
                vals = np.asarray(point_ds["u10"].values, dtype=float).reshape(-1)
                if vals.size == n_obs:
                    out.loc[idx, "u10"] = vals
            if "v10" in point_ds.data_vars:
                vals = np.asarray(point_ds["v10"].values, dtype=float).reshape(-1)
                if vals.size == n_obs:
                    out.loc[idx, "v10"] = vals
        finally:
            if ds is not None:
                ds.close()

    u = pd.to_numeric(out["u10"], errors="coerce")
    v = pd.to_numeric(out["v10"], errors="coerce")
    out["wind_speed"] = np.sqrt(u * u + v * v)
    out["rh2m_is_available"] = out["rh2m"].notna()
    out["wind_is_available"] = out["u10"].notna() & out["v10"].notna()
    out = out.drop(columns=["_weather_run_id"], errors="ignore")
    return out
