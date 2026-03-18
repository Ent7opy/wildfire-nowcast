"""HRRR (High-Resolution Rapid Refresh) weather ingest for CONUS.

Downloads surface-level fields from the NOAA open-data S3 bucket using
byte-range requests so that only the required variables are transferred
(the full wrfsubhf files are 200-400 MB each).

CONUS bounds (approximate):  lon [-125, -66],  lat [22, 50]
Source:  s3://noaa-hrrr-bdp-pds/  (public, no auth required)
Cycles:  hourly (t00z – t23z)
Horizon: 0 – 18 h forecast (wrfsubhf{FH:02d}.grib2)
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import xarray as xr

from ingest.config import REPO_ROOT
from ingest.logging_utils import log_event
from ingest.weather_repository import (
    create_weather_run_record,
    finalize_weather_run_record,
)

sys.path.append(str(REPO_ROOT))

from api.core.grid import (
    DEFAULT_CELL_SIZE_DEG,
    DEFAULT_CRS,
    GridSpec,
    grid_coords,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("hrrr_ingest")

# HRRR S3 open-data base URL (no authentication required)
HRRR_S3_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HRRR_MODEL_NAME = "hrrr_3km"

# HRRR CONUS approximate bounds
HRRR_CONUS_LON_MIN = -125.0
HRRR_CONUS_LON_MAX = -66.0
HRRR_CONUS_LAT_MIN = 22.0
HRRR_CONUS_LAT_MAX = 50.0

# Variables extracted from wrfsubhf (surface hourly fields)
# Keys are our canonical names; values are search strings in the .idx file.
HRRR_VARIABLE_IDX_KEYS: dict[str, str] = {
    "u10":  "UGRD:10 m above ground",
    "v10":  "VGRD:10 m above ground",
    "t2m":  "TMP:2 m above ground",
    "rh2m": "RH:2 m above ground",
}
HRRR_VARIABLE_IDX_PRECIP = "tp"
HRRR_PRECIP_IDX_KEY = "APCP:surface"

ANALYSIS_CELL_SIZE_DEG = DEFAULT_CELL_SIZE_DEG
ANALYSIS_CRS = DEFAULT_CRS


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def snap_to_hrrr_cycle(dt: datetime) -> datetime:
    """Snap to the latest preceding HRRR cycle (hourly)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def build_hrrr_urls(run_time: datetime, forecast_hour: int) -> tuple[str, str]:
    """Return ``(grib_url, idx_url)`` for an HRRR wrfsubhf file."""
    date_str = run_time.strftime("%Y%m%d")
    hour_str = run_time.strftime("%H")
    stem = f"hrrr.t{hour_str}z.wrfsubhf{forecast_hour:02d}.grib2"
    base = f"{HRRR_S3_BASE}/hrrr.{date_str}/conus/{stem}"
    return base, f"{base}.idx"


def is_conus_bbox(bbox: tuple[float, float, float, float]) -> bool:
    """Return True if the bbox is entirely within approximate CONUS bounds."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        min_lon >= HRRR_CONUS_LON_MIN
        and max_lon <= HRRR_CONUS_LON_MAX
        and min_lat >= HRRR_CONUS_LAT_MIN
        and max_lat <= HRRR_CONUS_LAT_MAX
    )


# ---------------------------------------------------------------------------
# .idx file parsing & byte-range download
# ---------------------------------------------------------------------------

def parse_hrrr_idx(idx_text: str, variables: dict[str, str]) -> list[dict]:
    """Parse HRRR .idx content and return byte ranges for the requested variables.

    Parameters
    ----------
    idx_text : str
        Contents of the ``.grib2.idx`` file.
    variables : dict[str, str]
        Mapping of canonical variable name → idx search key (e.g. "UGRD:10 m above ground").

    Returns
    -------
    list of dicts with keys: ``canonical``, ``start_byte``, ``end_byte`` (-1 = EOF).
    """
    lines = [ln.strip() for ln in idx_text.splitlines() if ln.strip()]
    records: list[dict] = []
    for line in lines:
        parts = line.split(":")
        if len(parts) < 6:
            continue
        try:
            records.append(
                {
                    "num": int(parts[0]),
                    "offset": int(parts[1]),
                    "var_level": f"{parts[3]}:{parts[4]}",
                }
            )
        except (ValueError, IndexError):
            continue

    result: list[dict] = []
    for i, rec in enumerate(records):
        for canonical, search_key in variables.items():
            if search_key in rec["var_level"]:
                start = rec["offset"]
                end = records[i + 1]["offset"] - 1 if i + 1 < len(records) else -1
                result.append({"canonical": canonical, "start_byte": start, "end_byte": end})
                break  # each record matches at most one variable

    return result


def _download_bytes_range(
    client: httpx.Client,
    url: str,
    start: int,
    end: int,
    *,
    timeout: float = 60.0,
) -> bytes:
    """Download a byte range from a URL."""
    headers = {"Range": f"bytes={start}-{end}" if end != -1 else f"bytes={start}-"}
    response = client.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content


def download_hrrr_variable_gribs(
    grib_url: str,
    idx_url: str,
    variables: dict[str, str],
    download_dir: Path,
    *,
    client: httpx.Client,
    timeout: float = 60.0,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> dict[str, Path]:
    """Download individual GRIB2 messages for requested variables via byte-range.

    Returns a dict mapping canonical variable name → path to a single-message GRIB2 file.
    """
    # 1. Download the index file (tiny, ~few KB)
    LOGGER.info("Fetching HRRR index: %s", idx_url)
    idx_text: str = ""
    for attempt in range(max_attempts):
        try:
            resp = client.get(idx_url, timeout=timeout)
            resp.raise_for_status()
            idx_text = resp.text
            break
        except Exception as exc:
            if attempt == max_attempts - 1:
                raise
            sleep_s = backoff_seconds * (2 ** attempt)
            LOGGER.warning("idx download attempt %d failed: %s; retrying in %.1fs", attempt + 1, exc, sleep_s)
            time.sleep(sleep_s)

    byte_ranges = parse_hrrr_idx(idx_text, variables)
    if not byte_ranges:
        raise ValueError(
            f"No matching HRRR variables found in idx {idx_url}. "
            f"Sought: {list(variables.values())}"
        )

    found = {r["canonical"] for r in byte_ranges}
    missing = set(variables.keys()) - found
    if missing:
        LOGGER.warning("HRRR idx missing variables: %s (available search keys: %s)", missing, list(variables.values()))

    # 2. Download each variable's byte range
    output: dict[str, Path] = {}
    for rec in byte_ranges:
        canonical = rec["canonical"]
        out_path = download_dir / f"hrrr_{canonical}.grib2"
        for attempt in range(max_attempts):
            try:
                data = _download_bytes_range(
                    client, grib_url, rec["start_byte"], rec["end_byte"], timeout=timeout
                )
                out_path.write_bytes(data)
                LOGGER.info(
                    "Downloaded HRRR %s (%d bytes)",
                    canonical,
                    len(data),
                )
                output[canonical] = out_path
                break
            except Exception as exc:
                if attempt == max_attempts - 1:
                    raise
                sleep_s = backoff_seconds * (2 ** attempt)
                LOGGER.warning(
                    "HRRR byte-range download attempt %d for %s failed: %s; retrying in %.1fs",
                    attempt + 1,
                    canonical,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)

    return output


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _open_hrrr_grib(path: Path, canonical: str) -> xr.DataArray:
    """Open a single-variable HRRR GRIB2 file and return a lat/lon DataArray.

    cfgrib reprojects HRRR's Lambert conformal grid to regular lat/lon automatically.
    """
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
        chunks=None,
    )
    ds.load()
    # Find the variable (may have a different internal name)
    if canonical not in ds.data_vars:
        if len(ds.data_vars) == 1:
            var_name = next(iter(ds.data_vars))
            ds = ds.rename({var_name: canonical})
        else:
            raise ValueError(
                f"Cannot find {canonical!r} in HRRR GRIB file {path}; "
                f"available: {list(ds.data_vars)}"
            )
    da = ds[canonical]
    # Rename lat/lon coordinates if needed
    rename = {}
    if "latitude" in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    ds.close()
    return da


def build_hrrr_dataset(
    grib_paths: dict[str, Path],
    run_time: datetime,
    forecast_hour: int,
    *,
    include_precip: bool = False,
) -> xr.Dataset:
    """Assemble individual HRRR GRIB fragments into a single time-indexed Dataset."""
    arrays: dict[str, xr.DataArray] = {}
    for canonical, path in grib_paths.items():
        if not include_precip and canonical == HRRR_VARIABLE_IDX_PRECIP:
            continue
        arrays[canonical] = _open_hrrr_grib(path, canonical)

    if not arrays:
        raise ValueError("No HRRR arrays loaded.")

    # Stack into a Dataset on a shared (lat, lon) grid.
    # All variables at the same forecast hour share the same spatial grid.
    ds = xr.Dataset(arrays)

    # Rename spatial dims to our convention if needed.
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.dims:
        ds = ds.rename({"longitude": "lon"})

    # HRRR longitudes may be in [0, 360]; normalise to [-180, 180].
    if "lon" in ds.coords and float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180).sortby("lon")
    if "lat" in ds.coords and not (ds["lat"][0] < ds["lat"][-1]):
        ds = ds.sortby("lat")

    # Attach time coordinate (valid time = run_time + forecast_hour).
    valid_time = run_time + timedelta(hours=forecast_hour)
    valid_time_utc = valid_time.astimezone(timezone.utc).replace(tzinfo=None)
    valid_time64 = np.datetime64(valid_time_utc, "ms")
    run_time64 = np.datetime64(run_time.astimezone(timezone.utc).replace(tzinfo=None), "ms")

    ds = ds.expand_dims(dim="time")
    ds = ds.assign_coords(
        time=[valid_time64],
        lead_time_hours=("time", [forecast_hour]),
        forecast_reference_time=run_time64,
    )
    ds = ds.transpose("time", "lat", "lon")
    return ds


def crop_hrrr_to_bbox(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    min_lon, min_lat, max_lon, max_lat = bbox
    return ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))


def regrid_hrrr_to_analysis_grid(ds: xr.Dataset, grid: GridSpec) -> xr.Dataset:
    target_lat, target_lon = grid_coords(grid)
    return ds.interp(lat=target_lat, lon=target_lon)


# ---------------------------------------------------------------------------
# Multi-step ingest (builds a single multi-time NetCDF like GFS)
# ---------------------------------------------------------------------------

def ingest_hrrr_for_bbox(
    bbox: tuple[float, float, float, float],
    forecast_time: datetime,
    output_dir: Path | str,
    *,
    horizon_hours: int = 18,
    step_hours: int = 1,
    include_precipitation: bool = False,
    request_timeout_seconds: int = 60,
    max_attempts: int = 3,
) -> int:
    """Ingest HRRR data for a CONUS bbox.

    Downloads one GRIB2 message per variable per forecast hour, assembles them
    into a single time-indexed NetCDF, and records the run in ``weather_runs``.

    Parameters
    ----------
    bbox
        (min_lon, min_lat, max_lon, max_lat) – must be within CONUS.
    forecast_time
        Desired run time; will be snapped to the nearest preceding hourly cycle.
    output_dir
        Directory for output NetCDF files.
    horizon_hours
        Maximum forecast horizon (0 – 18 for HRRR).
    step_hours
        Step between downloaded forecast hours (1 for all hours).
    include_precipitation
        Whether to include APCP (accumulated precipitation).
    request_timeout_seconds
        HTTP timeout per request.
    max_attempts
        Retry attempts per download.

    Returns
    -------
    int
        ID of the created ``weather_runs`` record.

    Raises
    ------
    ValueError
        If the bbox is outside CONUS.
    """
    if not is_conus_bbox(bbox):
        raise ValueError(
            f"HRRR ingest requires a CONUS bbox; got {bbox}. "
            "Use GFS ingest for non-CONUS regions."
        )

    output_dir = Path(output_dir)
    run_time = snap_to_hrrr_cycle(forecast_time)

    min_lon, min_lat, max_lon, max_lat = bbox
    grid = GridSpec.from_bbox(
        lat_min=min_lat,
        lat_max=max_lat,
        lon_min=min_lon,
        lon_max=max_lon,
        cell_size_deg=ANALYSIS_CELL_SIZE_DEG,
        crs=ANALYSIS_CRS,
    )

    canonical_vars = list(HRRR_VARIABLE_IDX_KEYS.keys())
    if include_precipitation:
        canonical_vars.append(HRRR_VARIABLE_IDX_PRECIP)

    LOGGER.info(
        "Starting HRRR ingest",
        extra={
            "run_time": run_time.isoformat(),
            "bbox": bbox,
            "horizon_hours": horizon_hours,
            "step_hours": step_hours,
        },
    )

    run_id = create_weather_run_record(
        model=HRRR_MODEL_NAME,
        run_time=run_time,
        horizon_hours=horizon_hours,
        step_hours=step_hours,
        bbox=bbox,
        variables=canonical_vars,
    )

    storage_path = ""
    try:
        forecast_hours = list(range(0, horizon_hours + 1, step_hours))
        time_datasets: list[xr.Dataset] = []

        variables_to_fetch = {**HRRR_VARIABLE_IDX_KEYS}
        if include_precipitation:
            variables_to_fetch[HRRR_VARIABLE_IDX_PRECIP] = HRRR_PRECIP_IDX_KEY

        with httpx.Client(timeout=request_timeout_seconds) as client:
            for fh in forecast_hours:
                grib_url, idx_url = build_hrrr_urls(run_time, fh)
                LOGGER.info("Downloading HRRR f%02d: %s", fh, grib_url)
                with tempfile.TemporaryDirectory(prefix="hrrr_fh_") as tmpdir:
                    grib_paths = download_hrrr_variable_gribs(
                        grib_url,
                        idx_url,
                        variables_to_fetch,
                        Path(tmpdir),
                        client=client,
                        timeout=float(request_timeout_seconds),
                        max_attempts=max_attempts,
                    )
                    ds_fh = build_hrrr_dataset(
                        grib_paths,
                        run_time,
                        fh,
                        include_precip=include_precipitation,
                    )
                    ds_fh = crop_hrrr_to_bbox(ds_fh, bbox)
                    ds_fh = regrid_hrrr_to_analysis_grid(ds_fh, grid)
                    ds_fh.load()
                    time_datasets.append(ds_fh)

        # Concatenate all forecast hours into a single dataset.
        ds_full = xr.concat(time_datasets, dim="time")
        ds_full = ds_full.transpose("time", "lat", "lon")

        # Save as NetCDF.
        region_label = f"bbox_{min_lon}_{min_lat}_{max_lon}_{max_lat}"
        target_dir = (
            output_dir
            / HRRR_MODEL_NAME
            / f"{run_time:%Y}"
            / f"{run_time:%m}"
            / f"{run_time:%d}"
            / f"{run_time:%H}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{HRRR_MODEL_NAME}_{run_time:%Y%m%dT%HZ}_0-{horizon_hours}h_{region_label}.nc"
        out_path = target_dir / filename
        ds_full.to_netcdf(out_path, engine="h5netcdf")
        storage_path = str(out_path)

        finalize_weather_run_record(
            run_id=run_id,
            storage_path=storage_path,
            status="completed",
            run_time=run_time,
            extra_metadata={
                "variables": list(ds_full.data_vars.keys()),
                "dimensions": {k: int(v) for k, v in ds_full.dims.items()},
                "run_time": run_time.isoformat(),
                "forecast_hours": forecast_hours,
            },
        )

        log_event(
            LOGGER,
            "hrrr_ingest.completed",
            "HRRR ingest completed",
            run_id=run_id,
            storage_path=storage_path,
            forecast_hours_count=len(forecast_hours),
        )
        return run_id

    except Exception as exc:
        LOGGER.exception("HRRR ingest failed")
        finalize_weather_run_record(
            run_id=run_id,
            storage_path=storage_path,
            status="failed",
            run_time=run_time,
            extra_metadata={"error": str(exc)},
        )
        raise
