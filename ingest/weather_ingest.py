"""CLI entrypoint for NOAA GFS weather ingestion."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.parse import urlencode

import httpx
import numpy as np
import xarray as xr

from ingest.config import REPO_ROOT, WeatherIngestSettings, weather_settings
from ingest.logging_utils import log_event
from ingest.weather_repository import (
    create_weather_run_record,
    finalize_weather_run_record,
)

# Ensure the API modules (and config.py) are importable when running from ingest/.
sys.path.append(str(REPO_ROOT))

from api.core.grid import (
    DEFAULT_CELL_SIZE_DEG,
    DEFAULT_CRS,
    GridSpec,
    grid_bounds,
    grid_coords,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("weather_ingest")

GFS_FILTER_VARIABLES = {
    # Variable short names used by NOMADS filter (levels supplied separately)
    "u10": "UGRD",
    "v10": "VGRD",
    "t2m": "TMP",
    "rh2m": "RH",
    "tp": "APCP",
}
GFS_FILTER_LEVELS = {
    "u10": "lev_10_m_above_ground",
    "v10": "lev_10_m_above_ground",
    "t2m": "lev_2_m_above_ground",
    "rh2m": "lev_2_m_above_ground",
    "tp": "lev_surface",
}
SHORT_NAME_MAP = {
    "u10": "10u",
    "v10": "10v",
    "t2m": "2t",
    "rh2m": "2r",
    "tp": "tp",
}

ANALYSIS_CRS = DEFAULT_CRS
ANALYSIS_CELL_SIZE_DEG = DEFAULT_CELL_SIZE_DEG


def build_grid_spec(settings: WeatherIngestSettings) -> GridSpec:
    """Build the canonical analysis grid snapped to 0.01°."""
    min_lon, min_lat, max_lon, max_lat = settings.bbox
    return GridSpec.from_bbox(
        lat_min=min_lat,
        lat_max=max_lat,
        lon_min=min_lon,
        lon_max=max_lon,
        cell_size_deg=ANALYSIS_CELL_SIZE_DEG,
        crs=ANALYSIS_CRS,
    )


def regrid_to_analysis_grid(ds: xr.Dataset, grid: GridSpec) -> xr.Dataset:
    """Interpolate dataset onto the canonical analysis grid."""
    target_lat, target_lon = grid_coords(grid)
    return ds.interp(lat=target_lat, lon=target_lon)


def attach_grid_metadata(ds: xr.Dataset, grid: GridSpec) -> xr.Dataset:
    """Attach grid attributes for downstream reconstruction."""
    grid_meta = {
        "crs": grid.crs,
        "cell_size_deg": grid.cell_size_deg,
        "origin_lat": grid.origin_lat,
        "origin_lon": grid.origin_lon,
        "n_lat": grid.n_lat,
        "n_lon": grid.n_lon,
    }
    ds = ds.assign_attrs(**grid_meta)
    return ds


def grid_metadata_dict(grid: GridSpec) -> dict:
    """Return a serializable grid metadata dictionary."""
    min_lon, min_lat, max_lon, max_lat = grid_bounds(grid)
    return {
        "crs": grid.crs,
        "cell_size_deg": grid.cell_size_deg,
        "origin_lat": grid.origin_lat,
        "origin_lon": grid.origin_lon,
        "n_lat": grid.n_lat,
        "n_lon": grid.n_lon,
        "bbox": {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        },
    }


def _response_snippet(response: httpx.Response, limit: int = 400) -> str:
    """Best-effort extract of a short response body snippet."""
    try:
        text = response.text
    except Exception:  # pragma: no cover - defensive
        text = ""
    if not text:
        try:
            text = response.content[:limit].decode("utf-8", errors="ignore")
        except Exception:  # pragma: no cover - defensive
            text = ""
    return text[:limit] if text else ""


def _extract_error_context(exc: Exception) -> dict:
    """Normalize HTTP error details for logging/metadata."""
    context: dict = {"error": str(exc)}
    weather_ctx = getattr(exc, "weather_context", None)
    if isinstance(weather_ctx, dict):
        context.update(weather_ctx)
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        if response is not None:
            context.setdefault("status_code", response.status_code)
            snippet = _response_snippet(response, limit=300)
            if snippet:
                context.setdefault("response_snippet", snippet)
            if response.request:
                context.setdefault("url", str(response.request.url))
    elif isinstance(exc, httpx.HTTPError) and exc.request:
        context.setdefault("url", str(exc.request.url))
    return context


def build_gfs_grib_url(
    run_time: datetime,
    forecast_hour: int,
    bbox: tuple[float, float, float, float],
    variables: Sequence[str],
    levels: Sequence[str] | None,
    base_url: str,
) -> str:
    """Construct a GFS URL for either NOMADS filter or direct object access."""
    dir_value = f"/gfs.{run_time:%Y%m%d}/{run_time:%H}/atmos"
    file_value = f"gfs.t{run_time:%H}z.pgrb2.0p25.f{forecast_hour:03d}"

    # If base_url points to the filter CGI, build query-string.
    if "filter_gfs" in base_url:
        params = {
            "dir": dir_value,
            "file": file_value,
            "leftlon": bbox[0],
            "rightlon": bbox[2],
            "toplat": bbox[3],
            "bottomlat": bbox[1],
        }
        for variable in variables:
            params[f"var_{variable}"] = "on"
        if levels:
            for level in levels:
                params[level] = "on"
        return f"{base_url}?{urlencode(params)}"

    # Otherwise assume direct object store layout (e.g., AWS noaa-gfs-bdp-pds).
    base_url = base_url.rstrip("/")
    return f"{base_url}{dir_value}/{file_value}"


def download_grib_files(
    settings: WeatherIngestSettings,
    run_time: datetime,
    variables: Sequence[str],
    levels: Sequence[str] | None,
    download_dir: Path,
    base_urls: Sequence[str],
    *,
    max_attempts_per_url: int = 3,
    backoff_seconds: float = 1.0,
) -> list[Path]:
    """Download GRIB2 files for the configured forecast horizon."""
    download_dir.mkdir(parents=True, exist_ok=True)
    forecast_hours = range(0, settings.horizon_hours + 1, settings.step_hours)
    paths: list[Path] = []

    timeout = settings.request_timeout_seconds
    with httpx.Client(timeout=timeout) as client:
        for forecast_hour in forecast_hours:
            target_path = download_dir / f"{settings.model_name}_f{forecast_hour:03d}.grib2"

            last_error: Exception | None = None
            for base_url in base_urls:
                url = build_gfs_grib_url(
                    run_time, forecast_hour, settings.bbox, variables, levels, base_url
                )
                for attempt in range(max_attempts_per_url):
                    try:
                        LOGGER.info(
                            "Downloading GFS GRIB %s (f%03d)", url, forecast_hour
                        )
                        with client.stream("GET", url) as response:
                            if response.is_error:
                                snippet = _response_snippet(response, limit=300)
                                context = {
                                    "url": url,
                                    "status_code": response.status_code,
                                    "response_snippet": snippet,
                                    "forecast_hour": forecast_hour,
                                }
                                LOGGER.error(
                                    "HTTP error for %s status=%s snippet=%s",
                                    url,
                                    response.status_code,
                                    snippet,
                                )
                                http_exc = httpx.HTTPStatusError(
                                    f"Error response {response.status_code} while requesting {url}",
                                    request=response.request,
                                    response=response,
                                )
                                setattr(http_exc, "weather_context", context)
                                raise http_exc
                            with target_path.open("wb") as fh:
                                for chunk in response.iter_bytes():
                                    fh.write(chunk)
                        LOGGER.info(
                            "Downloaded GFS GRIB",
                            extra={
                                "url": url,
                                "forecast_hour": forecast_hour,
                                "target_path": str(target_path),
                            },
                        )
                        paths.append(target_path)
                        last_error = None
                        break
                    except Exception as exc:  # noqa: BLE001 - logged below
                        last_error = exc
                        context = getattr(exc, "weather_context", None)
                        if context is None:
                            context = {"url": url, "forecast_hour": forecast_hour}
                            if isinstance(exc, httpx.HTTPError) and exc.response is not None:
                                context["status_code"] = exc.response.status_code
                                try:
                                    resp_text = exc.response.text
                                    if resp_text:
                                        context["response_snippet"] = resp_text[:300]
                                except Exception:  # pragma: no cover - defensive
                                    pass
                            setattr(exc, "weather_context", context)
                        status_code = context.get("status_code")
                        snippet = context.get("response_snippet")
                        if status_code:
                            LOGGER.warning(
                                "Download attempt failed for %s (status=%s snippet=%s)",
                                url,
                                status_code,
                                snippet,
                            )
                        else:
                            LOGGER.warning("Download attempt failed for %s: %s", url, exc)
                        if attempt == max_attempts_per_url - 1:
                            LOGGER.warning(
                                "Exhausted attempts for forecast hour %s via base %s",
                                forecast_hour,
                                base_url,
                            )
                        else:
                            sleep_s = backoff_seconds * (2**attempt)
                            LOGGER.warning(
                                "Retrying download for forecast hour %s (attempt %s) after %.1fs",
                                forecast_hour,
                                attempt + 2,
                                sleep_s,
                            )
                            time.sleep(sleep_s)
                if last_error is None:
                    break

            if last_error is not None:
                LOGGER.exception("Failed to download forecast hour %s", forecast_hour)
                raise last_error

    paths.sort()
    return paths


def _open_variable_dataset(
    grib_paths: Iterable[Path],
    short_name: str,
) -> xr.Dataset:
    """Open GRIB files for a specific shortName via cfgrib."""
    # Some backends may expose slightly different names; map a few alternates.
    alt_names: dict[str, list[str]] = {
        "10u": ["u10", "UGRD_10maboveground"],
        "10v": ["v10", "VGRD_10maboveground"],
        "2t": ["t2m", "TMP_2maboveground"],
        "2r": ["rh2m", "RH_2maboveground"],
        "tp": ["apcp", "APCP_surface"],
    }

    datasets: list[xr.Dataset] = []
    for path in grib_paths:
        ds_single = xr.open_dataset(
            path,
            engine="cfgrib",
            chunks=None,  # ensure eager arrays, no dask
            backend_kwargs={
                "filter_by_keys": {"shortName": short_name},
                "indexpath": "",
            },
        )
        ds_single.load()  # materialize into memory to avoid chunk manager issues

        if short_name not in ds_single.data_vars:
            # Try renaming if only one variable exists or an alternate name is present.
            alt_candidates = alt_names.get(short_name, [])
            found_name = None
            for candidate in alt_candidates:
                if candidate in ds_single.data_vars:
                    found_name = candidate
                    break
            if found_name is None and len(ds_single.data_vars) == 1:
                found_name = list(ds_single.data_vars.keys())[0]
            if found_name:
                ds_single = ds_single.rename({found_name: short_name})

        datasets.append(ds_single)

    if not datasets:
        raise ValueError(f"No datasets opened for shortName={short_name}")

    ds = xr.concat(datasets, dim="step")
    if "number" in ds.dims:
        ds = ds.squeeze("number", drop=True)
    if "valid_time" in ds:
        valid_time = ds["valid_time"]
    elif "time" in ds and "step" in ds:
        valid_time = ds["time"] + ds["step"]
    else:
        raise ValueError(f"Cannot resolve valid_time for shortName={short_name}")

    ds = ds.assign_coords(time=("step", valid_time.data))
    ds = ds.swap_dims({"step": "time"})
    drop_vars = [var for var in ("step", "valid_time") if var in ds]
    ds = ds.drop_vars(drop_vars, errors="ignore")
    if "heightAboveGround" in ds.dims:
        ds = ds.squeeze("heightAboveGround", drop=True)
    if "heightAboveGround" in ds.coords:
        ds = ds.drop_vars("heightAboveGround", errors="ignore")
    if "surface" in ds.dims:
        ds = ds.squeeze("surface", drop=True)
    return ds


def build_weather_dataset(
    grib_paths: list[Path],
    run_time: datetime,
    *,
    include_precip: bool,
) -> xr.Dataset:
    """Load GRIB files and normalize them into a dataset."""
    if not grib_paths:
        raise ValueError("No GRIB files provided")

    canonical_vars = ["u10", "v10", "t2m", "rh2m"]
    if include_precip:
        canonical_vars.append("tp")

    data_arrays = {}

    for canonical in canonical_vars:
        short_name = SHORT_NAME_MAP[canonical]
        ds_var = _open_variable_dataset(grib_paths, short_name)
        if short_name not in ds_var:
            raise ValueError(f"Variable {short_name} missing from GRIB data")
        data_arrays[canonical] = ds_var[short_name].rename(canonical)

    ds = xr.Dataset(data_arrays)
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    ds = ds.transpose("time", "lat", "lon")

    # Normalize time coordinates to UTC ns precision to avoid tz/precision warnings.
    run_time_utc = run_time.astimezone(timezone.utc).replace(tzinfo=None)
    run_time64 = np.datetime64(run_time_utc, "ns")
    ds = ds.assign_coords(time=ds["time"].astype("datetime64[ns]"))
    ds = ds.assign_coords(forecast_reference_time=run_time64)
    lead_time = (ds["time"].values - run_time64) / np.timedelta64(1, "h")
    ds = ds.assign_coords(lead_time_hours=("time", lead_time.astype(int)))
    return ds


def crop_to_bbox(
    ds: xr.Dataset, settings: WeatherIngestSettings, *, target_bbox: tuple[float, float, float, float] | None = None
) -> xr.Dataset:
    """Enforce bbox subset to keep stored NetCDF aligned with metadata."""
    if target_bbox is None:
        min_lon, min_lat, max_lon, max_lat = (
            settings.bbox_min_lon,
            settings.bbox_min_lat,
            settings.bbox_max_lon,
            settings.bbox_max_lat,
        )
    else:
        min_lon, min_lat, max_lon, max_lat = target_bbox

    lon = ds["lon"]
    if float(lon.max()) > 180:
        ds = ds.assign_coords(lon=((lon + 180) % 360) - 180).sortby("lon")
    lat = ds["lat"]
    if not (lat[0] < lat[-1]):
        ds = ds.sortby("lat")
    return ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))


def save_dataset_to_netcdf(
    ds: xr.Dataset,
    settings: WeatherIngestSettings,
    run_time: datetime,
    *,
    region_bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """Persist dataset to NetCDF following the directory layout."""
    bbox = region_bbox or (
        settings.bbox_min_lon,
        settings.bbox_min_lat,
        settings.bbox_max_lon,
        settings.bbox_max_lat,
    )
    min_lon, min_lat, max_lon, max_lat = bbox

    region_label = "global"
    if not (min_lon == -180 and max_lon == 180 and min_lat == -90 and max_lat == 90):
        region_label = f"bbox_{min_lon}_{min_lat}_{max_lon}_{max_lat}"

    target_dir = (
        Path(settings.base_dir)
        / settings.model_name
        / f"{run_time:%Y}"
        / f"{run_time:%m}"
        / f"{run_time:%d}"
        / f"{run_time:%H}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{settings.model_name}_{run_time:%Y%m%dT%HZ}_"
        f"0-{settings.horizon_hours}h_{region_label}.nc"
    )
    target_path = target_dir / filename
    ds.to_netcdf(target_path, engine="h5netcdf")
    return target_path


def _compute_field_stats(ds: xr.Dataset, variables: Sequence[str]) -> dict:
    """Calculate min/max/mean for selected variables."""
    stats: dict[str, dict[str, float]] = {}
    for var in variables:
        if var not in ds:
            continue
        arr = np.asarray(ds[var].values, dtype=float)
        stats[var] = {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
        }
    return stats


def _validate_weather_dataset(
    ds: xr.Dataset, settings: WeatherIngestSettings, variables: Sequence[str]
) -> None:
    """Log validation details for weather datasets."""
    missing_vars = sorted([v for v in variables if v not in ds.data_vars])
    if missing_vars:
        log_event(
            LOGGER,
            "weather.validation",
            "Missing expected variables",
            level="warning",
            missing_vars=missing_vars,
        )

    missing_dims = sorted([dim for dim in ("time", "lat", "lon") if dim not in ds.dims])
    if missing_dims:
        log_event(
            LOGGER,
            "weather.validation",
            "Dataset missing expected dimensions",
            level="warning",
            missing_dims=missing_dims,
        )

    if "time" not in ds or "lead_time_hours" not in ds:
        log_event(
            LOGGER,
            "weather.validation",
            "Dataset missing time coverage metadata",
            level="warning",
        )
        return

    time_values = ds["time"].values
    lead_values = ds["lead_time_hours"].values
    lead_min = int(np.nanmin(lead_values))
    lead_max = int(np.nanmax(lead_values))
    expected_max = min(settings.horizon_hours, 72)
    expected_count = (settings.horizon_hours // settings.step_hours) + 1
    time_count = int(ds.dims.get("time", 0))

    if lead_min > 0 or lead_max < expected_max or time_count < expected_count:
        log_event(
            LOGGER,
            "weather.validation",
            "Time dimension does not fully cover requested horizon",
            level="warning",
            lead_min=lead_min,
            lead_max=lead_max,
            expected_max=expected_max,
            time_count=time_count,
            expected_count=expected_count,
        )

    if lead_max > 72:
        log_event(
            LOGGER,
            "weather.validation",
            "Lead time exceeds 72h sanity window",
            level="warning",
            lead_max=lead_max,
        )

    field_stats = _compute_field_stats(ds, variables)
    log_event(
        LOGGER,
        "weather.stats",
        "Weather dataset stats",
        time_range={
            "start": str(time_values.min()),
            "end": str(time_values.max()),
        },
        lead_time_hours={"min": lead_min, "max": lead_max, "expected_max": expected_max},
        field_stats=field_stats,
        time_count=time_count,
        dims={k: int(v) for k, v in ds.dims.items()},
    )


def snap_to_gfs_cycle(dt: datetime) -> datetime:
    """Snap a datetime to the latest preceding GFS cycle (00, 06, 12, 18 UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    hour_block = (dt.hour // 6) * 6
    return dt.replace(hour=hour_block, minute=0, second=0, microsecond=0)


def _resolve_run_time(cli_value: str | None, configured: datetime | None) -> datetime:
    """Pick run_time from CLI, config, or latest 6h cycle."""
    if cli_value:
        value = cli_value.rstrip("Z")
        run_dt = datetime.fromisoformat(value)
        return snap_to_gfs_cycle(run_dt)
    if configured:
        return snap_to_gfs_cycle(configured)

    return snap_to_gfs_cycle(datetime.now(timezone.utc))


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GFS weather ingestion pipeline.")
    parser.add_argument("--run-time", type=str, default=None, help="ISO8601 model run time (UTC).")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override bounding box (lon/lat).",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=None,
        help="Override forecast horizon in hours (default: settings).",
    )
    parser.add_argument(
        "--step-hours",
        type=int,
        default=None,
        help="Override forecast step in hours (default: settings).",
    )
    parser.add_argument(
        "--include-precip",
        action="store_true",
        help="Include accumulated precipitation (APCP).",
    )
    parser.add_argument(
        "--patch-mode",
        action="store_true",
        help="Optimize for small AOIs: 24h horizon, 6h steps, no precipitation.",
    )
    return parser.parse_args(argv)


def _apply_overrides(
    settings: WeatherIngestSettings,
    args: argparse.Namespace,
) -> WeatherIngestSettings:
    updates = {}
    if args.bbox:
        updates["bbox_min_lon"] = args.bbox[0]
        updates["bbox_min_lat"] = args.bbox[1]
        updates["bbox_max_lon"] = args.bbox[2]
        updates["bbox_max_lat"] = args.bbox[3]
    if args.horizon_hours is not None:
        updates["horizon_hours"] = args.horizon_hours
    if args.step_hours is not None:
        updates["step_hours"] = args.step_hours
    if args.include_precip:
        updates["include_precipitation"] = True
    return settings.model_copy(update=updates)


def ingest_weather_for_bbox(
    bbox: tuple[float, float, float, float],
    forecast_time: datetime,
    output_dir: Path | str,
    *,
    horizon_hours: int = 72,
    step_hours: int = 3,
    include_precipitation: bool = False,
    model_name: str = "gfs_0p25",
    request_timeout_seconds: int = 60,
    patch_mode: bool = False,
) -> int:
    """Ingest weather data for the given bbox and forecast time.

    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat).
        forecast_time: The model run time (UTC).
        output_dir: Directory to store the NetCDF output.
        horizon_hours: Forecast horizon in hours (default: 72).
        step_hours: Forecast step in hours (default: 3).
        include_precipitation: Whether to include precipitation data (default: False).
        model_name: Weather model identifier (default: "gfs_0p25").
        request_timeout_seconds: HTTP request timeout (default: 60).
        patch_mode: Enable optimizations for small AOIs (<50km x 50km).
                    Reduces horizon to 24h, uses 6h steps, skips precipitation,
                    and adds spatial margin for subsetting. (default: False).

    Returns:
        weather_run_id: The database ID of the created weather run record.

    Raises:
        Exception: If the ingestion fails after fallback attempts.
    """
    output_dir = Path(output_dir)
    original_time = forecast_time
    forecast_time = snap_to_gfs_cycle(forecast_time)
    
    # Adjust horizon to cover the original requested time range if we snapped back
    if original_time > forecast_time:
        diff_hours = (original_time - forecast_time).total_seconds() / 3600
        # Increase horizon by at least the difference, rounded up to next step
        horizon_hours += int(np.ceil(diff_hours))
        LOGGER.info(
            "Snapped forecast_time %s -> %s, increased horizon to %dh to maintain coverage",
            original_time.isoformat(),
            forecast_time.isoformat(),
            horizon_hours,
        )

    # Apply patch mode optimizations for small AOIs
    if patch_mode:
        horizon_hours = 24
        step_hours = 6
        include_precipitation = False
        LOGGER.info(
            "Patch mode enabled: horizon_hours=24, step_hours=6, precipitation=False"
        )

    min_lon, min_lat, max_lon, max_lat = bbox
    
    # For patch mode, add spatial margin to download bbox to ensure adequate coverage
    # for interpolation, but final grid will be cropped to original bbox
    download_bbox = bbox
    if patch_mode:
        margin_deg = 0.5  # ~50km at equator
        download_bbox = (
            min_lon - margin_deg,
            min_lat - margin_deg,
            max_lon + margin_deg,
            max_lat + margin_deg,
        )
        LOGGER.info(
            "Patch mode: downloading with margin bbox=%s, final bbox=%s",
            download_bbox,
            bbox,
        )
    
    grid_spec = GridSpec.from_bbox(
        lat_min=min_lat,
        lat_max=max_lat,
        lon_min=min_lon,
        lon_max=max_lon,
        cell_size_deg=ANALYSIS_CELL_SIZE_DEG,
        crs=ANALYSIS_CRS,
    )
    grid_meta = grid_metadata_dict(grid_spec)

    canonical_variables = ["u10", "v10", "t2m", "rh2m"]
    if include_precipitation:
        canonical_variables.append("tp")

    variables = [GFS_FILTER_VARIABLES[name] for name in canonical_variables]
    level_params = sorted(
        {lvl for name in canonical_variables if (lvl := GFS_FILTER_LEVELS.get(name))}
    )

    LOGGER.info(
        "Starting weather ingestion",
        extra={
            "run_time": forecast_time.isoformat(),
            "bbox": bbox,
            "horizon_hours": horizon_hours,
            "step_hours": step_hours,
            "include_precipitation": include_precipitation,
            "grid": grid_meta,
        },
    )

    run_id = create_weather_run_record(
        model=model_name,
        run_time=forecast_time,
        horizon_hours=horizon_hours,
        step_hours=step_hours,
        bbox=bbox,
        variables=canonical_variables,
    )

    storage_path = ""
    base_urls = [weather_settings.gfs_base_url_primary]
    if weather_settings.gfs_base_url_fallback:
        base_urls.append(weather_settings.gfs_base_url_fallback)

    # Use download_bbox for NOMADS filter, but keep original bbox for final output
    dl_min_lon, dl_min_lat, dl_max_lon, dl_max_lat = download_bbox
    settings = WeatherIngestSettings(
        model_name=model_name,
        base_dir=output_dir,
        bbox_min_lon=dl_min_lon,
        bbox_min_lat=dl_min_lat,
        bbox_max_lon=dl_max_lon,
        bbox_max_lat=dl_max_lat,
        horizon_hours=horizon_hours,
        step_hours=step_hours,
        include_precipitation=include_precipitation,
        request_timeout_seconds=request_timeout_seconds,
    )

    def _attempt_ingest(selected_run_time: datetime) -> None:
        nonlocal storage_path
        with tempfile.TemporaryDirectory(prefix="gfs_grib_") as tmpdir:
            grib_paths = download_grib_files(
                settings,
                selected_run_time,
                variables,
                level_params,
                Path(tmpdir),
                base_urls,
            )
            dataset = build_weather_dataset(
                grib_paths,
                selected_run_time,
                include_precip=include_precipitation,
            )
            # Crop to original bbox (not download_bbox with margin)
            dataset = crop_to_bbox(dataset, settings, target_bbox=bbox)
            dataset = regrid_to_analysis_grid(dataset, grid_spec)
            dataset = attach_grid_metadata(dataset, grid_spec)
            _validate_weather_dataset(dataset, settings, canonical_variables)
            storage_path = str(
                save_dataset_to_netcdf(
                    dataset,
                    settings,
                    selected_run_time,
                    region_bbox=grid_bounds(grid_spec),
                )
            )
            finalize_weather_run_record(
                run_id=run_id,
                storage_path=storage_path,
                status="completed",
                run_time=selected_run_time,
                extra_metadata={
                    "variables": list(dataset.data_vars.keys()),
                    "dimensions": {k: int(v) for k, v in dataset.dims.items()},
                    "run_time": selected_run_time.isoformat(),
                    "grid": grid_meta,
                },
            )

    try:
        _attempt_ingest(forecast_time)
        LOGGER.info(
            "Weather ingest completed",
            extra={"run_id": run_id, "storage_path": storage_path},
        )
        return run_id
    except httpx.HTTPStatusError as exc:
        error_ctx = _extract_error_context(exc)
        prev_run_time = forecast_time - timedelta(hours=6)
        LOGGER.warning(
            "Primary cycle %s failed (status %s). Falling back to previous cycle %s",
            forecast_time.isoformat(),
            error_ctx.get("status_code", "unknown"),
            prev_run_time.isoformat(),
        )
        try:
            _attempt_ingest(prev_run_time)
            LOGGER.info(
                "Weather ingest completed via fallback cycle",
                extra={
                    "run_id": run_id,
                    "storage_path": storage_path,
                    "fallback_run_time": prev_run_time.isoformat(),
                },
            )
            return run_id
        except Exception as fallback_exc:
            fallback_ctx = _extract_error_context(fallback_exc)
            LOGGER.exception("Weather ingest failed after fallback cycle")
            finalize_weather_run_record(
                run_id=run_id,
                storage_path=storage_path,
                status="failed",
                run_time=prev_run_time,
                extra_metadata={
                    "error": str(fallback_exc),
                    "fallback_run_time": prev_run_time.isoformat(),
                    "primary_error": error_ctx,
                    "fallback_error": fallback_ctx,
                },
            )
            raise
    except Exception as exc:
        LOGGER.exception("Weather ingest failed")
        finalize_weather_run_record(
            run_id=run_id,
            storage_path=storage_path,
            status="failed",
            run_time=forecast_time,
            extra_metadata=_extract_error_context(exc),
        )
        raise


def run_weather_ingest(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    settings = _apply_overrides(weather_settings, args)
    run_time = _resolve_run_time(args.run_time, settings.run_time)

    try:
        ingest_weather_for_bbox(
            bbox=settings.bbox,
            forecast_time=run_time,
            output_dir=settings.base_dir,
            horizon_hours=settings.horizon_hours,
            step_hours=settings.step_hours,
            include_precipitation=settings.include_precipitation,
            model_name=settings.model_name,
            request_timeout_seconds=settings.request_timeout_seconds,
            patch_mode=args.patch_mode,
        )
        return 0
    except Exception:
        return 1


def main(argv: List[str] | None = None) -> None:
    exit_code = run_weather_ingest(argv)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])

