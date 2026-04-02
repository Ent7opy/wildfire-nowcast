"""Background weather ingestion that stores GFS values in a DB point cache.

Downloads GFS GRIB files to a temporary directory, extracts weather variables
at GFS 0.25° grid points near recent fire detections, and bulk-inserts the
values into ``weather_point_cache``.  No persistent files are written — the
GRIB tempdir is cleaned up automatically.

A ``weather_runs`` row is created (with ``file_format='point_cache'``) so that
``api.data_status`` freshness reporting continues to work without changes.

Usage (standalone)::

    uv run --project ingest -m ingest.weather_point_ingest

Or via the orchestrator ``weather`` job.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from ingest.config import REPO_ROOT, WeatherIngestSettings, weather_settings
from ingest.weather_ingest import (
    GFS_FILTER_LEVELS,
    GFS_FILTER_VARIABLES,
    build_weather_dataset,
    download_grib_files,
    snap_to_gfs_cycle,
)
from ingest.weather_repository import (
    FILE_FORMAT_POINT_CACHE,
    GFS_MODEL_NAME,
    bbox_from_grid_points,
    bulk_insert_weather_point_cache,
    create_weather_run_record,
    finalize_weather_run_record,
    query_fire_detection_grid_points,
)

# Ensure the API modules are importable when running from ingest/.
sys.path.append(str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("weather_point_ingest")

# Default forecast hours to cache (analysis + up to 24 h out at 6 h steps).
DEFAULT_FORECAST_HOURS: tuple[int, ...] = (0, 6, 12, 18, 24)


def _extract_records_at_grid_points(
    ds: xr.Dataset,
    grid_points: Sequence[tuple[float, float]],
    variables: Sequence[str],
    forecast_hours: Sequence[int],
) -> list[dict]:
    """Extract weather values at *grid_points* for each *forecast_hour*.

    Uses ``sel(method="nearest")`` on the native GFS grid — at 0.25° the snap
    error is at most ~14 km, which is the intrinsic resolution.

    Returns a list of dicts ready for ``bulk_insert_weather_point_cache``.
    """
    records: list[dict] = []

    for fh in forecast_hours:
        # Find nearest time step in the dataset for this forecast hour.
        if "lead_time_hours" in ds.coords:
            target_idx = int(np.argmin(np.abs(ds["lead_time_hours"].values - fh)))
            ds_t = ds.isel(time=target_idx)
            actual_fh = int(ds["lead_time_hours"].values[target_idx])
        elif "time" in ds.dims and ds.dims["time"] > 0:
            ds_t = ds.isel(time=min(fh // 6, ds.dims["time"] - 1))
            actual_fh = fh
        else:
            ds_t = ds
            actual_fh = 0

        for lat_g, lon_g in grid_points:
            pt = ds_t.sel(lat=lat_g, lon=lon_g, method="nearest")
            rec: dict = {
                "forecast_hour": actual_fh,
                "lat_grid": lat_g,
                "lon_grid": lon_g,
            }
            for var in variables:
                if var in pt.data_vars:
                    val = float(pt[var].values)
                    rec[var] = val if not np.isnan(val) else None
                else:
                    rec[var] = None
            records.append(rec)

    return records


# Representative CONUS GFS grid points used when no fire detections are present.
# These are evenly distributed across the continental US and are already on the
# 0.25° GFS grid (each coordinate satisfies x % 0.25 == 0).
BOOTSTRAP_SEED_POINTS: list[tuple[float, float]] = [
    (48.0,  -122.0),  # Pacific NW
    (34.0,  -118.0),  # Southern California
    (39.75, -105.0),  # Colorado
    (30.0,   -90.0),  # Gulf Coast
    (41.75,  -87.75), # Great Lakes
]


def ingest_weather_points(
    *,
    forecast_time: datetime | None = None,
    detection_lookback_hours: float = 48.0,
    horizon_hours: int = 24,
    step_hours: int = 6,
    model_name: str = GFS_MODEL_NAME,
    include_precipitation: bool = True,
    request_timeout_seconds: int = 60,
    max_fallback_age_hours: int = 12,
) -> int:
    """Download GFS and cache weather values at fire-detection grid points.

    Returns the ``weather_runs.id`` of the created record.

    Raises on unrecoverable download failure (after one fallback cycle attempt).

    Note:
        GFS APCP is unavailable at forecast_hour=0 (analysis step). ``tp`` will
        be NULL for fh=0 even when ``include_precipitation=True``.
    """
    now = datetime.now(timezone.utc)
    run_time = snap_to_gfs_cycle(forecast_time or now)

    # ── 1. Collect target grid points ────────────────────────────────────
    grid_points = query_fire_detection_grid_points(
        lookback_hours=detection_lookback_hours,
    )
    if not grid_points:
        LOGGER.warning(
            "No fire detections in the last %.0f hours — nothing to cache.",
            detection_lookback_hours,
        )
        LOGGER.warning("Using bootstrap seed points for initial weather coverage.")
        grid_points = BOOTSTRAP_SEED_POINTS

    LOGGER.info(
        "Caching weather for %d unique GFS grid points (detection lookback=%.0fh)",
        len(grid_points),
        detection_lookback_hours,
    )

    # ── 2. Determine bbox that covers all grid points (with margin) ──────
    bbox = bbox_from_grid_points(grid_points)

    # ── 3. Prepare canonical variable lists ──────────────────────────────
    canonical_variables = ["u10", "v10", "t2m", "rh2m"]
    if include_precipitation:
        canonical_variables.append("tp")

    gfs_variables = [GFS_FILTER_VARIABLES[name] for name in canonical_variables]
    level_params = sorted(
        {lvl for name in canonical_variables if (lvl := GFS_FILTER_LEVELS.get(name))}
    )

    # ── 4. Create weather_runs tracking row ──────────────────────────────
    run_id = create_weather_run_record(
        model=model_name,
        run_time=run_time,
        horizon_hours=horizon_hours,
        step_hours=step_hours,
        bbox=bbox,
        variables=canonical_variables,
        file_format=FILE_FORMAT_POINT_CACHE,
    )

    # ── 5. Build download settings ───────────────────────────────────────
    settings = WeatherIngestSettings(
        WEATHER_MODEL=model_name,
        WEATHER_BASE_DIR=str(Path(tempfile.gettempdir()) / "gfs_point_cache"),
        WEATHER_BBOX_MIN_LON=bbox[0],
        WEATHER_BBOX_MIN_LAT=bbox[1],
        WEATHER_BBOX_MAX_LON=bbox[2],
        WEATHER_BBOX_MAX_LAT=bbox[3],
        WEATHER_HORIZON_HOURS=horizon_hours,
        WEATHER_STEP_HOURS=step_hours,
        WEATHER_INCLUDE_PRECIP=include_precipitation,
        WEATHER_REQUEST_TIMEOUT_SECONDS=request_timeout_seconds,
    )

    base_urls = [weather_settings.gfs_base_url_primary]
    if weather_settings.gfs_base_url_fallback:
        base_urls.append(weather_settings.gfs_base_url_fallback)

    forecast_hours = tuple(range(0, horizon_hours + 1, step_hours))

    # ── 6. Download, extract, insert ─────────────────────────────────────
    def _attempt(selected_run_time: datetime) -> int:
        with tempfile.TemporaryDirectory(prefix="gfs_point_") as tmpdir:
            grib_paths = download_grib_files(
                settings,
                selected_run_time,
                gfs_variables,
                level_params,
                Path(tmpdir),
                base_urls,
            )
            dataset = build_weather_dataset(
                grib_paths,
                selected_run_time,
                include_precip=include_precipitation,
            )

            records = _extract_records_at_grid_points(
                dataset, grid_points, canonical_variables, forecast_hours,
            )
            inserted = bulk_insert_weather_point_cache(run_id, records)

            finalize_weather_run_record(
                run_id=run_id,
                storage_path="",
                status="completed",
                run_time=selected_run_time,
                extra_metadata={
                    "file_format": FILE_FORMAT_POINT_CACHE,
                    "point_cache_rows": inserted,
                    "grid_points": len(grid_points),
                    "variables": canonical_variables,
                    "forecast_hours": list(forecast_hours),
                },
            )
            return inserted

    try:
        inserted = _attempt(run_time)
        LOGGER.info(
            "Weather point ingest completed: run_id=%d, rows=%d, grid_points=%d",
            run_id,
            inserted,
            len(grid_points),
        )
        return run_id

    except Exception as primary_exc:
        # Fallback to previous GFS cycle.
        prev_run_time = run_time - timedelta(hours=6)
        fallback_age = (now - prev_run_time).total_seconds() / 3600

        if fallback_age > max_fallback_age_hours:
            LOGGER.error(
                "Primary cycle %s failed and fallback %s is %.1fh old (max %dh). Giving up.",
                run_time, prev_run_time, fallback_age, max_fallback_age_hours,
            )
            finalize_weather_run_record(
                run_id=run_id, storage_path="", status="failed",
                extra_metadata={"error": str(primary_exc), "fallback_skipped": True},
            )
            raise

        LOGGER.warning(
            "Primary cycle %s failed; falling back to %s",
            run_time, prev_run_time,
        )
        try:
            inserted = _attempt(prev_run_time)
            LOGGER.info(
                "Weather point ingest completed via fallback: run_id=%d, rows=%d",
                run_id, inserted,
            )
            return run_id
        except Exception as fallback_exc:
            LOGGER.exception("Weather point ingest failed after fallback")
            finalize_weather_run_record(
                run_id=run_id, storage_path="", status="failed",
                extra_metadata={
                    "error": str(fallback_exc),
                    "primary_error": str(primary_exc),
                },
            )
            raise


def main() -> None:
    """CLI entry point."""
    try:
        run_id = ingest_weather_points()
        LOGGER.info("Done. weather_runs.id = %d", run_id)
    except Exception:
        LOGGER.exception("Weather point ingest failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
