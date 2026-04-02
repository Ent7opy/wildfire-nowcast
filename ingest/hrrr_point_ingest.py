"""HRRR point cache ingest — stores HRRR 3km weather values at
GFS-snapped 0.25° grid points near recent fire detections.

Like weather_point_ingest (GFS path) but uses HRRR for CONUS fires:
hourly cycles, 3km source resolution, CONUS-only coverage.

Data is stored at GFS 0.25° grid coordinates (snapped via snap_to_gfs_grid)
so that _query_point_cache in api/core/weather.py works for both models
with the same lookup key.

Usage (standalone)::

    uv run --project ingest -m ingest.hrrr_point_ingest
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import xarray as xr

from ingest.config import REPO_ROOT
from ingest.hrrr_ingest import (
    ANALYSIS_CELL_SIZE_DEG,
    HRRR_MODEL_NAME,
    HRRR_PRECIP_IDX_KEY,
    HRRR_VARIABLE_IDX_KEYS,
    HRRR_VARIABLE_IDX_PRECIP,
    build_hrrr_dataset,
    build_hrrr_urls,
    crop_hrrr_to_bbox,
    download_hrrr_variable_gribs,
    is_conus_bbox,
    regrid_hrrr_to_analysis_grid,
    snap_to_hrrr_cycle,
)
from ingest.weather_point_ingest import _extract_records_at_grid_points
from ingest.weather_repository import (
    FILE_FORMAT_POINT_CACHE,
    GFS_GRID_DEG,
    STATUS_COMPLETED,
    STATUS_FAILED,
    bbox_from_grid_points,
    bulk_insert_weather_point_cache,
    create_weather_run_record,
    finalize_weather_run_record,
    query_fire_detection_grid_points,
)

# Ensure the API modules are importable when running from ingest/.
sys.path.append(str(REPO_ROOT))

from api.core.grid import GridSpec  # noqa: E402 — after sys.path append

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("hrrr_point_ingest")

# Default forecast hours to cache (analysis + sub-daily up to 18 h).
DEFAULT_HRRR_FORECAST_HOURS: tuple[int, ...] = (0, 1, 2, 3, 6, 9, 12, 18)


def ingest_hrrr_points(
    *,
    forecast_time: datetime | None = None,
    detection_lookback_hours: float = 48.0,
    horizon_hours: int = 18,
    step_hours: int = 1,
    model_name: str = HRRR_MODEL_NAME,
    include_precipitation: bool = True,
    request_timeout_seconds: int = 60,
    max_fallback_age_hours: int = 3,
) -> int:
    """Download HRRR and cache weather values at fire-detection grid points.

    Returns the weather_runs.id of the created record.
    Raises ValueError if detection bbox is outside CONUS.
    Raises on unrecoverable download failure after one fallback cycle.
    """
    now = datetime.now(timezone.utc)
    run_time = snap_to_hrrr_cycle(forecast_time or now)

    # ── 1. Collect target grid points ────────────────────────────────────
    grid_points = query_fire_detection_grid_points(
        lookback_hours=detection_lookback_hours,
    )
    if not grid_points:
        LOGGER.warning(
            "No fire detections in the last %.0f hours — nothing to cache for HRRR.",
            detection_lookback_hours,
        )
        # HRRR does NOT use bootstrap seed points (unlike GFS which covers global).
        # HRRR is only triggered when CONUS detections exist anyway — if there are
        # none, record the attempt and return early.
        run_id = create_weather_run_record(
            model=model_name,
            run_time=run_time,
            horizon_hours=horizon_hours,
            step_hours=step_hours,
            bbox=(None, None, None, None),
            variables=[],
            file_format=FILE_FORMAT_POINT_CACHE,
        )
        finalize_weather_run_record(
            run_id=run_id,
            storage_path="",
            status=STATUS_COMPLETED,
            extra_metadata={"point_cache_rows": 0, "grid_points": 0},
        )
        return run_id

    LOGGER.info(
        "Caching HRRR weather for %d unique GFS grid points (detection lookback=%.0fh)",
        len(grid_points),
        detection_lookback_hours,
    )

    # ── 2. Determine bbox that covers all grid points (with margin) ──────
    bbox = bbox_from_grid_points(grid_points)

    # ── 3. CONUS check — HRRR is CONUS-only ─────────────────────────────
    if not is_conus_bbox(bbox):
        raise ValueError(
            f"HRRR ingest requires CONUS bbox; got {bbox}. "
            "Use GFS ingest for non-CONUS regions."
        )

    # ── 4. Prepare canonical variable lists ──────────────────────────────
    canonical_variables: list[str] = list(HRRR_VARIABLE_IDX_KEYS.keys())
    if include_precipitation:
        canonical_variables.append(HRRR_VARIABLE_IDX_PRECIP)

    variables_to_fetch: dict[str, str] = {**HRRR_VARIABLE_IDX_KEYS}
    if include_precipitation:
        variables_to_fetch[HRRR_VARIABLE_IDX_PRECIP] = HRRR_PRECIP_IDX_KEY

    forecast_hours = tuple(range(0, horizon_hours + 1, step_hours))

    # ── 5. Create weather_runs tracking row ──────────────────────────────
    run_id = create_weather_run_record(
        model=model_name,
        run_time=run_time,
        horizon_hours=horizon_hours,
        step_hours=step_hours,
        bbox=bbox,
        variables=canonical_variables,
        file_format="point_cache",
    )

    # ── 6. Download, regrid, extract, insert ─────────────────────────────
    def _attempt(selected_run_time: datetime) -> int:
        time_datasets: list[xr.Dataset] = []

        # A fresh client per attempt is intentional: on fallback, we want a clean
        # connection pool rather than reusing one that may have seen errors.
        with httpx.Client(timeout=float(request_timeout_seconds)) as client:
            for fh in forecast_hours:
                grib_url, idx_url = build_hrrr_urls(selected_run_time, fh)
                LOGGER.info(
                    "Downloading HRRR f%02d for point cache: %s", fh, grib_url
                )
                with tempfile.TemporaryDirectory(prefix="hrrr_pt_fh_") as tmpdir:
                    grib_paths = download_hrrr_variable_gribs(
                        grib_url,
                        idx_url,
                        variables_to_fetch,
                        Path(tmpdir),
                        client=client,
                        timeout=float(request_timeout_seconds),
                    )
                    ds_fh = build_hrrr_dataset(
                        grib_paths,
                        selected_run_time,
                        fh,
                        include_precip=include_precipitation,
                    )
                    ds_fh = crop_hrrr_to_bbox(ds_fh, bbox)

                    # Regrid HRRR's native Lambert Conformal projection to a
                    # regular 0.01° lat/lon grid so that sel(method="nearest")
                    # in _extract_records_at_grid_points can snap GFS 0.25°
                    # target points to the nearest regridded coordinate.
                    grid = GridSpec.from_bbox(
                        bbox,
                        cell_size_deg=ANALYSIS_CELL_SIZE_DEG,
                    )
                    ds_fh = regrid_hrrr_to_analysis_grid(ds_fh, grid)
                    # Force eager load before the tempdir is deleted — cfgrib arrays are
                    # lazy and backed by the on-disk GRIB file; reading after cleanup fails.
                    ds_fh.load()
                    time_datasets.append(ds_fh)

        # Concatenate all forecast hours into one multi-time Dataset.
        combined_ds = xr.concat(time_datasets, dim="time")
        combined_ds = combined_ds.transpose("time", "lat", "lon")

        records = _extract_records_at_grid_points(
            combined_ds, grid_points, canonical_variables, forecast_hours
        )
        inserted = bulk_insert_weather_point_cache(run_id, records)

        finalize_weather_run_record(
            run_id=run_id,
            storage_path="",
            status=STATUS_COMPLETED,
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

    # ── 7. Attempt primary cycle; fall back to previous hour on failure ───
    try:
        inserted = _attempt(run_time)
        LOGGER.info(
            "HRRR point ingest completed: run_id=%d, rows=%d, grid_points=%d",
            run_id,
            inserted,
            len(grid_points),
        )
        return run_id

    except Exception as primary_exc:
        prev_run_time = run_time - timedelta(hours=1)
        fallback_age = (now - prev_run_time).total_seconds() / 3600

        if fallback_age > max_fallback_age_hours:
            LOGGER.error(
                "Primary HRRR cycle %s failed and fallback %s is %.1fh old "
                "(max %dh). Giving up.",
                run_time,
                prev_run_time,
                fallback_age,
                max_fallback_age_hours,
            )
            finalize_weather_run_record(
                run_id=run_id,
                storage_path="",
                status=STATUS_FAILED,
                extra_metadata={
                    "error": str(primary_exc),
                    "fallback_skipped": True,
                },
            )
            raise

        LOGGER.warning(
            "Primary HRRR cycle %s failed; falling back to %s",
            run_time,
            prev_run_time,
        )
        try:
            inserted = _attempt(prev_run_time)
            LOGGER.info(
                "HRRR point ingest completed via fallback: run_id=%d, rows=%d",
                run_id,
                inserted,
            )
            return run_id
        except Exception as fallback_exc:
            LOGGER.exception("HRRR point ingest failed after fallback")
            finalize_weather_run_record(
                run_id=run_id,
                storage_path="",
                status=STATUS_FAILED,
                extra_metadata={
                    "error": str(fallback_exc),
                    "primary_error": str(primary_exc),
                },
            )
            raise


def main() -> None:
    """CLI entry point."""
    try:
        run_id = ingest_hrrr_points()
        LOGGER.info("Done. weather_runs.id = %d", run_id)
    except Exception:
        LOGGER.exception("HRRR point ingest failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
