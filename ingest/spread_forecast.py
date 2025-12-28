"""CLI for running wildfire spread forecasts and persisting products."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.ops import unary_union

from ingest.config import REPO_ROOT
from ml.spread.contract import DEFAULT_HORIZONS_HOURS, SpreadForecast

# Ensure the API modules are importable
sys.path.append(str(REPO_ROOT))
from api.core.grid import GridSpec, GridWindow, get_grid_window_for_bbox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("spread_forecast_ingest")


def _as_datetime64_utc_naive(dt: datetime) -> np.datetime64:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(dt, "ns")


def _select_probability_slice_by_horizon(
    forecast: SpreadForecast, horizon_hours: int
) -> np.ndarray:
    """Return a 2D (lat, lon) array for a given horizon.

    The model contract recommends a `lead_time_hours` coordinate aligned to the `time`
    dimension, but this coordinate is not necessarily an xarray index. We therefore
    select by time index rather than `.sel(lead_time_hours=...)`.
    """
    da = forecast.probabilities

    # Preferred path: select the matching time index via the lead_time_hours coord.
    if "lead_time_hours" in da.coords:
        lead = np.asarray(da.coords["lead_time_hours"].values)
        lead_int = lead.astype(int, copy=False)
        matches = np.where(lead_int == int(horizon_hours))[0]
        if matches.size:
            return np.asarray(da.isel(time=int(matches[0])).values)

    # Fallback: select by absolute time.
    target = forecast.forecast_reference_time + timedelta(hours=int(horizon_hours))
    time_coord = np.asarray(da.coords["time"].values)
    if time_coord.dtype.kind != "M":
        raise ValueError(
            "Cannot select probability slice by horizon: `time` coord is not datetime64 "
            f"(dtype={time_coord.dtype!s})."
        )

    # Ensure target has same unit as time coordinate to avoid unit-mismatch surprises.
    time_unit = np.datetime_data(time_coord.dtype)[0]
    target64 = np.datetime64(_as_datetime64_utc_naive(target), time_unit)

    matches = np.where(time_coord == target64)[0]
    if matches.size:
        return np.asarray(da.isel(time=int(matches[0])).values)

    # Never silently "snap" to a nearby horizon; this can generate incorrect rasters/contours.
    # Provide a helpful error message for debugging / CLI usage.
    available_h = None
    if "lead_time_hours" in da.coords:
        try:
            available_h = [int(x) for x in np.asarray(da.coords["lead_time_hours"].values).astype(int)]
        except Exception:
            available_h = None
    if available_h is None:
        try:
            # Derive lead times from time coordinate.
            ref64 = np.datetime64(_as_datetime64_utc_naive(forecast.forecast_reference_time), time_unit)
            delta_hours = (time_coord - ref64).astype("timedelta64[h]").astype(int)
            available_h = [int(x) for x in delta_hours.tolist()]
        except Exception:
            available_h = None

    raise ValueError(
        f"Requested horizon_hours={int(horizon_hours)} does not exist in forecast output. "
        + (f"Available horizons: {sorted(set(available_h))}." if available_h is not None else "")
    )


def convert_to_cog(in_path: Path) -> Path:
    """Convert GeoTIFF to Cloud Optimized GeoTIFF."""
    from rio_cogeo import cog_profiles, cog_translate

    out_path = in_path.with_name(in_path.stem + "_cog.tif")
    profile = cog_profiles.get("deflate")
    cog_translate(
        in_path,
        out_path,
        profile,
        in_memory=False,
        quiet=True,
    )
    LOGGER.info("Wrote COG to %s", out_path)
    return out_path


def generate_contours(
    data: np.ndarray, transform: rasterio.Affine, thresholds: Sequence[float]
) -> list[dict]:
    """Generate contours as GeoJSON for specified thresholds."""
    from rasterio import features

    all_contours = []
    for t in thresholds:
        # Create a mask for values >= threshold
        mask = (data >= t).astype(np.uint8)
        # Extract shapes (polygons) for the mask
        shapes = list(features.shapes(mask, mask=mask, transform=transform))

        polygons = [shape(geom) for geom, val in shapes if val == 1]
        if polygons:
            # Union all polygons for this threshold.
            merged = unary_union(polygons)
            # Normalize type to MultiPolygon to match DB column type.
            if merged.geom_type == "Polygon":
                merged = MultiPolygon([merged])
            elif merged.geom_type == "MultiPolygon":
                pass
            else:
                # If union yields GEOMETRYCOLLECTION (or other), keep only polygonal parts.
                poly_parts = [g for g in getattr(merged, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
                if not poly_parts:
                    merged = MultiPolygon([])
                else:
                    flattened = []
                    for g in poly_parts:
                        if g.geom_type == "Polygon":
                            flattened.append(g)
                        else:
                            flattened.extend(list(g.geoms))
                    merged = MultiPolygon(flattened)
            geom_geojson = json.dumps(mapping(merged))
        else:
            # Always emit an entry per threshold so downstream/UI can rely on presence.
            # This encodes MULTIPOLYGON EMPTY in GeoJSON form.
            geom_geojson = json.dumps({"type": "MultiPolygon", "coordinates": []})

        all_contours.append({"threshold": t, "geom_geojson": geom_geojson})
    return all_contours


def _transform_for_grid_window(grid: GridSpec, window: GridWindow) -> rasterio.Affine:
    """Return a north-up raster transform for a bbox-window on the region grid.

    Notes
    -----
    - `GridSpec.origin_lat/origin_lon` are the **southern/western cell edges**.
    - `GridWindow` indices are half-open `(i0:i1, j0:j1)` in analysis order.
    - GeoTIFF expects north-up with origin at the **north-west cell edge**.
    """
    cell = float(grid.cell_size_deg)
    west = float(grid.origin_lon + window.j0 * cell)
    north = float(grid.origin_lat + window.i1 * cell)
    return from_origin(west=west, north=north, xsize=cell, ysize=cell)


def _transform_from_latlon_centers(
    lat_centers: np.ndarray, lon_centers: np.ndarray, *, fallback_cell_size_deg: float
) -> rasterio.Affine:
    """Derive a north-up transform from 1D cell-center coordinates."""
    lat = np.asarray(lat_centers, dtype=float)
    lon = np.asarray(lon_centers, dtype=float)
    if lat.size == 0 or lon.size == 0:
        # Degenerate; caller should avoid writing rasters for empty windows.
        return from_origin(west=0.0, north=0.0, xsize=float(fallback_cell_size_deg), ysize=float(fallback_cell_size_deg))

    # Prefer coord-derived cell sizes (robust if the window is clipped/snapped differently),
    # but fall back to the grid cell size for 1-pixel edges.
    dx = float(np.median(np.diff(lon))) if lon.size > 1 else float(fallback_cell_size_deg)
    dy = float(np.median(np.diff(lat))) if lat.size > 1 else float(fallback_cell_size_deg)

    # Sanity: use absolute sizes; sign is handled by north-up convention.
    dx = abs(dx) if dx else float(fallback_cell_size_deg)
    dy = abs(dy) if dy else float(fallback_cell_size_deg)

    west_edge = float(lon.min() - dx / 2.0)
    north_edge = float(lat.max() + dy / 2.0)
    return from_origin(west=west_edge, north=north_edge, xsize=dx, ysize=dy)


def _effective_transform_for_forecast_window(
    forecast: SpreadForecast,
    grid: GridSpec,
    window: GridWindow,
    *,
    atol: float = 1e-12,
) -> rasterio.Affine:
    """Choose the transform that should be used for both rasters and contours.

    We *prefer* the grid-index-derived transform (stable + snapped), but if the forecast's
    lat/lon coordinates don't match the requested window (even slightly), we must derive a
    transform from the forecast coordinates to avoid georeferencing drift.

    This function exists so that raster saving and contour generation cannot diverge.
    """
    height = int(np.asarray(window.lat).size)
    width = int(np.asarray(window.lon).size)

    grid_transform = _transform_for_grid_window(grid, window)

    try:
        f_lat = np.asarray(forecast.probabilities.coords["lat"].values)
        f_lon = np.asarray(forecast.probabilities.coords["lon"].values)
        if int(f_lat.size) != height or int(f_lon.size) != width:
            raise ValueError("forecast coord lengths do not match window shape")

        w_lat = np.asarray(window.lat)
        w_lon = np.asarray(window.lon)
        if not np.allclose(f_lat, w_lat, rtol=0.0, atol=float(atol)) or not np.allclose(
            f_lon, w_lon, rtol=0.0, atol=float(atol)
        ):
            raise ValueError("forecast coords do not match window coords")
    except Exception:
        return _transform_from_latlon_centers(
            np.asarray(forecast.probabilities.coords["lat"].values),
            np.asarray(forecast.probabilities.coords["lon"].values),
            fallback_cell_size_deg=float(grid.cell_size_deg),
        )

    return grid_transform


def save_forecast_rasters(
    forecast: SpreadForecast,
    grid: GridSpec,
    window: GridWindow,
    run_dir: Path,
    emit_cog: bool = True,
) -> list[dict]:
    """Save probability grids as (COG) GeoTIFFs."""
    run_dir.mkdir(parents=True, exist_ok=True)

    raster_records = []

    # IMPORTANT: forecast.probabilities is computed on the bbox-window, not the full region grid.
    # Therefore, both `transform` and `profile.height/width` must be window-shaped.
    height = int(window.lat.size)
    width = int(window.lon.size)

    transform = _effective_transform_for_forecast_window(forecast, grid, window)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": transform,
        "nodata": -1.0,
    }

    for h in forecast.horizons_hours:
        # Extract data for horizon and flip latitude (analysis S->N to raster N->S)
        # probabilities dims are (time, lat, lon)
        data = _select_probability_slice_by_horizon(forecast, int(h))
        data_flipped = np.flipud(data)

        filename = f"spread_h{h:03d}.tif"
        out_path = run_dir / filename

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data_flipped, 1)

        final_path = out_path
        if emit_cog:
            final_path = convert_to_cog(out_path)
            # Remove the non-COG intermediate file
            out_path.unlink()

        raster_records.append(
            {
                "horizon_hours": h,
                "file_format": "COG" if emit_cog else "GTiff",
                "storage_path": str(final_path.relative_to(REPO_ROOT)),
            }
        )

    return raster_records


def build_contour_records(
    *,
    forecast: SpreadForecast,
    grid: GridSpec,
    window: GridWindow,
    thresholds: Sequence[float],
) -> list[dict]:
    """Generate contour records for the model's actual output horizons.

    Important: this intentionally iterates over `forecast.horizons_hours` (not a user-supplied
    horizon list) to prevent producing contours for horizons that do not exist.
    """
    # IMPORTANT: Contours must use the *same* transform as the saved rasters.
    transform = _effective_transform_for_forecast_window(forecast, grid, window)

    contour_records: list[dict] = []
    for h in forecast.horizons_hours:
        data = _select_probability_slice_by_horizon(forecast, int(h))
        data_flipped = np.flipud(data)
        contours = generate_contours(data_flipped, transform, thresholds)
        for c in contours:
            c["horizon_hours"] = int(h)
            contour_records.append(c)
    return contour_records


def main():
    parser = argparse.ArgumentParser(description="Run spread forecast and persist products.")
    parser.add_argument("--region", type=str, required=True, help="Region name.")
    parser.add_argument(
        "--bbox", type=float, nargs=4, required=True, help="min_lon min_lat max_lon max_lat"
    )
    parser.add_argument("--time", type=str, help="Reference time (ISO). Defaults to now.")
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS_HOURS)
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--no-cog", action="store_true", help="Disable COG conversion.")

    args = parser.parse_args()

    # Import the model runner lazily so importing this module for utility functions
    # (e.g. contour generation / raster persistence) doesn't require optional ML deps.
    from ml.spread.service import SpreadForecastRequest, run_spread_forecast

    # Import persistence layer lazily to avoid requiring DB deps in unit tests
    # that only exercise contours/rasters.
    from ingest.spread_repository import (
        create_spread_forecast_run,
        finalize_spread_forecast_run,
        insert_spread_forecast_contours,
        insert_spread_forecast_rasters,
    )
    from api.fires.service import get_region_grid_spec

    ref_time = datetime.fromisoformat(args.time) if args.time else datetime.now(timezone.utc)
    if ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=timezone.utc)

    bbox = tuple(args.bbox)

    # 1. Create run record
    # For now we assume HeuristicSpreadModelV0 as it's the only one implemented
    run_id = create_spread_forecast_run(
        region_name=args.region,
        model_name="HeuristicSpreadModelV0",
        model_version="v0",
        forecast_reference_time=ref_time,
        bbox=bbox,
    )

    try:
        # 2. Run forecast
        request = SpreadForecastRequest(
            region_name=args.region,
            bbox=bbox,
            forecast_reference_time=ref_time,
            horizons_hours=args.horizons,
        )
        LOGGER.info(f"Running forecast for run_id={run_id}...")
        forecast = run_spread_forecast(request)

        # 3. Persist rasters
        grid = get_region_grid_spec(args.region)
        window = get_grid_window_for_bbox(grid, bbox, clip=True)

        run_dir = REPO_ROOT / "data" / "forecasts" / args.region / f"run_{run_id}"
        raster_records = save_forecast_rasters(
            forecast, grid, window, run_dir, emit_cog=not args.no_cog
        )
        insert_spread_forecast_rasters(run_id, raster_records)

        # 4. Generate and persist contours
        if sorted(set(args.horizons)) != sorted(set(forecast.horizons_hours)):
            LOGGER.warning(
                "Requested horizons %s were adjusted by the model to %s; generating contours for model outputs only.",
                sorted(set(args.horizons)),
                sorted(set(forecast.horizons_hours)),
            )
        contour_records = build_contour_records(
            forecast=forecast, grid=grid, window=window, thresholds=args.thresholds
        )

        insert_spread_forecast_contours(run_id, contour_records)

        # 5. Finalize
        finalize_spread_forecast_run(run_id, status="completed")
        LOGGER.info(f"Forecast run_id={run_id} completed successfully.")

    except Exception as e:
        LOGGER.exception(f"Forecast run_id={run_id} failed.")
        finalize_spread_forecast_run(run_id, status="failed", extra_metadata={"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()

