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
from ml.spread.service import SpreadForecastRequest, run_spread_forecast
from ingest.spread_repository import (
    create_spread_forecast_run,
    finalize_spread_forecast_run,
    insert_spread_forecast_contours,
    insert_spread_forecast_rasters,
)

# Ensure the API modules are importable
sys.path.append(str(REPO_ROOT))
from api.core.grid import GridSpec
from api.fires.service import get_region_grid_spec

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
    target64 = _as_datetime64_utc_naive(target)
    return np.asarray(da.sel(time=target64, method="nearest").values)


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


def save_forecast_rasters(
    forecast: SpreadForecast, grid: GridSpec, run_dir: Path, emit_cog: bool = True
) -> list[dict]:
    """Save probability grids as (COG) GeoTIFFs."""
    run_dir.mkdir(parents=True, exist_ok=True)

    raster_records = []

    # GeoTIFF transform: north-up (origin is top-left)
    # Analysis grid origin is bottom-left (southern/western edges).
    north = grid.origin_lat + grid.n_lat * grid.cell_size_deg
    transform = from_origin(
        west=grid.origin_lon,
        north=north,
        xsize=grid.cell_size_deg,
        ysize=grid.cell_size_deg,
    )

    profile = {
        "driver": "GTiff",
        "height": grid.n_lat,
        "width": grid.n_lon,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
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

        run_dir = REPO_ROOT / "data" / "forecasts" / args.region / f"run_{run_id}"
        raster_records = save_forecast_rasters(forecast, grid, run_dir, emit_cog=not args.no_cog)
        insert_spread_forecast_rasters(run_id, raster_records)

        # 4. Generate and persist contours
        # We need the transform for contours too
        north = grid.origin_lat + grid.n_lat * grid.cell_size_deg
        transform = from_origin(
            west=grid.origin_lon,
            north=north,
            xsize=grid.cell_size_deg,
            ysize=grid.cell_size_deg,
        )

        contour_records = []
        for h in args.horizons:
            data = _select_probability_slice_by_horizon(forecast, int(h))
            data_flipped = np.flipud(data)
            contours = generate_contours(data_flipped, transform, args.thresholds)
            for c in contours:
                c["horizon_hours"] = h
                contour_records.append(c)

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

