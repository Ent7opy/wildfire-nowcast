"""Smoke check for canonical 0.01° EPSG:4326 grid alignment across DEM and weather."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import rasterio
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "api"))

from api.core.grid import GridSpec, grid_bounds  # noqa: E402
from api.terrain.repo import get_latest_dem_metadata_for_region  # noqa: E402
from ingest.dem_preprocess import main as dem_main  # noqa: E402
from ingest.weather_ingest import run_weather_ingest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test grid alignment for DEM and weather.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=[5.123, 35.456, 5.987, 35.999],
        help="Test bbox (lon/lat, unsnapped is fine).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="smoke_grid",
        help="Region name to use for DEM ingest.",
    )
    parser.add_argument(
        "--run-time",
        type=str,
        default=None,
        help="Weather run time (ISO8601 UTC). If omitted, latest 6h cycle is used.",
    )
    parser.add_argument(
        "--include-precip",
        action="store_true",
        help="Include precipitation in weather ingest.",
    )
    parser.add_argument(
        "--weather-model",
        type=str,
        default="gfs_0p25",
        help="Weather model name (for locating output).",
    )
    return parser.parse_args()


def _print_header(title: str) -> None:
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)


def _run_dem(region: str, bbox: Tuple[float, float, float, float]) -> Path:
    dem_args = [
        "--bbox",
        str(bbox[0]),
        str(bbox[1]),
        str(bbox[2]),
        str(bbox[3]),
        "--region-name",
        region,
    ]
    dem_main(dem_args)
    out_path = REPO_ROOT / "data" / "dem" / region / f"dem_{region}_epsg4326_0p01deg.tif"
    return out_path


def _run_weather(bbox: Tuple[float, float, float, float], run_time: Optional[str], include_precip: bool) -> int:
    args = ["--bbox", str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]
    if run_time:
        args.extend(["--run-time", run_time])
    if include_precip:
        args.append("--include-precip")
    return run_weather_ingest(args)


def _latest_weather_nc(model: str) -> Optional[Path]:
    root = REPO_ROOT / "data" / "weather" / model
    if not root.exists():
        return None
    nc_files = sorted(root.rglob("*.nc"), key=lambda p: p.stat().st_mtime, reverse=True)
    return nc_files[0] if nc_files else None


def _describe_dem(path: Path) -> None:
    if not path.exists():
        print(f"DEM not found at {path}")
        return
    with rasterio.open(path) as ds:
        bounds = ds.bounds
        res_x, res_y = ds.res
        print(f"DEM path: {path}")
        print(f"CRS: {ds.crs}")
        print(f"Resolution (deg): {res_x}, {res_y}")
        print(f"Size: width={ds.width}, height={ds.height}")
        print(f"Bounds: {bounds}")


def _describe_weather(path: Path) -> None:
    if not path or not path.exists():
        print(f"Weather NetCDF not found at {path}")
        return
    ds = xr.open_dataset(path)
    lat_step = float(ds["lat"].diff("lat").median())
    lon_step = float(ds["lon"].diff("lon").median())
    attrs = {k: ds.attrs.get(k) for k in ["crs", "cell_size_deg", "origin_lat", "origin_lon", "n_lat", "n_lon"]}
    lat_range = (float(ds["lat"].min()), float(ds["lat"].max()))
    lon_range = (float(ds["lon"].min()), float(ds["lon"].max()))
    print(f"Weather path: {path}")
    print(f"Grid attrs: {attrs}")
    print(f"Lat step ~ {lat_step}, Lon step ~ {lon_step}")
    print(f"Lat range: {lat_range}")
    print(f"Lon range: {lon_range}")
    ds.close()


def _describe_metadata(region: str) -> None:
    try:
        md = get_latest_dem_metadata_for_region(region)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not fetch terrain metadata: {exc}")
        return
    if md is None:
        print("No terrain_metadata row found for region.")
        return
    print(
        "terrain_metadata:",
        {
            "crs_epsg": md.crs_epsg,
            "cell_size_deg": md.cell_size_deg,
            "origin_lat": md.origin_lat,
            "origin_lon": md.origin_lon,
            "grid_n_lat": md.grid_n_lat,
            "grid_n_lon": md.grid_n_lon,
            "bbox": md.bbox,
        },
    )


def main() -> None:
    args = _parse_args()
    bbox = tuple(args.bbox)  # type: ignore[assignment]

    grid = GridSpec.from_bbox(
        lat_min=bbox[1],
        lat_max=bbox[3],
        lon_min=bbox[0],
        lon_max=bbox[2],
    )
    snapped_bounds = grid_bounds(grid)

    _print_header("Planned GridSpec")
    print(
        {
            "origin_lat": grid.origin_lat,
            "origin_lon": grid.origin_lon,
            "cell_size_deg": grid.cell_size_deg,
            "n_lat": grid.n_lat,
            "n_lon": grid.n_lon,
            "bounds": snapped_bounds,
        }
    )

    _print_header("DEM ingest")
    dem_path = _run_dem(args.region, bbox)  # uses snapping internally
    _describe_dem(dem_path)

    _print_header("Weather ingest")
    weather_code = _run_weather(bbox, args.run_time, args.include_precip)
    print(f"Weather ingest exit code: {weather_code}")
    weather_path = _latest_weather_nc(args.weather_model)
    _describe_weather(weather_path if weather_path else Path("N/A"))

    _print_header("terrain_metadata row")
    _describe_metadata(args.region)


if __name__ == "__main__":
    main()

