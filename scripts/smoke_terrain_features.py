"""Smoke check for deterministic slope/aspect generation aligned to the DEM grid."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "api"))

from api.terrain.repo import get_latest_dem_metadata_for_region  # noqa: E402
from ingest.dem_preprocess import main as dem_main  # noqa: E402
from ingest.terrain_features import main as terrain_features_main  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test slope/aspect output alignment + determinism."
    )
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
        help="Region name to use for DEM + terrain features ingest.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute even if outputs exist.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_alignment(dem_path: Path, slope_path: Path, aspect_path: Path) -> None:
    with rasterio.open(dem_path) as dem, rasterio.open(slope_path) as slope, rasterio.open(
        aspect_path
    ) as aspect:
        for name, ds in [("slope", slope), ("aspect", aspect)]:
            assert ds.crs == dem.crs, f"{name} CRS mismatch: {ds.crs} vs {dem.crs}"
            assert ds.transform == dem.transform, f"{name} transform mismatch"
            assert ds.width == dem.width and ds.height == dem.height, f"{name} shape mismatch"
            assert ds.nodata is not None, f"{name} missing nodata"


def _validate_value_ranges(slope_path: Path, aspect_path: Path) -> None:
    with rasterio.open(slope_path) as slope_ds, rasterio.open(aspect_path) as aspect_ds:
        slope = slope_ds.read(1, masked=True).astype(float)
        aspect = aspect_ds.read(1, masked=True).astype(float)

        slope_vals = slope.compressed()
        assert slope_vals.size > 0, "Slope has no valid pixels"
        assert float(np.nanmin(slope_vals)) >= -1e-6
        assert float(np.nanmax(slope_vals)) <= 90.0 + 1e-6

        aspect_vals = aspect.compressed()
        # Aspect can be fully masked in pathological cases, but usually should have data.
        assert aspect_vals.size > 0, "Aspect has no valid pixels"
        assert float(np.nanmin(aspect_vals)) >= 0.0 - 1e-6
        assert float(np.nanmax(aspect_vals)) < 360.0 + 1e-6


def _run_dem(region: str, bbox: Tuple[float, float, float, float]) -> None:
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


def _run_features(region: str, *, recompute: bool) -> None:
    args = ["--region-name", region]
    if recompute:
        args.append("--recompute")
    terrain_features_main(args)


def main() -> None:
    args = _parse_args()
    bbox = tuple(args.bbox)  # type: ignore[assignment]

    print("\n======== DEM ingest ========")
    _run_dem(args.region, bbox)
    dem_md = get_latest_dem_metadata_for_region(args.region)
    if dem_md is None:
        raise RuntimeError("Expected terrain_metadata after DEM ingest, but none found.")
    dem_path = Path(dem_md.raster_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM raster not found at {dem_path}")

    out_dir = REPO_ROOT / "data" / "terrain" / args.region
    slope_path = out_dir / f"slope_{args.region}_epsg4326_0p01deg.tif"
    aspect_path = out_dir / f"aspect_{args.region}_epsg4326_0p01deg.tif"

    print("\n======== Terrain features ingest (run 1) ========")
    _run_features(args.region, recompute=True if args.recompute else False)
    if not slope_path.exists() or not aspect_path.exists():
        raise FileNotFoundError("Expected slope/aspect outputs, but files were not found.")

    _validate_alignment(dem_path, slope_path, aspect_path)
    _validate_value_ranges(slope_path, aspect_path)

    h1_slope = _sha256(slope_path)
    h1_aspect = _sha256(aspect_path)
    print(f"Hashes run1 slope={h1_slope[:12]} aspect={h1_aspect[:12]}")

    print("\n======== Terrain features ingest (run 2, deterministic check) ========")
    _run_features(args.region, recompute=True)
    h2_slope = _sha256(slope_path)
    h2_aspect = _sha256(aspect_path)
    print(f"Hashes run2 slope={h2_slope[:12]} aspect={h2_aspect[:12]}")

    if h1_slope != h2_slope or h1_aspect != h2_aspect:
        raise AssertionError("Outputs are not deterministic (file hashes changed between runs).")

    print("\nOK: slope/aspect aligned and deterministic.")
    print(f"DEM:    {dem_path}")
    print(f"Slope:  {slope_path}")
    print(f"Aspect: {aspect_path}")


if __name__ == "__main__":
    main()

