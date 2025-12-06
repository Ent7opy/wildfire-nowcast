"""Example script to query DEM elevations for sample points."""

from __future__ import annotations

from typing import Iterable, Tuple

from api.terrain.dem_loader import load_dem_for_bbox


def _print_samples(
    da,
    points: Iterable[Tuple[float, float]],
) -> None:
    for lon, lat in points:
        value = da.sel(x=lon, y=lat, method="nearest").item()
        print(f"Elevation at ({lon:.4f}, {lat:.4f}): {value:.2f} m")


def main() -> None:
    region_name = "test_region"
    sample_bbox = (6.0, 35.5, 7.0, 36.0)
    sample_points = [(6.2, 35.7), (6.6, 35.85)]

    da = load_dem_for_bbox(region_name, sample_bbox)
    print(f"Loaded DEM clip for region={region_name}, shape={da.shape}, CRS={da.rio.crs}")
    _print_samples(da, sample_points)


if __name__ == "__main__":
    main()

