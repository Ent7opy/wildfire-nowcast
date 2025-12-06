"""Copernicus GLO-30 DEM stitching, reprojection, and persistence."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from dem_stitcher import stitch_dem
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)

from ingest.config import REPO_ROOT, WeatherIngestSettings
from ingest.logging_utils import log_event

# Ensure the API modules (and config.py) are importable when running from ingest/.
sys.path.append(str(REPO_ROOT / "api"))

from api.terrain.repo import TerrainMetadataCreate, insert_terrain_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("dem_preprocess")

# Ensure the API modules (and config.py) are importable when running from ingest/.
sys.path.append(str(REPO_ROOT / "api"))

DEG_PER_METER = 1 / 111_320.0  # Rough conversion at the equator
_weather_defaults = WeatherIngestSettings()


class DemIngestSettings(BaseSettings):
    """Environment-driven configuration for DEM preprocessing."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    data_dir: Path = Field(
        default=REPO_ROOT / "data" / "dem",
        validation_alias="DEM_DATA_DIR",
    )
    region_name: str = Field(default="test_region", validation_alias="DEM_REGION_NAME")
    source: str = Field(default="copernicus_glo30", validation_alias="DEM_SOURCE")
    bbox_min_lon: float = Field(
        default=float(_weather_defaults.bbox_min_lon),
        validation_alias="DEM_BBOX_MIN_LON",
    )
    bbox_min_lat: float = Field(
        default=float(_weather_defaults.bbox_min_lat),
        validation_alias="DEM_BBOX_MIN_LAT",
    )
    bbox_max_lon: float = Field(
        default=float(_weather_defaults.bbox_max_lon),
        validation_alias="DEM_BBOX_MAX_LON",
    )
    bbox_max_lat: float = Field(
        default=float(_weather_defaults.bbox_max_lat),
        validation_alias="DEM_BBOX_MAX_LAT",
    )
    bbox_override: str | None = Field(default=None, validation_alias="DEM_BBOX")
    target_crs_epsg: int = Field(default=4326, validation_alias="DEM_TARGET_CRS")
    target_resolution_m: float = Field(
        default=1000.0,
        validation_alias="DEM_TARGET_RES_M",
    )

    @model_validator(mode="after")
    def _apply_bbox_override(self) -> "DemIngestSettings":
        if self.bbox_override:
            parts = [segment.strip() for segment in str(self.bbox_override).split(",")]
            if len(parts) != 4:
                msg = "DEM_BBOX must be 'min_lon,min_lat,max_lon,max_lat'"
                raise ValueError(msg)
            min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
            self.bbox_min_lon = min_lon
            self.bbox_min_lat = min_lat
            self.bbox_max_lon = max_lon
            self.bbox_max_lat = max_lat
        return self

    @field_validator("data_dir", mode="before")
    @classmethod
    def _coerce_data_dir(cls, value: object) -> Path:
        if value is None:
            return REPO_ROOT / "data" / "dem"
        return Path(value)

    @field_validator("target_crs_epsg", mode="before")
    @classmethod
    def _normalize_epsg(cls, value: object) -> int:
        if isinstance(value, str) and value.upper().startswith("EPSG:"):
            return int(value.split(":", 1)[1])
        return int(value)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(self.bbox_min_lon),
            float(self.bbox_min_lat),
            float(self.bbox_max_lon),
            float(self.bbox_max_lat),
        )


def load_raw_dem(bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, dict]:
    """Fetch Copernicus GLO-30 DEM tiles for the requested bounds."""
    LOGGER.info("Stitching DEM for bounds=%s", bounds)
    data, profile = stitch_dem(
        bounds,
        dem_name="glo_30",
        dst_ellipsoidal_height=False,
        dst_area_or_point="Point",
    )
    return np.asarray(data), profile


def _target_resolution_deg(settings: DemIngestSettings) -> float:
    return float(settings.target_resolution_m) * DEG_PER_METER


def reproject_and_resample_to_project_grid(
    data: np.ndarray, profile: dict, settings: DemIngestSettings
) -> tuple[np.ndarray, dict]:
    """Reproject DEM into target CRS and resolution."""
    src_crs = CRS.from_user_input(profile["crs"])
    dst_crs = CRS.from_epsg(settings.target_crs_epsg)
    src_bounds = array_bounds(profile["height"], profile["width"], profile["transform"])

    res = None
    if dst_crs.is_geographic:
        deg_res = _target_resolution_deg(settings)
        res = (deg_res, deg_res)

    transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        profile["width"],
        profile["height"],
        *src_bounds,
        resolution=res,
    )

    dst_profile = profile.copy()
    dst_profile.update(
        {
            "driver": "GTiff",
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "count": 1,
        }
    )

    destination = np.empty((height, width), dtype=data.dtype)
    reproject(
        source=data,
        destination=destination,
        src_transform=profile["transform"],
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    return destination, dst_profile


def _summarize_dem(data: np.ndarray, settings: DemIngestSettings) -> None:
    """Log DEM coverage and elevation sanity checks."""
    array = np.asarray(data)
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        log_event(
            LOGGER,
            "dem.validation",
            "DEM contains no finite elevation values",
            level="warning",
            region=settings.region_name,
            bbox=settings.bbox,
        )
        return

    coverage = float(finite_mask.mean())
    min_val = float(np.nanmin(array))
    max_val = float(np.nanmax(array))

    log_event(
        LOGGER,
        "dem.stats",
        "DEM coverage and elevation stats",
        region=settings.region_name,
        coverage=coverage,
        min_elevation=min_val,
        max_elevation=max_val,
    )

    if coverage < 0.9:
        log_event(
            LOGGER,
            "dem.validation",
            "DEM coverage below expected threshold",
            level="warning",
            coverage=coverage,
            gap_fraction=float(1.0 - coverage),
            bbox=settings.bbox,
        )


def save_dem_to_geotiff(
    data: np.ndarray, profile: dict, settings: DemIngestSettings
) -> Path:
    """Persist the DEM to GeoTIFF: dem_{region}_epsg{crs}_{res_m}m.tif."""
    out_dir = settings.data_dir / settings.region_name
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"dem_{settings.region_name}_epsg{settings.target_crs_epsg}_"
        f"{int(settings.target_resolution_m)}m.tif"
    )
    out_path = out_dir / filename

    write_profile = profile.copy()
    write_profile.update({"driver": "GTiff", "count": 1, "dtype": data.dtype})

    with rasterio.open(out_path, "w", **write_profile) as dst:
        dst.write(data, 1)

    LOGGER.info("Wrote DEM to %s", out_path)
    return out_path


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DEM stitching + reprojection pipeline.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override bounding box (lon/lat, EPSG:4326).",
    )
    parser.add_argument("--region-name", type=str, default=None, help="Region label override.")
    parser.add_argument("--target-res-m", type=float, default=None, help="Target resolution in meters.")
    parser.add_argument("--target-crs", type=int, default=None, help="Target CRS EPSG code.")
    parser.add_argument(
        "--cog",
        action="store_true",
        help="Also emit a Cloud Optimized GeoTIFF alongside the GeoTIFF.",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(
    settings: DemIngestSettings, args: argparse.Namespace
) -> DemIngestSettings:
    updates: dict[str, object] = {}
    if args.bbox:
        updates["bbox_min_lon"] = args.bbox[0]
        updates["bbox_min_lat"] = args.bbox[1]
        updates["bbox_max_lon"] = args.bbox[2]
        updates["bbox_max_lat"] = args.bbox[3]
    if args.region_name:
        updates["region_name"] = args.region_name
    if args.target_res_m is not None:
        updates["target_resolution_m"] = args.target_res_m
    if args.target_crs is not None:
        updates["target_crs_epsg"] = args.target_crs
    if not updates:
        return settings
    return settings.model_copy(update=updates)


def _bounds_to_epsg4326(profile: dict) -> tuple[float, float, float, float]:
    dst_bounds = array_bounds(profile["height"], profile["width"], profile["transform"])
    return transform_bounds(profile["crs"], "EPSG:4326", *dst_bounds, densify_pts=21)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = _apply_cli_overrides(DemIngestSettings(), args)

    raw_data, raw_profile = load_raw_dem(settings.bbox)
    warped_data, warped_profile = reproject_and_resample_to_project_grid(
        raw_data, raw_profile, settings
    )
    _summarize_dem(warped_data, settings)
    geotiff_path = save_dem_to_geotiff(warped_data, warped_profile, settings)
    final_path = convert_to_cog(geotiff_path) if args.cog else geotiff_path

    bbox_4326 = _bounds_to_epsg4326(warped_profile)
    metadata = TerrainMetadataCreate(
        region_name=settings.region_name,
        dem_source=settings.source,
        crs_epsg=settings.target_crs_epsg,
        resolution_m=float(settings.target_resolution_m),
        bbox=bbox_4326,
        raster_path=str(final_path),
    )
    inserted = insert_terrain_metadata(metadata)

    transform = warped_profile["transform"]
    resolution = (abs(transform.a), abs(transform.e))
    LOGGER.info(
        "DEM ingest complete: path=%s crs=%s resolution=(%.5f, %.5f) deg",
        final_path,
        warped_profile["crs"],
        resolution[0],
        resolution[1],
    )
    print(f"DEM written to: {final_path}")
    print(f"CRS: {warped_profile['crs']}")
    print(f"Resolution (deg): {resolution}")
    print(f"Inserted terrain_metadata id={inserted.id}")


if __name__ == "__main__":
    main()

