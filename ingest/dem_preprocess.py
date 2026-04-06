"""Copernicus GLO-30 DEM stitching, reprojection, and persistence."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from dem_stitcher import stitch_dem
from scipy.ndimage import gaussian_filter1d
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from ingest.config import REPO_ROOT, WeatherIngestSettings
from ingest.logging_utils import log_event

# Ensure the API modules (and config.py) are importable when running from ingest/.
sys.path.append(str(REPO_ROOT))

from api.core.grid import DEFAULT_CELL_SIZE_DEG, DEFAULT_CRS, GridSpec, grid_bounds
from api.terrain.repo import TerrainMetadataCreate, insert_terrain_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("dem_preprocess")

CANONICAL_CRS = DEFAULT_CRS
CANONICAL_CELL_SIZE_DEG = DEFAULT_CELL_SIZE_DEG
METERS_PER_DEG_AT_EQUATOR = 111_320.0  # Rough conversion at the equator
CANONICAL_RESOLUTION_M = CANONICAL_CELL_SIZE_DEG * METERS_PER_DEG_AT_EQUATOR
CANONICAL_EPSG = 4326
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
        default=CANONICAL_RESOLUTION_M,
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

    @property
    def grid_spec(self) -> GridSpec:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return GridSpec.from_bbox(
            lat_min=min_lat,
            lat_max=max_lat,
            lon_min=min_lon,
            lon_max=max_lon,
            cell_size_deg=CANONICAL_CELL_SIZE_DEG,
            crs=CANONICAL_CRS,
        )


# ---------------------------------------------------------------------------
# Seam blending: Copernicus GLO-30 tiles are 1-degree; boundaries fall on
# integer degree lines.  After stitching, small elevation jumps at these seams
# propagate as spurious slope/aspect features.  We detect boundary pixel rows/
# cols from the geotransform and apply a narrow 1-D Gaussian blur across the
# seam to smooth discontinuities without degrading the interior.
# ---------------------------------------------------------------------------

# Default half-width (in pixels) of the corridor blended on each side of a seam.
_SEAM_BLEND_HALF_WIDTH: int = 3
# Gaussian sigma (pixels) for seam smoothing.
_SEAM_BLEND_SIGMA: float = 1.0
# Slope discontinuity (degrees) threshold for QA warnings at tile seams.
_SEAM_SLOPE_WARN_THRESHOLD_DEG: float = 5.0


def _integer_degree_pixel_indices(
    origin: float,
    pixel_size: float,
    n_pixels: int,
) -> list[int]:
    """Return pixel indices where integer-degree tile boundaries fall.

    Parameters
    ----------
    origin : float
        Coordinate (lon or lat) of the west or north edge of pixel 0.
    pixel_size : float
        Signed pixel size in degrees (negative for north-up rows).
    n_pixels : int
        Number of pixels along this axis.

    Returns
    -------
    List of pixel indices that straddle an integer-degree line.
    Only interior boundaries are returned (not the edges of the raster).
    """
    abs_size = abs(pixel_size)
    if abs_size == 0:
        return []

    # Collect candidates grouped by the integer-degree boundary they belong to.
    # At native GLO-30 resolution (~0.000278 deg/pixel) multiple pixels may fall
    # within the 0.6*cell_size threshold for the same boundary; we keep only the
    # single closest pixel per boundary to avoid overlapping Gaussian applications.
    best_per_boundary: dict[int, tuple[int, float]] = {}  # nearest_int -> (idx, dist)
    for i in range(n_pixels):
        coord = origin + (i + 0.5) * pixel_size  # center of pixel i
        nearest_int = round(coord)
        dist = abs(coord - nearest_int)
        if dist < abs_size * 0.6:
            # Exclude raster-edge boundaries (first/last few pixels)
            if _SEAM_BLEND_HALF_WIDTH < i < n_pixels - _SEAM_BLEND_HALF_WIDTH:
                prev = best_per_boundary.get(nearest_int)
                if prev is None or dist < prev[1]:
                    best_per_boundary[nearest_int] = (i, dist)

    return sorted(idx for idx, _ in best_per_boundary.values())


def blend_tile_seams(
    data: np.ndarray,
    profile: dict,
    *,
    half_width: int = _SEAM_BLEND_HALF_WIDTH,
    sigma: float = _SEAM_BLEND_SIGMA,
) -> np.ndarray:
    """Apply Gaussian smoothing along tile-boundary seam lines.

    Only pixels within *half_width* of an integer-degree boundary are touched.
    The rest of the DEM is left unchanged.

    Parameters
    ----------
    data : np.ndarray
        2-D elevation array (height x width), float or compatible.
    profile : dict
        Rasterio-style profile with ``transform``, ``width``, ``height``.
    half_width : int
        Number of pixels on each side of the seam to include in blending.
    sigma : float
        Standard deviation (in pixels) for the 1-D Gaussian kernel.

    Returns
    -------
    np.ndarray with the same shape and dtype as *data*.
    """
    transform = profile["transform"]
    height, width = data.shape[-2], data.shape[-1]

    # Work on a float copy so we never mutate the caller's array.
    blended = np.array(data, dtype=np.float64)

    # --- Vertical seam lines (boundaries along longitude = integer degrees) ---
    col_indices = _integer_degree_pixel_indices(
        origin=float(transform.c),  # west edge of col 0
        pixel_size=float(transform.a),  # positive (east)
        n_pixels=width,
    )
    for ci in col_indices:
        lo = max(ci - half_width, 0)
        hi = min(ci + half_width + 1, width)
        strip = blended[:, lo:hi].copy()
        # 1-D Gaussian along the column (horizontal) axis within the strip.
        finite_mask = np.isfinite(strip)
        if finite_mask.any():
            fill_val = np.nanmean(strip) if finite_mask.any() else 0.0
            smoothed = gaussian_filter1d(
                np.where(finite_mask, strip, fill_val), sigma=sigma, axis=1
            )
            blended[:, lo:hi] = np.where(finite_mask, smoothed, strip)

    # --- Horizontal seam lines (boundaries along latitude = integer degrees) ---
    row_indices = _integer_degree_pixel_indices(
        origin=float(transform.f),  # north edge of row 0
        pixel_size=float(transform.e),  # negative (south)
        n_pixels=height,
    )
    for ri in row_indices:
        lo = max(ri - half_width, 0)
        hi = min(ri + half_width + 1, height)
        strip = blended[lo:hi, :].copy()
        finite_mask = np.isfinite(strip)
        if finite_mask.any():
            fill_val = np.nanmean(strip) if finite_mask.any() else 0.0
            smoothed = gaussian_filter1d(
                np.where(finite_mask, strip, fill_val), sigma=sigma, axis=0
            )
            blended[lo:hi, :] = np.where(finite_mask, smoothed, strip)

    n_seams = len(col_indices) + len(row_indices)
    if n_seams:
        LOGGER.info(
            "Blended %d tile seam(s) (half_width=%d, sigma=%.1f): "
            "%d vertical, %d horizontal",
            n_seams, half_width, sigma,
            len(col_indices), len(row_indices),
        )
    else:
        LOGGER.debug("No interior tile seams detected; blending skipped.")

    return blended.astype(data.dtype)


def check_seam_quality(
    data: np.ndarray,
    profile: dict,
    *,
    slope_threshold_deg: float = _SEAM_SLOPE_WARN_THRESHOLD_DEG,
) -> list[dict]:
    """QA check: flag tile seams where the cross-seam slope exceeds a threshold.

    Computes the pixel-to-pixel elevation difference across each seam line and
    converts to approximate slope degrees.  Returns a list of warning dicts
    (empty means all seams are clean).
    """
    transform = profile["transform"]
    height, width = data.shape[-2], data.shape[-1]
    # NOTE: This equatorial approximation overestimates cell width at higher
    # latitudes (by ~cos(lat)).  Acceptable for a QA threshold check but not
    # for precise slope computation — see api/terrain/features_math.py for the
    # lat-corrected version.
    cell_m = abs(float(transform.a)) * METERS_PER_DEG_AT_EQUATOR

    warnings: list[dict] = []

    col_indices = _integer_degree_pixel_indices(
        float(transform.c), float(transform.a), width,
    )
    for ci in col_indices:
        if ci < 1 or ci >= width - 1:
            continue
        # Check the max pixel-to-pixel diff in a small corridor around the
        # boundary pixel (the actual tile edge may be at ci or ci+1).
        lo = max(ci - 1, 0)
        hi = min(ci + 2, width)
        corridor = data[:, lo:hi].astype(float)
        diff = np.abs(np.diff(corridor, axis=1))
        slope_deg = np.rad2deg(np.arctan2(diff, cell_m))
        max_slope = float(np.nanmax(slope_deg))
        if max_slope > slope_threshold_deg:
            coord = float(transform.c) + ci * float(transform.a)
            warn = {
                "axis": "longitude",
                "boundary_deg": round(coord, 6),
                "pixel_index": ci,
                "max_cross_seam_slope_deg": round(max_slope, 2),
            }
            warnings.append(warn)
            log_event(
                LOGGER,
                "dem.seam_quality",
                "Cross-seam slope exceeds threshold",
                level="warning",
                **warn,
                threshold_deg=slope_threshold_deg,
            )

    row_indices = _integer_degree_pixel_indices(
        float(transform.f), float(transform.e), height,
    )
    for ri in row_indices:
        if ri < 1 or ri >= height - 1:
            continue
        lo = max(ri - 1, 0)
        hi = min(ri + 2, height)
        corridor = data[lo:hi, :].astype(float)
        diff = np.abs(np.diff(corridor, axis=0))
        slope_deg = np.rad2deg(np.arctan2(diff, cell_m))
        max_slope = float(np.nanmax(slope_deg))
        if max_slope > slope_threshold_deg:
            coord = float(transform.f) + ri * float(transform.e)
            warn = {
                "axis": "latitude",
                "boundary_deg": round(coord, 6),
                "pixel_index": ri,
                "max_cross_seam_slope_deg": round(max_slope, 2),
            }
            warnings.append(warn)
            log_event(
                LOGGER,
                "dem.seam_quality",
                "Cross-seam slope exceeds threshold",
                level="warning",
                **warn,
                threshold_deg=slope_threshold_deg,
            )

    if not warnings:
        LOGGER.info("Seam QA passed: no cross-seam slope discontinuities above %.1f°", slope_threshold_deg)

    return warnings


def load_raw_dem(bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, dict]:
    """Fetch Copernicus GLO-30 DEM tiles for the requested bounds.

    After stitching, applies seam blending along tile boundaries and runs a
    QA check for residual cross-seam discontinuities.
    """
    LOGGER.info("Stitching DEM for bounds=%s", bounds)
    data, profile = stitch_dem(
        bounds,
        dem_name="glo_30",
        dst_ellipsoidal_height=False,
        dst_area_or_point="Point",
    )
    raw = np.asarray(data)

    # Blend tile-boundary seams to suppress elevation jumps.
    blended = blend_tile_seams(raw, profile)

    # QA: warn if any seam still has an abnormal discontinuity after blending.
    check_seam_quality(blended, profile)

    return blended, profile


def _grid_transform(grid: GridSpec):
    north = grid.origin_lat + grid.n_lat * grid.cell_size_deg
    return from_origin(
        west=grid.origin_lon,
        north=north,
        xsize=grid.cell_size_deg,
        ysize=grid.cell_size_deg,
    )


def resample_to_grid(
    data: np.ndarray, profile: dict, grid: GridSpec
) -> tuple[np.ndarray, dict]:
    """Reproject and resample DEM into the canonical analysis grid."""
    src_crs = CRS.from_user_input(profile["crs"])
    dst_crs = CRS.from_string(grid.crs)

    transform = _grid_transform(grid)
    dst_profile = profile.copy()
    dst_profile.update(
        {
            "driver": "GTiff",
            "crs": dst_crs,
            "transform": transform,
            "width": grid.n_lon,
            "height": grid.n_lat,
            "count": 1,
        }
    )

    destination = np.empty((grid.n_lat, grid.n_lon), dtype=data.dtype)
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
    """Persist the DEM to GeoTIFF: dem_{region}_epsg{crs}_0p01deg.tif."""
    out_dir = settings.data_dir / settings.region_name
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"dem_{settings.region_name}_epsg{CANONICAL_EPSG}_0p01deg.tif"
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
    if not updates:
        return settings
    return settings.model_copy(update=updates)


def ingest_terrain_for_bbox(
    bbox: tuple[float, float, float, float],
    output_dir: Path | str,
    region_name: str | None = None,
    emit_cog: bool = False,
) -> int:
    """Download DEM, compute terrain features (slope, aspect), persist to DB.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
        output_dir: Directory for output rasters (e.g., data/terrain).
        region_name: Optional region label (defaults to bbox-based name).
        emit_cog: If True, convert outputs to Cloud Optimized GeoTIFFs.

    Returns:
        terrain_features_metadata.id from the database.

    Raises:
        ValueError: If bbox is invalid or DEM download fails.
        FileNotFoundError: If intermediate files are missing.
    """
    from api.terrain.features_math import compute_slope_aspect
    from api.terrain.features_repo import (
        TerrainFeaturesMetadataCreate,
        insert_terrain_features_metadata,
    )

    output_dir = Path(output_dir)
    min_lon, min_lat, max_lon, max_lat = bbox

    if region_name is None:
        region_name = f"bbox_{min_lon:.2f}_{min_lat:.2f}_{max_lon:.2f}_{max_lat:.2f}"

    grid = GridSpec.from_bbox(
        lat_min=min_lat,
        lat_max=max_lat,
        lon_min=min_lon,
        lon_max=max_lon,
        cell_size_deg=CANONICAL_CELL_SIZE_DEG,
        crs=CANONICAL_CRS,
    )

    raw_data, raw_profile = load_raw_dem(grid_bounds(grid))
    warped_data, warped_profile = resample_to_grid(raw_data, raw_profile, grid)

    settings_for_logging = DemIngestSettings(
        region_name=region_name,
        bbox_min_lon=min_lon,
        bbox_min_lat=min_lat,
        bbox_max_lon=max_lon,
        bbox_max_lat=max_lat,
        data_dir=output_dir,
    )
    _summarize_dem(warped_data, settings_for_logging)

    out_dir = output_dir / region_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dem_filename = f"dem_{region_name}_epsg{CANONICAL_EPSG}_0p01deg.tif"
    dem_path = out_dir / dem_filename

    write_profile = warped_profile.copy()
    write_profile.update({"driver": "GTiff", "count": 1, "dtype": warped_data.dtype})
    with rasterio.open(dem_path, "w", **write_profile) as dst:
        dst.write(warped_data, 1)
    LOGGER.info("Wrote DEM to %s", dem_path)

    final_dem_path = convert_to_cog(dem_path) if emit_cog else dem_path

    bbox_4326 = grid_bounds(grid)
    resolution_m = grid.cell_size_deg * METERS_PER_DEG_AT_EQUATOR
    dem_metadata_obj = TerrainMetadataCreate(
        region_name=region_name,
        dem_source="copernicus_glo30",
        crs_epsg=CANONICAL_EPSG,
        resolution_m=float(resolution_m),
        bbox=bbox_4326,
        raster_path=str(final_dem_path),
        cell_size_deg=grid.cell_size_deg,
        origin_lat=grid.origin_lat,
        origin_lon=grid.origin_lon,
        grid_n_lat=grid.n_lat,
        grid_n_lon=grid.n_lon,
    )
    dem_metadata = insert_terrain_metadata(dem_metadata_obj)
    LOGGER.info("Inserted terrain_metadata id=%s", dem_metadata.id)

    with rasterio.open(final_dem_path) as src:
        z_ma = src.read(1, masked=True)
        z = np.asarray(z_ma.filled(np.nan), dtype=float)
        cell_deg = float(abs(src.transform.a))
        lat_centers = src.transform.f + (np.arange(src.height) + 0.5) * src.transform.e

        slope_deg, aspect_deg = compute_slope_aspect(
            z, cell_size_deg=cell_deg, lat_centers_deg=lat_centers
        )

        nodata_value = -9999.0
        slope_out = np.where(np.isfinite(slope_deg), slope_deg, nodata_value).astype(np.float32)
        aspect_out = np.where(np.isfinite(aspect_deg), aspect_deg, nodata_value).astype(np.float32)

        slope_path = out_dir / f"slope_{region_name}_epsg{CANONICAL_EPSG}_0p01deg.tif"
        aspect_path = out_dir / f"aspect_{region_name}_epsg{CANONICAL_EPSG}_0p01deg.tif"

        slope_profile = src.profile.copy()
        slope_profile.update({"driver": "GTiff", "count": 1, "dtype": "float32", "nodata": nodata_value})
        with rasterio.open(slope_path, "w", **slope_profile) as dst:
            dst.write(slope_out, 1)
        with rasterio.open(aspect_path, "w", **slope_profile) as dst:
            dst.write(aspect_out, 1)

        final_slope_path = convert_to_cog(slope_path) if emit_cog else slope_path
        final_aspect_path = convert_to_cog(aspect_path) if emit_cog else aspect_path

        mask = np.isfinite(slope_deg)
        coverage = float(mask.mean()) if mask.any() else 0.0
        s_min = float(np.min(slope_deg[mask])) if mask.any() else None
        s_max = float(np.max(slope_deg[mask])) if mask.any() else None
        a_min = float(np.min(aspect_deg[mask])) if mask.any() else None
        a_max = float(np.max(aspect_deg[mask])) if mask.any() else None

        log_event(
            LOGGER,
            "terrain_features.computed",
            "Computed slope/aspect for bbox",
            region=region_name,
            slope={"min": s_min, "max": s_max, "units": "degrees"},
            aspect={"min": a_min, "max": a_max, "units": "degrees"},
            coverage_fraction=coverage,
        )

        origin_lon = float(src.transform.c)
        north_edge = float(src.transform.f)
        origin_lat = float(north_edge - src.height * cell_deg)

    features_metadata_obj = TerrainFeaturesMetadataCreate(
        region_name=region_name,
        source_dem_metadata_id=int(dem_metadata.id),
        slope_path=str(final_slope_path),
        aspect_path=str(final_aspect_path),
        crs_epsg=CANONICAL_EPSG,
        cell_size_deg=float(cell_deg),
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        grid_n_lat=int(grid.n_lat),
        grid_n_lon=int(grid.n_lon),
        bbox=bbox_4326,
        slope_min=s_min,
        slope_max=s_max,
        aspect_min=a_min,
        aspect_max=a_max,
        coverage_fraction=coverage,
        nodata_value=nodata_value,
    )
    features_metadata = insert_terrain_features_metadata(features_metadata_obj)

    LOGGER.info(
        "Terrain ingest complete: features_id=%s dem_path=%s slope_path=%s aspect_path=%s",
        features_metadata.id,
        final_dem_path,
        final_slope_path,
        final_aspect_path,
    )
    return int(features_metadata.id)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = _apply_cli_overrides(DemIngestSettings(), args)

    features_id = ingest_terrain_for_bbox(
        bbox=settings.bbox,
        output_dir=settings.data_dir,
        region_name=settings.region_name,
        emit_cog=bool(args.cog),
    )

    print("Terrain features ingested successfully.")
    print(f"terrain_features_metadata id={features_id}")


if __name__ == "__main__":
    main()

