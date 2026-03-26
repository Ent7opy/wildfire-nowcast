"""Compute slope & aspect rasters aligned to the canonical DEM grid."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ingest.config import REPO_ROOT
from ingest.logging_utils import log_event

# Ensure the API modules (and config.py) are importable when running from ingest/.
sys.path.append(str(REPO_ROOT))

from api.core.grid import DEFAULT_CELL_SIZE_DEG, DEFAULT_CRS  # noqa: E402
from api.terrain.dem_loader import grid_spec_from_metadata  # noqa: E402
from api.terrain.features_math import compute_slope_aspect  # noqa: E402
from api.terrain.features_repo import (  # noqa: E402
    TerrainFeaturesMetadataCreate,
    insert_terrain_features_metadata,
)
from api.terrain.repo import find_fallback_dem, get_latest_dem_metadata_for_region  # noqa: E402
from api.terrain.validate import validate_raster_matches_grid, validate_terrain_stack  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("terrain_features")

CANONICAL_EPSG = 4326
CANONICAL_CRS = DEFAULT_CRS
CANONICAL_CELL_SIZE_DEG = DEFAULT_CELL_SIZE_DEG


class TerrainFeaturesSettings(BaseSettings):
    """Environment-driven configuration for slope/aspect derivation."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    region_name: str = Field(
        default="test_region",
        validation_alias="TERRAIN_FEATURES_REGION_NAME",
    )
    data_dir: Path = Field(
        default=REPO_ROOT / "data" / "terrain",
        validation_alias="TERRAIN_FEATURES_DATA_DIR",
    )
    output_dir: Path | None = Field(
        default=None,
        validation_alias="TERRAIN_FEATURES_OUTPUT_DIR",
    )
    recompute: bool = Field(default=False, validation_alias="TERRAIN_FEATURES_RECOMPUTE")
    nodata_value: float = Field(default=-9999.0, validation_alias="TERRAIN_FEATURES_NODATA")

    @field_validator("data_dir", mode="before")
    @classmethod
    def _coerce_data_dir(cls, value: object) -> Path:
        if value is None:
            return REPO_ROOT / "data" / "terrain"
        return Path(value)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_output_dir(cls, value: object) -> Path | None:
        if value in (None, ""):
            return None
        return Path(value)

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir or self.data_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute slope/aspect rasters from the latest preprocessed DEM."
    )
    parser.add_argument("--region-name", type=str, default=None, help="Region label override.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute even if slope/aspect files already exist.",
    )
    parser.add_argument(
        "--cog",
        action="store_true",
        help="Also emit Cloud Optimized GeoTIFFs alongside the GeoTIFFs.",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(
    settings: TerrainFeaturesSettings, args: argparse.Namespace
) -> tuple[TerrainFeaturesSettings, bool]:
    updates: dict[str, object] = {}
    if args.region_name:
        updates["region_name"] = args.region_name
    if args.recompute:
        updates["recompute"] = True
    next_settings = settings.model_copy(update=updates) if updates else settings
    return next_settings, bool(args.cog)


def _validate_dem_alignment(
    *,
    crs_epsg: int | None,
    transform: rasterio.Affine,
    width: int,
    height: int,
    expected_n_lon: int | None,
    expected_n_lat: int | None,
) -> float:
    """Validate DEM is on canonical EPSG:4326 0.01° north-up grid.

    Returns cell size in degrees.
    """
    if crs_epsg != CANONICAL_EPSG:
        raise ValueError(f"Expected DEM CRS EPSG:{CANONICAL_EPSG}, got EPSG:{crs_epsg}")
    # no rotation/shear
    if not (abs(transform.b) < 1e-12 and abs(transform.d) < 1e-12):
        raise ValueError(f"Expected north-up grid (no rotation), got transform={transform}")
    cell_x = float(abs(transform.a))
    cell_y = float(abs(transform.e))
    if not np.isclose(cell_x, cell_y, rtol=0, atol=1e-12):
        raise ValueError(f"Expected square pixels, got (x,y)=({cell_x},{cell_y}) deg")
    if not np.isclose(cell_x, CANONICAL_CELL_SIZE_DEG, rtol=0, atol=1e-9):
        raise ValueError(
            f"Expected cell size {CANONICAL_CELL_SIZE_DEG}°, got {cell_x}° (transform={transform})"
        )
    if expected_n_lon is not None and width != int(expected_n_lon):
        raise ValueError(f"DEM width mismatch: got {width}, expected {expected_n_lon}")
    if expected_n_lat is not None and height != int(expected_n_lat):
        raise ValueError(f"DEM height mismatch: got {height}, expected {expected_n_lat}")
    return cell_x


def _lat_centers_from_transform(transform: rasterio.Affine, height: int) -> np.ndarray:
    """Row-center latitudes for a north-up raster (row 0 is north)."""
    # For north-up rasters, transform.e is negative; this yields decreasing lats by row.
    return transform.f + (np.arange(height) + 0.5) * transform.e


def _compute_stats(arr: np.ndarray, nodata_value: float) -> tuple[float | None, float | None, float | None, float]:
    mask = np.isfinite(arr) & (arr != nodata_value)
    if not bool(mask.any()):
        return None, None, None, 0.0
    vals = arr[mask]
    return float(np.min(vals)), float(np.max(vals)), float(np.mean(vals)), float(mask.mean())


def _write_aligned_geotiff(
    *,
    out_path: Path,
    data: np.ndarray,
    profile: dict,
    nodata_value: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_profile = profile.copy()
    write_profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "nodata": float(nodata_value),
        }
    )
    with rasterio.open(out_path, "w", **write_profile) as dst:
        dst.write(data.astype(np.float32), 1)


def _convert_to_cog(in_path: Path) -> Path:
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
    return out_path



def _write_flat_terrain_stub(
    *,
    settings: TerrainFeaturesSettings,
    dem_metadata: object,
    slope_path: Path,
    aspect_path: Path,
    reason: str,
) -> None:
    """Write zero slope/aspect GeoTIFFs and register a fallback terrain_features_metadata row.

    The row is marked ``terrain_fallback_used=True`` so downstream consumers can detect it.
    """
    grid_spec = grid_spec_from_metadata(dem_metadata)
    n_lat = int(grid_spec.n_lat)
    n_lon = int(grid_spec.n_lon)

    # Build a north-up transform matching the DEM grid exactly.
    from rasterio.transform import from_bounds

    min_lon, min_lat, max_lon, max_lat = dem_metadata.bbox
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, n_lon, n_lat)

    flat_profile = {
        "driver": "GTiff",
        "crs": f"EPSG:{CANONICAL_EPSG}",
        "transform": transform,
        "width": n_lon,
        "height": n_lat,
        "count": 1,
        "dtype": "float32",
        "nodata": float(settings.nodata_value),
    }
    flat_data = np.zeros((n_lat, n_lon), dtype=np.float32)

    _write_aligned_geotiff(
        out_path=slope_path,
        data=flat_data,
        profile=flat_profile,
        nodata_value=settings.nodata_value,
    )
    _write_aligned_geotiff(
        out_path=aspect_path,
        data=flat_data,
        profile=flat_profile,
        nodata_value=settings.nodata_value,
    )

    log_event(
        LOGGER,
        "terrain_features.flat_stub",
        "Wrote flat-terrain stub (slope=0, aspect=0)",
        region=settings.region_name,
        reason=reason,
        bbox=list(dem_metadata.bbox),
        slope_path=str(slope_path),
        aspect_path=str(aspect_path),
    )

    inserted = insert_terrain_features_metadata(
        TerrainFeaturesMetadataCreate(
            region_name=settings.region_name,
            source_dem_metadata_id=int(dem_metadata.id),
            slope_path=str(slope_path),
            aspect_path=str(aspect_path),
            crs_epsg=CANONICAL_EPSG,
            cell_size_deg=float(grid_spec.cell_size_deg),
            origin_lat=float(grid_spec.origin_lat),
            origin_lon=float(grid_spec.origin_lon),
            grid_n_lat=n_lat,
            grid_n_lon=n_lon,
            bbox=dem_metadata.bbox,
            slope_min=0.0,
            slope_max=0.0,
            aspect_min=0.0,
            aspect_max=0.0,
            coverage_fraction=0.0,
            nodata_value=float(settings.nodata_value),
            terrain_fallback_used=True,
        )
    )
    LOGGER.warning(
        "Flat-terrain stub registered as terrain_features_metadata id=%s "
        "(terrain_fallback_used=True). Reason: %s. "
        "Mitigation: run DEM ingest for region '%s' to replace stub with real data.",
        inserted.id,
        reason,
        settings.region_name,
    )
    print(f"Slope (flat stub) written to: {slope_path}")
    print(f"Aspect (flat stub) written to: {aspect_path}")
    print(f"Inserted terrain_features_metadata id={inserted.id} [terrain_fallback_used=True]")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    settings, emit_cog = _apply_cli_overrides(TerrainFeaturesSettings(), args)

    dem_metadata = get_latest_dem_metadata_for_region(settings.region_name)
    if dem_metadata is None:
        LOGGER.warning(
            "No DEM metadata found for region '%s'. "
            "Cannot compute terrain features or write a stub (no source_dem_metadata_id). "
            "Mitigation: run DEM ingest for this region first.",
            settings.region_name,
        )
        return

    dem_path = Path(dem_metadata.raster_path)
    out_dir = settings.resolved_output_dir / settings.region_name
    slope_path = out_dir / f"slope_{settings.region_name}_epsg{CANONICAL_EPSG}_0p01deg.tif"
    aspect_path = out_dir / f"aspect_{settings.region_name}_epsg{CANONICAL_EPSG}_0p01deg.tif"

    if not dem_path.exists():
        # Resolution-ladder fallback: try any other registered DEM whose file exists.
        fallback_md = find_fallback_dem([settings.region_name], skip_path=dem_path)
        if fallback_md is not None:
            LOGGER.warning(
                "Primary DEM raster missing at %s for region '%s' (bbox=%s). "
                "Falling back to lower-resolution DEM: %s (%.0f m). "
                "Mitigation: re-ingest the missing DEM tile.",
                dem_path,
                settings.region_name,
                list(dem_metadata.bbox),
                fallback_md.raster_path,
                fallback_md.resolution_m,
            )
            dem_metadata = fallback_md
            dem_path = Path(fallback_md.raster_path)
        else:
            # No usable DEM on disk — write flat-terrain stub and exit without crash.
            LOGGER.warning(
                "DEM raster missing at %s for region '%s' (bbox=%s) "
                "and no lower-resolution fallback found. "
                "Writing flat-terrain stub (slope=0, aspect=0, terrain_fallback_used=True). "
                "Mitigation: run DEM ingest for region '%s'.",
                dem_path,
                settings.region_name,
                list(dem_metadata.bbox),
                settings.region_name,
            )
            if not settings.recompute and slope_path.exists() and aspect_path.exists():
                LOGGER.info(
                    "Flat stub already exists; skipping (use --recompute to override)."
                )
                return
            _write_flat_terrain_stub(
                settings=settings,
                dem_metadata=dem_metadata,
                slope_path=slope_path,
                aspect_path=aspect_path,
                reason=f"dem_file_missing:{dem_path}",
            )
            return

    # ── happy path: real DEM on disk ─────────────────────────────────────────
    if not settings.recompute and slope_path.exists() and aspect_path.exists():
        LOGGER.info("Slope/aspect already exist. Skipping (use --recompute to override).")
        print(f"Slope: {slope_path}")
        print(f"Aspect: {aspect_path}")
        return

    grid_spec = grid_spec_from_metadata(dem_metadata)
    # Fail fast if the DEM is not exactly on the expected grid.
    validate_raster_matches_grid(dem_path, grid_spec, strict=True)
    with rasterio.open(dem_path) as src:
        z_ma = src.read(1, masked=True)
        z = np.asarray(z_ma.filled(np.nan), dtype=float)

        crs_epsg = src.crs.to_epsg() if src.crs is not None else None
        cell_deg = _validate_dem_alignment(
            crs_epsg=crs_epsg,
            transform=src.transform,
            width=src.width,
            height=src.height,
            expected_n_lon=grid_spec.n_lon,
            expected_n_lat=grid_spec.n_lat,
        )
        lat_centers = _lat_centers_from_transform(src.transform, src.height)

        slope_deg, aspect_deg = compute_slope_aspect(
            z, cell_size_deg=cell_deg, lat_centers_deg=lat_centers
        )

        slope_out = np.where(np.isfinite(slope_deg), slope_deg, np.nan).astype(np.float32)
        aspect_out = np.where(np.isfinite(aspect_deg), aspect_deg, np.nan).astype(np.float32)

        slope_out = np.where(np.isfinite(slope_out), slope_out, settings.nodata_value)
        aspect_out = np.where(np.isfinite(aspect_out), aspect_out, settings.nodata_value)

        _artifacts: list[Path] = []
        try:
            _write_aligned_geotiff(
                out_path=slope_path,
                data=slope_out,
                profile=src.profile,
                nodata_value=settings.nodata_value,
            )
            _artifacts.append(slope_path)
            _write_aligned_geotiff(
                out_path=aspect_path,
                data=aspect_out,
                profile=src.profile,
                nodata_value=settings.nodata_value,
            )
            _artifacts.append(aspect_path)

            final_slope_path = _convert_to_cog(slope_path) if emit_cog else slope_path
            final_aspect_path = _convert_to_cog(aspect_path) if emit_cog else aspect_path
            if emit_cog:
                _artifacts.extend([final_slope_path, final_aspect_path])

            # Fail fast if outputs are misaligned with DEM/grid.
            validate_terrain_stack(dem_path, final_slope_path, final_aspect_path, grid_spec, strict=True)
        except Exception:
            for p in _artifacts:
                try:
                    p.unlink()
                    log_event(
                        LOGGER,
                        "terrain_features.cleanup",
                        "Removed orphaned terrain artifact after validation failure",
                        level="warning",
                        path=str(p),
                        region=settings.region_name,
                    )
                except FileNotFoundError:
                    pass
            raise

        s_min, s_max, s_mean, coverage = _compute_stats(slope_out, settings.nodata_value)
        a_min, a_max, _a_mean, _a_cov = _compute_stats(aspect_out, settings.nodata_value)

        log_event(
            LOGGER,
            "terrain_features.stats",
            "Computed slope/aspect stats",
            region=settings.region_name,
            slope={"min": s_min, "max": s_max, "mean": s_mean, "units": "degrees"},
            aspect={"min": a_min, "max": a_max, "units": "degrees", "convention": "clockwise_from_north_downslope"},
            coverage_fraction=coverage,
        )

        # Derive origin edges from transform to match raster alignment exactly.
        origin_lon = float(src.transform.c)
        north_edge = float(src.transform.f)
        origin_lat = float(north_edge - src.height * cell_deg)

    inserted = insert_terrain_features_metadata(
        TerrainFeaturesMetadataCreate(
            region_name=settings.region_name,
            source_dem_metadata_id=int(dem_metadata.id),
            slope_path=str(final_slope_path),
            aspect_path=str(final_aspect_path),
            crs_epsg=CANONICAL_EPSG,
            cell_size_deg=float(cell_deg),
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            grid_n_lat=int(grid_spec.n_lat),
            grid_n_lon=int(grid_spec.n_lon),
            bbox=dem_metadata.bbox,
            slope_min=s_min,
            slope_max=s_max,
            aspect_min=a_min,
            aspect_max=a_max,
            coverage_fraction=coverage,
            nodata_value=float(settings.nodata_value),
        )
    )

    LOGGER.info(
        "Terrain features complete: slope=%s aspect=%s metadata_id=%s",
        final_slope_path,
        final_aspect_path,
        inserted.id,
    )
    print(f"Slope written to: {final_slope_path}")
    print(f"Aspect written to: {final_aspect_path}")
    print(f"Inserted terrain_features_metadata id={inserted.id}")


if __name__ == "__main__":
    main()

