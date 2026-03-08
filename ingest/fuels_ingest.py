"""UNSAFE synthetic fuel/moisture feature ingestion (deprecated).

WARNING: UNSAFE_FOR_PRODUCTION
This module generates deterministic synthetic priors and must not be used for
production denoiser or forecast training.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sqlalchemy as sa
import xarray as xr

from api.db import get_engine
from ingest.config import fuel_settings

LOGGER = logging.getLogger(__name__)
UNSAFE_FOR_PRODUCTION = True
_ALLOW_UNSAFE_ENV = "ALLOW_UNSAFE_SYNTHETIC_FUELS"


def _assert_not_production_synthetic(*, allow_unsafe_synthetic: bool) -> None:
    if not UNSAFE_FOR_PRODUCTION:
        return
    env_allow = str(os.getenv(_ALLOW_UNSAFE_ENV, "false")).strip().lower() in {"1", "true", "yes", "on"}
    if allow_unsafe_synthetic or env_allow:
        LOGGER.warning(
            "WARNING: UNSAFE_FOR_PRODUCTION synthetic fuel ingestion enabled via explicit override."
        )
        return
    raise RuntimeError(
        "WARNING: UNSAFE_FOR_PRODUCTION synthetic fuel/moisture cube is deprecated. "
        "Use authoritative ingestors (lfmc_ecland_ingest.py and dfmc_sjsu_ingest.py) instead. "
        "To run this script only for controlled local tests, pass --allow-unsafe-synthetic "
        f"or set {_ALLOW_UNSAFE_ENV}=true."
    )


def _parse_run_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _grid_from_bbox(
    bbox: tuple[float, float, float, float],
    *,
    resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"Invalid bbox: {bbox!r}")
    lat = np.arange(min_lat + resolution_deg / 2.0, max_lat, resolution_deg, dtype=np.float32)
    lon = np.arange(min_lon + resolution_deg / 2.0, max_lon, resolution_deg, dtype=np.float32)
    if lat.size == 0 or lon.size == 0:
        raise ValueError(f"Empty output grid for bbox={bbox!r} resolution_deg={resolution_deg}")
    return lat, lon


def _build_feature_cube(
    bbox: tuple[float, float, float, float],
    run_time: datetime,
    *,
    resolution_deg: float,
) -> xr.Dataset:
    lat, lon = _grid_from_bbox(bbox, resolution_deg=resolution_deg)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    # Deterministic synthetic priors; external-provider integration can overwrite these fields.
    day = float(run_time.timetuple().tm_yday)
    seasonal = 0.5 + 0.5 * np.sin(2.0 * np.pi * day / 365.0)

    ndvi = np.clip(0.4 + 0.25 * np.sin(np.radians(lat2d * 2.0)) * np.cos(np.radians(lon2d)), 0.0, 1.0)
    lfmc = np.clip(70.0 + 40.0 * seasonal + 15.0 * ndvi, 20.0, 200.0)
    dfmc = np.clip(8.0 + 10.0 * (1.0 - seasonal) + 3.0 * (1.0 - ndvi), 2.0, 35.0)
    precip_24h = np.clip(1.0 + 4.0 * np.maximum(0.0, np.sin(np.radians(lat2d + lon2d))), 0.0, 80.0)

    coords = {"lat": lat, "lon": lon}
    return xr.Dataset(
        data_vars={
            "ndvi": (("lat", "lon"), ndvi.astype(np.float32, copy=False)),
            "lfmc": (("lat", "lon"), lfmc.astype(np.float32, copy=False)),
            "dfmc": (("lat", "lon"), dfmc.astype(np.float32, copy=False)),
            "precip_24h": (("lat", "lon"), precip_24h.astype(np.float32, copy=False)),
        },
        coords=coords,
        attrs={
            "run_time": run_time.isoformat(),
            "bbox": [float(v) for v in bbox],
            "provider": fuel_settings.provider_url,
            "resolution_deg": float(resolution_deg),
        },
    )


def _record_run(
    *,
    run_time: datetime,
    bbox: tuple[float, float, float, float],
    storage_path: Path,
    provider: str,
) -> int | None:
    stmt = sa.text(
        """
        INSERT INTO fuel_moisture_runs (
            run_time,
            bbox_min_lon,
            bbox_min_lat,
            bbox_max_lon,
            bbox_max_lat,
            status,
            storage_path,
            provider
        )
        VALUES (
            :run_time,
            :min_lon,
            :min_lat,
            :max_lon,
            :max_lat,
            'completed',
            :storage_path,
            :provider
        )
        RETURNING id
        """
    )
    try:
        with get_engine().begin() as conn:
            row = conn.execute(
                stmt,
                {
                    "run_time": run_time,
                    "min_lon": float(bbox[0]),
                    "min_lat": float(bbox[1]),
                    "max_lon": float(bbox[2]),
                    "max_lat": float(bbox[3]),
                    "storage_path": str(storage_path),
                    "provider": provider,
                },
            ).mappings().first()
        return int(row["id"]) if row and row.get("id") is not None else None
    except Exception:
        LOGGER.warning(
            "Skipping fuel_moisture_runs DB insert (table may be unavailable).",
            exc_info=True,
        )
        return None


def ingest_fuel_moisture_for_bbox(
    *,
    bbox: tuple[float, float, float, float],
    run_time: datetime | None = None,
    output_dir: Path | None = None,
    resolution_deg: float = 0.01,
    allow_unsafe_synthetic: bool = False,
) -> dict[str, Any]:
    """Build and persist a cached fuel/moisture feature cube for an AOI."""
    _assert_not_production_synthetic(allow_unsafe_synthetic=allow_unsafe_synthetic)
    resolved_time = (run_time or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out_root = output_dir or fuel_settings.cache_root
    out_root.mkdir(parents=True, exist_ok=True)

    ts = resolved_time.strftime("%Y%m%dT%HZ")
    bbox_token = "_".join(f"{float(v):.4f}" for v in bbox)
    out_path = out_root / f"fuels_{ts}_bbox_{bbox_token}.nc"

    ds = _build_feature_cube(bbox=bbox, run_time=resolved_time, resolution_deg=float(resolution_deg))
    ds.to_netcdf(out_path)
    run_id = _record_run(
        run_time=resolved_time,
        bbox=bbox,
        storage_path=out_path,
        provider=fuel_settings.provider_url,
    )
    LOGGER.info("Fuel/moisture cube written: %s", out_path)
    return {"run_id": run_id, "storage_path": str(out_path), "bbox": bbox, "run_time": resolved_time.isoformat()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest/calculate fuel moisture features for an AOI.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        required=True,
        help="AOI bbox in WGS84.",
    )
    parser.add_argument(
        "--run-time",
        type=str,
        default=None,
        help="ISO8601 reference time (default: current UTC hour).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=fuel_settings.cache_root,
        help="Output directory for fuel feature cubes.",
    )
    parser.add_argument(
        "--resolution-deg",
        type=float,
        default=0.01,
        help="Grid resolution in degrees.",
    )
    parser.add_argument(
        "--allow-unsafe-synthetic",
        action="store_true",
        help="Explicitly allow deprecated synthetic cube generation for local tests only.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    result = ingest_fuel_moisture_for_bbox(
        bbox=tuple(float(v) for v in args.bbox),
        run_time=_parse_run_time(args.run_time),
        output_dir=args.output_dir,
        resolution_deg=float(args.resolution_deg),
        allow_unsafe_synthetic=bool(args.allow_unsafe_synthetic),
    )
    LOGGER.info("Fuel ingestion completed: %s", result)


if __name__ == "__main__":
    main()
