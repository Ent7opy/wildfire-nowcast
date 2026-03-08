"""Backfill authoritative LULC classes from ESA WorldCover tiles.

This script augments `fire_detections` with categorical land-cover fields
(`landcover_class`, `landcover_label`) and synchronized `landcover_score`
for a target space-time window.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import numpy as np
import rasterio
from sqlalchemy import text

from api.db import get_engine

LOGGER = logging.getLogger("lulc_worldcover_ingest")

_WORLDCOVER_S3_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# ESA WorldCover class labels.
_CLASS_LABELS: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Keep scoring aligned with existing denoiser landcover priors.
_CLASS_SCORES: dict[int, float] = {
    10: 1.0,
    20: 1.0,
    30: 1.0,
    40: 0.7,
    50: 0.1,
    60: 0.1,
    70: 0.1,
    80: 0.1,
    90: 0.1,
    95: 0.1,
    100: 0.1,
}


@dataclass(frozen=True)
class TileWorkItem:
    tile_id: str
    tile_lat0: int
    tile_lon0: int
    detection_count: int

    @property
    def tile_min_lat(self) -> float:
        return float(self.tile_lat0)

    @property
    def tile_max_lat(self) -> float:
        return float(self.tile_lat0 + 3)

    @property
    def tile_min_lon(self) -> float:
        return float(self.tile_lon0)

    @property
    def tile_max_lon(self) -> float:
        return float(self.tile_lon0 + 3)


def _format_tile_id(*, lat0: int, lon0: int) -> str:
    lat_prefix = "N" if lat0 >= 0 else "S"
    lon_prefix = "E" if lon0 >= 0 else "W"
    return f"{lat_prefix}{abs(lat0):02d}{lon_prefix}{abs(lon0):03d}"


def _tile_path(*, version: str, year: int, tile_id: str) -> str:
    return f"{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile_id}_Map.tif"


def _ensure_columns() -> None:
    stmt = text(
        """
        ALTER TABLE fire_detections
            ADD COLUMN IF NOT EXISTS landcover_class integer,
            ADD COLUMN IF NOT EXISTS landcover_label text
        """
    )
    with get_engine().begin() as conn:
        conn.execute(stmt)


def _list_tile_work(
    *,
    start_time: datetime,
    end_time: datetime,
    bbox: tuple[float, float, float, float],
    force: bool,
) -> list[TileWorkItem]:
    min_lon, min_lat, max_lon, max_lat = bbox
    stmt = text(
        """
        WITH candidates AS (
            SELECT
                floor(lat / 3.0)::int * 3 AS tile_lat0,
                floor(lon / 3.0)::int * 3 AS tile_lon0
            FROM fire_detections
            WHERE acq_time BETWEEN :start_time AND :end_time
              AND lon BETWEEN :min_lon AND :max_lon
              AND lat BETWEEN :min_lat AND :max_lat
              AND (:force OR landcover_class IS NULL)
        )
        SELECT
            tile_lat0,
            tile_lon0,
            COUNT(*) AS n
        FROM candidates
        GROUP BY tile_lat0, tile_lon0
        ORDER BY n DESC
        """
    )
    params = {
        "start_time": start_time,
        "end_time": end_time,
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
        "force": bool(force),
    }
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, params).mappings().all()

    out: list[TileWorkItem] = []
    for row in rows:
        lat0 = int(row["tile_lat0"])
        lon0 = int(row["tile_lon0"])
        out.append(
            TileWorkItem(
                tile_id=_format_tile_id(lat0=lat0, lon0=lon0),
                tile_lat0=lat0,
                tile_lon0=lon0,
                detection_count=int(row["n"]),
            )
        )
    return out


def _download_tile(*, url: str, local_path: Path, timeout_seconds: int = 120) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        return

    with urlopen(url, timeout=timeout_seconds) as resp:
        tmp = local_path.with_suffix(local_path.suffix + ".part")
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(local_path)


def _sample_classes(src: rasterio.io.DatasetReader, coords: list[tuple[float, float]]) -> np.ndarray:
    # rasterio returns arrays with one value per coordinate for a single-band raster.
    return np.fromiter((int(v[0]) for v in src.sample(coords)), dtype=np.int32, count=len(coords))


def _iter_chunks(seq: list[dict], size: int) -> Iterable[list[dict]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _backfill_tile(
    *,
    item: TileWorkItem,
    tile_path: Path,
    start_time: datetime,
    end_time: datetime,
    bbox: tuple[float, float, float, float],
    force: bool,
    query_batch_size: int,
    update_batch_size: int,
    source_tag: str,
    source_version: str,
) -> tuple[int, int]:
    min_lon, min_lat, max_lon, max_lat = bbox

    select_stmt = text(
        """
        SELECT id, lat, lon
        FROM fire_detections
        WHERE acq_time BETWEEN :start_time AND :end_time
          AND lon BETWEEN :bbox_min_lon AND :bbox_max_lon
          AND lat BETWEEN :bbox_min_lat AND :bbox_max_lat
          AND lon >= :tile_min_lon
          AND lon < :tile_max_lon
          AND lat >= :tile_min_lat
          AND lat < :tile_max_lat
          AND (:force OR landcover_class IS NULL)
        ORDER BY id
        """
    )

    update_stmt = text(
        """
        UPDATE fire_detections
        SET
            landcover_class = :landcover_class,
            landcover_label = :landcover_label,
            landcover_score = :landcover_score,
            raw_properties = COALESCE(raw_properties, '{}'::jsonb)
                || jsonb_build_object(
                    'landcover_class', :landcover_class,
                    'landcover_label', :landcover_label,
                    'landcover_source', :landcover_source,
                    'landcover_version', :landcover_version
                )
        WHERE id = :id
        """
    )

    params = {
        "start_time": start_time,
        "end_time": end_time,
        "bbox_min_lon": float(min_lon),
        "bbox_min_lat": float(min_lat),
        "bbox_max_lon": float(max_lon),
        "bbox_max_lat": float(max_lat),
        "tile_min_lon": item.tile_min_lon,
        "tile_max_lon": item.tile_max_lon,
        "tile_min_lat": item.tile_min_lat,
        "tile_max_lat": item.tile_max_lat,
        "force": bool(force),
    }

    rows_seen = 0
    rows_updated = 0

    with (
        rasterio.open(tile_path) as src,
        get_engine().connect() as read_conn,
        get_engine().begin() as write_conn,
    ):
        result = read_conn.execution_options(stream_results=True).execute(select_stmt, params)
        while True:
            chunk = result.fetchmany(query_batch_size)
            if not chunk:
                break

            rows_seen += len(chunk)
            ids = [int(r.id) for r in chunk]
            coords = [(float(r.lon), float(r.lat)) for r in chunk]
            classes = _sample_classes(src, coords)

            updates: list[dict] = []
            nodata = src.nodata
            for det_id, lc in zip(ids, classes):
                if nodata is not None and math.isclose(float(lc), float(nodata)):
                    continue
                if lc <= 0:
                    continue
                label = _CLASS_LABELS.get(int(lc), "Unknown")
                score = float(_CLASS_SCORES.get(int(lc), 0.5))
                updates.append(
                    {
                        "id": det_id,
                        "landcover_class": int(lc),
                        "landcover_label": label,
                        "landcover_score": score,
                        "landcover_source": source_tag,
                        "landcover_version": source_version,
                    }
                )

            for update_batch in _iter_chunks(updates, update_batch_size):
                write_conn.execute(update_stmt, update_batch)
                rows_updated += len(update_batch)

    return rows_seen, rows_updated


def backfill_worldcover(
    *,
    start_time: datetime,
    end_time: datetime,
    bbox: tuple[float, float, float, float],
    version: str,
    year: int,
    force: bool,
    max_tiles: int,
    query_batch_size: int,
    update_batch_size: int,
    cache_dir: Path,
) -> dict[str, int]:
    _ensure_columns()

    work = _list_tile_work(
        start_time=start_time,
        end_time=end_time,
        bbox=bbox,
        force=force,
    )
    if max_tiles > 0:
        work = work[:max_tiles]

    LOGGER.info("WorldCover worklist: %s tiles", len(work))

    rows_seen_total = 0
    rows_updated_total = 0
    tiles_processed = 0

    for idx, item in enumerate(work, start=1):
        rel_key = _tile_path(version=version, year=year, tile_id=item.tile_id)
        url = f"{_WORLDCOVER_S3_BASE}/{rel_key}"
        local_path = cache_dir / Path(rel_key).name

        try:
            _download_tile(url=url, local_path=local_path)
        except Exception as exc:
            LOGGER.warning("Skipping tile %s (download failed): %s", item.tile_id, exc)
            continue

        tile_seen, tile_updated = _backfill_tile(
            item=item,
            tile_path=local_path,
            start_time=start_time,
            end_time=end_time,
            bbox=bbox,
            force=force,
            query_batch_size=query_batch_size,
            update_batch_size=update_batch_size,
            source_tag="esa_worldcover",
            source_version=f"{version}_{year}",
        )
        rows_seen_total += tile_seen
        rows_updated_total += tile_updated
        tiles_processed += 1

        LOGGER.info(
            "Tile %s (%s/%s) detections=%s updated=%s",
            item.tile_id,
            idx,
            len(work),
            tile_seen,
            tile_updated,
        )

    with get_engine().begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE acq_time BETWEEN :start_time AND :end_time
                          AND lon BETWEEN :min_lon AND :max_lon
                          AND lat BETWEEN :min_lat AND :max_lat
                          AND landcover_class = 40
                    ) AS cropland_rows,
                    COUNT(*) FILTER (
                        WHERE acq_time BETWEEN :start_time AND :end_time
                          AND lon BETWEEN :min_lon AND :max_lon
                          AND lat BETWEEN :min_lat AND :max_lat
                          AND landcover_class IS NOT NULL
                    ) AS classified_rows
                FROM fire_detections
                """
            ),
            {
                "start_time": start_time,
                "end_time": end_time,
                "min_lon": float(bbox[0]),
                "min_lat": float(bbox[1]),
                "max_lon": float(bbox[2]),
                "max_lat": float(bbox[3]),
            },
        ).mappings().first()

    return {
        "tiles_requested": len(work),
        "tiles_processed": tiles_processed,
        "rows_seen": int(rows_seen_total),
        "rows_updated": int(rows_updated_total),
        "classified_rows": int(row["classified_rows"] or 0),
        "cropland_rows": int(row["cropland_rows"] or 0),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill LULC classes from ESA WorldCover.")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=[-179.5, 18.0, -66.0, 72.0],
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    )
    parser.add_argument("--version", type=str, default="v200")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--force", action="store_true", help="Recompute even if landcover_class is present")
    parser.add_argument("--max-tiles", type=int, default=0, help="Optional cap for tile count (0 = all)")
    parser.add_argument("--query-batch-size", type=int, default=50000)
    parser.add_argument("--update-batch-size", type=int, default=5000)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/lulc/worldcover/v200_2021_tiles",
        help="Local cache for downloaded WorldCover tiles.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    start_time = datetime.strptime(args.start, "%Y-%m-%d")
    end_time = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)

    result = backfill_worldcover(
        start_time=start_time,
        end_time=end_time,
        bbox=tuple(args.bbox),
        version=str(args.version),
        year=int(args.year),
        force=bool(args.force),
        max_tiles=int(args.max_tiles),
        query_batch_size=int(args.query_batch_size),
        update_batch_size=int(args.update_batch_size),
        cache_dir=Path(args.cache_dir),
    )
    print(result)


if __name__ == "__main__":
    main()
