"""Seed script to load Natural Earth 10m populated places into PostGIS.

Download the Natural Earth 10m Populated Places dataset from:
  https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-populated-places/

The zip contains ne_10m_populated_places.shp and ne_10m_populated_places.geojson.
Pass either file as the positional argument. Shapefile input requires ogr2ogr (GDAL).

Usage:
    uv run --project api scripts/seed_ne_populated_places.py /path/to/ne_10m_populated_places.geojson
    uv run --project api scripts/seed_ne_populated_places.py /path/to/ne_10m_populated_places.shp
    uv run --project api scripts/seed_ne_populated_places.py /path/to/file.geojson --truncate
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text  # noqa: E402

from api.db import get_engine  # noqa: E402

LOGGER = logging.getLogger("seed_ne_populated_places")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_BATCH_SIZE = 500


def _shp_to_geojson(shp_path: Path) -> dict[str, Any]:
    """Convert shapefile to GeoJSON via ogr2ogr (requires GDAL)."""
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            ["ogr2ogr", "-f", "GeoJSON", str(tmp_path), str(shp_path), "-overwrite"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stderr:
            LOGGER.debug("ogr2ogr stderr: %s", result.stderr)
        return json.loads(tmp_path.read_text())
    except FileNotFoundError:
        raise RuntimeError(
            "ogr2ogr not found. Install GDAL or convert the shapefile to GeoJSON manually "
            "before running this script."
        ) from None
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_geojson(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".shp":
        LOGGER.info("Converting shapefile to GeoJSON via ogr2ogr...")
        return _shp_to_geojson(path)
    return json.loads(path.read_text())


def _extract_row(feat: dict[str, Any]) -> dict[str, Any] | None:
    # Normalize to uppercase once — Natural Earth ships uppercase; ogr2ogr may lowercase.
    props = {k.upper(): v for k, v in (feat.get("properties") or {}).items()}
    geom = feat.get("geometry")
    if not geom or geom.get("type") != "Point":
        return None
    coords = geom.get("coordinates") or []
    if len(coords) < 2:
        return None
    lon, lat = float(coords[0]), float(coords[1])
    name = props.get("NAME")
    if not name:
        return None
    pop_raw = props.get("POP_MAX")
    adm0 = props.get("ADM0_A3")
    adm1 = props.get("ADM1NAME")
    return {
        "name": str(name),
        "lon": lon,
        "lat": lat,
        "pop_max": int(pop_raw) if pop_raw is not None else None,
        "adm0_a3": str(adm0)[:3] if adm0 else None,
        "adm1name": str(adm1) if adm1 else None,
    }


def seed(path: Path, *, truncate: bool = False) -> int:
    """Load Natural Earth populated places from GeoJSON/shapefile.

    Returns number of rows inserted.
    """
    data = _load_geojson(path)
    features = data.get("features") or []
    if not features:
        LOGGER.warning("No features found in %s", path)
        return 0

    rows = [r for f in features if (r := _extract_row(f)) is not None]
    LOGGER.info("Parsed %d valid place records from %d features.", len(rows), len(features))

    insert_stmt = text(
        """
        INSERT INTO ne_populated_places (name, geom, pop_max, adm0_a3, adm1name)
        VALUES (
            :name,
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
            :pop_max,
            :adm0_a3,
            :adm1name
        )
        """
    )

    with get_engine().begin() as conn:
        if truncate:
            conn.execute(text("TRUNCATE TABLE ne_populated_places RESTART IDENTITY"))
            LOGGER.info("Truncated ne_populated_places.")
        for batch_start in range(0, len(rows), _BATCH_SIZE):
            batch = rows[batch_start : batch_start + _BATCH_SIZE]
            conn.execute(insert_stmt, batch)
            LOGGER.info(
                "Inserted rows %d–%d / %d",
                batch_start + 1,
                batch_start + len(batch),
                len(rows),
            )

    LOGGER.info("Done. %d rows loaded into ne_populated_places.", len(rows))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Natural Earth 10m populated places into PostGIS."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to ne_10m_populated_places.geojson or .shp",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate the table before inserting (safe to re-run after download update)",
    )
    args = parser.parse_args()

    if not args.path.exists():
        LOGGER.error("File not found: %s", args.path)
        sys.exit(1)

    seed(args.path, truncate=args.truncate)


if __name__ == "__main__":
    main()
