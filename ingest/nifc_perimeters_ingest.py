"""Ingest NIFC fire perimeters for ground truth denoiser labeling.

Downloads wildfire perimeter polygons from the NIFC WFIGS ArcGIS REST API
and loads them into the ``fire_perimeters`` table for spatial
cross-referencing with FIRMS detections.

Usage::

    # Ingest all US perimeters for 2024
    python -m ingest.nifc_perimeters_ingest --year 2024

    # Ingest perimeters within a bounding box
    python -m ingest.nifc_perimeters_ingest --year 2024 --bbox -125 24 -66 50

    # Ingest multiple years
    python -m ingest.nifc_perimeters_ingest --year 2023 --year 2024
"""

import argparse
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from sqlalchemy import JSON, bindparam, text

from ingest.repository import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("nifc_perimeters_ingest")

# NIFC WFIGS Interagency Perimeters – has full date attributes.
NIFC_FEATURE_SERVER = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
    "/WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)
MAX_RECORD_COUNT = 2000  # ArcGIS server page size limit


def _epoch_ms_to_dt(epoch_ms: Optional[int]) -> Optional[datetime]:
    """Convert ArcGIS epoch-millisecond timestamp to timezone-aware datetime."""
    if epoch_ms is None:
        return None
    try:
        return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    except (OSError, ValueError, OverflowError):
        return None


def _geometry_to_wkt(geometry: Dict[str, Any]) -> Optional[str]:
    """Convert an ArcGIS JSON geometry to WKT MULTIPOLYGON."""
    if not geometry:
        return None

    # Handle "rings" format (single polygon with possible holes)
    rings = geometry.get("rings")
    if rings:
        parts = []
        for ring in rings:
            coords = ", ".join(f"{pt[0]} {pt[1]}" for pt in ring)
            parts.append(f"({coords})")
        polygon_wkt = "(" + ", ".join(parts) + ")"
        return f"MULTIPOLYGON({polygon_wkt})"

    return None


def fetch_nifc_perimeters(
    year: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    timeout_seconds: float = 120.0,
) -> List[Dict[str, Any]]:
    """Fetch all NIFC fire perimeters for a given year via paginated queries.

    Parameters
    ----------
    year : int
        Fire year to query (filters on ``attr_FireDiscoveryDateTime``).
    bbox : tuple, optional
        (min_lon, min_lat, max_lon, max_lat) spatial filter.
    timeout_seconds : float
        HTTP request timeout.

    Returns
    -------
    list[dict]
        Raw feature dicts from the ArcGIS API.
    """
    where_clause = (
        f"attr_FireDiscoveryDateTime >= DATE '{year}-01-01' "
        f"AND attr_FireDiscoveryDateTime < DATE '{year + 1}-01-01'"
    )

    geometry_filter = ""
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        geometry_filter = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    all_features: List[Dict[str, Any]] = []
    offset = 0

    while True:
        params: Dict[str, Any] = {
            "where": where_clause,
            "outFields": "*",
            "returnGeometry": "true",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": MAX_RECORD_COUNT,
            "outSR": "4326",
        }
        if geometry_filter:
            params["geometry"] = geometry_filter
            params["geometryType"] = "esriGeometryEnvelope"
            params["spatialRel"] = "esriSpatialRelIntersects"
            params["inSR"] = "4326"

        LOGGER.info(
            "Fetching NIFC perimeters: year=%d offset=%d bbox=%s",
            year, offset, bbox,
        )

        response = httpx.get(
            NIFC_FEATURE_SERVER,
            params=params,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            LOGGER.error("NIFC API error: %s", data["error"])
            break

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        LOGGER.info("  Received %d features (total so far: %d)", len(features), len(all_features))

        # Check if there are more pages
        if len(features) < MAX_RECORD_COUNT:
            break
        offset += MAX_RECORD_COUNT

    LOGGER.info("Total features fetched for year %d: %d", year, len(all_features))
    return all_features


def _parse_feature(feature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a single ArcGIS feature into a row dict for insertion."""
    attrs = feature.get("attributes", {})
    geometry = feature.get("geometry")

    wkt = _geometry_to_wkt(geometry)
    if not wkt:
        return None

    # Extract fire dates – WFIGS uses epoch-ms timestamps
    fire_start = _epoch_ms_to_dt(attrs.get("attr_FireDiscoveryDateTime"))
    fire_end = (
        _epoch_ms_to_dt(attrs.get("attr_ContainmentDateTime"))
        or _epoch_ms_to_dt(attrs.get("attr_FireOutDateTime"))
        or _epoch_ms_to_dt(attrs.get("attr_ControlDateTime"))
    )

    source_id = str(
        attrs.get("attr_IrwinID")
        or attrs.get("poly_IRWINID")
        or attrs.get("attr_UniqueFireIdentifier")
        or attrs.get("OBJECTID", "")
    )

    acres = (
        attrs.get("attr_CalculatedAcres")
        or attrs.get("poly_GISAcres")
        or attrs.get("attr_FinalAcres")
        or attrs.get("attr_IncidentSize")
    )
    if acres is not None:
        try:
            acres = float(acres)
            if math.isnan(acres):
                acres = None
        except (ValueError, TypeError):
            acres = None

    fire_name = (
        attrs.get("attr_IncidentName")
        or attrs.get("poly_IncidentName")
    )

    cause = attrs.get("attr_FireCause") or attrs.get("attr_FireCauseGeneral")
    state = attrs.get("attr_POOState")

    # Store full attributes as meta for traceability
    meta = {}
    for k, v in attrs.items():
        if v is not None and v != "" and not isinstance(v, (bytes, bytearray)):
            meta[k] = v

    return {
        "wkt": wkt,
        "source": "NIFC",
        "source_id": source_id,
        "fire_name": fire_name,
        "fire_start": fire_start,
        "fire_end": fire_end,
        "acres": acres,
        "cause": cause,
        "state": state,
        "meta": meta,
    }


def ingest_perimeters(features: List[Dict[str, Any]]) -> int:
    """Parse and insert NIFC perimeter features into the DB.

    Returns the number of rows inserted.
    """
    engine = get_engine()
    rows = []
    skipped = 0

    for feat in features:
        parsed = _parse_feature(feat)
        if parsed is None:
            skipped += 1
            continue
        rows.append(parsed)

    if not rows:
        LOGGER.warning("No valid perimeters to ingest (skipped %d).", skipped)
        return 0

    LOGGER.info("Parsed %d perimeters (%d skipped).", len(rows), skipped)

    insert_stmt = text("""
        INSERT INTO fire_perimeters (geom, source, source_id, fire_name, fire_start, fire_end, acres, cause, state, meta)
        VALUES (
            ST_Multi(ST_GeomFromText(:wkt, 4326)),
            :source, :source_id, :fire_name, :fire_start, :fire_end, :acres, :cause, :state, :meta
        )
        ON CONFLICT (source, source_id) DO UPDATE SET
            geom = EXCLUDED.geom,
            fire_name = EXCLUDED.fire_name,
            fire_start = EXCLUDED.fire_start,
            fire_end = EXCLUDED.fire_end,
            acres = EXCLUDED.acres,
            cause = EXCLUDED.cause,
            state = EXCLUDED.state,
            meta = EXCLUDED.meta
    """).bindparams(bindparam("meta", type_=JSON))

    batch_size = 100
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            conn.execute(insert_stmt, batch)
            inserted += len(batch)
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(rows):
                LOGGER.info("  Inserted %d / %d perimeters...", min(i + batch_size, len(rows)), len(rows))

    LOGGER.info("Successfully ingested %d perimeters.", inserted)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest NIFC fire perimeters for ground truth labeling."
    )
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        required=True,
        help="Fire year(s) to ingest (can be repeated, e.g. --year 2024 --year 2025).",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box spatial filter.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: 120).",
    )

    args = parser.parse_args()
    bbox = tuple(args.bbox) if args.bbox else None

    total = 0
    for year in args.year:
        features = fetch_nifc_perimeters(year, bbox=bbox, timeout_seconds=args.timeout)
        total += ingest_perimeters(features)

    LOGGER.info("Done. Total perimeters ingested: %d", total)


if __name__ == "__main__":
    main()
