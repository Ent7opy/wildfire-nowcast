"""Ingest authoritative Canada wildfire perimeters from CWFIS WFS."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import text

from ingest.repository import get_engine

LOGGER = logging.getLogger("cwfis_authority_ingest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

CWFIS_PROFILES: dict[str, dict[str, Any]] = {
    "cwfis_nbac_historical": {
        "wfs_url": "https://cwfis.cfs.nrcan.gc.ca/geoserver/wfs",
        "layer": "public:nbac",
        "source_uri": "https://cwfis.cfs.nrcan.gc.ca/geoserver/wfs",
        "date_field": "capdate",
        "tier": "gold",
        "is_authoritative": True,
        "map_method": "NBAC Composite",
        "poly_source": "NBAC",
    }
}


@dataclass
class FetchStats:
    records_fetched: int = 0
    pages: int = 0


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_geo_date(value: Any) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    token = raw.replace("Z", "")
    try:
        dt = datetime.fromisoformat(token)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_multipolygon_geojson(geometry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not geometry:
        return None
    gtype = str(geometry.get("type") or "").strip()
    coords = geometry.get("coordinates")
    if not isinstance(coords, list):
        return None
    if gtype == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": coords}
    if gtype == "Polygon":
        return {"type": "MultiPolygon", "coordinates": [coords]}
    return None


def _source_object_id(props: dict[str, Any]) -> str | None:
    for key in ("__gid", "gid", "id"):
        if key in props and props[key] is not None and str(props[key]).strip():
            return str(props[key])
    year = props.get("year")
    nfireid = props.get("nfireid")
    admin_area = props.get("admin_area")
    if year is None or nfireid is None:
        return None
    return f"{int(year)}:{int(nfireid)}:{(str(admin_area or '').strip() or 'na')}"


def _record_time(props: dict[str, Any]) -> datetime | None:
    for key in ("capdate", "ag_edate", "hs_edate", "ag_sdate", "hs_sdate"):
        dt = _parse_geo_date(props.get(key))
        if dt is not None:
            return dt
    return None


def _extract_record(
    feature: dict[str, Any],
    *,
    source_profile: str,
    source_layer: str,
    run_id: str,
    profile: dict[str, Any],
) -> dict[str, Any] | None:
    props = feature.get("properties") or {}
    if not isinstance(props, dict):
        return None
    source_id = _source_object_id(props)
    if source_id is None:
        return None
    geojson = _to_multipolygon_geojson(feature.get("geometry"))
    if geojson is None:
        return None

    start_dt = _parse_geo_date(props.get("ag_sdate")) or _parse_geo_date(props.get("hs_sdate"))
    end_dt = _parse_geo_date(props.get("capdate")) or _parse_geo_date(props.get("ag_edate")) or _parse_geo_date(
        props.get("hs_edate")
    )

    return {
        "source_profile": source_profile,
        "source_layer": source_layer,
        "source_object_id": source_id,
        "poly_irwinid": None,
        "poly_sourceglobalid": None,
        "poly_featurestatus": "Certified",
        "poly_featureaccess": "Public",
        "poly_isvisible": "Yes",
        "attr_isvalid": 1,
        "attr_isquarantined": 0,
        "poly_source": str(profile["poly_source"]),
        "poly_mapmethod": str(profile["map_method"]),
        "attr_firediscoverydatetime": start_dt,
        "poly_polygondatetime": end_dt,
        "attr_containmentdatetime": end_dt,
        "attr_controldatetime": end_dt,
        "tier": str(profile["tier"]),
        "is_authoritative": bool(profile["is_authoritative"]),
        "geom_json": json.dumps(geojson),
        "raw_attributes": json.dumps(props, default=str),
        "run_id": run_id,
    }


def _create_run(*, run_id: str, source_profile: str, source_uri: str, source_layer: str) -> None:
    stmt = text(
        """
        INSERT INTO authoritative_perimeter_ingest_runs (
            run_id,
            source_profile,
            source_uri,
            source_layer,
            status,
            started_at,
            created_at,
            updated_at
        )
        VALUES (
            :run_id,
            :source_profile,
            :source_uri,
            :source_layer,
            'running',
            NOW(),
            NOW(),
            NOW()
        )
        """
    )
    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": run_id,
                "source_profile": source_profile,
                "source_uri": source_uri,
                "source_layer": source_layer,
            },
        )


def _finish_run(
    *,
    run_id: str,
    status: str,
    source_last_edit: datetime | None,
    records_fetched: int,
    records_upserted: int,
    records_skipped: int,
    pages: int,
    where_clause: str,
    error_text: str | None = None,
) -> None:
    stmt = text(
        """
        UPDATE authoritative_perimeter_ingest_runs
        SET
            status = :status,
            source_last_edit = :source_last_edit,
            records_fetched = :records_fetched,
            records_upserted = :records_upserted,
            records_skipped = :records_skipped,
            metrics_json = CAST(:metrics_json AS json),
            error_text = :error_text,
            finished_at = NOW(),
            updated_at = NOW()
        WHERE run_id = :run_id
        """
    )
    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": run_id,
                "status": status,
                "source_last_edit": source_last_edit,
                "records_fetched": records_fetched,
                "records_upserted": records_upserted,
                "records_skipped": records_skipped,
                "metrics_json": json.dumps({"pages": pages, "where_clause": where_clause}),
                "error_text": error_text,
            },
        )


def _upsert_perimeters(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    stmt = text(
        """
        INSERT INTO authoritative_perimeters (
            source_profile,
            source_layer,
            source_object_id,
            poly_irwinid,
            poly_sourceglobalid,
            poly_featurestatus,
            poly_featureaccess,
            poly_isvisible,
            attr_isvalid,
            attr_isquarantined,
            poly_source,
            poly_mapmethod,
            attr_firediscoverydatetime,
            poly_polygondatetime,
            attr_containmentdatetime,
            attr_controldatetime,
            tier,
            is_authoritative,
            geom,
            raw_attributes,
            run_id,
            last_seen_at,
            created_at,
            updated_at
        )
        VALUES (
            :source_profile,
            :source_layer,
            :source_object_id,
            :poly_irwinid,
            :poly_sourceglobalid,
            :poly_featurestatus,
            :poly_featureaccess,
            :poly_isvisible,
            :attr_isvalid,
            :attr_isquarantined,
            :poly_source,
            :poly_mapmethod,
            :attr_firediscoverydatetime,
            :poly_polygondatetime,
            :attr_containmentdatetime,
            :attr_controldatetime,
            :tier,
            :is_authoritative,
            ST_SetSRID(ST_GeomFromGeoJSON(:geom_json), 4326),
            CAST(:raw_attributes AS json),
            :run_id,
            NOW(),
            NOW(),
            NOW()
        )
        ON CONFLICT (source_profile, source_layer, source_object_id) DO UPDATE SET
            poly_irwinid = EXCLUDED.poly_irwinid,
            poly_sourceglobalid = EXCLUDED.poly_sourceglobalid,
            poly_featurestatus = EXCLUDED.poly_featurestatus,
            poly_featureaccess = EXCLUDED.poly_featureaccess,
            poly_isvisible = EXCLUDED.poly_isvisible,
            attr_isvalid = EXCLUDED.attr_isvalid,
            attr_isquarantined = EXCLUDED.attr_isquarantined,
            poly_source = EXCLUDED.poly_source,
            poly_mapmethod = EXCLUDED.poly_mapmethod,
            attr_firediscoverydatetime = EXCLUDED.attr_firediscoverydatetime,
            poly_polygondatetime = EXCLUDED.poly_polygondatetime,
            attr_containmentdatetime = EXCLUDED.attr_containmentdatetime,
            attr_controldatetime = EXCLUDED.attr_controldatetime,
            tier = EXCLUDED.tier,
            is_authoritative = EXCLUDED.is_authoritative,
            geom = EXCLUDED.geom,
            raw_attributes = EXCLUDED.raw_attributes,
            run_id = EXCLUDED.run_id,
            last_seen_at = NOW(),
            updated_at = NOW()
        """
    )
    with get_engine().begin() as conn:
        result = conn.execute(stmt, rows)
    return int(result.rowcount or 0)


def _fetch_features(
    client: httpx.Client,
    *,
    wfs_url: str,
    layer: str,
    page_size: int,
    max_pages: int,
    timeout_seconds: float,
    cql_filter: str | None,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[list[dict[str, Any]], FetchStats]:
    start_index = 0
    stats = FetchStats()
    features: list[dict[str, Any]] = []
    while True:
        params: dict[str, Any] = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": layer,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": int(page_size),
            "startIndex": int(start_index),
        }
        if cql_filter:
            params["CQL_FILTER"] = cql_filter
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            params["bbox"] = f"{min_lon},{min_lat},{max_lon},{max_lat},EPSG:4326"
        resp = client.get(wfs_url, params=params, timeout=timeout_seconds)
        resp.raise_for_status()
        payload = resp.json()
        chunk = list(payload.get("features") or [])
        features.extend(chunk)
        stats.records_fetched += len(chunk)
        stats.pages += 1
        LOGGER.info("Fetched page=%s size=%s total=%s", stats.pages, len(chunk), stats.records_fetched)
        if not chunk or len(chunk) < page_size:
            break
        if max_pages > 0 and stats.pages >= max_pages:
            LOGGER.info("Stopping at max_pages=%s", max_pages)
            break
        start_index += page_size
    return features, stats


def _in_window(row_time: datetime | None, start: datetime | None, end: datetime | None) -> bool:
    if row_time is None:
        return start is None and end is None
    if start is not None and row_time < start:
        return False
    if end is not None and row_time > end:
        return False
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest authoritative CWFIS perimeters.")
    parser.add_argument("--source-profile", required=True, choices=sorted(CWFIS_PROFILES.keys()))
    parser.add_argument("--start", default=None, help="ISO datetime lower bound.")
    parser.add_argument("--end", default=None, help="ISO datetime upper bound.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means unlimited")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profile = CWFIS_PROFILES[args.source_profile]
    start_time = _parse_dt(args.start)
    end_time = _parse_dt(args.end)
    if start_time and end_time and end_time < start_time:
        raise SystemExit("--end must be >= --start")

    cql_filter: str | None = None
    if start_time or end_time:
        lo_year = start_time.year if start_time else 1900
        hi_year = end_time.year if end_time else 3000
        cql_filter = f"year >= {lo_year} AND year <= {hi_year}"

    run_id = f"{args.source_profile}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    source_uri = str(profile["source_uri"])
    source_layer = str(profile["layer"])
    if not args.dry_run:
        _create_run(run_id=run_id, source_profile=args.source_profile, source_uri=source_uri, source_layer=source_layer)

    fetched = 0
    upserted = 0
    skipped = 0
    source_last_edit = datetime.now(timezone.utc)
    stats = FetchStats()
    where_clause = cql_filter or "1=1"

    try:
        with httpx.Client() as client:
            features, stats = _fetch_features(
                client,
                wfs_url=str(profile["wfs_url"]),
                layer=source_layer,
                page_size=max(1, min(10000, int(args.page_size))),
                max_pages=max(0, int(args.max_pages)),
                timeout_seconds=float(args.timeout_seconds),
                cql_filter=cql_filter,
                bbox=tuple(args.bbox) if args.bbox else None,
            )
        fetched = len(features)

        rows: list[dict[str, Any]] = []
        for feature in features:
            props = feature.get("properties") or {}
            row_time = _record_time(props if isinstance(props, dict) else {})
            if not _in_window(row_time, start_time, end_time):
                continue
            row = _extract_record(
                feature,
                source_profile=args.source_profile,
                source_layer=source_layer,
                run_id=run_id,
                profile=profile,
            )
            if row is None:
                skipped += 1
                continue
            rows.append(row)

        if not args.dry_run:
            upserted = _upsert_perimeters(rows)
            _finish_run(
                run_id=run_id,
                status="succeeded",
                source_last_edit=source_last_edit,
                records_fetched=fetched,
                records_upserted=upserted,
                records_skipped=skipped,
                pages=stats.pages,
                where_clause=where_clause,
            )

        summary = {
            "run_id": run_id,
            "source_profile": args.source_profile,
            "source_layer": source_layer,
            "source_last_edit": source_last_edit.isoformat(),
            "records_fetched": fetched,
            "records_upserted": upserted,
            "records_skipped": skipped,
            "pages": stats.pages,
            "where_clause": where_clause,
            "dry_run": bool(args.dry_run),
        }
        print(json.dumps(summary))
    except Exception as exc:
        if not args.dry_run:
            _finish_run(
                run_id=run_id,
                status="failed",
                source_last_edit=source_last_edit,
                records_fetched=fetched,
                records_upserted=upserted,
                records_skipped=skipped,
                pages=stats.pages,
                where_clause=where_clause,
                error_text=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
