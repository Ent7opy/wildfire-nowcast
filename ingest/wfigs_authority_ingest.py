"""Ingest authoritative WFIGS perimeters with provenance and tier classification."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import text

from ingest.perimeter_authority import (
    log_authority_conflict,
    record_authority_conflict,
    should_overwrite,
)
from ingest.repository import get_engine

LOGGER = logging.getLogger("wfigs_authority_ingest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

WFIGS_PROFILES: dict[str, dict[str, str]] = {
    "wfigs_current_interagency_perimeters": {
        "query_url": (
            "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
            "WFIGS_Current_Interagency_Fire_Perimeters/FeatureServer/0/query"
        ),
        "layer": "WFIGS_Current_Interagency_Fire_Perimeters",
    },
    "wfigs_2025_ytd_perimeters": {
        "query_url": (
            "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
            "WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0/query"
        ),
        "layer": "WFIGS_Interagency_Perimeters_YearToDate",
    },
    "wfigs_interagency_perimeters_full_history": {
        "query_url": (
            "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
            "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
        ),
        "layer": "WFIGS_Interagency_Perimeters",
    },
}


@dataclass
class FetchStats:
    records_fetched: int = 0
    http_429_count: int = 0
    max_backoff_seconds: int = 0
    pages: int = 0


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _arcgis_date_literal(value: datetime) -> str:
    as_utc = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    as_utc = as_utc.astimezone(timezone.utc)
    return as_utc.strftime("DATE '%Y-%m-%d %H:%M:%S'")


def _epoch_ms_to_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        epoch = int(value)
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(epoch / 1000.0, tz=timezone.utc)
    except (OSError, ValueError, OverflowError):
        return None


def _normalize_yes(value: Any) -> bool:
    if value is None:
        return False
    token = str(value).strip().lower()
    return token in {"y", "yes", "true", "1"}


def _normalize_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_str(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _close_ring(ring: list[list[float]]) -> list[list[float]]:
    if not ring:
        return ring
    first = ring[0]
    last = ring[-1]
    if len(first) >= 2 and len(last) >= 2 and (first[0] != last[0] or first[1] != last[1]):
        return ring + [first]
    return ring


def _arcgis_to_multipolygon_geojson(geometry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not geometry:
        return None
    rings = geometry.get("rings")
    if not isinstance(rings, list) or not rings:
        return None

    cleaned_rings: list[list[list[float]]] = []
    for ring in rings:
        if not isinstance(ring, list) or len(ring) < 4:
            continue
        pts: list[list[float]] = []
        for pt in ring:
            if not isinstance(pt, list) or len(pt) < 2:
                continue
            pts.append([float(pt[0]), float(pt[1])])
        if len(pts) < 4:
            continue
        cleaned_rings.append(_close_ring(pts))

    if not cleaned_rings:
        return None
    # ArcGIS ring orientation semantics are not guaranteed here; keep each ring as a polygon shell.
    return {"type": "MultiPolygon", "coordinates": [[ring] for ring in cleaned_rings]}


def _classify_tier(attrs: dict[str, Any]) -> tuple[str, bool]:
    status = (_normalize_str(attrs.get("poly_FeatureStatus")) or "").lower()
    access = (_normalize_str(attrs.get("poly_FeatureAccess")) or "").lower()
    visible = _normalize_yes(attrs.get("poly_IsVisible"))
    is_valid = _normalize_int(attrs.get("attr_IsValid"), default=1) == 1
    quarantined = _normalize_int(attrs.get("attr_IsQuarantined"), default=0) == 1
    poly_source = (_normalize_str(attrs.get("poly_Source")) or "").upper()

    silver_ok = (
        status in {"approved", "certified"}
        and access == "public"
        and visible
        and is_valid
        and not quarantined
    )
    if silver_ok and poly_source in {"FFP", "FODR"}:
        return "gold", True
    if silver_ok:
        return "silver", True
    if quarantined:
        return "blocked", False
    return "bronze", False


def _source_object_id(attrs: dict[str, Any]) -> str | None:
    for key in ("OBJECTID", "objectid", "ObjectId"):
        value = attrs.get(key)
        if value is not None and str(value).strip():
            return str(value)
    gid = attrs.get("GlobalID") or attrs.get("globalid")
    if gid is not None and str(gid).strip():
        return str(gid)
    return None


def _extract_record(
    feature: dict[str, Any],
    *,
    source_profile: str,
    source_layer: str,
    run_id: str,
) -> dict[str, Any] | None:
    attrs = feature.get("attributes") or {}
    if not isinstance(attrs, dict):
        return None

    source_id = _source_object_id(attrs)
    if source_id is None:
        return None

    geojson = _arcgis_to_multipolygon_geojson(feature.get("geometry"))
    if geojson is None:
        return None

    tier, authoritative = _classify_tier(attrs)
    return {
        "source_profile": source_profile,
        "source_layer": source_layer,
        "source_object_id": source_id,
        "poly_irwinid": _normalize_str(attrs.get("poly_IRWINID") or attrs.get("attr_IrwinID")),
        "poly_sourceglobalid": _normalize_str(attrs.get("poly_SourceGlobalID")),
        "poly_featurestatus": _normalize_str(attrs.get("poly_FeatureStatus")),
        "poly_featureaccess": _normalize_str(attrs.get("poly_FeatureAccess")),
        "poly_isvisible": _normalize_str(attrs.get("poly_IsVisible")),
        "attr_isvalid": _normalize_int(attrs.get("attr_IsValid")),
        "attr_isquarantined": _normalize_int(attrs.get("attr_IsQuarantined")),
        "poly_source": _normalize_str(attrs.get("poly_Source")),
        "poly_mapmethod": _normalize_str(attrs.get("poly_MapMethod")),
        "attr_firediscoverydatetime": _epoch_ms_to_dt(attrs.get("attr_FireDiscoveryDateTime")),
        "poly_polygondatetime": _epoch_ms_to_dt(attrs.get("poly_PolygonDateTime")),
        "attr_containmentdatetime": _epoch_ms_to_dt(attrs.get("attr_ContainmentDateTime")),
        "attr_controldatetime": _epoch_ms_to_dt(attrs.get("attr_ControlDateTime")),
        "tier": tier,
        "is_authoritative": bool(authoritative),
        "geom_json": json.dumps(geojson),
        "raw_attributes": json.dumps(attrs, default=str),
        "run_id": run_id,
    }


def _latest_successful_run(source_profile: str) -> dict[str, Any] | None:
    stmt = text(
        """
        SELECT run_id, source_last_edit, finished_at
        FROM authoritative_perimeter_ingest_runs
        WHERE source_profile = :source_profile
          AND status = 'succeeded'
        ORDER BY finished_at DESC NULLS LAST, started_at DESC
        LIMIT 1
        """
    )
    with get_engine().begin() as conn:
        row = conn.execute(stmt, {"source_profile": source_profile}).mappings().first()
    return dict(row) if row else None


def _service_metadata(
    client: httpx.Client,
    *,
    query_url: str,
    api_key: str | None,
    timeout_seconds: float,
) -> tuple[datetime | None, int]:
    metadata_url = query_url.rsplit("/query", 1)[0]
    params: dict[str, Any] = {"f": "json"}
    if api_key:
        params["token"] = api_key
    response = client.get(metadata_url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()

    editing = payload.get("editingInfo") or {}
    last_edit = _epoch_ms_to_dt(editing.get("lastEditDate"))
    max_record_count = int(payload.get("maxRecordCount") or 2000)
    max_record_count = max(1, min(2000, max_record_count))
    return last_edit, max_record_count


def _build_where(
    *,
    start_time: datetime | None,
    end_time: datetime | None,
    incremental_field: str,
    checkpoint_time: datetime | None,
) -> str:
    clauses = ["1=1"]
    if start_time is not None:
        clauses.append(f"attr_FireDiscoveryDateTime >= {_arcgis_date_literal(start_time)}")
    if end_time is not None:
        clauses.append(f"attr_FireDiscoveryDateTime <= {_arcgis_date_literal(end_time)}")
    if checkpoint_time is not None:
        # Keep a small overlap to avoid missing records around sync boundaries.
        anchor = checkpoint_time - timedelta(minutes=1)
        clauses.append(f"{incremental_field} >= {_arcgis_date_literal(anchor)}")
    return " AND ".join(clauses)


def _create_run(
    *,
    run_id: str,
    source_profile: str,
    source_uri: str,
    source_layer: str,
) -> None:
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
    http_429_count: int,
    max_backoff_seconds: int,
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
            http_429_count = :http_429_count,
            max_backoff_seconds = :max_backoff_seconds,
            metrics_json = CAST(:metrics_json AS json),
            error_text = :error_text,
            finished_at = NOW(),
            updated_at = NOW()
        WHERE run_id = :run_id
        """
    )
    metrics = {
        "pages": pages,
        "where_clause": where_clause,
    }
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
                "http_429_count": http_429_count,
                "max_backoff_seconds": max_backoff_seconds,
                "metrics_json": json.dumps(metrics),
                "error_text": error_text,
            },
        )


def _upsert_perimeters(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    # Authority-aware upsert: only overwrite when the incoming tier is equal
    # or higher authority (alphabetically: blocked > bronze > gold > silver,
    # but we use the WHERE clause to compare by tier rank).
    insert_stmt = text(
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

    existing_tier_stmt = text(
        """
        SELECT tier FROM authoritative_perimeters
        WHERE source_profile = :source_profile
          AND source_layer = :source_layer
          AND source_object_id = :source_object_id
        """
    )

    upserted = 0
    authority_rejected = 0
    with get_engine().begin() as conn:
        accepted: list[dict[str, Any]] = []
        for row in rows:
            result = conn.execute(
                existing_tier_stmt,
                {
                    "source_profile": row["source_profile"],
                    "source_layer": row["source_layer"],
                    "source_object_id": row["source_object_id"],
                },
            ).fetchone()
            existing_tier = result[0] if result else None

            if existing_tier is not None and not should_overwrite(
                row["tier"], existing_tier
            ):
                authority_rejected += 1
                log_authority_conflict(
                    source=row["source_profile"],
                    source_id=row["source_object_id"],
                    incoming_tier=row["tier"],
                    existing_tier=existing_tier,
                )
                record_authority_conflict(
                    conn,
                    table_name="authoritative_perimeters",
                    source=row["source_profile"],
                    source_id=row["source_object_id"],
                    incoming_tier=row["tier"],
                    existing_tier=existing_tier,
                    outcome="rejected",
                    run_id=row.get("run_id"),
                )
                continue

            if existing_tier is not None:
                record_authority_conflict(
                    conn,
                    table_name="authoritative_perimeters",
                    source=row["source_profile"],
                    source_id=row["source_object_id"],
                    incoming_tier=row["tier"],
                    existing_tier=existing_tier,
                    outcome="accepted",
                    run_id=row.get("run_id"),
                )
            accepted.append(row)

        if accepted:
            result = conn.execute(insert_stmt, accepted)
            upserted = int(result.rowcount or 0)

        if authority_rejected:
            LOGGER.warning(
                "WFIGS authority conflicts: %d records rejected (lower authority).",
                authority_rejected,
            )

    return upserted


def _fetch_features(
    client: httpx.Client,
    *,
    query_url: str,
    where_clause: str,
    bbox: tuple[float, float, float, float] | None,
    page_size: int,
    max_pages: int,
    timeout_seconds: float,
    api_key: str | None,
) -> tuple[list[dict[str, Any]], FetchStats]:
    offset = 0
    all_features: list[dict[str, Any]] = []
    stats = FetchStats()

    while True:
        params: dict[str, Any] = {
            "where": where_clause,
            "outFields": "*",
            "returnGeometry": "true",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "outSR": "4326",
            "orderByFields": "OBJECTID ASC",
        }
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            params["geometry"] = f"{min_lon},{min_lat},{max_lon},{max_lat}"
            params["geometryType"] = "esriGeometryEnvelope"
            params["spatialRel"] = "esriSpatialRelIntersects"
            params["inSR"] = "4326"
        if api_key:
            params["token"] = api_key

        retries = 0
        while True:
            response = client.get(query_url, params=params, timeout=timeout_seconds)
            if response.status_code == 429:
                retries += 1
                stats.http_429_count += 1
                backoff = min(60, 5 * retries)
                stats.max_backoff_seconds = max(stats.max_backoff_seconds, backoff)
                LOGGER.warning("WFIGS returned 429; backing off for %ss", backoff)
                time.sleep(backoff)
                continue
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(f"WFIGS query failed: {payload['error']}")
            break

        features = list(payload.get("features") or [])
        all_features.extend(features)
        stats.records_fetched += len(features)
        stats.pages += 1
        LOGGER.info(
            "Fetched page=%s size=%s total=%s",
            stats.pages,
            len(features),
            stats.records_fetched,
        )

        if not features or len(features) < page_size:
            break
        if max_pages > 0 and stats.pages >= max_pages:
            LOGGER.info("Stopping at max_pages=%s", max_pages)
            break
        offset += page_size

    return all_features, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest authoritative WFIGS perimeters.")
    parser.add_argument(
        "--source-profile",
        required=True,
        choices=sorted(WFIGS_PROFILES.keys()),
    )
    parser.add_argument("--start", default=None, help="ISO datetime lower bound on discovery time.")
    parser.add_argument("--end", default=None, help="ISO datetime upper bound on discovery time.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
    )
    parser.add_argument("--api-key", default=os.getenv("WFIGS_API_KEY", ""))
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means unlimited")
    parser.add_argument(
        "--incremental-field",
        default="poly_DateCurrent",
        choices=["poly_DateCurrent", "poly_PolygonDateTime"],
    )
    parser.add_argument(
        "--checkpoint-ms",
        type=int,
        default=None,
        help="Optional explicit incremental checkpoint (epoch milliseconds).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit incremental checkpoint (ISO datetime).",
    )
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profile_cfg = WFIGS_PROFILES[args.source_profile]
    source_uri = profile_cfg["query_url"]
    source_layer = profile_cfg["layer"]
    start_time = _parse_dt(args.start)
    end_time = _parse_dt(args.end)
    if start_time and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time and end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    if start_time and end_time and end_time < start_time:
        raise SystemExit("--end must be >= --start")

    checkpoint_time = _parse_dt(args.checkpoint)
    if checkpoint_time and checkpoint_time.tzinfo is None:
        checkpoint_time = checkpoint_time.replace(tzinfo=timezone.utc)
    if checkpoint_time is None and args.checkpoint_ms is not None:
        checkpoint_time = _epoch_ms_to_dt(args.checkpoint_ms)

    # Only auto-incremental when caller did not pin a historical window.
    # For backfills (--start/--end), default should be exact window semantics.
    auto_incremental = (start_time is None and end_time is None and not args.full_refresh)
    if checkpoint_time is None and auto_incremental:
        latest = _latest_successful_run(args.source_profile)
        if latest and latest.get("source_last_edit") is not None:
            checkpoint_time = latest["source_last_edit"]

    run_id = f"{args.source_profile}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    if not args.dry_run:
        _create_run(
            run_id=run_id,
            source_profile=args.source_profile,
            source_uri=source_uri,
            source_layer=source_layer,
        )

    where_clause = _build_where(
        start_time=start_time,
        end_time=end_time,
        incremental_field=args.incremental_field,
        checkpoint_time=checkpoint_time,
    )
    LOGGER.info("WFIGS ingest profile=%s where=%s", args.source_profile, where_clause)

    source_last_edit: datetime | None = None
    fetched = 0
    upserted = 0
    skipped = 0
    stats = FetchStats()

    try:
        with httpx.Client() as client:
            source_last_edit, max_record_count = _service_metadata(
                client,
                query_url=source_uri,
                api_key=args.api_key or None,
                timeout_seconds=float(args.timeout_seconds),
            )
            page_size = max(1, min(2000, int(args.page_size), int(max_record_count)))
            features, stats = _fetch_features(
                client,
                query_url=source_uri,
                where_clause=where_clause,
                bbox=tuple(args.bbox) if args.bbox else None,
                page_size=page_size,
                max_pages=max(0, int(args.max_pages)),
                timeout_seconds=float(args.timeout_seconds),
                api_key=args.api_key or None,
            )

        fetched = len(features)
        rows: list[dict[str, Any]] = []
        for feature in features:
            row = _extract_record(
                feature,
                source_profile=args.source_profile,
                source_layer=source_layer,
                run_id=run_id,
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
                http_429_count=stats.http_429_count,
                max_backoff_seconds=stats.max_backoff_seconds,
                pages=stats.pages,
                where_clause=where_clause,
            )

        summary = {
            "run_id": run_id,
            "source_profile": args.source_profile,
            "source_layer": source_layer,
            "source_last_edit": source_last_edit.isoformat() if source_last_edit else None,
            "records_fetched": fetched,
            "records_upserted": upserted,
            "records_skipped": skipped,
            "http_429_count": stats.http_429_count,
            "max_backoff_seconds": stats.max_backoff_seconds,
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
                http_429_count=stats.http_429_count,
                max_backoff_seconds=stats.max_backoff_seconds,
                pages=stats.pages,
                where_clause=where_clause,
                error_text=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
