"""Ingest Copernicus EMS Rapid Mapping wildfire activation AOIs as silver perimeters."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import text

from ingest.perimeter_authority import (
    log_authority_conflict,
    record_authority_conflict,
    should_overwrite,
)
from ingest.repository import get_engine

LOGGER = logging.getLogger("copernicus_ems_authority_ingest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SOURCE_PROFILE = "copernicus_ems_wildfire_activations"
SOURCE_LAYER = "public_activations_aois"
SOURCE_URI = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"
LIST_ENDPOINT = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/"
DETAIL_ENDPOINT = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"


@dataclass
class FetchStats:
    activations_listed: int = 0
    activations_selected: int = 0
    activations_loaded: int = 0
    records_built: int = 0
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


def _normalize_yes(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def _create_run(*, run_id: str) -> None:
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
                "source_profile": SOURCE_PROFILE,
                "source_uri": SOURCE_URI,
                "source_layer": SOURCE_LAYER,
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
    metrics: dict[str, Any],
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
    merged = dict(metrics)
    merged["pages"] = pages
    merged["where_clause"] = where_clause
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
                "metrics_json": json.dumps(merged, default=str),
                "error_text": error_text,
            },
        )


def _upsert_perimeters(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

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
            ST_Multi(ST_SetSRID(ST_GeomFromText(:geom_wkt), 4326)),
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
                "Copernicus EMS authority conflicts: %d records rejected (lower authority).",
                authority_rejected,
            )

    return upserted


def _fetch_activation_summaries(
    client: httpx.Client,
    *,
    category: str,
    page_size: int,
    max_pages: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], FetchStats]:
    params = {"category": category, "limit": int(page_size), "offset": 0}
    stats = FetchStats()
    out: list[dict[str, Any]] = []
    while True:
        resp = client.get(LIST_ENDPOINT, params=params, timeout=timeout_seconds)
        resp.raise_for_status()
        payload = resp.json()
        chunk = list(payload.get("results") or [])
        out.extend(chunk)
        stats.activations_listed += len(chunk)
        stats.pages += 1
        LOGGER.info("Listed page=%s size=%s total=%s", stats.pages, len(chunk), stats.activations_listed)
        if not payload.get("next"):
            break
        if max_pages > 0 and stats.pages >= max_pages:
            LOGGER.info("Stopping at max_pages=%s", max_pages)
            break
        params["offset"] += int(page_size)
    return out, stats


def _fetch_activation_detail(client: httpx.Client, *, code: str, timeout_seconds: float) -> dict[str, Any] | None:
    resp = client.get(DETAIL_ENDPOINT, params={"code": code}, timeout=timeout_seconds)
    resp.raise_for_status()
    payload = resp.json()
    results = list(payload.get("results") or [])
    if not results:
        return None
    return results[0]


def _in_window(event_time: datetime | None, start: datetime | None, end: datetime | None) -> bool:
    if event_time is None:
        return False
    if start is not None and event_time < start:
        return False
    if end is not None and event_time > end:
        return False
    return True


def _build_records(
    detail: dict[str, Any],
    *,
    run_id: str,
    source_last_edit: datetime | None,
) -> list[dict[str, Any]]:
    code = str(detail.get("code") or "").strip()
    if not code:
        return []

    event_time = _parse_dt(detail.get("eventTime"))
    activation_time = _parse_dt(detail.get("activationTime"))
    last_update = _parse_dt(detail.get("lastUpdate")) or source_last_edit
    closed = bool(detail.get("closed"))
    feature_status = "Certified" if closed else "Approved"

    common = {
        "source_profile": SOURCE_PROFILE,
        "source_layer": SOURCE_LAYER,
        "poly_irwinid": code,
        "poly_sourceglobalid": None,
        "poly_featurestatus": feature_status,
        "poly_featureaccess": "Public",
        "poly_isvisible": _normalize_yes(True),
        "attr_isvalid": 1,
        "attr_isquarantined": 0,
        "poly_source": "COPERNICUS_EMS",
        "poly_mapmethod": "Rapid Mapping AOI",
        "attr_firediscoverydatetime": event_time,
        "poly_polygondatetime": activation_time,
        "attr_containmentdatetime": last_update if closed else None,
        "attr_controldatetime": last_update if closed else None,
        "tier": "silver",
        "is_authoritative": True,
        "run_id": run_id,
    }

    rows: list[dict[str, Any]] = []
    aois = list(detail.get("aois") or [])
    for aoi in aois:
        aoi_name = str(aoi.get("name") or "").strip() or "unknown"
        aoi_number = int(aoi.get("number") or 0)
        source_object_id = f"{code}:AOI:{aoi_number or aoi_name}"
        geom_wkt = str(aoi.get("extent") or "").strip()
        if not geom_wkt:
            continue
        raw_payload = {
            "activation": detail,
            "aoi": aoi,
        }
        rows.append(
            {
                **common,
                "source_object_id": source_object_id,
                "geom_wkt": geom_wkt,
                "raw_attributes": json.dumps(raw_payload, default=str),
            }
        )

    if rows:
        return rows

    extent_wkt = str(detail.get("extent") or "").strip()
    if extent_wkt:
        rows.append(
            {
                **common,
                "source_object_id": f"{code}:ACTIVATION_EXTENT",
                "geom_wkt": extent_wkt,
                "raw_attributes": json.dumps({"activation": detail}, default=str),
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Copernicus EMS wildfire activation AOIs.")
    parser.add_argument("--start", default=None, help="ISO datetime lower bound on eventTime")
    parser.add_argument("--end", default=None, help="ISO datetime upper bound on eventTime")
    parser.add_argument("--category", default="Wildfire")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start_time = _parse_dt(args.start)
    end_time = _parse_dt(args.end)
    if start_time and end_time and end_time < start_time:
        raise SystemExit("--end must be >= --start")

    run_id = f"{SOURCE_PROFILE}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    if not args.dry_run:
        _create_run(run_id=run_id)

    stats = FetchStats()
    records_fetched = 0
    records_upserted = 0
    records_skipped = 0
    source_last_edit: datetime | None = None
    where_clause = f"category={args.category}, start={args.start or ''}, end={args.end or ''}"

    try:
        with httpx.Client() as client:
            summaries, list_stats = _fetch_activation_summaries(
                client,
                category=str(args.category),
                page_size=max(1, min(200, int(args.page_size))),
                max_pages=max(0, int(args.max_pages)),
                timeout_seconds=float(args.timeout_seconds),
            )
            stats.activations_listed = list_stats.activations_listed
            stats.pages = list_stats.pages

            selected: list[dict[str, Any]] = []
            for item in summaries:
                event_time = _parse_dt(item.get("eventTime"))
                if _in_window(event_time, start_time, end_time):
                    selected.append(item)
            stats.activations_selected = len(selected)

            rows: list[dict[str, Any]] = []
            for item in selected:
                code = str(item.get("code") or "").strip()
                if not code:
                    continue
                list_last_update = _parse_dt(item.get("lastUpdate"))
                if list_last_update is not None and (source_last_edit is None or list_last_update > source_last_edit):
                    source_last_edit = list_last_update
                detail = _fetch_activation_detail(client, code=code, timeout_seconds=float(args.timeout_seconds))
                if detail is None:
                    records_skipped += 1
                    continue
                source_last_edit = _parse_dt(detail.get("lastUpdate")) or source_last_edit
                records = _build_records(detail, run_id=run_id, source_last_edit=source_last_edit)
                if not records:
                    records_skipped += 1
                    continue
                rows.extend(records)
                stats.records_built += len(records)
                stats.activations_loaded += 1

        records_fetched = stats.records_built
        if not args.dry_run:
            records_upserted = _upsert_perimeters(rows)
            _finish_run(
                run_id=run_id,
                status="succeeded",
                source_last_edit=source_last_edit,
                records_fetched=records_fetched,
                records_upserted=records_upserted,
                records_skipped=records_skipped,
                pages=stats.pages,
                metrics={
                    "activations_listed": stats.activations_listed,
                    "activations_selected": stats.activations_selected,
                    "activations_loaded": stats.activations_loaded,
                    "records_built": stats.records_built,
                },
                where_clause=where_clause,
            )

        summary = {
            "run_id": run_id,
            "source_profile": SOURCE_PROFILE,
            "source_layer": SOURCE_LAYER,
            "source_last_edit": source_last_edit.isoformat() if source_last_edit else None,
            "records_fetched": records_fetched,
            "records_upserted": records_upserted,
            "records_skipped": records_skipped,
            "pages": stats.pages,
            "activations_listed": stats.activations_listed,
            "activations_selected": stats.activations_selected,
            "activations_loaded": stats.activations_loaded,
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
                records_fetched=records_fetched,
                records_upserted=records_upserted,
                records_skipped=records_skipped,
                pages=stats.pages,
                metrics={
                    "activations_listed": stats.activations_listed,
                    "activations_selected": stats.activations_selected,
                    "activations_loaded": stats.activations_loaded,
                    "records_built": stats.records_built,
                },
                where_clause=where_clause,
                error_text=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
