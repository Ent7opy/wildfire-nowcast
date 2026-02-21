"""Event/front builder for denoiser v2.

This module deterministically maps detections to two hierarchical IDs:
- front_id: overpass-local cluster key (sensor/source + time bucket + small spatial cell)
- event_id: multi-day linkage key (sensor/source + day bucket + coarse spatial cell)

The mapping is idempotent: rerunning with the same inputs rewrites the same IDs.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("denoiser_eventize")


@dataclass(frozen=True)
class EventizeParams:
    front_time_bucket_minutes: int = 30
    front_cell_deg: float = 0.05
    event_cell_deg: float = 0.2
    event_link_days: int = 3


def _build_where_clause(
    batch_id: Optional[int],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    source_like: Optional[str],
) -> tuple[str, dict[str, object]]:
    clauses = ["TRUE"]
    params: dict[str, object] = {}

    if batch_id is not None:
        clauses.append("ingest_batch_id = :batch_id")
        params["batch_id"] = int(batch_id)

    if start_time is not None:
        clauses.append("acq_time >= :start_time")
        params["start_time"] = start_time

    if end_time is not None:
        clauses.append("acq_time <= :end_time")
        params["end_time"] = end_time

    if source_like:
        clauses.append("source LIKE :source_like")
        params["source_like"] = source_like

    return " AND ".join(clauses), params


def eventize_detections(
    engine: Engine,
    *,
    batch_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    source_like: Optional[str] = None,
    params: EventizeParams = EventizeParams(),
    dry_run: bool = False,
) -> dict[str, int]:
    """Build front/event assignments and persist mappings."""
    where_clause, query_params = _build_where_clause(batch_id, start_time, end_time, source_like)

    if batch_id is None and start_time is None and end_time is None:
        raise ValueError("Provide at least one selector: --batch-id or --start/--end")

    query_params = {
        **query_params,
        "front_time_bucket_minutes": int(params.front_time_bucket_minutes),
        "front_cell_deg": float(params.front_cell_deg),
        "event_cell_deg": float(params.event_cell_deg),
        "event_link_days": int(params.event_link_days),
    }

    create_temp_sql = text(
        f"""
        CREATE TEMP TABLE tmp_eventize_selected ON COMMIT DROP AS
        SELECT
            id,
            source,
            sensor,
            acq_time,
            lat,
            lon,
            frp,
            confidence,
            geom,
            md5(
                concat_ws(
                    '|',
                    COALESCE(source, ''),
                    COALESCE(sensor, ''),
                    floor(extract(epoch from acq_time) / (:front_time_bucket_minutes * 60))::text,
                    floor(lat / :front_cell_deg)::text,
                    floor(lon / :front_cell_deg)::text
                )
            ) AS front_id,
            md5(
                concat_ws(
                    '|',
                    COALESCE(source, ''),
                    COALESCE(sensor, ''),
                    floor(extract(epoch from acq_time) / (86400 * :event_link_days))::text,
                    floor(lat / :event_cell_deg)::text,
                    floor(lon / :event_cell_deg)::text
                )
            ) AS event_id
        FROM fire_detections
        WHERE {where_clause}
        """
    )

    count_sql = text("SELECT COUNT(*) AS n_rows, COUNT(DISTINCT front_id) AS n_fronts, COUNT(DISTINCT event_id) AS n_events FROM tmp_eventize_selected")

    front_upsert_sql = text(
        """
        INSERT INTO fire_fronts (
            front_id,
            source,
            sensor,
            overpass_start,
            overpass_end,
            detection_count,
            frp_max,
            frp_mean,
            confidence_max,
            geom,
            updated_at
        )
        SELECT
            front_id,
            MAX(source) AS source,
            MAX(sensor) AS sensor,
            MIN(acq_time) AS overpass_start,
            MAX(acq_time) AS overpass_end,
            COUNT(*)::int AS detection_count,
            MAX(frp) AS frp_max,
            AVG(frp) AS frp_mean,
            MAX(confidence) AS confidence_max,
            ST_ConvexHull(ST_Collect(geom)) AS geom,
            NOW() AS updated_at
        FROM tmp_eventize_selected
        GROUP BY front_id
        ON CONFLICT (front_id) DO UPDATE SET
            source = EXCLUDED.source,
            sensor = EXCLUDED.sensor,
            overpass_start = EXCLUDED.overpass_start,
            overpass_end = EXCLUDED.overpass_end,
            detection_count = EXCLUDED.detection_count,
            frp_max = EXCLUDED.frp_max,
            frp_mean = EXCLUDED.frp_mean,
            confidence_max = EXCLUDED.confidence_max,
            geom = EXCLUDED.geom,
            updated_at = NOW()
        """
    )

    event_upsert_sql = text(
        """
        INSERT INTO fire_events (
            event_id,
            source,
            sensor,
            start_time,
            end_time,
            detection_count,
            front_count,
            geom,
            updated_at
        )
        SELECT
            event_id,
            MAX(source) AS source,
            MAX(sensor) AS sensor,
            MIN(acq_time) AS start_time,
            MAX(acq_time) AS end_time,
            COUNT(*)::int AS detection_count,
            COUNT(DISTINCT front_id)::int AS front_count,
            ST_ConvexHull(ST_Collect(geom)) AS geom,
            NOW() AS updated_at
        FROM tmp_eventize_selected
        GROUP BY event_id
        ON CONFLICT (event_id) DO UPDATE SET
            source = EXCLUDED.source,
            sensor = EXCLUDED.sensor,
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            detection_count = EXCLUDED.detection_count,
            front_count = EXCLUDED.front_count,
            geom = EXCLUDED.geom,
            updated_at = NOW()
        """
    )

    memberships_upsert_sql = text(
        """
        INSERT INTO fire_event_memberships (
            fire_detection_id,
            front_id,
            event_id,
            member_role,
            linked_at
        )
        SELECT
            id,
            front_id,
            event_id,
            'member' AS member_role,
            NOW() AS linked_at
        FROM tmp_eventize_selected
        ON CONFLICT (fire_detection_id) DO UPDATE SET
            front_id = EXCLUDED.front_id,
            event_id = EXCLUDED.event_id,
            member_role = EXCLUDED.member_role,
            linked_at = EXCLUDED.linked_at
        """
    )

    detections_update_sql = text(
        """
        UPDATE fire_detections d
        SET
            front_id = s.front_id,
            event_id = s.event_id
        FROM tmp_eventize_selected s
        WHERE d.id = s.id
        """
    )

    with engine.begin() as conn:
        conn.execute(create_temp_sql, query_params)
        stats_row = conn.execute(count_sql).mappings().first()
        if stats_row is None:
            return {"rows": 0, "fronts": 0, "events": 0, "updated_detections": 0}

        rows = int(stats_row["n_rows"])
        fronts = int(stats_row["n_fronts"])
        events = int(stats_row["n_events"])

        if dry_run:
            return {
                "rows": rows,
                "fronts": fronts,
                "events": events,
                "updated_detections": 0,
            }

        conn.execute(front_upsert_sql)
        conn.execute(event_upsert_sql)
        conn.execute(memberships_upsert_sql)
        update_result = conn.execute(detections_update_sql)

    return {
        "rows": rows,
        "fronts": fronts,
        "events": events,
        "updated_detections": int(update_result.rowcount or 0),
    }


def _parse_dt(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event/front mappings for denoiser v2.")
    parser.add_argument("--batch-id", type=int, default=None, help="Restrict to a single ingest batch")
    parser.add_argument("--start", type=str, default=None, help="ISO datetime start")
    parser.add_argument("--end", type=str, default=None, help="ISO datetime end")
    parser.add_argument("--source-like", type=str, default=None, help="Optional SQL LIKE filter for source")
    parser.add_argument("--front-time-bucket-minutes", type=int, default=30)
    parser.add_argument("--front-cell-deg", type=float, default=0.05)
    parser.add_argument("--event-cell-deg", type=float, default=0.2)
    parser.add_argument("--event-link-days", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    params = EventizeParams(
        front_time_bucket_minutes=args.front_time_bucket_minutes,
        front_cell_deg=args.front_cell_deg,
        event_cell_deg=args.event_cell_deg,
        event_link_days=args.event_link_days,
    )

    stats = eventize_detections(
        get_engine(),
        batch_id=args.batch_id,
        start_time=_parse_dt(args.start),
        end_time=_parse_dt(args.end),
        source_like=args.source_like,
        params=params,
        dry_run=args.dry_run,
    )
    LOGGER.info("Eventize stats: %s", stats)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
