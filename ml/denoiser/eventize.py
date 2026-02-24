"""Event/front association for denoiser v2.

This module deterministically maps detections to two hierarchical IDs using explicit
spatial-temporal linkage:
- front_id: overpass-local detection components
- event_id: multi-day front components

The mapping is idempotent: rerunning with the same inputs rewrites the same IDs.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    front_link_radius_m: float = 2500.0
    front_max_gap_minutes: int = 45
    event_link_radius_m: float = 10000.0
    event_max_gap_days: int = 11
    static_persistence_threshold: float = 0.85
    strict_static_split: bool = True

    def __post_init__(self) -> None:
        if float(self.front_link_radius_m) <= 0:
            raise ValueError("front_link_radius_m must be > 0")
        if int(self.front_max_gap_minutes) <= 0:
            raise ValueError("front_max_gap_minutes must be > 0")
        if float(self.event_link_radius_m) <= 0:
            raise ValueError("event_link_radius_m must be > 0")
        if int(self.event_max_gap_days) <= 0:
            raise ValueError("event_max_gap_days must be > 0")
        threshold = float(self.static_persistence_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("static_persistence_threshold must be in [0, 1]")


def _is_static_like(
    false_source_masked: bool | None,
    persistence_score: float | None,
    threshold: float,
) -> bool:
    return bool(false_source_masked) or float(persistence_score or 0.0) >= float(threshold)


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


def _log_step(step: str, started_at: float, *, rows: int | None = None) -> None:
    elapsed = time.perf_counter() - started_at
    suffix = ""
    if rows is not None:
        suffix = f", rows={rows}"
    LOGGER.info("%s completed in %.3fs%s", step, elapsed, suffix)


def _eventize_single_window(
    engine: Engine,
    *,
    batch_id: Optional[int],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    source_like: Optional[str],
    params: EventizeParams,
    dry_run: bool,
    apply_start_time: Optional[datetime] = None,
    apply_end_time: Optional[datetime] = None,
) -> dict[str, int]:
    where_clause, query_params = _build_where_clause(batch_id, start_time, end_time, source_like)

    query_params = {
        **query_params,
        "front_link_radius_m": float(params.front_link_radius_m),
        "front_max_gap_minutes": int(params.front_max_gap_minutes),
        "event_link_radius_m": float(params.event_link_radius_m),
        "event_max_gap_days": int(params.event_max_gap_days),
        "static_persistence_threshold": float(params.static_persistence_threshold),
        "strict_static_split": bool(params.strict_static_split),
        "apply_start_time": apply_start_time,
        "apply_end_time": apply_end_time,
    }

    create_selected_sql = text(
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
            COALESCE(false_source_masked, FALSE) AS false_source_masked,
            COALESCE(persistence_score, 0.0) AS persistence_score,
            (
                COALESCE(false_source_masked, FALSE)
                OR COALESCE(persistence_score, 0.0) >= :static_persistence_threshold
            ) AS is_static_like
        FROM fire_detections
        WHERE {where_clause}
        """
    )

    create_target_ids_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_target_ids ON COMMIT DROP AS
        SELECT id
        FROM tmp_eventize_selected
        WHERE (
            :apply_start_time IS NULL
            OR acq_time >= :apply_start_time
        )
          AND (
            :apply_end_time IS NULL
            OR acq_time <= :apply_end_time
          )
        """
    )

    create_front_edges_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_front_edges ON COMMIT DROP AS
        SELECT
            a.id AS left_id,
            b.id AS right_id
        FROM tmp_eventize_selected a
        JOIN tmp_eventize_selected b
          ON a.id < b.id
         AND COALESCE(a.source, '') = COALESCE(b.source, '')
         AND COALESCE(a.sensor, '') = COALESCE(b.sensor, '')
         AND ABS(EXTRACT(EPOCH FROM (a.acq_time - b.acq_time))) <= (:front_max_gap_minutes * 60)
         AND ST_DWithin(a.geom::geography, b.geom::geography, :front_link_radius_m)
         AND (
            :strict_static_split = FALSE
            OR a.is_static_like = b.is_static_like
         )
        """
    )

    create_front_components_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_front_components ON COMMIT DROP AS
        WITH RECURSIVE graph AS (
            SELECT
                s.id AS node_id,
                s.id AS root_id
            FROM tmp_eventize_selected s

            UNION

            SELECT
                e.dst_id AS node_id,
                g.root_id
            FROM graph g
            JOIN (
                SELECT left_id AS src_id, right_id AS dst_id
                FROM tmp_eventize_front_edges
                UNION ALL
                SELECT right_id AS src_id, left_id AS dst_id
                FROM tmp_eventize_front_edges
            ) e ON e.src_id = g.node_id
        ),
        normalized AS (
            SELECT
                node_id,
                MIN(root_id)::bigint AS component_anchor_id
            FROM graph
            GROUP BY node_id
        )
        SELECT
            n.node_id AS id,
            n.component_anchor_id,
            s.source,
            s.sensor,
            s.acq_time,
            s.frp,
            s.confidence,
            s.geom,
            s.is_static_like
        FROM normalized n
        JOIN tmp_eventize_selected s ON s.id = n.node_id
        """
    )

    create_front_summary_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_front_summary ON COMMIT DROP AS
        SELECT
            md5(
                concat_ws(
                    '|',
                    'front_v2',
                    COALESCE(MAX(source), ''),
                    COALESCE(MAX(sensor), ''),
                    component_anchor_id::text,
                    to_char(MIN(acq_time) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US')
                )
            ) AS front_id,
            component_anchor_id,
            MAX(source) AS source,
            MAX(sensor) AS sensor,
            MIN(acq_time) AS overpass_start,
            MAX(acq_time) AS overpass_end,
            COUNT(*)::int AS detection_count,
            MAX(frp) AS frp_max,
            AVG(frp) AS frp_mean,
            MAX(confidence) AS confidence_max,
            ST_ConvexHull(ST_Collect(geom)) AS geom,
            AVG(CASE WHEN is_static_like THEN 1.0 ELSE 0.0 END) >= 0.5 AS front_static_like
        FROM tmp_eventize_front_components
        GROUP BY component_anchor_id
        """
    )

    create_detection_front_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_detection_front ON COMMIT DROP AS
        SELECT
            c.id,
            f.front_id
        FROM tmp_eventize_front_components c
        JOIN tmp_eventize_front_summary f
          ON f.component_anchor_id = c.component_anchor_id
        """
    )

    create_front_nodes_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_front_nodes ON COMMIT DROP AS
        SELECT
            f.front_id,
            f.component_anchor_id AS front_anchor_id,
            f.source,
            f.sensor,
            f.overpass_start,
            f.overpass_end,
            f.geom,
            ST_Centroid(f.geom) AS centroid_geom,
            f.front_static_like
        FROM tmp_eventize_front_summary f
        """
    )

    create_event_edges_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_event_edges ON COMMIT DROP AS
        SELECT
            a.front_id AS left_front_id,
            b.front_id AS right_front_id
        FROM tmp_eventize_front_nodes a
        JOIN tmp_eventize_front_nodes b
          ON a.front_id < b.front_id
         AND COALESCE(a.source, '') = COALESCE(b.source, '')
         AND COALESCE(a.sensor, '') = COALESCE(b.sensor, '')
         AND a.overpass_start <= b.overpass_end + make_interval(days => :event_max_gap_days)
         AND b.overpass_start <= a.overpass_end + make_interval(days => :event_max_gap_days)
         AND ST_DWithin(a.centroid_geom::geography, b.centroid_geom::geography, :event_link_radius_m)
         AND (
            :strict_static_split = FALSE
            OR a.front_static_like = b.front_static_like
         )
        """
    )

    create_event_edges_undir_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_event_edges_undir ON COMMIT DROP AS
        SELECT left_front_id AS src_front_id, right_front_id AS dst_front_id
        FROM tmp_eventize_event_edges
        UNION ALL
        SELECT right_front_id AS src_front_id, left_front_id AS dst_front_id
        FROM tmp_eventize_event_edges
        """
    )

    create_event_components_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_event_components ON COMMIT DROP AS
        WITH RECURSIVE graph AS (
            SELECT
                n.front_id AS node_front_id,
                n.front_anchor_id AS root_anchor_id
            FROM tmp_eventize_front_nodes n

            UNION

            SELECT
                e.dst_front_id AS node_front_id,
                g.root_anchor_id
            FROM graph g
            JOIN tmp_eventize_event_edges_undir e ON e.src_front_id = g.node_front_id
        )
        SELECT
            node_front_id AS front_id,
            MIN(root_anchor_id)::bigint AS component_anchor_id
        FROM graph
        GROUP BY node_front_id
        """
    )

    create_event_summary_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_event_summary ON COMMIT DROP AS
        WITH component_rows AS (
            SELECT
                ec.front_id,
                ec.component_anchor_id,
                fn.source,
                fn.sensor,
                fn.overpass_start,
                fn.overpass_end,
                fn.geom,
                fn.front_static_like
            FROM tmp_eventize_event_components ec
            JOIN tmp_eventize_front_nodes fn ON fn.front_id = ec.front_id
        ),
        event_ids AS (
            SELECT
                component_anchor_id,
                md5(
                    concat_ws(
                        '|',
                        'event_v2',
                        COALESCE(MAX(source), ''),
                        COALESCE(MAX(sensor), ''),
                        component_anchor_id::text,
                        to_char(MIN(overpass_start) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US')
                    )
                ) AS event_id
            FROM component_rows
            GROUP BY component_anchor_id
        )
        SELECT
            e.event_id,
            c.component_anchor_id,
            MAX(c.source) AS source,
            MAX(c.sensor) AS sensor,
            MIN(c.overpass_start) AS start_time,
            MAX(c.overpass_end) AS end_time,
            COUNT(df.id)::int AS detection_count,
            COUNT(DISTINCT c.front_id)::int AS front_count,
            ST_ConvexHull(ST_Collect(c.geom)) AS geom,
            AVG(CASE WHEN c.front_static_like THEN 1.0 ELSE 0.0 END) AS static_front_ratio
        FROM component_rows c
        JOIN event_ids e ON e.component_anchor_id = c.component_anchor_id
        LEFT JOIN tmp_eventize_detection_front df ON df.front_id = c.front_id
        GROUP BY e.event_id, c.component_anchor_id
        """
    )

    create_front_event_map_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_front_event_map ON COMMIT DROP AS
        SELECT
            ec.front_id,
            es.event_id
        FROM tmp_eventize_event_components ec
        JOIN tmp_eventize_event_summary es
          ON es.component_anchor_id = ec.component_anchor_id
        """
    )

    create_detection_event_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_detection_event ON COMMIT DROP AS
        SELECT
            df.id,
            df.front_id,
            fe.event_id
        FROM tmp_eventize_detection_front df
        JOIN tmp_eventize_front_event_map fe
          ON fe.front_id = df.front_id
        """
    )

    create_assignments_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_assignments ON COMMIT DROP AS
        SELECT
            de.id,
            de.front_id,
            de.event_id
        FROM tmp_eventize_detection_event de
        JOIN tmp_eventize_target_ids t ON t.id = de.id
        """
    )

    create_target_fronts_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_target_fronts ON COMMIT DROP AS
        SELECT DISTINCT front_id
        FROM tmp_eventize_assignments
        """
    )

    create_target_events_sql = text(
        """
        CREATE TEMP TABLE tmp_eventize_target_events ON COMMIT DROP AS
        SELECT DISTINCT event_id
        FROM tmp_eventize_assignments
        """
    )

    count_sql = text(
        """
        SELECT
            (SELECT COUNT(*) FROM tmp_eventize_assignments) AS n_rows,
            (SELECT COUNT(*) FROM tmp_eventize_target_fronts) AS n_fronts,
            (SELECT COUNT(*) FROM tmp_eventize_target_events) AS n_events
        """
    )

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
            f.front_id,
            f.source,
            f.sensor,
            f.overpass_start,
            f.overpass_end,
            f.detection_count,
            f.frp_max,
            f.frp_mean,
            f.confidence_max,
            f.geom,
            NOW() AS updated_at
        FROM tmp_eventize_front_summary f
        JOIN tmp_eventize_target_fronts tf ON tf.front_id = f.front_id
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
        WHERE
            fire_fronts.source IS DISTINCT FROM EXCLUDED.source
            OR fire_fronts.sensor IS DISTINCT FROM EXCLUDED.sensor
            OR fire_fronts.overpass_start IS DISTINCT FROM EXCLUDED.overpass_start
            OR fire_fronts.overpass_end IS DISTINCT FROM EXCLUDED.overpass_end
            OR fire_fronts.detection_count IS DISTINCT FROM EXCLUDED.detection_count
            OR fire_fronts.frp_max IS DISTINCT FROM EXCLUDED.frp_max
            OR fire_fronts.frp_mean IS DISTINCT FROM EXCLUDED.frp_mean
            OR fire_fronts.confidence_max IS DISTINCT FROM EXCLUDED.confidence_max
            OR fire_fronts.geom IS DISTINCT FROM EXCLUDED.geom
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
            e.event_id,
            e.source,
            e.sensor,
            e.start_time,
            e.end_time,
            e.detection_count,
            e.front_count,
            e.geom,
            NOW() AS updated_at
        FROM tmp_eventize_event_summary e
        JOIN tmp_eventize_target_events te ON te.event_id = e.event_id
        ON CONFLICT (event_id) DO UPDATE SET
            source = EXCLUDED.source,
            sensor = EXCLUDED.sensor,
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            detection_count = EXCLUDED.detection_count,
            front_count = EXCLUDED.front_count,
            geom = EXCLUDED.geom,
            updated_at = NOW()
        WHERE
            fire_events.source IS DISTINCT FROM EXCLUDED.source
            OR fire_events.sensor IS DISTINCT FROM EXCLUDED.sensor
            OR fire_events.start_time IS DISTINCT FROM EXCLUDED.start_time
            OR fire_events.end_time IS DISTINCT FROM EXCLUDED.end_time
            OR fire_events.detection_count IS DISTINCT FROM EXCLUDED.detection_count
            OR fire_events.front_count IS DISTINCT FROM EXCLUDED.front_count
            OR fire_events.geom IS DISTINCT FROM EXCLUDED.geom
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
        FROM tmp_eventize_assignments
        ON CONFLICT (fire_detection_id) DO UPDATE SET
            front_id = EXCLUDED.front_id,
            event_id = EXCLUDED.event_id,
            member_role = EXCLUDED.member_role,
            linked_at = EXCLUDED.linked_at
        WHERE
            fire_event_memberships.front_id IS DISTINCT FROM EXCLUDED.front_id
            OR fire_event_memberships.event_id IS DISTINCT FROM EXCLUDED.event_id
            OR fire_event_memberships.member_role IS DISTINCT FROM EXCLUDED.member_role
        """
    )

    detections_update_sql = text(
        """
        UPDATE fire_detections d
        SET
            front_id = a.front_id,
            event_id = a.event_id
        FROM tmp_eventize_assignments a
        WHERE d.id = a.id
          AND (
            d.front_id IS DISTINCT FROM a.front_id
            OR d.event_id IS DISTINCT FROM a.event_id
          )
        """
    )

    create_temp_indexes_sql = [
        text("CREATE UNIQUE INDEX tmp_eventize_selected_id_idx ON tmp_eventize_selected (id)"),
        text("CREATE INDEX tmp_eventize_selected_src_sensor_time_idx ON tmp_eventize_selected (source, sensor, acq_time)"),
        text("CREATE INDEX tmp_eventize_selected_static_idx ON tmp_eventize_selected (is_static_like)"),
        text("CREATE INDEX tmp_eventize_selected_geog_idx ON tmp_eventize_selected USING gist ((geom::geography))"),
    ]

    post_build_indexes_sql = [
        text("CREATE UNIQUE INDEX tmp_eventize_target_ids_idx ON tmp_eventize_target_ids (id)"),
        text("CREATE INDEX tmp_eventize_front_edges_left_idx ON tmp_eventize_front_edges (left_id)"),
        text("CREATE INDEX tmp_eventize_front_edges_right_idx ON tmp_eventize_front_edges (right_id)"),
        text("CREATE UNIQUE INDEX tmp_eventize_front_summary_id_idx ON tmp_eventize_front_summary (front_id)"),
        text("CREATE UNIQUE INDEX tmp_eventize_detection_front_id_idx ON tmp_eventize_detection_front (id)"),
        text("CREATE UNIQUE INDEX tmp_eventize_front_nodes_id_idx ON tmp_eventize_front_nodes (front_id)"),
        text("CREATE INDEX tmp_eventize_event_edges_left_idx ON tmp_eventize_event_edges (left_front_id)"),
        text("CREATE INDEX tmp_eventize_event_edges_right_idx ON tmp_eventize_event_edges (right_front_id)"),
        text("CREATE INDEX tmp_eventize_event_edges_undir_src_idx ON tmp_eventize_event_edges_undir (src_front_id)"),
        text("CREATE UNIQUE INDEX tmp_eventize_event_summary_id_idx ON tmp_eventize_event_summary (event_id)"),
        text("CREATE UNIQUE INDEX tmp_eventize_detection_event_id_idx ON tmp_eventize_detection_event (id)"),
        text("CREATE UNIQUE INDEX tmp_eventize_assignments_id_idx ON tmp_eventize_assignments (id)"),
        text("CREATE UNIQUE INDEX tmp_eventize_target_fronts_idx ON tmp_eventize_target_fronts (front_id)"),
        text("CREATE UNIQUE INDEX tmp_eventize_target_events_idx ON tmp_eventize_target_events (event_id)"),
    ]

    with engine.begin() as conn:
        started = time.perf_counter()
        conn.execute(create_selected_sql, query_params)
        _log_step("eventize.create_selected", started)

        started = time.perf_counter()
        for stmt in create_temp_indexes_sql:
            conn.execute(stmt)
        _log_step("eventize.selected_indexes", started)

        started = time.perf_counter()
        conn.execute(create_target_ids_sql, query_params)
        conn.execute(create_front_edges_sql, query_params)
        conn.execute(create_front_components_sql)
        conn.execute(create_front_summary_sql)
        conn.execute(create_detection_front_sql)
        conn.execute(create_front_nodes_sql)
        conn.execute(create_event_edges_sql, query_params)
        conn.execute(create_event_edges_undir_sql)
        conn.execute(create_event_components_sql)
        conn.execute(create_event_summary_sql)
        conn.execute(create_front_event_map_sql)
        conn.execute(create_detection_event_sql)
        conn.execute(create_assignments_sql)
        conn.execute(create_target_fronts_sql)
        conn.execute(create_target_events_sql)
        _log_step("eventize.build_components", started)

        started = time.perf_counter()
        for stmt in post_build_indexes_sql:
            conn.execute(stmt)
        _log_step("eventize.component_indexes", started)

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

        started = time.perf_counter()
        front_result = conn.execute(front_upsert_sql)
        _log_step("eventize.upsert_fronts", started, rows=int(front_result.rowcount or 0))

        started = time.perf_counter()
        event_result = conn.execute(event_upsert_sql)
        _log_step("eventize.upsert_events", started, rows=int(event_result.rowcount or 0))

        started = time.perf_counter()
        member_result = conn.execute(memberships_upsert_sql)
        _log_step("eventize.upsert_memberships", started, rows=int(member_result.rowcount or 0))

        started = time.perf_counter()
        update_result = conn.execute(detections_update_sql)
        _log_step("eventize.update_detections", started, rows=int(update_result.rowcount or 0))

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


def eventize_detections(
    engine: Engine,
    *,
    batch_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    source_like: Optional[str] = None,
    params: EventizeParams = EventizeParams(),
    dry_run: bool = False,
    chunk_days: int = 0,
) -> dict[str, int]:
    """Build front/event assignments and persist mappings."""
    if batch_id is None and start_time is None and end_time is None:
        raise ValueError("Provide at least one selector: --batch-id or --start/--end")

    if batch_id is not None or start_time is None or end_time is None or int(chunk_days) <= 0:
        return _eventize_single_window(
            engine,
            batch_id=batch_id,
            start_time=start_time,
            end_time=end_time,
            source_like=source_like,
            params=params,
            dry_run=dry_run,
        )

    if start_time > end_time:
        raise ValueError("--start must be <= --end")

    chunk_delta = timedelta(days=int(chunk_days))
    link_back_delta = timedelta(days=int(params.event_max_gap_days))
    cursor = start_time

    totals = {
        "rows": 0,
        "fronts": 0,
        "events": 0,
        "updated_detections": 0,
        "chunks": 0,
    }

    while cursor <= end_time:
        chunk_end = min(end_time, cursor + chunk_delta)
        expanded_start = max(start_time, cursor - link_back_delta)
        LOGGER.info(
            "Eventize chunk window: expanded=%s -> %s, apply=%s -> %s",
            expanded_start.isoformat(),
            chunk_end.isoformat(),
            cursor.isoformat(),
            chunk_end.isoformat(),
        )
        chunk_stats = _eventize_single_window(
            engine,
            batch_id=None,
            start_time=expanded_start,
            end_time=chunk_end,
            source_like=source_like,
            params=params,
            dry_run=dry_run,
            apply_start_time=cursor,
            apply_end_time=chunk_end,
        )
        totals["rows"] += int(chunk_stats.get("rows", 0))
        totals["fronts"] += int(chunk_stats.get("fronts", 0))
        totals["events"] += int(chunk_stats.get("events", 0))
        totals["updated_detections"] += int(chunk_stats.get("updated_detections", 0))
        totals["chunks"] += 1

        if chunk_end >= end_time:
            break
        cursor = chunk_end + timedelta(microseconds=1)

    LOGGER.info("Eventize chunked totals: %s", totals)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event/front mappings for denoiser v2.")
    parser.add_argument("--batch-id", type=int, default=None, help="Restrict to a single ingest batch")
    parser.add_argument("--start", type=str, default=None, help="ISO datetime start")
    parser.add_argument("--end", type=str, default=None, help="ISO datetime end")
    parser.add_argument("--source-like", type=str, default=None, help="Optional SQL LIKE filter for source")
    parser.add_argument("--event-front-radius-m", type=float, default=2500.0)
    parser.add_argument("--event-front-max-gap-minutes", type=int, default=45)
    parser.add_argument("--event-link-radius-m", type=float, default=10000.0)
    parser.add_argument("--event-link-max-gap-days", type=int, default=11)
    parser.add_argument("--event-static-persistence-threshold", type=float, default=0.85)
    parser.add_argument("--event-strict-static-split", action="store_true")
    parser.add_argument("--no-event-strict-static-split", action="store_true")
    parser.add_argument("--chunk-days", type=int, default=0, help="Optional chunking window in days")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    strict_static_split = True
    if args.no_event_strict_static_split:
        strict_static_split = False
    elif args.event_strict_static_split:
        strict_static_split = True

    params = EventizeParams(
        front_link_radius_m=float(args.event_front_radius_m),
        front_max_gap_minutes=int(args.event_front_max_gap_minutes),
        event_link_radius_m=float(args.event_link_radius_m),
        event_max_gap_days=int(args.event_link_max_gap_days),
        static_persistence_threshold=float(args.event_static_persistence_threshold),
        strict_static_split=bool(strict_static_split),
    )

    stats = eventize_detections(
        get_engine(),
        batch_id=args.batch_id,
        start_time=_parse_dt(args.start),
        end_time=_parse_dt(args.end),
        source_like=args.source_like,
        params=params,
        dry_run=args.dry_run,
        chunk_days=args.chunk_days,
    )
    LOGGER.info("Eventize stats: %s", stats)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
