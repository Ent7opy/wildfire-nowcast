"""Ground-truth + weak-supervision labeling for denoiser v2."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.core.grid import DEFAULT_CELL_SIZE_DEG
from api.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("denoiser_label_v2")


DEFAULT_PARAMS = {
    "positive_buffer_m": 2315.0,
    "positive_time_pad_hours": 48,
    "positive_confidence_floor": 30.0,
    "negative_industrial_radius_m": 1000.0,
    "negative_far_dist_m": 10000.0,
    "negative_time_pad_days": 30,
    "negative_frp_floor_mw": 5.0,
    "chronic_static_days_threshold": 200,
    "chronic_static_window_days": 365,
    "biophysical_landcover_max": 0.1,
    "probable_positive_frp_mw": 100.0,
    "probable_positive_confidence": 70.0,
    "probable_positive_landcover_min": 0.5,
}


def _check_perimeter_coverage(
    engine: Engine,
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
) -> int:
    min_lon, min_lat, max_lon, max_lat = aoi_bbox
    stmt = text(
        """
        SELECT COUNT(*) AS n
        FROM fire_perimeters
        WHERE geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
          AND fire_start IS NOT NULL
          AND fire_start <= :end_time
          AND (fire_end IS NULL OR fire_end >= :start_time)
        """
    )
    with engine.begin() as conn:
        row = conn.execute(
            stmt,
            {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "start_time": start_time,
                "end_time": end_time,
            },
        ).mappings().first()
    return int(row["n"]) if row else 0


def _get_perimeter_coverage_bbox(
    engine: Engine,
    start_time: datetime,
    end_time: datetime,
) -> Optional[Tuple[float, float, float, float]]:
    stmt = text(
        """
        SELECT
            ST_XMin(ST_Extent(geom)) AS min_lon,
            ST_YMin(ST_Extent(geom)) AS min_lat,
            ST_XMax(ST_Extent(geom)) AS max_lon,
            ST_YMax(ST_Extent(geom)) AS max_lat
        FROM fire_perimeters
        WHERE fire_start IS NOT NULL
          AND fire_start <= :end_time
          AND (fire_end IS NULL OR fire_end >= :start_time)
        """
    )
    with engine.begin() as conn:
        row = conn.execute(
            stmt,
            {
                "start_time": start_time,
                "end_time": end_time,
            },
        ).mappings().first()
    if row and row["min_lon"] is not None:
        return (
            float(row["min_lon"]),
            float(row["min_lat"]),
            float(row["max_lon"]),
            float(row["max_lat"]),
        )
    return None


def _log_step(step: str, started_at: float, *, rows: int | None = None) -> None:
    elapsed = time.perf_counter() - started_at
    suffix = ""
    if rows is not None:
        suffix = f", rows={rows}"
    LOGGER.info("%s completed in %.3fs%s", step, elapsed, suffix)


def _label_single_window(
    engine: Engine,
    *,
    aoi_bbox: Tuple[float, float, float, float],
    coverage_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    rule_version: str,
    params: Dict,
) -> dict[str, int]:
    min_lon, min_lat, max_lon, max_lat = aoi_bbox
    cov_min_lon, cov_min_lat, cov_max_lon, cov_max_lat = coverage_bbox

    p = params
    meters_to_deg = 1.0 / 111000.0
    query_params = {
        "start_time": start_time,
        "end_time": end_time,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "cov_min_lon": cov_min_lon,
        "cov_min_lat": cov_min_lat,
        "cov_max_lon": cov_max_lon,
        "cov_max_lat": cov_max_lat,
        "positive_buffer_m": float(p["positive_buffer_m"]),
        "positive_buffer_deg": float(p["positive_buffer_m"]) * meters_to_deg,
        "positive_time_pad_hours": int(p["positive_time_pad_hours"]),
        "positive_confidence_floor": float(p["positive_confidence_floor"]),
        "negative_industrial_radius_m": float(p["negative_industrial_radius_m"]),
        "negative_industrial_radius_deg": float(p["negative_industrial_radius_m"]) * meters_to_deg,
        "negative_far_dist_m": float(p["negative_far_dist_m"]),
        "negative_far_dist_deg": float(p["negative_far_dist_m"]) * meters_to_deg,
        "negative_time_pad_days": int(p["negative_time_pad_days"]),
        "negative_frp_floor_mw": float(p["negative_frp_floor_mw"]),
        "chronic_static_days_threshold": int(p["chronic_static_days_threshold"]),
        "chronic_static_window_days": int(p["chronic_static_window_days"]),
        "grid_size": float(DEFAULT_CELL_SIZE_DEG),
        "biophysical_landcover_max": float(p["biophysical_landcover_max"]),
        "probable_positive_frp_mw": float(p["probable_positive_frp_mw"]),
        "probable_positive_confidence": float(p["probable_positive_confidence"]),
        "probable_positive_landcover_min": float(p["probable_positive_landcover_min"]),
        "rule_version": rule_version,
        "source": "ground_truth_v2",
        "rule_params": json.dumps(p),
        "labeled_at": datetime.utcnow(),
    }

    create_candidates_sql = text(
        """
        CREATE TEMP TABLE tmp_label_candidates ON COMMIT DROP AS
        SELECT
            id,
            event_id,
            lat,
            lon,
            acq_time,
            confidence,
            frp,
            landcover_score,
            geom,
            (
                lon >= :cov_min_lon
                AND lon <= :cov_max_lon
                AND lat >= :cov_min_lat
                AND lat <= :cov_max_lat
            ) AS in_coverage
        FROM fire_detections
        WHERE acq_time BETWEEN :start_time AND :end_time
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
        """
    )

    create_candidates_indexes = [
        text("CREATE UNIQUE INDEX tmp_label_candidates_id_idx ON tmp_label_candidates (id)"),
        text("CREATE INDEX tmp_label_candidates_cov_idx ON tmp_label_candidates (in_coverage)"),
    ]

    create_chronic_cells_sql = text(
        """
        CREATE TEMP TABLE tmp_label_chronic_cells ON COMMIT DROP AS
        SELECT
            floor(lat / :grid_size) AS i_lat,
            floor(lon / :grid_size) AS j_lon
        FROM fire_detections
        WHERE acq_time BETWEEN (
            :start_time - make_interval(days => :chronic_static_window_days)
        ) AND :end_time
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT date(acq_time)) >= :chronic_static_days_threshold
        """
    )

    create_chronic_idx_sql = text(
        "CREATE INDEX tmp_label_chronic_cells_idx ON tmp_label_chronic_cells (i_lat, j_lon)"
    )

    create_positive_sql = text(
        """
        CREATE TEMP TABLE tmp_label_positive_ids ON COMMIT DROP AS
        SELECT DISTINCT c.id
        FROM tmp_label_candidates c
        JOIN fire_perimeters fp
          ON fp.geom && ST_Expand(c.geom, :positive_buffer_deg)
         AND ST_DWithin(c.geom::geography, fp.geom::geography, :positive_buffer_m)
        WHERE c.in_coverage
          AND c.acq_time >= fp.fire_start - make_interval(hours => :positive_time_pad_hours)
          AND (
            fp.fire_end IS NULL
            OR c.acq_time <= fp.fire_end + make_interval(hours => :positive_time_pad_hours)
          )
          AND COALESCE(c.confidence, 0) >= :positive_confidence_floor
        """
    )

    create_industrial_sql = text(
        """
        CREATE TEMP TABLE tmp_label_industrial_ids ON COMMIT DROP AS
        SELECT DISTINCT c.id
        FROM tmp_label_candidates c
        JOIN industrial_sources i
          ON i.geom && ST_Expand(c.geom, :negative_industrial_radius_deg)
         AND ST_DWithin(c.geom::geography, i.geom::geography, :negative_industrial_radius_m)
        WHERE c.in_coverage
        """
    )

    create_far_low_sql = text(
        """
        CREATE TEMP TABLE tmp_label_far_low_ids ON COMMIT DROP AS
        SELECT c.id
        FROM tmp_label_candidates c
        WHERE c.in_coverage
          AND COALESCE(c.frp, 0) < :negative_frp_floor_mw
          AND NOT EXISTS (
            SELECT 1
            FROM fire_perimeters fp
            WHERE fp.geom && ST_Expand(c.geom, :negative_far_dist_deg)
              AND ST_DWithin(c.geom::geography, fp.geom::geography, :negative_far_dist_m)
              AND c.acq_time >= fp.fire_start - make_interval(days => :negative_time_pad_days)
              AND (
                fp.fire_end IS NULL
                OR c.acq_time <= fp.fire_end + make_interval(days => :negative_time_pad_days)
              )
          )
        """
    )

    create_chronic_ids_sql = text(
        """
        CREATE TEMP TABLE tmp_label_chronic_ids ON COMMIT DROP AS
        SELECT c.id
        FROM tmp_label_candidates c
        JOIN tmp_label_chronic_cells cc
          ON floor(c.lat / :grid_size) = cc.i_lat
         AND floor(c.lon / :grid_size) = cc.j_lon
        WHERE c.in_coverage
        """
    )

    create_negative_sql = text(
        """
        CREATE TEMP TABLE tmp_label_negative_ids ON COMMIT DROP AS
        SELECT id FROM tmp_label_industrial_ids
        UNION
        SELECT id FROM tmp_label_far_low_ids
        UNION
        SELECT id FROM tmp_label_chronic_ids
        UNION
        SELECT c.id
        FROM tmp_label_candidates c
        WHERE c.in_coverage
          AND COALESCE(c.landcover_score, 0.5) <= :biophysical_landcover_max
        """
    )

    create_probable_sql = text(
        """
        CREATE TEMP TABLE tmp_label_probable_positive_ids ON COMMIT DROP AS
        SELECT c.id
        FROM tmp_label_candidates c
        WHERE c.in_coverage
          AND COALESCE(c.frp, 0) >= :probable_positive_frp_mw
          AND COALESCE(c.confidence, 0) >= :probable_positive_confidence
          AND COALESCE(c.landcover_score, 0.5) >= :probable_positive_landcover_min
        """
    )

    create_final_sql = text(
        """
        CREATE TEMP TABLE tmp_label_final ON COMMIT DROP AS
        SELECT
            c.id AS fire_detection_id,
            c.event_id,
            CASE
                WHEN p.id IS NOT NULL THEN 'POSITIVE'
                WHEN n.id IS NOT NULL THEN 'NEGATIVE'
                WHEN pp.id IS NOT NULL THEN 'PROBABLE_POSITIVE'
                ELSE 'UNKNOWN'
            END AS label,
            CASE
                WHEN p.id IS NULL AND n.id IS NULL AND pp.id IS NOT NULL THEN TRUE
                ELSE FALSE
            END AS weak_supervision
        FROM tmp_label_candidates c
        LEFT JOIN tmp_label_positive_ids p ON p.id = c.id
        LEFT JOIN tmp_label_negative_ids n ON n.id = c.id
        LEFT JOIN tmp_label_probable_positive_ids pp ON pp.id = c.id
        """
    )

    upsert_sql = text(
        """
        INSERT INTO denoiser_labels_v2 (
            fire_detection_id,
            event_id,
            label,
            rule_version,
            source,
            rule_params,
            weak_supervision,
            labeled_at
        )
        SELECT
            fire_detection_id,
            event_id,
            label,
            :rule_version,
            :source,
            :rule_params::jsonb,
            weak_supervision,
            :labeled_at
        FROM tmp_label_final
        ON CONFLICT (fire_detection_id, rule_version) DO UPDATE SET
            event_id = EXCLUDED.event_id,
            label = EXCLUDED.label,
            source = EXCLUDED.source,
            rule_params = EXCLUDED.rule_params,
            weak_supervision = EXCLUDED.weak_supervision,
            labeled_at = EXCLUDED.labeled_at
        """
    )

    counts_sql = text(
        """
        SELECT label, COUNT(*) AS n
        FROM tmp_label_final
        GROUP BY label
        """
    )

    with engine.begin() as conn:
        started = time.perf_counter()
        conn.execute(create_candidates_sql, query_params)
        _log_step("label_v2.create_candidates", started)

        started = time.perf_counter()
        for stmt in create_candidates_indexes:
            conn.execute(stmt)
        _log_step("label_v2.candidate_indexes", started)

        total_rows = int(conn.execute(text("SELECT COUNT(*) FROM tmp_label_candidates")).scalar_one() or 0)
        if total_rows == 0:
            return {}

        started = time.perf_counter()
        conn.execute(create_chronic_cells_sql, query_params)
        conn.execute(create_chronic_idx_sql)
        _log_step("label_v2.chronic_cells", started)

        started = time.perf_counter()
        conn.execute(create_positive_sql, query_params)
        _log_step("label_v2.positive", started)

        started = time.perf_counter()
        conn.execute(create_industrial_sql, query_params)
        _log_step("label_v2.negative_industrial", started)

        started = time.perf_counter()
        conn.execute(create_far_low_sql, query_params)
        _log_step("label_v2.negative_far_low", started)

        started = time.perf_counter()
        conn.execute(create_chronic_ids_sql, query_params)
        _log_step("label_v2.negative_chronic", started)

        started = time.perf_counter()
        conn.execute(create_negative_sql, query_params)
        _log_step("label_v2.negative_union", started)

        started = time.perf_counter()
        conn.execute(create_probable_sql, query_params)
        _log_step("label_v2.probable_positive", started)

        started = time.perf_counter()
        conn.execute(create_final_sql)
        _log_step("label_v2.finalize_labels", started)

        started = time.perf_counter()
        upsert_result = conn.execute(upsert_sql, query_params)
        _log_step("label_v2.upsert", started, rows=int(upsert_result.rowcount or 0))

        rows = conn.execute(counts_sql).mappings().all()

    counts = {str(row["label"]): int(row["n"]) for row in rows}
    LOGGER.info("Label v2 counts: %s", counts)
    return counts


def label_detections_v2(
    engine: Engine,
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    *,
    rule_version: str = "v2_default",
    params: Optional[Dict] = None,
    chunk_days: int = 0,
) -> dict[str, int]:
    p = {**DEFAULT_PARAMS, **(params or {})}

    coverage_count = _check_perimeter_coverage(engine, aoi_bbox, start_time, end_time)
    if coverage_count == 0:
        raise SystemExit(
            "No perimeter coverage found for selected window. Load perimeter data before labeling v2."
        )

    coverage_bbox = _get_perimeter_coverage_bbox(engine, start_time, end_time)
    if coverage_bbox is None:
        coverage_bbox = aoi_bbox

    if int(chunk_days) <= 0:
        counts = _label_single_window(
            engine,
            aoi_bbox=aoi_bbox,
            coverage_bbox=coverage_bbox,
            start_time=start_time,
            end_time=end_time,
            rule_version=rule_version,
            params=p,
        )
        if not counts:
            raise SystemExit("No detections found in selected window for labeling v2.")
        return counts

    if start_time > end_time:
        raise ValueError("--start must be <= --end")

    totals: dict[str, int] = {}
    chunk_delta = timedelta(days=int(chunk_days))
    cursor = start_time
    chunks = 0

    while cursor <= end_time:
        chunk_end = min(end_time, cursor + chunk_delta)
        LOGGER.info("Label v2 chunk window: %s -> %s", cursor.isoformat(), chunk_end.isoformat())

        counts = _label_single_window(
            engine,
            aoi_bbox=aoi_bbox,
            coverage_bbox=coverage_bbox,
            start_time=cursor,
            end_time=chunk_end,
            rule_version=rule_version,
            params=p,
        )
        if counts:
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + int(value)

        chunks += 1
        if chunk_end >= end_time:
            break
        cursor = chunk_end + timedelta(microseconds=1)

    if not totals:
        raise SystemExit("No detections found in selected window for labeling v2.")

    totals["_chunks"] = chunks
    LOGGER.info("Label v2 chunked totals: %s", totals)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Label detections for denoiser v2.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    )
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--version", type=str, default="v2_default")
    parser.add_argument("--chunk-days", type=int, default=0, help="Optional chunking window in days")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)

    counts = label_detections_v2(
        get_engine(),
        tuple(args.bbox),
        start,
        end,
        rule_version=args.version,
        chunk_days=args.chunk_days,
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
