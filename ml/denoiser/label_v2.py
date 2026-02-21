"""Ground-truth + weak-supervision labeling for denoiser v2."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
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


def label_detections_v2(
    engine: Engine,
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    *,
    rule_version: str = "v2_default",
    params: Optional[Dict] = None,
) -> dict[str, int]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    min_lon, min_lat, max_lon, max_lat = aoi_bbox

    coverage_count = _check_perimeter_coverage(engine, aoi_bbox, start_time, end_time)
    if coverage_count == 0:
        raise SystemExit(
            "No perimeter coverage found for selected window. Load perimeter data before labeling v2."
        )

    coverage_bbox = _get_perimeter_coverage_bbox(engine, start_time, end_time)
    if coverage_bbox is None:
        coverage_bbox = aoi_bbox

    detection_query = text(
        """
        SELECT
            id,
            event_id,
            lat,
            lon,
            acq_time,
            confidence,
            frp,
            landcover_score
        FROM fire_detections
        WHERE acq_time BETWEEN :start_time AND :end_time
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
        """
    )

    with engine.begin() as conn:
        df = pd.read_sql(
            detection_query,
            conn,
            params={
                "start_time": start_time,
                "end_time": end_time,
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            },
        )

    if df.empty:
        raise SystemExit("No detections found in selected window for labeling v2.")

    df["label"] = "UNKNOWN"
    df["weak_supervision"] = False

    # Keep unknown outside trusted perimeter-coverage region.
    cov_min_lon, cov_min_lat, cov_max_lon, cov_max_lat = coverage_bbox
    in_coverage = (
        (df["lon"] >= cov_min_lon)
        & (df["lon"] <= cov_max_lon)
        & (df["lat"] >= cov_min_lat)
        & (df["lat"] <= cov_max_lat)
    )

    positive_query = text(
        """
        SELECT DISTINCT d.id
        FROM fire_detections d
        JOIN fire_perimeters fp
          ON ST_DWithin(d.geom::geography, fp.geom::geography, :buffer_m)
        WHERE d.id = ANY(:ids)
          AND d.acq_time >= fp.fire_start - make_interval(hours => :pad_h)
          AND (fp.fire_end IS NULL OR d.acq_time <= fp.fire_end + make_interval(hours => :pad_h))
          AND COALESCE(d.confidence, 0) >= :confidence_floor
        """
    )

    candidates = df.loc[in_coverage, "id"].astype(int).tolist()
    positive_ids: set[int] = set()

    with engine.begin() as conn:
        for i in range(0, len(candidates), 2000):
            chunk = candidates[i : i + 2000]
            if not chunk:
                continue
            rows = conn.execute(
                positive_query,
                {
                    "ids": chunk,
                    "buffer_m": float(p["positive_buffer_m"]),
                    "pad_h": int(p["positive_time_pad_hours"]),
                    "confidence_floor": float(p["positive_confidence_floor"]),
                },
            )
            positive_ids.update(int(r[0]) for r in rows)

    df.loc[df["id"].isin(positive_ids), "label"] = "POSITIVE"

    industrial_query = text(
        """
        SELECT d.id
        FROM fire_detections d
        WHERE d.id = ANY(:ids)
          AND EXISTS (
            SELECT 1
            FROM industrial_sources i
            WHERE ST_DWithin(d.geom::geography, i.geom::geography, :radius_m)
          )
        """
    )

    far_low_frp_query = text(
        """
        SELECT d.id
        FROM fire_detections d
        WHERE d.id = ANY(:ids)
          AND COALESCE(d.frp, 0) < :frp_floor
          AND NOT EXISTS (
            SELECT 1
            FROM fire_perimeters fp
            WHERE ST_DWithin(d.geom::geography, fp.geom::geography, :far_dist_m)
              AND d.acq_time >= fp.fire_start - make_interval(days => :pad_d)
              AND (fp.fire_end IS NULL OR d.acq_time <= fp.fire_end + make_interval(days => :pad_d))
          )
        """
    )

    chronic_query = text(
        f"""
        WITH chronic_cells AS (
            SELECT
                floor(lat / :grid_size) AS i_lat,
                floor(lon / :grid_size) AS j_lon,
                COUNT(DISTINCT date(acq_time)) AS n_days
            FROM fire_detections
            WHERE acq_time BETWEEN :start_time - interval '{int(p["chronic_static_window_days"])} days' AND :end_time
            GROUP BY 1, 2
            HAVING COUNT(DISTINCT date(acq_time)) >= :threshold_days
        )
        SELECT d.id
        FROM fire_detections d
        JOIN chronic_cells c
          ON floor(d.lat / :grid_size) = c.i_lat
         AND floor(d.lon / :grid_size) = c.j_lon
        WHERE d.id = ANY(:ids)
        """
    )

    negative_candidates = df.loc[(df["label"] == "UNKNOWN") & in_coverage, "id"].astype(int).tolist()
    negative_ids: set[int] = set()

    with engine.begin() as conn:
        for i in range(0, len(negative_candidates), 2000):
            chunk = negative_candidates[i : i + 2000]
            if not chunk:
                continue

            for row in conn.execute(
                industrial_query,
                {
                    "ids": chunk,
                    "radius_m": float(p["negative_industrial_radius_m"]),
                },
            ):
                negative_ids.add(int(row[0]))

            for row in conn.execute(
                far_low_frp_query,
                {
                    "ids": chunk,
                    "frp_floor": float(p["negative_frp_floor_mw"]),
                    "far_dist_m": float(p["negative_far_dist_m"]),
                    "pad_d": int(p["negative_time_pad_days"]),
                },
            ):
                negative_ids.add(int(row[0]))

            for row in conn.execute(
                chronic_query,
                {
                    "ids": chunk,
                    "start_time": start_time,
                    "end_time": end_time,
                    "grid_size": float(DEFAULT_CELL_SIZE_DEG),
                    "threshold_days": int(p["chronic_static_days_threshold"]),
                },
            ):
                negative_ids.add(int(row[0]))

    # Biophysical impossibility fallback based on low land-cover plausibility signal.
    biophysical_ids = set(
        df.loc[
            (df["label"] == "UNKNOWN")
            & in_coverage
            & (df["landcover_score"].fillna(0.5) <= float(p["biophysical_landcover_max"])),
            "id",
        ].astype(int)
    )
    negative_ids |= biophysical_ids

    df.loc[df["id"].isin(negative_ids) & (df["label"] == "UNKNOWN"), "label"] = "NEGATIVE"

    probable_positive_mask = (
        (df["label"] == "UNKNOWN")
        & in_coverage
        & (df["frp"].fillna(0) >= float(p["probable_positive_frp_mw"]))
        & (df["confidence"].fillna(0) >= float(p["probable_positive_confidence"]))
        & (df["landcover_score"].fillna(0.5) >= float(p["probable_positive_landcover_min"]))
    )
    df.loc[probable_positive_mask, "label"] = "PROBABLE_POSITIVE"
    df.loc[probable_positive_mask, "weak_supervision"] = True

    upsert_stmt = text(
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
        VALUES (
            :fire_detection_id,
            :event_id,
            :label,
            :rule_version,
            :source,
            :rule_params,
            :weak_supervision,
            :labeled_at
        )
        ON CONFLICT (fire_detection_id, rule_version) DO UPDATE SET
            event_id = EXCLUDED.event_id,
            label = EXCLUDED.label,
            source = EXCLUDED.source,
            rule_params = EXCLUDED.rule_params,
            weak_supervision = EXCLUDED.weak_supervision,
            labeled_at = EXCLUDED.labeled_at
        """
    )

    now = datetime.utcnow()
    payload = json.dumps(p)
    rows = [
        {
            "fire_detection_id": int(row.id),
            "event_id": row.event_id,
            "label": row.label,
            "rule_version": rule_version,
            "source": "ground_truth_v2",
            "rule_params": payload,
            "weak_supervision": bool(row.weak_supervision),
            "labeled_at": now,
        }
        for row in df.itertuples(index=False)
    ]

    with engine.begin() as conn:
        for i in range(0, len(rows), 2000):
            conn.execute(upsert_stmt, rows[i : i + 2000])

    counts = {k: int(v) for k, v in df["label"].value_counts(dropna=False).to_dict().items()}
    LOGGER.info("Label v2 counts: %s", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Label detections for denoiser v2.")
    parser.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--version", type=str, default="v2_default")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)

    counts = label_detections_v2(
        get_engine(),
        tuple(args.bbox),
        start,
        end,
        rule_version=args.version,
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
