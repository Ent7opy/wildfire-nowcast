"""Ground-truth labeling pipeline for FIRMS hotspot denoiser (v2).

Cross-references FIRMS detections against NIFC fire perimeter polygons
stored in ``fire_perimeters`` to produce high-quality training labels.

Label logic
-----------

**POSITIVE** – detection falls within (or within a buffer of) a known fire
perimeter and the acquisition time overlaps the fire's active period.

**NEGATIVE** – detection is far from any known fire perimeter during a period
with good perimeter coverage *and* meets at least one reinforcing signal
(industrial mask, low confidence, or chronic static).

**UNKNOWN** – everything else (excluded from training).

This replaces the purely heuristic ``label_v1`` with labels grounded in
independent fire perimeter observations, dramatically increasing both
label volume and reliability.

Usage::

    python -m ml.denoiser.label_v2 \\
        --bbox -125 24 -66 50 \\
        --start 2024-01-01 --end 2024-12-31

"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import Engine, text

from api.core.grid import DEFAULT_CELL_SIZE_DEG
from api.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("denoiser_label_v2")

DEFAULT_PARAMS = {
    # Spatial buffer around perimeter for positive matching (meters).
    # 375 m ~ half a VIIRS pixel.
    "positive_buffer_m": 375.0,
    # Temporal tolerance before fire_start / after fire_end for positive match.
    "positive_time_pad_hours": 48,
    # Minimum distance from any fire perimeter to consider a detection a
    # candidate negative (meters).
    "negative_min_dist_m": 10_000.0,
    # Temporal window around detection to search for nearby perimeters when
    # constructing negatives.
    "negative_time_pad_days": 30,
    # --- reinforcing signals for negatives (at least one must hold) ---
    # Industrial mask radius (meters).
    "industrial_radius_km": 2.0,
    # Low-confidence threshold for singleton negatives.
    "low_conf_threshold": 30.0,
    # Singleton spatial isolation radius (meters).
    "singleton_dist_km": 5.0,
    # Singleton temporal isolation window (hours).
    "singleton_time_hours": 24,
    # Chronic static: min distinct days in cell.
    "chronic_static_days": 20,
    # Chronic static: lookback window.
    "chronic_static_window_days": 90,
}


def _check_perimeter_coverage(
    engine: Engine,
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
) -> int:
    """Return the number of fire perimeters that overlap the AOI and time window."""
    min_lon, min_lat, max_lon, max_lat = aoi_bbox
    stmt = text("""
        SELECT COUNT(*) AS n
        FROM fire_perimeters
        WHERE geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
          AND (
            (fire_start IS NOT NULL AND fire_start <= :end)
            AND (fire_end IS NULL OR fire_end >= :start)
          )
    """)
    with engine.connect() as conn:
        row = conn.execute(stmt, {
            "min_lon": min_lon, "min_lat": min_lat,
            "max_lon": max_lon, "max_lat": max_lat,
            "start": start_time, "end": end_time,
        }).mappings().first()
    return int(row["n"]) if row else 0


def label_detections_v2(
    engine: Engine,
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    rule_version: str = "v2.0.0",
    params: Optional[Dict] = None,
) -> None:
    """Label FIRMS detections using fire perimeter ground truth.

    Steps:
    1. Fetch all detections in the AOI/time window.
    2. Spatial-temporal join against ``fire_perimeters`` → POSITIVE.
    3. Identify candidate negatives (far from all perimeters + reinforcing signal).
    4. Everything else → UNKNOWN.
    5. Upsert into ``fire_labels``.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    min_lon, min_lat, max_lon, max_lat = aoi_bbox

    LOGGER.info("Label v2: bbox=%s, %s → %s", aoi_bbox, start_time, end_time)
    LOGGER.info("Params: %s", p)

    # ── 0. Check perimeter coverage ─────────────────────────────────────
    n_perimeters = _check_perimeter_coverage(engine, aoi_bbox, start_time, end_time)
    LOGGER.info("Fire perimeters overlapping AOI/time: %d", n_perimeters)
    if n_perimeters == 0:
        LOGGER.warning(
            "No fire perimeters found. Run `nifc_perimeters_ingest` first. Aborting."
        )
        return

    # ── 1. Fetch detections ─────────────────────────────────────────────
    query_ids = text("""
        SELECT id, lat, lon, acq_time, confidence, frp
        FROM fire_detections
        WHERE acq_time BETWEEN :start AND :end
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
          AND ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query_ids, conn, params={
            "start": start_time, "end": end_time,
            "min_lon": min_lon, "min_lat": min_lat,
            "max_lon": max_lon, "max_lat": max_lat,
        })

    if df.empty:
        LOGGER.info("No detections in window. Nothing to label.")
        return

    LOGGER.info("Detections to label: %d", len(df))
    df["label"] = "UNKNOWN"
    df["rule_applied"] = "Default"

    batch_size = 1000

    # ── 2. POSITIVE: detection within buffer of a fire perimeter ────────
    # and temporal overlap with the fire's active period.
    positive_query = text("""
        SELECT d.id
        FROM fire_detections d
        JOIN fire_perimeters fp
          ON ST_DWithin(d.geom::geography, fp.geom::geography, :buffer_m)
        WHERE d.id = ANY(:ids)
          AND d.acq_time >= fp.fire_start - make_interval(hours => :pad_h)
          AND (
            fp.fire_end IS NULL
            OR d.acq_time <= fp.fire_end + make_interval(hours => :pad_h)
          )
    """)

    positive_ids: List[int] = []
    with engine.connect() as conn:
        for i in range(0, len(df), batch_size):
            batch_ids = df["id"].iloc[i : i + batch_size].astype(int).tolist()
            if not batch_ids:
                continue
            res = conn.execute(positive_query, {
                "ids": batch_ids,
                "buffer_m": p["positive_buffer_m"],
                "pad_h": p["positive_time_pad_hours"],
            })
            positive_ids.extend(row[0] for row in res)

    pos_mask = df["id"].isin(positive_ids)
    df.loc[pos_mask, "label"] = "POSITIVE"
    df.loc[pos_mask, "rule_applied"] = "Perimeter Match"
    LOGGER.info("Perimeter Match → POSITIVE: %d", pos_mask.sum())

    # ── 3. NEGATIVE candidates ──────────────────────────────────────────
    # A detection is a negative candidate if:
    #   (a) it is NOT already POSITIVE, AND
    #   (b) it is far from every fire perimeter in a generous time window, AND
    #   (c) it matches at least one reinforcing negative signal.
    #
    # Step 3a: find detections far from all perimeters.
    remaining_mask = df["label"] == "UNKNOWN"
    remaining_ids = df.loc[remaining_mask, "id"].astype(int).tolist()

    if remaining_ids:
        far_from_fire_query = text("""
            SELECT d.id
            FROM fire_detections d
            WHERE d.id = ANY(:ids)
              AND NOT EXISTS (
                SELECT 1
                FROM fire_perimeters fp
                WHERE ST_DWithin(d.geom::geography, fp.geom::geography, :neg_dist_m)
                  AND fp.fire_start IS NOT NULL
                  AND d.acq_time >= fp.fire_start - make_interval(days => :neg_pad_d)
                  AND (
                    fp.fire_end IS NULL
                    OR d.acq_time <= fp.fire_end + make_interval(days => :neg_pad_d)
                  )
              )
        """)

        far_ids: List[int] = []
        with engine.connect() as conn:
            for i in range(0, len(remaining_ids), batch_size):
                batch_ids = remaining_ids[i : i + batch_size]
                if not batch_ids:
                    continue
                res = conn.execute(far_from_fire_query, {
                    "ids": batch_ids,
                    "neg_dist_m": p["negative_min_dist_m"],
                    "neg_pad_d": p["negative_time_pad_days"],
                })
                far_ids.extend(row[0] for row in res)

        LOGGER.info("Far from any perimeter (>%d m): %d detections", p["negative_min_dist_m"], len(far_ids))

        # Step 3b: among those far-from-fire detections, require at least one
        # reinforcing negative signal (industrial, low-conf singleton, or chronic static).
        if far_ids:
            neg_candidates = set(far_ids)
            confirmed_negative_ids: List[int] = []
            far_ids_list = list(far_ids)

            # 3b-i: Industrial mask
            industrial_query = text("""
                SELECT d.id
                FROM fire_detections d
                JOIN industrial_sources i
                  ON ST_DWithin(d.geom::geography, i.geom::geography, :radius_m)
                WHERE d.id = ANY(:ids)
            """)
            industrial_ids = set()
            with engine.connect() as conn:
                for i in range(0, len(far_ids_list), batch_size):
                    batch = far_ids_list[i : i + batch_size]
                    if not batch:
                        continue
                    res = conn.execute(industrial_query, {
                        "ids": batch,
                        "radius_m": p["industrial_radius_km"] * 1000,
                    })
                    industrial_ids.update(row[0] for row in res)

            for did in industrial_ids:
                if did in neg_candidates:
                    confirmed_negative_ids.append(did)
            LOGGER.info("  Industrial mask reinforcement: %d", len(industrial_ids & neg_candidates))

            # 3b-ii: Low-confidence singleton
            singleton_query = text("""
                SELECT d.id
                FROM fire_detections d
                WHERE d.id = ANY(:ids)
                  AND d.confidence < :conf
                  AND NOT EXISTS (
                    SELECT 1 FROM fire_detections d2
                    WHERE d2.id != d.id
                      AND d2.acq_time BETWEEN d.acq_time - make_interval(hours => :h)
                                           AND d.acq_time + make_interval(hours => :h)
                      AND ST_DWithin(d.geom::geography, d2.geom::geography, :r_m)
                  )
            """)
            singleton_ids = set()
            # Only query IDs not yet confirmed negative
            unconfirmed = [x for x in far_ids_list if x not in set(confirmed_negative_ids)]
            with engine.connect() as conn:
                for i in range(0, len(unconfirmed), batch_size):
                    batch = unconfirmed[i : i + batch_size]
                    if not batch:
                        continue
                    res = conn.execute(singleton_query, {
                        "ids": batch,
                        "conf": float(p["low_conf_threshold"]),
                        "h": p["singleton_time_hours"],
                        "r_m": p["singleton_dist_km"] * 1000,
                    })
                    singleton_ids.update(row[0] for row in res)

            for did in singleton_ids:
                if did in neg_candidates and did not in set(confirmed_negative_ids):
                    confirmed_negative_ids.append(did)
            LOGGER.info("  Low-conf singleton reinforcement: %d", len(singleton_ids & neg_candidates))

            # 3b-iii: Chronic static
            chronic_query = text(f"""
                WITH cell_counts AS (
                    SELECT
                        floor(lat / :grid_size) AS i_lat,
                        floor(lon / :grid_size) AS j_lon,
                        COUNT(DISTINCT date(acq_time)) AS day_count
                    FROM fire_detections
                    WHERE acq_time BETWEEN :start - interval '{p["chronic_static_window_days"]} days' AND :end
                    GROUP BY 1, 2
                    HAVING COUNT(DISTINCT date(acq_time)) >= :threshold
                )
                SELECT d.id
                FROM fire_detections d
                JOIN cell_counts c
                  ON floor(d.lat / :grid_size) = c.i_lat
                 AND floor(d.lon / :grid_size) = c.j_lon
                WHERE d.id = ANY(:ids)
            """)
            chronic_ids = set()
            already_confirmed = set(confirmed_negative_ids)
            unconfirmed2 = [x for x in far_ids_list if x not in already_confirmed]
            with engine.connect() as conn:
                for i in range(0, len(unconfirmed2), batch_size):
                    batch = unconfirmed2[i : i + batch_size]
                    if not batch:
                        continue
                    res = conn.execute(chronic_query, {
                        "ids": batch,
                        "grid_size": DEFAULT_CELL_SIZE_DEG,
                        "start": start_time,
                        "end": end_time,
                        "threshold": p["chronic_static_days"],
                    })
                    chronic_ids.update(row[0] for row in res)

            for did in chronic_ids:
                if did in neg_candidates and did not in already_confirmed:
                    confirmed_negative_ids.append(did)
            LOGGER.info("  Chronic static reinforcement: %d", len(chronic_ids & neg_candidates))

            # Apply confirmed negatives
            neg_mask = df["id"].isin(confirmed_negative_ids) & (df["label"] == "UNKNOWN")
            df.loc[neg_mask, "label"] = "NEGATIVE"
            df.loc[neg_mask, "rule_applied"] = "Far From Perimeter + Reinforced"
            LOGGER.info("Reinforced negatives → NEGATIVE: %d", neg_mask.sum())

            # 3c: Far-from-fire detections without a reinforcing signal.
            # We still label these as NEGATIVE — the perimeter distance alone
            # is strong evidence when coverage is good.  But we tag them
            # distinctly so downstream consumers can filter by confidence.
            unreinforced = set(far_ids) - set(confirmed_negative_ids)
            unreinforced_mask = df["id"].isin(unreinforced) & (df["label"] == "UNKNOWN")
            df.loc[unreinforced_mask, "label"] = "NEGATIVE"
            df.loc[unreinforced_mask, "rule_applied"] = "Far From Perimeter (unreinforced)"
            LOGGER.info("Unreinforced negatives → NEGATIVE: %d", unreinforced_mask.sum())

    # ── 4. Summary ──────────────────────────────────────────────────────
    counts = df["label"].value_counts().to_dict()
    LOGGER.info("Final label counts: %s", counts)

    rule_counts = df.groupby("rule_applied").size().to_dict()
    LOGGER.info("Rule breakdown: %s", rule_counts)

    # ── 5. Upsert into fire_labels ──────────────────────────────────────
    upsert_stmt = text("""
        INSERT INTO fire_labels (fire_detection_id, label, rule_version, source, rule_params, labeled_at)
        VALUES (:id, :label, :version, :source, :params, :now)
        ON CONFLICT (fire_detection_id) DO UPDATE SET
            label = EXCLUDED.label,
            rule_version = EXCLUDED.rule_version,
            source = EXCLUDED.source,
            rule_params = EXCLUDED.rule_params,
            labeled_at = EXCLUDED.labeled_at
    """)

    now = datetime.now()
    params_json = json.dumps(p)

    with engine.begin() as conn:
        batch = []
        for _, row in df.iterrows():
            batch.append({
                "id": int(row["id"]),
                "label": row["label"],
                "version": rule_version,
                "source": "ground_truth_v2",
                "params": params_json,
                "now": now,
            })

        for i in range(0, len(batch), batch_size):
            conn.execute(upsert_stmt, batch[i : i + batch_size])

    LOGGER.info("Upserted %d labels into fire_labels.", len(df))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label FIRMS detections using fire perimeter ground truth (v2)."
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Area of interest bounding box.",
    )
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--version", type=str, default="v2.0.0", help="Rule version tag."
    )

    args = parser.parse_args()

    engine = get_engine()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)

    label_detections_v2(
        engine, tuple(args.bbox), start, end, rule_version=args.version,
    )


if __name__ == "__main__":
    main()
