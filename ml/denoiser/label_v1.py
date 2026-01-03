"""DB-first labeling pipeline for FIRMS hotspot denoiser (v1)."""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text, Engine
from api.db import get_engine
from api.core.grid import DEFAULT_CELL_SIZE_DEG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("denoiser_label_v1")

# Default Parameters from docs/denoiser_labels.md
DEFAULT_PARAMS = {
    "industrial_radius_km": 2.0,
    "chronic_static_days": 20,
    "chronic_static_window_days": 90,
    "persistence_window_hours": 72,
    "cluster_dist_km": 2.0,
    "cluster_time_hours": 24,
    "low_conf_threshold": 30.0,
    "singleton_dist_km": 5.0,
    "singleton_time_hours": 24,
}

def label_detections_v1(
    engine: Engine,
    aoi_bbox: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    rule_version: str = "v1.0.0",
    params: Optional[Dict] = None
):
    """Run labeling heuristics and upsert into fire_labels."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    min_lon, min_lat, max_lon, max_lat = aoi_bbox

    LOGGER.info(f"Starting labeling for {aoi_bbox} from {start_time} to {end_time}")
    LOGGER.info(f"Rule version: {rule_version}, params: {p}")

    # 1. Preflight: confirm we actually have detections in this window.
    preflight_stmt = text(
        """
        SELECT
            COUNT(*) AS n,
            MIN(acq_time) AS min_time,
            MAX(acq_time) AS max_time
        FROM fire_detections
        WHERE acq_time BETWEEN :start AND :end
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
          AND ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            preflight_stmt,
            {
                "start": start_time,
                "end": end_time,
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            },
        ).mappings().first()

    n = int(row["n"]) if row and row["n"] is not None else 0
    if n == 0:
        LOGGER.info(
            "No detections found in the specified range. Common causes: "
            "(1) FIRMS *_NRT sources typically retain only ~7 days, so historical windows return 0 rows; "
            "(2) your ingest day-range/window doesn't overlap the label window; "
            "(3) bbox mismatch. Re-run ingest for a recent 7-day window to validate plumbing, or ingest historical detections via an archive flow."
        )
        return

    LOGGER.info(
        "Preflight: found %s detections in range (min_acq_time=%s, max_acq_time=%s).",
        n,
        row["min_time"],
        row["max_time"],
    )

    # 2. Fetch detection IDs in scope
    query_ids = text("""
        SELECT id, lat, lon, acq_time, confidence, frp 
        FROM fire_detections
        WHERE acq_time BETWEEN :start AND :end
          AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
          AND ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query_ids, conn, params={
            "start": start_time,
            "end": end_time,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat
        })

    LOGGER.info(f"Labeling {len(df)} detections...")

    # Initialize labels as UNKNOWN
    df["label"] = "UNKNOWN"
    df["rule_applied"] = "Default"

    # 2. Rule: Industrial Mask (NEGATIVE)
    # We do this via a spatial join for efficiency
    with engine.connect() as conn:
        industrial_query = text("""
            SELECT d.id
            FROM fire_detections d
            JOIN industrial_sources i ON ST_DWithin(d.geom::geography, i.geom::geography, :radius_m)
            WHERE d.id = ANY(:ids)
        """)
        # Batch the IDs to avoid query length limits
        batch_size = 1000
        industrial_ids = []
        for i in range(0, len(df), batch_size):
            batch_ids = df["id"].iloc[i:i+batch_size].astype(int).tolist()
            if batch_ids:
                res = conn.execute(industrial_query, {"radius_m": p["industrial_radius_km"] * 1000, "ids": batch_ids})
                industrial_ids.extend([row[0] for row in res])
        
        df.loc[df["id"].isin(industrial_ids), "label"] = "NEGATIVE"
        df.loc[df["id"].isin(industrial_ids), "rule_applied"] = "Industrial Mask"
        LOGGER.info(f"Industrial Mask: marked {len(industrial_ids)} as NEGATIVE")

    # 3. Rule: Low-Conf Singleton (NEGATIVE)
    # confidence < threshold AND no other detections within (5km, 24h)
    low_conf_mask = (df["label"] == "UNKNOWN") & (df["confidence"] < p["low_conf_threshold"])
    if low_conf_mask.any():
        low_conf_ids = df.loc[low_conf_mask, "id"].tolist()
        singleton_ids: List[int] = []
        # Set-based SQL (batched) instead of one query per detection id.
        with engine.connect() as conn:
            singleton_query = text("""
                SELECT d.id
                FROM fire_detections d
                WHERE d.id = ANY(:ids)
                  AND d.confidence < :conf
                  AND NOT EXISTS (
                    SELECT 1 FROM fire_detections d2
                    WHERE d2.id != d.id
                      AND d2.acq_time BETWEEN d.acq_time - make_interval(hours => :h) AND d.acq_time + make_interval(hours => :h)
                      AND ST_DWithin(d.geom::geography, d2.geom::geography, :r_m)
                  )
            """)
            for i in range(0, len(low_conf_ids), batch_size):
                batch_ids = [int(x) for x in low_conf_ids[i : i + batch_size]]
                if not batch_ids:
                    continue
                res = conn.execute(
                    singleton_query,
                    {
                        "ids": batch_ids,
                        "conf": float(p["low_conf_threshold"]),
                        "h": p["singleton_time_hours"],
                        "r_m": p["singleton_dist_km"] * 1000,
                    },
                )
                singleton_ids.extend([row[0] for row in res])
        
        df.loc[df["id"].isin(singleton_ids), "label"] = "NEGATIVE"
        df.loc[df["id"].isin(singleton_ids), "rule_applied"] = "Low-Conf Singleton"
        LOGGER.info(f"Low-Conf Singleton: marked {len(singleton_ids)} as NEGATIVE")

    # 4. Rule: Chronic Static (NEGATIVE)
    # same cell on >= 20 distinct days in 90 days AND no adjacent cell detections
    # This is more complex. We'll simplify for v1: just check cell count.
    with engine.connect() as conn:
        chronic_query = text(f"""
            WITH cell_counts AS (
                SELECT 
                    floor(lat / :grid_size) as i_lat,
                    floor(lon / :grid_size) as j_lon,
                    COUNT(DISTINCT date(acq_time)) as day_count
                FROM fire_detections
                WHERE acq_time BETWEEN :start - interval '{p["chronic_static_window_days"]} days' AND :end
                GROUP BY 1, 2
                HAVING COUNT(DISTINCT date(acq_time)) >= :threshold
            )
            SELECT d.id
            FROM fire_detections d
            JOIN cell_counts c ON floor(d.lat / :grid_size) = c.i_lat AND floor(d.lon / :grid_size) = c.j_lon
            WHERE d.id = ANY(:ids)
        """)
        chronic_ids = []
        for i in range(0, len(df), batch_size):
            batch_ids = df["id"].iloc[i:i+batch_size].astype(int).tolist()
            if batch_ids:
                res = conn.execute(chronic_query, {
                    "grid_size": DEFAULT_CELL_SIZE_DEG,
                    "start": start_time,
                    "end": end_time,
                    "threshold": p["chronic_static_days"],
                    "ids": batch_ids
                })
                chronic_ids.extend([row[0] for row in res])
        
        # Only apply to UNKNOWN
        chronic_mask = df["id"].isin(chronic_ids) & (df["label"] == "UNKNOWN")
        df.loc[chronic_mask, "label"] = "NEGATIVE"
        df.loc[chronic_mask, "rule_applied"] = "Chronic Static"
        LOGGER.info(f"Chronic Static: marked {df[chronic_mask].shape[0]} as NEGATIVE")

    # 5. Rule: Persistent Cluster / Cluster Growth (POSITIVE)
    # For v1, we use a simpler cluster definition: >= 2 detections within 2km/24h
    # AND NOT already marked NEGATIVE.
    with engine.connect() as conn:
        cluster_query = text("""
            SELECT d.id
            FROM fire_detections d
            WHERE d.id = ANY(:ids)
              AND EXISTS (
                SELECT 1 FROM fire_detections d2
                WHERE d2.id != d.id
                  AND d2.acq_time BETWEEN d.acq_time - make_interval(hours => :h) AND d.acq_time + make_interval(hours => :h)
                  AND ST_DWithin(d.geom::geography, d2.geom::geography, :r_m)
              )
        """)
        cluster_ids = []
        for i in range(0, len(df), batch_size):
            batch_ids = df["id"].iloc[i:i+batch_size].astype(int).tolist()
            if batch_ids:
                res = conn.execute(cluster_query, {
                    "h": p["cluster_time_hours"],
                    "r_m": p["cluster_dist_km"] * 1000,
                    "ids": batch_ids
                })
                cluster_ids.extend([row[0] for row in res])
        
        # Only apply to UNKNOWN
        pos_mask = df["id"].isin(cluster_ids) & (df["label"] == "UNKNOWN")
        df.loc[pos_mask, "label"] = "POSITIVE"
        df.loc[pos_mask, "rule_applied"] = "Persistent Cluster"
        LOGGER.info(f"Persistent Cluster: marked {df[pos_mask].shape[0]} as POSITIVE")

    # 6. Final counts
    counts = df["label"].value_counts().to_dict()
    LOGGER.info(f"Final Label counts: {counts}")

    # 7. Upsert into fire_labels
    import json
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
        # Prepare batch for execution
        batch = []
        for _, row in df.iterrows():
            batch.append({
                "id": int(row["id"]),
                "label": row["label"],
                "version": rule_version,
                "source": "heuristic_v1",
                "params": params_json,
                "now": now
            })
        
        # Batch insert
        for i in range(0, len(batch), batch_size):
            conn.execute(upsert_stmt, batch[i:i+batch_size])
    
    LOGGER.info(f"Successfully upserted {len(df)} labels into DB.")

def main():
    parser = argparse.ArgumentParser(description="Label detections using heuristics v1.")
    parser.add_argument("--bbox", type=float, nargs=4, required=True, help="min_lon min_lat max_lon max_lat")
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Rule version")
    
    args = parser.parse_args()
    
    engine = get_engine()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)
    
    label_detections_v1(engine, tuple(args.bbox), start, end, rule_version=args.version)

if __name__ == "__main__":
    main()

