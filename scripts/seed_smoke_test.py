"""Seed dummy data for denoiser smoke test."""

import pandas as pd
from datetime import datetime, timedelta
from api.db import get_engine
from sqlalchemy import text

def seed():
    engine = get_engine()
    
    # 1. Create labels table if it doesn't exist (it won't)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fire_labels"))
        conn.execute(text("""
            CREATE TABLE fire_labels (
                fire_detection_id BIGINT PRIMARY KEY,
                label VARCHAR(32) NOT NULL
            )
        """))
    
    # 2. Insert dummy detections if none exist or just insert some specifically for the test
    # We'll insert a few points
    t0 = datetime.now() - timedelta(days=5)
    
    detections = [
        # (lat, lon, acq_time, confidence, frp, brightness, bright_t31, source, sensor)
        (40.0, -120.0, t0, 90.0, 100.0, 320.0, 300.0, "firms_viirs", "VIIRS"),
        (40.01, -120.01, t0 + timedelta(hours=1), 80.0, 50.0, 310.0, 295.0, "firms_viirs", "VIIRS"),
        (30.0, -110.0, t0, 20.0, 10.0, 290.0, 280.0, "firms_modis", "MODIS"),
    ]
    
    with engine.begin() as conn:
        # Clear existing for clean test
        conn.execute(text("DELETE FROM fire_detections WHERE source LIKE 'smoke_test%'"))
        
        ids = []
        for i, (lat, lon, ts, conf, frp, bright, t31, src, sensor) in enumerate(detections):
            res = conn.execute(text("""
                INSERT INTO fire_detections (
                    geom, lat, lon, acq_time, confidence, frp, brightness, bright_t31, source, sensor, dedupe_hash
                ) VALUES (
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                    :lat, :lon, :ts, :conf, :frp, :bright, :t31, :src, :sensor, :hash
                ) RETURNING id
            """), {
                "lat": lat, "lon": lon, "ts": ts, "conf": conf, "frp": frp, 
                "bright": bright, "t31": t31, "src": f"smoke_test_{i}", "sensor": sensor,
                "hash": f"smoke_{i}"
            })
            ids.append(res.scalar())
        
        # 3. Insert labels
        conn.execute(text("INSERT INTO fire_labels (fire_detection_id, label) VALUES (:id, :label)"), [
            {"id": ids[0], "label": "POSITIVE"},
            {"id": ids[1], "label": "POSITIVE"},
            {"id": ids[2], "label": "NEGATIVE"},
        ])
        
    print(f"Seeded {len(ids)} detections and labels for smoke test.")

if __name__ == "__main__":
    seed()
