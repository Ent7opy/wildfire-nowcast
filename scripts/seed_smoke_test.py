"""Seed dummy data for denoiser smoke test."""

from datetime import datetime, timedelta
from api.db import get_engine
from sqlalchemy import text

def seed():
    engine = get_engine()

    t0 = datetime.now() - timedelta(days=5)
    
    detections = [
        # (lat, lon, acq_time, confidence, frp, brightness, bright_t31, source, sensor)
        (40.0, -120.0, t0, 90.0, 100.0, 320.0, 300.0, "firms_viirs", "VIIRS"),
        (40.01, -120.01, t0 + timedelta(hours=1), 80.0, 50.0, 310.0, 295.0, "firms_viirs", "VIIRS"),
        (30.0, -110.0, t0, 20.0, 10.0, 290.0, 280.0, "firms_modis", "MODIS"),
    ]
    
    with engine.begin() as conn:
        # Clear existing smoke test data for clean re-runs
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
        
        # Insert labels into denoiser_labels_v2
        conn.execute(text("""
            INSERT INTO denoiser_labels_v2 (fire_detection_id, label, rule_version, source)
            VALUES (:id, :label, :version, :source)
            ON CONFLICT (fire_detection_id, rule_version) DO UPDATE SET
                label = EXCLUDED.label
        """), [
            {"id": ids[0], "label": "POSITIVE", "version": "smoke_test", "source": "smoke_test"},
            {"id": ids[1], "label": "POSITIVE", "version": "smoke_test", "source": "smoke_test"},
            {"id": ids[2], "label": "NEGATIVE", "version": "smoke_test", "source": "smoke_test"},
        ])
        
    print(f"Seeded {len(ids)} detections and labels for smoke test.")

if __name__ == "__main__":
    seed()
