"""Integration test for composite fire likelihood scoring pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from api.db import get_engine
from api.fires.repo import (
    update_false_source_masking,
    update_persistence_scores,
    update_landcover_scores,
    update_weather_scores,
    update_fire_likelihood,
    list_fire_detections_bbox_time,
)


@pytest.fixture
def test_batch_id():
    """Create a test ingest batch and return its ID."""
    stmt = text("""
        INSERT INTO ingest_batches (source, source_uri, started_at, status)
        VALUES ('test_source', 'test_uri', now(), 'in_progress')
        RETURNING id
    """)
    with get_engine().begin() as conn:
        result = conn.execute(stmt)
        batch_id = result.scalar()
    yield batch_id
    # Cleanup: delete batch and related detections
    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM fire_detections WHERE ingest_batch_id = :batch_id"), {"batch_id": batch_id})
        conn.execute(text("DELETE FROM ingest_batches WHERE id = :batch_id"), {"batch_id": batch_id})


@pytest.fixture
def insert_test_detections(test_batch_id):
    """Insert test fire detections into the database."""
    detections = [
        {
            "lat": 42.5,
            "lon": 23.0,
            "acq_time": datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
            "sensor": "VIIRS",
            "source": "VIIRS_SNPP_NRT",
            "confidence": 90.0,
            "brightness": 330.0,
            "frp": 15.5,
            "batch_id": test_batch_id,
            "dedupe_hash": "test_det_1",
        },
        {
            "lat": 42.51,
            "lon": 23.01,
            "acq_time": datetime(2026, 1, 15, 13, 0, tzinfo=timezone.utc),
            "sensor": "MODIS",
            "source": "MODIS_NRT",
            "confidence": 85.0,
            "brightness": 320.0,
            "frp": 12.0,
            "batch_id": test_batch_id,
            "dedupe_hash": "test_det_2",
        },
        {
            "lat": 40.0,
            "lon": -74.0,
            "acq_time": datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc),
            "sensor": "VIIRS",
            "source": "VIIRS_SNPP_NRT",
            "confidence": 50.0,
            "brightness": 310.0,
            "frp": 5.0,
            "batch_id": test_batch_id,
            "dedupe_hash": "test_det_3",
        },
    ]

    stmt = text("""
        INSERT INTO fire_detections 
        (geom, lat, lon, acq_time, sensor, source, confidence, brightness, frp, ingest_batch_id, confidence_score, dedupe_hash)
        VALUES 
        (ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :lat, :lon, :acq_time, :sensor, :source, :confidence, :brightness, :frp, :batch_id, :confidence / 100.0, :dedupe_hash)
        RETURNING id
    """)

    detection_ids = []
    with get_engine().begin() as conn:
        for det in detections:
            result = conn.execute(stmt, det)
            detection_ids.append(result.scalar())

    return detection_ids


@pytest.mark.integration
def test_composite_likelihood_pipeline(check_likelihood_schema, test_batch_id, insert_test_detections):
    """Test end-to-end fire likelihood scoring pipeline.

    Validates:
    1. All component scores are computed (persistence, landcover, weather, false_source_masked)
    2. Composite fire_likelihood is computed from components
    3. Scores are persisted to database
    4. Scores can be queried via API repo function
    """
    detection_ids = insert_test_detections

    # Step 1: Update false source masking
    masked_count = update_false_source_masking(test_batch_id)
    assert masked_count >= 0

    # Step 2: Update persistence scores
    persistence_count = update_persistence_scores(test_batch_id)
    assert persistence_count == len(detection_ids)

    # Step 3: Update landcover scores
    landcover_count = update_landcover_scores(test_batch_id)
    assert landcover_count == len(detection_ids)

    # Step 4: Update weather scores
    weather_count = update_weather_scores(test_batch_id)
    assert weather_count == len(detection_ids)

    # Step 5: Update composite fire likelihood
    likelihood_count = update_fire_likelihood(test_batch_id)
    assert likelihood_count == len(detection_ids)

    # Step 6: Query detections and verify all scores are populated
    stmt = text("""
        SELECT 
            id,
            confidence_score,
            persistence_score,
            landcover_score,
            weather_score,
            false_source_masked,
            fire_likelihood
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
        ORDER BY id
    """)

    with get_engine().begin() as conn:
        result = conn.execute(stmt, {"batch_id": test_batch_id})
        rows = result.mappings().all()

    assert len(rows) == len(detection_ids)

    for row in rows:
        # All component scores should be populated
        assert row["confidence_score"] is not None
        assert row["persistence_score"] is not None
        assert row["landcover_score"] is not None
        assert row["weather_score"] is not None
        assert row["false_source_masked"] is not None
        assert row["fire_likelihood"] is not None

        # Scores should be in valid range [0, 1]
        assert 0.0 <= row["confidence_score"] <= 1.0
        assert 0.0 <= row["persistence_score"] <= 1.0
        assert 0.0 <= row["landcover_score"] <= 1.0
        assert 0.0 <= row["weather_score"] <= 1.0
        assert 0.0 <= row["fire_likelihood"] <= 1.0

        # If masked, fire_likelihood should be 0
        if row["false_source_masked"]:
            assert row["fire_likelihood"] == 0.0

    # Step 7: Verify scores are returned via list_fire_detections_bbox_time
    # Query a large bbox covering all test detections
    detections = list_fire_detections_bbox_time(
        bbox=(-180.0, -90.0, 180.0, 90.0),
        start_time=datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 16, 0, 0, tzinfo=timezone.utc),
        columns=[
            "id",
            "confidence_score",
            "persistence_score",
            "landcover_score",
            "weather_score",
            "false_source_masked",
            "fire_likelihood",
        ],
        limit=100,
    )

    # Filter to only our test detections
    test_detections = [d for d in detections if d["id"] in detection_ids]
    assert len(test_detections) == len(detection_ids)

    for det in test_detections:
        # Verify all likelihood-related columns are present in API response
        assert "confidence_score" in det
        assert "persistence_score" in det
        assert "landcover_score" in det
        assert "weather_score" in det
        assert "false_source_masked" in det
        assert "fire_likelihood" in det


@pytest.mark.integration
def test_false_source_masking_sets_likelihood_zero(check_likelihood_schema, test_batch_id):
    """Test that detections near industrial sources get fire_likelihood=0."""
    # Insert an industrial source
    industrial_stmt = text("""
        INSERT INTO industrial_sources (geom, name, source_type, metadata)
        VALUES (ST_SetSRID(ST_MakePoint(25.0, 45.0), 4326), 'Test Plant', 'power_plant', '{}')
        RETURNING id
    """)

    with get_engine().begin() as conn:
        result = conn.execute(industrial_stmt)
        industrial_id = result.scalar()

    # Insert a detection near the industrial source (within 500m)
    detection_stmt = text("""
        INSERT INTO fire_detections 
        (geom, lat, lon, acq_time, sensor, source, confidence, brightness, frp, ingest_batch_id, confidence_score, dedupe_hash)
        VALUES 
        (ST_SetSRID(ST_MakePoint(25.001, 45.001), 4326), 45.001, 25.001, now(), 'VIIRS', 'VIIRS_SNPP_NRT', 80.0, 320.0, 10.0, :batch_id, 0.8, 'test_det_industrial')
        RETURNING id
    """)

    with get_engine().begin() as conn:
        result = conn.execute(detection_stmt, {"batch_id": test_batch_id})
        detection_id = result.scalar()

    try:
        # Run the scoring pipeline
        update_false_source_masking(test_batch_id)
        update_persistence_scores(test_batch_id)
        update_landcover_scores(test_batch_id)
        update_weather_scores(test_batch_id)
        update_fire_likelihood(test_batch_id)

        # Query the detection and verify it's masked with likelihood=0
        query_stmt = text("""
            SELECT false_source_masked, fire_likelihood
            FROM fire_detections
            WHERE id = :detection_id
        """)

        with get_engine().begin() as conn:
            result = conn.execute(query_stmt, {"detection_id": detection_id})
            row = result.mappings().first()

        assert row["false_source_masked"] is True
        assert row["fire_likelihood"] == 0.0

    finally:
        # Cleanup
        with get_engine().begin() as conn:
            conn.execute(text("DELETE FROM fire_detections WHERE id = :detection_id"), {"detection_id": detection_id})
            conn.execute(text("DELETE FROM industrial_sources WHERE id = :industrial_id"), {"industrial_id": industrial_id})
