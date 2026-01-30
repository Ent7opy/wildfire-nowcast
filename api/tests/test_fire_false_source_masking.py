"""Tests for false-source masking of fire detections."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from api.fires.scoring import mask_false_sources


def test_mask_false_sources_detection_near_industrial():
    """Test that detections near industrial sources are marked as masked."""
    detections = [
        {
            "id": 1,
            "lat": 42.0,
            "lon": 21.0,
        }
    ]

    # Mock count check result (table exists and has data)
    mock_count_result = MagicMock()
    mock_count_result.mappings.return_value.first.return_value = {"count": 5}

    # Mock main query result to return detection_id 1 as near industrial source
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {"detection_id": 1}
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_count_result

    mock_begin_conn = MagicMock()
    mock_begin_conn.__enter__.return_value.execute.return_value = mock_result

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.connect.return_value = mock_conn
        mock_engine.return_value.begin.return_value = mock_begin_conn
        masked = mask_false_sources(detections)

    assert 1 in masked
    assert masked[1] is True, "Detection near industrial source should be masked"


def test_mask_false_sources_detection_far_from_industrial():
    """Test that detections far from industrial sources are not masked."""
    detections = [
        {
            "id": 2,
            "lat": 42.0,
            "lon": 21.0,
        }
    ]

    # Mock count check result (table exists and has data)
    mock_count_result = MagicMock()
    mock_count_result.mappings.return_value.first.return_value = {"count": 5}

    # Mock database query to return empty result (no nearby industrial sources)
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = []

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_count_result

    mock_begin_conn = MagicMock()
    mock_begin_conn.__enter__.return_value.execute.return_value = mock_result

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.connect.return_value = mock_conn
        mock_engine.return_value.begin.return_value = mock_begin_conn
        masked = mask_false_sources(detections)

    assert 2 in masked
    assert masked[2] is False, "Detection far from industrial sources should not be masked"


def test_mask_false_sources_mixed_detections():
    """Test that mixed detections (some near, some far) are correctly classified."""
    detections = [
        {"id": 3, "lat": 42.0, "lon": 21.0},
        {"id": 4, "lat": 42.1, "lon": 21.1},
        {"id": 5, "lat": 42.2, "lon": 21.2},
    ]

    # Mock count check result (table exists and has data)
    mock_count_result = MagicMock()
    mock_count_result.mappings.return_value.first.return_value = {"count": 5}

    # Mock database query to return only detection_id 3 and 5 as near industrial sources
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {"detection_id": 3},
        {"detection_id": 5},
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_count_result

    mock_begin_conn = MagicMock()
    mock_begin_conn.__enter__.return_value.execute.return_value = mock_result

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.connect.return_value = mock_conn
        mock_engine.return_value.begin.return_value = mock_begin_conn
        masked = mask_false_sources(detections)

    assert masked[3] is True, "Detection 3 should be masked"
    assert masked[4] is False, "Detection 4 should not be masked"
    assert masked[5] is True, "Detection 5 should be masked"


def test_mask_false_sources_empty_input():
    """Test that empty input returns empty dict."""
    masked = mask_false_sources([])
    assert masked == {}


def test_mask_false_sources_custom_radius():
    """Test that custom radius is passed to query."""
    detections = [{"id": 6, "lat": 42.0, "lon": 21.0}]

    # Mock count check result (table exists and has data)
    mock_count_result = MagicMock()
    mock_count_result.mappings.return_value.first.return_value = {"count": 5}

    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = []

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_count_result

    mock_begin_conn = MagicMock()
    mock_execute = mock_begin_conn.__enter__.return_value.execute
    mock_execute.return_value = mock_result

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.connect.return_value = mock_conn
        mock_engine.return_value.begin.return_value = mock_begin_conn
        mask_false_sources(detections, radius_m=1000.0)

    # Verify that the query was called with custom radius
    call_args = mock_execute.call_args
    assert call_args[0][1]["radius_m"] == 1000.0


def test_mask_false_sources_empty_table():
    """Test that all detections pass through unmasked when industrial_sources table is empty."""
    detections = [
        {"id": 7, "lat": 42.0, "lon": 21.0},
        {"id": 8, "lat": 42.1, "lon": 21.1},
    ]

    # Mock count check result (table exists but is empty)
    mock_count_result = MagicMock()
    mock_count_result.mappings.return_value.first.return_value = {"count": 0}

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_count_result

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.connect.return_value = mock_conn
        masked = mask_false_sources(detections)

    assert masked[7] is False, "Detection should be unmasked when table is empty"
    assert masked[8] is False, "Detection should be unmasked when table is empty"


def test_mask_false_sources_missing_table():
    """Test that all detections pass through unmasked when industrial_sources table doesn't exist."""
    detections = [
        {"id": 9, "lat": 42.0, "lon": 21.0},
    ]

    # Mock connection that raises an exception (table doesn't exist)
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.side_effect = Exception("relation 'industrial_sources' does not exist")

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.connect.return_value = mock_conn
        masked = mask_false_sources(detections)

    assert masked[9] is False, "Detection should be unmasked when table is missing"


@pytest.mark.integration
def test_mask_false_sources_integration(check_likelihood_schema):
    """Integration test that validates spatial query against real database."""
    from sqlalchemy import text
    from api.db import get_engine

    # Seed test data in industrial_sources table
    with get_engine().begin() as conn:
        # Clean up any existing test data
        conn.execute(text("DELETE FROM industrial_sources WHERE name LIKE 'test_source_%'"))
        conn.execute(text("DELETE FROM fire_detections WHERE lat BETWEEN 41.99 AND 42.01"))

        # Insert test industrial source at (42.0, 21.0)
        conn.execute(
            text("""
                INSERT INTO industrial_sources (name, type, source, source_version, geom)
                VALUES (
                    'test_source_1',
                    'power_plant',
                    'test',
                    'v1',
                    ST_SetSRID(ST_MakePoint(21.0, 42.0), 4326)
                )
            """)
        )

        # Insert test fire detections
        # Detection 1: Within 500m of industrial source (at same location)
        conn.execute(
            text("""
                INSERT INTO fire_detections (
                    geom, lat, lon, acq_time, sensor, source, confidence,
                    brightness, frp, raw_properties, dedupe_hash
                )
                VALUES (
                    ST_SetSRID(ST_MakePoint(21.0, 42.0), 4326),
                    42.0,
                    21.0,
                    :acq_time,
                    'VIIRS',
                    'test',
                    90.0,
                    350.0,
                    10.0,
                    '{}',
                    'test_det_1'
                )
                RETURNING id
            """),
            {"acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)}
        )
        det_1_id = conn.execute(text("SELECT id FROM fire_detections WHERE dedupe_hash = 'test_det_1'")).scalar_one()

        # Detection 2: Far from industrial source (>1km away, approximately 0.01 degrees)
        conn.execute(
            text("""
                INSERT INTO fire_detections (
                    geom, lat, lon, acq_time, sensor, source, confidence,
                    brightness, frp, raw_properties, dedupe_hash
                )
                VALUES (
                    ST_SetSRID(ST_MakePoint(21.02, 42.01), 4326),
                    42.01,
                    21.02,
                    :acq_time,
                    'VIIRS',
                    'test',
                    85.0,
                    320.0,
                    8.0,
                    '{}',
                    'test_det_2'
                )
                RETURNING id
            """),
            {"acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)}
        )
        det_2_id = conn.execute(text("SELECT id FROM fire_detections WHERE dedupe_hash = 'test_det_2'")).scalar_one()

    try:
        # Call mask_false_sources without mocks
        detections = [
            {"id": det_1_id, "lat": 42.0, "lon": 21.0},
            {"id": det_2_id, "lat": 42.01, "lon": 21.02},
        ]

        masked = mask_false_sources(detections, radius_m=500.0)

        # Validate results
        assert det_1_id in masked, "Detection 1 should be in results"
        assert det_2_id in masked, "Detection 2 should be in results"
        assert masked[det_1_id] is True, "Detection 1 (at industrial source) should be masked"
        assert masked[det_2_id] is False, "Detection 2 (>1km away) should not be masked"

    finally:
        # Clean up test data
        with get_engine().begin() as conn:
            conn.execute(text("DELETE FROM fire_detections WHERE dedupe_hash IN ('test_det_1', 'test_det_2')"))
            conn.execute(text("DELETE FROM industrial_sources WHERE name LIKE 'test_source_%'"))
