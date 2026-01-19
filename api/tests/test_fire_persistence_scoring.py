"""Tests for fire detection persistence scoring."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from api.fires.scoring import compute_persistence_scores


def test_compute_persistence_scores_isolated_detection():
    """Test that isolated detections receive score ≤ 0.2."""
    detections = [
        {
            "id": 1,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            "sensor": "VIIRS",
        }
    ]

    # Mock database query to return cluster_size=1 (isolated)
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {
            "detection_id": 1,
            "cluster_size": 1,
            "sensor_count": 1,
            "sensors": ["VIIRS"],
        }
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        scores = compute_persistence_scores(detections)

    assert 1 in scores
    assert scores[1] <= 0.2, f"Isolated detection should have score ≤0.2, got {scores[1]}"


def test_compute_persistence_scores_multi_sensor_bonus():
    """Test that multi-sensor detections receive bonus (+0.1)."""
    detections = [
        {
            "id": 2,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            "sensor": "VIIRS",
        }
    ]

    # Mock database query to return cluster with 2 sensors
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {
            "detection_id": 2,
            "cluster_size": 3,
            "sensor_count": 2,
            "sensors": ["VIIRS", "Terra"],
        }
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        scores = compute_persistence_scores(detections)

    assert 2 in scores
    # Cluster size 3 should give base_score ~0.4, multi-sensor adds 0.1 → 0.5
    assert scores[2] >= 0.4, f"Multi-sensor cluster should have score ≥0.4, got {scores[2]}"


def test_compute_persistence_scores_large_cluster():
    """Test that large clusters receive high persistence scores."""
    detections = [
        {
            "id": 3,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            "sensor": "VIIRS",
        }
    ]

    # Mock database query to return large cluster (10+ detections)
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {
            "detection_id": 3,
            "cluster_size": 15,
            "sensor_count": 1,
            "sensors": ["VIIRS"],
        }
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        scores = compute_persistence_scores(detections)

    assert 3 in scores
    assert scores[3] >= 0.7, f"Large cluster should have score ≥0.7, got {scores[3]}"


def test_compute_persistence_scores_empty_input():
    """Test that empty input returns empty dict."""
    scores = compute_persistence_scores([])
    assert scores == {}


def test_compute_persistence_scores_invalid_time_window():
    """Test that invalid time window raises ValueError."""
    detections = [
        {
            "id": 1,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            "sensor": "VIIRS",
        }
    ]

    with pytest.raises(ValueError, match="Invalid time_window_hours"):
        compute_persistence_scores(detections, time_window_hours=(72.0, 24.0))


def test_compute_persistence_scores_no_cluster_found():
    """Test that detections with no nearby detections receive isolated score."""
    detections = [
        {
            "id": 4,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            "sensor": "VIIRS",
        }
    ]

    # Mock database query to return no clusters (empty result)
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = []

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        scores = compute_persistence_scores(detections)

    assert 4 in scores
    assert scores[4] == 0.2, f"Detection with no cluster should have score 0.2, got {scores[4]}"
