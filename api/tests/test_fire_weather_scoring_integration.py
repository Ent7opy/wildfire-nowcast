"""Tests for weather score batch update integration."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from api.fires.repo import update_weather_scores


def test_update_weather_scores_empty_batch():
    """Test that empty batch returns 0 updates."""
    # Mock database query to return no detections
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = []

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result

    with patch("api.fires.repo.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        count = update_weather_scores(batch_id=1)

    assert count == 0


def test_update_weather_scores_batch_with_detections():
    """Test that batch with detections computes and updates scores."""
    # Mock database query to return detections
    mock_select_result = MagicMock()
    mock_select_result.mappings.return_value.all.return_value = [
        {
            "id": 1,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        },
        {
            "id": 2,
            "lat": 42.1,
            "lon": 21.1,
            "acq_time": datetime(2025, 1, 1, 12, 30, tzinfo=timezone.utc),
        },
    ]

    # Mock weather plausibility scoring to return scores
    with patch("api.fires.repo.get_engine") as mock_engine, \
         patch("api.fires.repo.compute_weather_plausibility_scores") as mock_compute:

        # Setup mocks
        mock_conn = MagicMock()
        mock_engine.return_value.begin.return_value = mock_conn

        # First call: SELECT detections
        mock_conn.__enter__.return_value.execute.side_effect = [
            mock_select_result,
            MagicMock(),  # Second call: UPDATE scores
        ]

        # Mock compute function to return scores
        mock_compute.return_value = {
            1: 0.7,
            2: 0.5,
        }

        count = update_weather_scores(batch_id=1)

    assert count == 2
    # Verify compute was called with detections
    mock_compute.assert_called_once()
    detections_arg = mock_compute.call_args[0][0]
    assert len(detections_arg) == 2
    assert detections_arg[0]["id"] == 1
    assert detections_arg[1]["id"] == 2
