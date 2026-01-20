"""Tests for fire detection weather plausibility scoring."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from api.fires.scoring import compute_weather_plausibility_scores


def _create_mock_weather_dataset(*, rh2m=None, u10=0.0, v10=0.0, tp=None):
    """Create a mock weather dataset for testing."""
    ref_time = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc).replace(tzinfo=None)
    time_coord = np.array([np.datetime64(ref_time, "ns")])

    data_vars = {
        "u10": (("time", "lat", "lon"), np.array([[[u10]]])),
        "v10": (("time", "lat", "lon"), np.array([[[v10]]])),
    }

    if rh2m is not None:
        data_vars["rh2m"] = (("time", "lat", "lon"), np.array([[[rh2m]]]))

    if tp is not None:
        data_vars["tp"] = (("time", "lat", "lon"), np.array([[[tp]]]))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": time_coord,
            "lat": [42.0],
            "lon": [21.0],
        },
    )
    return ds


def test_compute_weather_plausibility_high_rh_penalty():
    """Test that high relative humidity (>70%) results in penalty."""
    detections = [
        {
            "id": 1,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        }
    ]

    # Mock weather data with high RH (80%)
    mock_ds = _create_mock_weather_dataset(rh2m=80.0, u10=1.0, v10=1.0, tp=0.0)

    # Mock database query
    mock_db_result = MagicMock()
    mock_db_result.mappings.return_value.first.return_value = {
        "id": 1,
        "storage_path": "mock_path.nc",
        "run_time": datetime(2025, 1, 1, 6, 0, tzinfo=timezone.utc),
    }

    with patch("api.fires.scoring.get_engine") as mock_engine, \
         patch("api.fires.scoring.xr.open_dataset", return_value=mock_ds), \
         patch("api.fires.scoring.Path") as mock_path:

        mock_path.return_value.is_absolute.return_value = True
        mock_path.return_value.exists.return_value = True

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value.execute.return_value = mock_db_result
        mock_engine.return_value.connect.return_value = mock_conn

        scores = compute_weather_plausibility_scores(detections)

    assert 1 in scores
    # Base 0.5 - 0.3 (high RH penalty) = 0.2
    assert scores[1] <= 0.3, f"High RH should result in penalty, got {scores[1]}"


def test_compute_weather_plausibility_low_rh_bonus():
    """Test that low relative humidity (<40%) results in bonus."""
    detections = [
        {
            "id": 2,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        }
    ]

    # Mock weather data with low RH (30%)
    mock_ds = _create_mock_weather_dataset(rh2m=30.0, u10=1.0, v10=1.0, tp=0.0)

    mock_db_result = MagicMock()
    mock_db_result.mappings.return_value.first.return_value = {
        "id": 1,
        "storage_path": "mock_path.nc",
        "run_time": datetime(2025, 1, 1, 6, 0, tzinfo=timezone.utc),
    }

    with patch("api.fires.scoring.get_engine") as mock_engine, \
         patch("api.fires.scoring.xr.open_dataset", return_value=mock_ds), \
         patch("api.fires.scoring.Path") as mock_path:

        mock_path.return_value.is_absolute.return_value = True
        mock_path.return_value.exists.return_value = True

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value.execute.return_value = mock_db_result
        mock_engine.return_value.connect.return_value = mock_conn

        scores = compute_weather_plausibility_scores(detections)

    assert 2 in scores
    # Base 0.5 + 0.2 (low RH bonus) = 0.7
    assert scores[2] >= 0.6, f"Low RH should result in bonus, got {scores[2]}"


def test_compute_weather_plausibility_high_wind_bonus():
    """Test that moderate/high wind (>3 m/s) results in bonus."""
    detections = [
        {
            "id": 3,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        }
    ]

    # Mock weather data with high wind (u=4, v=3 → speed=5 m/s)
    mock_ds = _create_mock_weather_dataset(rh2m=50.0, u10=4.0, v10=3.0, tp=0.0)

    mock_db_result = MagicMock()
    mock_db_result.mappings.return_value.first.return_value = {
        "id": 1,
        "storage_path": "mock_path.nc",
        "run_time": datetime(2025, 1, 1, 6, 0, tzinfo=timezone.utc),
    }

    with patch("api.fires.scoring.get_engine") as mock_engine, \
         patch("api.fires.scoring.xr.open_dataset", return_value=mock_ds), \
         patch("api.fires.scoring.Path") as mock_path:

        mock_path.return_value.is_absolute.return_value = True
        mock_path.return_value.exists.return_value = True

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value.execute.return_value = mock_db_result
        mock_engine.return_value.connect.return_value = mock_conn

        scores = compute_weather_plausibility_scores(detections)

    assert 3 in scores
    # Base 0.5 + 0.1 (wind bonus) = 0.6
    assert scores[3] >= 0.5, f"High wind should result in bonus, got {scores[3]}"


def test_compute_weather_plausibility_no_weather_data():
    """Test that missing weather data returns neutral score (0.5)."""
    detections = [
        {
            "id": 4,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        }
    ]

    # Mock database query to return no weather run
    mock_db_result = MagicMock()
    mock_db_result.mappings.return_value.first.return_value = None

    with patch("api.fires.scoring.get_engine") as mock_engine:
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value.execute.return_value = mock_db_result
        mock_engine.return_value.connect.return_value = mock_conn

        scores = compute_weather_plausibility_scores(detections)

    assert 4 in scores
    assert scores[4] == 0.5, f"Missing weather data should give neutral score, got {scores[4]}"


def test_compute_weather_plausibility_combined_effects():
    """Test that multiple weather factors combine correctly."""
    detections = [
        {
            "id": 5,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        }
    ]

    # Mock weather data with low RH (30%) and high wind (5 m/s)
    # Should get both bonuses: +0.2 (low RH) + 0.1 (wind) = 0.8 total
    mock_ds = _create_mock_weather_dataset(rh2m=30.0, u10=3.0, v10=4.0, tp=0.0)

    mock_db_result = MagicMock()
    mock_db_result.mappings.return_value.first.return_value = {
        "id": 1,
        "storage_path": "mock_path.nc",
        "run_time": datetime(2025, 1, 1, 6, 0, tzinfo=timezone.utc),
    }

    with patch("api.fires.scoring.get_engine") as mock_engine, \
         patch("api.fires.scoring.xr.open_dataset", return_value=mock_ds), \
         patch("api.fires.scoring.Path") as mock_path:

        mock_path.return_value.is_absolute.return_value = True
        mock_path.return_value.exists.return_value = True

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value.execute.return_value = mock_db_result
        mock_engine.return_value.connect.return_value = mock_conn

        scores = compute_weather_plausibility_scores(detections)

    assert 5 in scores
    # Base 0.5 + 0.2 (low RH) + 0.1 (wind) = 0.8
    assert scores[5] >= 0.7, f"Combined favorable weather should give high score, got {scores[5]}"


def test_compute_weather_plausibility_empty_input():
    """Test that empty input returns empty dict."""
    scores = compute_weather_plausibility_scores([])
    assert scores == {}


def test_compute_weather_plausibility_score_clamping():
    """Test that scores are clamped to [0.1, 1.0] range."""
    detections = [
        {
            "id": 6,
            "lat": 42.0,
            "lon": 21.0,
            "acq_time": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        }
    ]

    # Mock weather data with very high RH and recent precipitation
    # Base 0.5 - 0.3 (high RH) - 0.2 (precip) = 0.0, should clamp to 0.1
    mock_ds = _create_mock_weather_dataset(rh2m=90.0, u10=0.0, v10=0.0, tp=0.015)

    mock_db_result = MagicMock()
    mock_db_result.mappings.return_value.first.return_value = {
        "id": 1,
        "storage_path": "mock_path.nc",
        "run_time": datetime(2025, 1, 1, 6, 0, tzinfo=timezone.utc),
    }

    with patch("api.fires.scoring.get_engine") as mock_engine, \
         patch("api.fires.scoring.xr.open_dataset", return_value=mock_ds), \
         patch("api.fires.scoring.Path") as mock_path:

        mock_path.return_value.is_absolute.return_value = True
        mock_path.return_value.exists.return_value = True

        mock_conn = MagicMock()
        mock_conn.__enter__.return_value.execute.return_value = mock_db_result
        mock_engine.return_value.connect.return_value = mock_conn

        scores = compute_weather_plausibility_scores(detections)

    assert 6 in scores
    assert scores[6] >= 0.1, f"Score should be clamped to minimum 0.1, got {scores[6]}"
    assert scores[6] <= 1.0, f"Score should be clamped to maximum 1.0, got {scores[6]}"
