"""Tests for fire detection land-cover plausibility scoring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from api.fires.landcover import compute_landcover_scores, LANDCOVER_SCORES


def test_compute_landcover_scores_forest():
    """Test that forest detections receive high score (1.0)."""
    detections = [
        {"id": 1, "lat": 42.0, "lon": 21.0},
    ]

    # Mock rasterio to return forest land-cover class (10)
    mock_src = MagicMock()
    mock_src.height = 1000
    mock_src.width = 1000
    mock_src.transform = MagicMock()
    mock_src.read.return_value = np.array([[10]], dtype=np.int32)  # Forest

    with patch("api.fires.landcover.rasterio.open") as mock_open:
        with patch("api.fires.landcover.rowcol", return_value=(500, 500)):
            mock_open.return_value.__enter__.return_value = mock_src
            scores = compute_landcover_scores(
                detections, landcover_path=Path("/fake/landcover.tif")
            )

    assert 1 in scores
    assert scores[1] == 1.0, f"Forest detection should have score 1.0, got {scores[1]}"


def test_compute_landcover_scores_urban():
    """Test that urban detections receive low score (0.1)."""
    detections = [
        {"id": 2, "lat": 40.0, "lon": -74.0},
    ]

    # Mock rasterio to return urban land-cover class (50)
    mock_src = MagicMock()
    mock_src.height = 1000
    mock_src.width = 1000
    mock_src.transform = MagicMock()
    mock_src.read.return_value = np.array([[50]], dtype=np.int32)  # Urban

    with patch("api.fires.landcover.rasterio.open") as mock_open:
        with patch("api.fires.landcover.rowcol", return_value=(500, 500)):
            mock_open.return_value.__enter__.return_value = mock_src
            scores = compute_landcover_scores(
                detections, landcover_path=Path("/fake/landcover.tif")
            )

    assert 2 in scores
    assert scores[2] == 0.1, f"Urban detection should have score 0.1, got {scores[2]}"


def test_compute_landcover_scores_cropland():
    """Test that cropland detections receive moderate score (0.7)."""
    detections = [
        {"id": 3, "lat": 38.0, "lon": -95.0},
    ]

    # Mock rasterio to return cropland land-cover class (40)
    mock_src = MagicMock()
    mock_src.height = 1000
    mock_src.width = 1000
    mock_src.transform = MagicMock()
    mock_src.read.return_value = np.array([[40]], dtype=np.int32)  # Cropland

    with patch("api.fires.landcover.rasterio.open") as mock_open:
        with patch("api.fires.landcover.rowcol", return_value=(500, 500)):
            mock_open.return_value.__enter__.return_value = mock_src
            scores = compute_landcover_scores(
                detections, landcover_path=Path("/fake/landcover.tif")
            )

    assert 3 in scores
    assert scores[3] == 0.7, f"Cropland detection should have score 0.7, got {scores[3]}"


def test_compute_landcover_scores_desert():
    """Test that desert detections receive low score (0.1)."""
    detections = [
        {"id": 4, "lat": 25.0, "lon": 45.0},
    ]

    # Mock rasterio to return bare/sparse vegetation class (60)
    mock_src = MagicMock()
    mock_src.height = 1000
    mock_src.width = 1000
    mock_src.transform = MagicMock()
    mock_src.read.return_value = np.array([[60]], dtype=np.int32)  # Desert

    with patch("api.fires.landcover.rasterio.open") as mock_open:
        with patch("api.fires.landcover.rowcol", return_value=(500, 500)):
            mock_open.return_value.__enter__.return_value = mock_src
            scores = compute_landcover_scores(
                detections, landcover_path=Path("/fake/landcover.tif")
            )

    assert 4 in scores
    assert scores[4] == 0.1, f"Desert detection should have score 0.1, got {scores[4]}"


def test_compute_landcover_scores_water():
    """Test that water detections receive low score (0.1)."""
    detections = [
        {"id": 5, "lat": 35.0, "lon": -80.0},
    ]

    # Mock rasterio to return water class (80)
    mock_src = MagicMock()
    mock_src.height = 1000
    mock_src.width = 1000
    mock_src.transform = MagicMock()
    mock_src.read.return_value = np.array([[80]], dtype=np.int32)  # Water

    with patch("api.fires.landcover.rasterio.open") as mock_open:
        with patch("api.fires.landcover.rowcol", return_value=(500, 500)):
            mock_open.return_value.__enter__.return_value = mock_src
            scores = compute_landcover_scores(
                detections, landcover_path=Path("/fake/landcover.tif")
            )

    assert 5 in scores
    assert scores[5] == 0.1, f"Water detection should have score 0.1, got {scores[5]}"


def test_compute_landcover_scores_unknown_class():
    """Test that unknown land-cover class defaults to neutral score (0.5)."""
    detections = [
        {"id": 6, "lat": 42.0, "lon": 21.0},
    ]

    # Mock rasterio to return unknown land-cover class (999)
    mock_src = MagicMock()
    mock_src.height = 1000
    mock_src.width = 1000
    mock_src.transform = MagicMock()
    mock_src.read.return_value = np.array([[999]], dtype=np.int32)  # Unknown

    with patch("api.fires.landcover.rasterio.open") as mock_open:
        with patch("api.fires.landcover.rowcol", return_value=(500, 500)):
            mock_open.return_value.__enter__.return_value = mock_src
            scores = compute_landcover_scores(
                detections, landcover_path=Path("/fake/landcover.tif")
            )

    assert 6 in scores
    assert scores[6] == 0.5, f"Unknown class should default to 0.5, got {scores[6]}"


def test_compute_landcover_scores_out_of_bounds():
    """Test that out-of-bounds detections receive neutral score (0.5)."""
    detections = [
        {"id": 7, "lat": 90.0, "lon": 180.0},
    ]

    # Mock rasterio with detection out of raster bounds
    mock_src = MagicMock()
    mock_src.height = 1000
    mock_src.width = 1000
    mock_src.transform = MagicMock()

    with patch("api.fires.landcover.rasterio.open") as mock_open:
        with patch("api.fires.landcover.rowcol", return_value=(2000, 2000)):  # Out of bounds
            mock_open.return_value.__enter__.return_value = mock_src
            scores = compute_landcover_scores(
                detections, landcover_path=Path("/fake/landcover.tif")
            )

    assert 7 in scores
    assert scores[7] == 0.5, f"Out-of-bounds detection should have score 0.5, got {scores[7]}"


def test_compute_landcover_scores_no_data():
    """Test that missing landcover data returns neutral scores (0.5)."""
    detections = [
        {"id": 8, "lat": 42.0, "lon": 21.0},
    ]

    # No landcover path provided and default path doesn't exist
    with patch("api.fires.landcover.get_landcover_path", return_value=None):
        scores = compute_landcover_scores(detections)

    assert 8 in scores
    assert scores[8] == 0.5, f"Missing data should default to 0.5, got {scores[8]}"


def test_compute_landcover_scores_empty_input():
    """Test that empty input returns empty dict."""
    scores = compute_landcover_scores([])
    assert scores == {}


def test_compute_landcover_scores_multiple_detections():
    """Test scoring multiple detections with different land-cover types."""
    detections = [
        {"id": 10, "lat": 42.0, "lon": 21.0},  # Forest
        {"id": 11, "lat": 40.0, "lon": -74.0},  # Urban
        {"id": 12, "lat": 38.0, "lon": -95.0},  # Cropland
    ]

    # Mock rasterio to return different classes based on detection
    mock_src = MagicMock()
    mock_src.height = 1000
    mock_src.width = 1000
    mock_src.transform = MagicMock()
    
    # Return different land-cover classes based on call order
    mock_src.read.side_effect = [
        np.array([[10]], dtype=np.int32),  # Forest
        np.array([[50]], dtype=np.int32),  # Urban
        np.array([[40]], dtype=np.int32),  # Cropland
    ]

    with patch("api.fires.landcover.rasterio.open") as mock_open:
        with patch("api.fires.landcover.rowcol", return_value=(500, 500)):
            mock_open.return_value.__enter__.return_value = mock_src
            scores = compute_landcover_scores(
                detections, landcover_path=Path("/fake/landcover.tif")
            )

    assert len(scores) == 3
    assert scores[10] == 1.0  # Forest
    assert scores[11] == 0.1  # Urban
    assert scores[12] == 0.7  # Cropland


def test_landcover_scores_definition():
    """Test that LANDCOVER_SCORES dict contains expected classes and ranges."""
    # Verify all scores are in [0, 1] range
    for class_id, score in LANDCOVER_SCORES.items():
        assert 0.0 <= score <= 1.0, f"Score for class {class_id} must be in [0, 1], got {score}"
    
    # Verify high-risk vegetation classes have score 1.0
    assert LANDCOVER_SCORES[10] == 1.0  # Tree cover
    assert LANDCOVER_SCORES[20] == 1.0  # Shrubland
    assert LANDCOVER_SCORES[30] == 1.0  # Grassland
    
    # Verify moderate-risk class has intermediate score
    assert LANDCOVER_SCORES[40] == 0.7  # Cropland
    
    # Verify low-risk classes have score 0.1
    assert LANDCOVER_SCORES[50] == 0.1  # Urban
    assert LANDCOVER_SCORES[60] == 0.1  # Desert
    assert LANDCOVER_SCORES[80] == 0.1  # Water
