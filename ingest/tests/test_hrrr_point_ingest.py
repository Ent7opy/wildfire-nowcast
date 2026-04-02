"""Unit tests for ingest/hrrr_point_ingest.py (no network or DB calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingest.hrrr_point_ingest import ingest_hrrr_points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A set of GFS-snapped points that fall squarely inside CONUS.
CONUS_POINTS: list[tuple[float, float]] = [
    (34.0, -118.0),   # Southern California
    (48.0, -122.0),   # Pacific NW
    (39.75, -105.0),  # Colorado
]

# A set of GFS-snapped points in Europe (outside CONUS).
EUROPE_POINTS: list[tuple[float, float]] = [
    (48.0, 11.0),   # Munich
    (51.5, -0.0),   # London
]


# ---------------------------------------------------------------------------
# Test: ValueError raised for non-CONUS bbox
# ---------------------------------------------------------------------------

@patch("ingest.hrrr_point_ingest.create_weather_run_record", return_value=42)
@patch("ingest.hrrr_point_ingest.finalize_weather_run_record")
@patch("ingest.hrrr_point_ingest.query_fire_detection_grid_points")
def test_raises_value_error_for_non_conus_bbox(
    mock_query: MagicMock,
    mock_finalize: MagicMock,
    mock_create: MagicMock,
) -> None:
    """ingest_hrrr_points raises ValueError when detection points are in Europe."""
    mock_query.return_value = EUROPE_POINTS

    with pytest.raises(ValueError, match="HRRR ingest requires CONUS bbox"):
        ingest_hrrr_points()

    # create_weather_run_record should NOT have been called because the bbox
    # check fires before the run record is created.
    mock_create.assert_not_called()
    mock_finalize.assert_not_called()


# ---------------------------------------------------------------------------
# Test: empty grid_points → returns run_id without downloading
# ---------------------------------------------------------------------------

@patch("ingest.hrrr_point_ingest.create_weather_run_record", return_value=7)
@patch("ingest.hrrr_point_ingest.finalize_weather_run_record")
@patch("ingest.hrrr_point_ingest.query_fire_detection_grid_points")
def test_empty_grid_points_returns_run_id(
    mock_query: MagicMock,
    mock_finalize: MagicMock,
    mock_create: MagicMock,
) -> None:
    """ingest_hrrr_points returns run_id and does not download when no detections."""
    mock_query.return_value = []

    run_id = ingest_hrrr_points()

    assert run_id == 7

    # A run record must still be created so data_status can show the attempt.
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "hrrr_3km"
    assert call_kwargs["file_format"] == "point_cache"

    # Must be finalized with status='completed' and zero rows.
    mock_finalize.assert_called_once()
    finalize_kwargs = mock_finalize.call_args.kwargs
    assert finalize_kwargs["status"] == "completed"
    assert finalize_kwargs["extra_metadata"]["point_cache_rows"] == 0
