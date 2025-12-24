from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np

from api.core.grid import GridSpec, GridWindow
from api.fires import service as fires_service


def _dummy_grid() -> GridSpec:
    return GridSpec(crs="EPSG:4326", cell_size_deg=0.1, origin_lat=0.0, origin_lon=0.0, n_lat=100, n_lon=100)


def _dummy_window() -> GridWindow:
    return GridWindow(
        i0=0, i1=10, j0=0, j1=10,
        lat=np.arange(0, 1, 0.1),
        lon=np.arange(0, 1, 0.1)
    )


def test_get_fire_cells_heatmap_threads_include_noise(monkeypatch):
    """Ensure include_noise is passed to the repo helper."""
    grid = _dummy_grid()
    win = _dummy_window()

    monkeypatch.setattr(fires_service, "get_region_grid_spec", lambda _: grid)
    monkeypatch.setattr(fires_service, "get_grid_window_for_bbox", lambda *_, **__: win)

    mock_list = MagicMock(return_value=[])
    monkeypatch.setattr(fires_service, "list_fire_detections_bbox_time", mock_list)

    fires_service.get_fire_cells_heatmap(
        region_name="test",
        bbox=(0, 0, 1, 1),
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        include_noise=True,
    )

    _, kwargs = mock_list.call_args
    assert kwargs["include_noise"] is True


def test_get_fire_cells_heatmap_weighting_shortcut(monkeypatch):
    """Ensure weight_by_denoised_score sets correct mode and value_col."""
    grid = _dummy_grid()
    win = _dummy_window()

    monkeypatch.setattr(fires_service, "get_region_grid_spec", lambda _: grid)
    monkeypatch.setattr(fires_service, "get_grid_window_for_bbox", lambda *_, **__: win)

    mock_list = MagicMock(return_value=[])
    monkeypatch.setattr(fires_service, "list_fire_detections_bbox_time", mock_list)

    fires_service.get_fire_cells_heatmap(
        region_name="test",
        bbox=(0, 0, 1, 1),
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        weight_by_denoised_score=True,
    )

    _, kwargs = mock_list.call_args
    # Should have switched to sum of denoised_score
    assert "denoised_score" in kwargs["columns"]


def test_get_fire_cells_heatmap_weighting_null_scores_do_not_produce_nan(monkeypatch):
    """Regression: NULL denoised_score must not poison heatmap with NaNs."""
    grid = _dummy_grid()
    win = _dummy_window()

    monkeypatch.setattr(fires_service, "get_region_grid_spec", lambda _: grid)
    monkeypatch.setattr(fires_service, "get_grid_window_for_bbox", lambda *_, **__: win)

    # One unscored detection (denoised_score=None) and one scored detection in same cell.
    detections = [
        {"lat": 0.05, "lon": 0.05, "denoised_score": None},
        {"lat": 0.05, "lon": 0.05, "denoised_score": 0.2},
    ]
    monkeypatch.setattr(fires_service, "list_fire_detections_bbox_time", lambda **_: detections)

    out = fires_service.get_fire_cells_heatmap(
        region_name="test",
        bbox=(0, 0, 1, 1),
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        weight_by_denoised_score=True,
    )

    assert np.isfinite(out.heatmap).all()
    # Unscored detection defaults to 1.0 weight, so expected sum is 1.2 in that cell.
    assert np.isclose(out.heatmap[0, 0], 1.2)

