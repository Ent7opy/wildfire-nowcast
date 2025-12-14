from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from api.core.grid import GridSpec, GridWindow
from api.fires import service as fires_service


def _dummy_grid() -> GridSpec:
    return GridSpec(crs="EPSG:4326", cell_size_deg=0.1, origin_lat=0.0, origin_lon=0.0, n_lat=100, n_lon=100)


def test_get_fire_cells_heatmap_empty_height_preserves_width(monkeypatch: pytest.MonkeyPatch):
    grid = _dummy_grid()
    win = GridWindow(
        i0=10,
        i1=10,
        j0=5,
        j1=10,
        lat=np.asarray([], dtype=float),
        lon=np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float),
    )

    monkeypatch.setattr(fires_service, "get_region_grid_spec", lambda region_name: grid)
    monkeypatch.setattr(fires_service, "get_grid_window_for_bbox", lambda grid, bbox, clip=True: win)

    # Ensure we don't hit the DB on an empty window.
    monkeypatch.setattr(
        fires_service,
        "list_fire_detections_bbox_time",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DB query should be skipped for empty windows")),
    )

    out = fires_service.get_fire_cells_heatmap(
        region_name="test",
        bbox=(0.0, 0.0, 1.0, 1.0),
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        include_points=True,
    )

    assert out.window is win
    assert out.heatmap.shape == (len(win.lat), len(win.lon))
    assert out.heatmap.shape == (0, 5)
    assert out.points == []


def test_get_fire_cells_heatmap_empty_width_preserves_height(monkeypatch: pytest.MonkeyPatch):
    grid = _dummy_grid()
    win = GridWindow(
        i0=10,
        i1=13,
        j0=7,
        j1=7,
        lat=np.asarray([1.0, 1.1, 1.2], dtype=float),
        lon=np.asarray([], dtype=float),
    )

    monkeypatch.setattr(fires_service, "get_region_grid_spec", lambda region_name: grid)
    monkeypatch.setattr(fires_service, "get_grid_window_for_bbox", lambda grid, bbox, clip=True: win)

    monkeypatch.setattr(
        fires_service,
        "list_fire_detections_bbox_time",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DB query should be skipped for empty windows")),
    )

    out = fires_service.get_fire_cells_heatmap(
        region_name="test",
        bbox=(0.0, 0.0, 1.0, 1.0),
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        include_points=False,
    )

    assert out.window is win
    assert out.heatmap.shape == (len(win.lat), len(win.lon))
    assert out.heatmap.shape == (3, 0)
    assert out.points is None
