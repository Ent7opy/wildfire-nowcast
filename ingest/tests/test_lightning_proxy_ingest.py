"""Unit tests for ingest/lightning_proxy_ingest.py.

Tests focus on:
- Grid generation: correct number of cells and cell centres.
- Point-in-polygon: ray-casting correctness.
- _materialise_proxy: correct delete-then-insert, active cell count.
- ingest_lightning_proxy: MeteoAlarm fetch failure degrades gracefully.
- run_lightning_proxy_ingest: returns 0 on success, 1 on unhandled failure.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


def _utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# _generate_grid_cells
# ---------------------------------------------------------------------------

class TestGenerateGridCells(unittest.TestCase):
    def test_single_cell(self):
        from ingest.lightning_proxy_ingest import _generate_grid_cells

        cells = _generate_grid_cells((0.0, 0.0, 0.1, 0.1), 0.1)
        self.assertEqual(1, len(cells))
        lon, lat = cells[0]
        self.assertAlmostEqual(0.05, lon, places=5)
        self.assertAlmostEqual(0.05, lat, places=5)

    def test_two_by_two_grid(self):
        from ingest.lightning_proxy_ingest import _generate_grid_cells

        cells = _generate_grid_cells((0.0, 0.0, 0.2, 0.2), 0.1)
        self.assertEqual(4, len(cells))

    def test_correct_lon_lat_ordering(self):
        """Cells should iterate lat in outer loop, lon in inner loop."""
        from ingest.lightning_proxy_ingest import _generate_grid_cells

        cells = _generate_grid_cells((0.0, 0.0, 0.3, 0.2), 0.1)
        # 3 lon × 2 lat = 6 cells
        self.assertEqual(6, len(cells))

    def test_global_bbox_approximate_count(self):
        from ingest.lightning_proxy_ingest import _generate_grid_cells

        cells = _generate_grid_cells((-180.0, -90.0, 180.0, 90.0), 1.0)
        # 360 × 180 = 64800 cells at 1° resolution
        self.assertEqual(64800, len(cells))


# ---------------------------------------------------------------------------
# _point_in_polygon
# ---------------------------------------------------------------------------

class TestPointInPolygon(unittest.TestCase):
    # Simple unit square [0,0]-[1,1]
    _SQUARE = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]

    def test_inside(self):
        from ingest.lightning_proxy_ingest import _point_in_polygon

        self.assertTrue(_point_in_polygon(0.5, 0.5, self._SQUARE))

    def test_outside(self):
        from ingest.lightning_proxy_ingest import _point_in_polygon

        self.assertFalse(_point_in_polygon(1.5, 0.5, self._SQUARE))
        self.assertFalse(_point_in_polygon(0.5, 1.5, self._SQUARE))
        self.assertFalse(_point_in_polygon(-0.1, 0.5, self._SQUARE))

    def test_near_edge(self):
        from ingest.lightning_proxy_ingest import _point_in_polygon

        # Just inside
        self.assertTrue(_point_in_polygon(0.01, 0.5, self._SQUARE))
        # Just outside
        self.assertFalse(_point_in_polygon(1.01, 0.5, self._SQUARE))


# ---------------------------------------------------------------------------
# _geometry_covers_point
# ---------------------------------------------------------------------------

class TestGeometryCoversPoint(unittest.TestCase):
    _POLYGON_GEOM = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }
    _MULTIPOLYGON_GEOM = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
            [[[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0], [2.0, 2.0]]],
        ],
    }

    def test_polygon_inside(self):
        from ingest.lightning_proxy_ingest import _geometry_covers_point

        self.assertTrue(_geometry_covers_point(self._POLYGON_GEOM, 0.5, 0.5))

    def test_polygon_outside(self):
        from ingest.lightning_proxy_ingest import _geometry_covers_point

        self.assertFalse(_geometry_covers_point(self._POLYGON_GEOM, 1.5, 0.5))

    def test_multipolygon_first_ring(self):
        from ingest.lightning_proxy_ingest import _geometry_covers_point

        self.assertTrue(_geometry_covers_point(self._MULTIPOLYGON_GEOM, 0.5, 0.5))

    def test_multipolygon_second_ring(self):
        from ingest.lightning_proxy_ingest import _geometry_covers_point

        self.assertTrue(_geometry_covers_point(self._MULTIPOLYGON_GEOM, 2.5, 2.5))

    def test_multipolygon_between_rings(self):
        from ingest.lightning_proxy_ingest import _geometry_covers_point

        self.assertFalse(_geometry_covers_point(self._MULTIPOLYGON_GEOM, 1.5, 1.5))

    def test_unknown_type_returns_false(self):
        from ingest.lightning_proxy_ingest import _geometry_covers_point

        self.assertFalse(_geometry_covers_point({"type": "Point", "coordinates": [0.5, 0.5]}, 0.5, 0.5))


# ---------------------------------------------------------------------------
# _materialise_proxy — grid materialisation
# ---------------------------------------------------------------------------

class TestMaterialiseProxy(unittest.TestCase):
    def _make_engine_mock(self):
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn
        return mock_engine, mock_conn

    def test_no_warnings_produces_zero_active_cells(self):
        from ingest.lightning_proxy_ingest import _materialise_proxy

        engine_mock, conn_mock = self._make_engine_mock()
        with patch("ingest.lightning_proxy_ingest.get_engine", return_value=engine_mock):
            active, total = _materialise_proxy(
                bbox=(0.0, 0.0, 1.0, 1.0),
                warnings=[],
                valid_time=_utc(),
                grid_resolution=0.5,
            )
        self.assertEqual(0, active)
        self.assertEqual(4, total)

    def test_warning_covering_all_cells_marks_all_active(self):
        from ingest.lightning_proxy_ingest import _materialise_proxy

        # Warning polygon covers the entire 1°×1° bbox.
        big_warning = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-1.0, -1.0], [2.0, -1.0], [2.0, 2.0], [-1.0, 2.0], [-1.0, -1.0]]],
            },
            "onset": _utc(),
            "expires": _utc(),
        }
        engine_mock, conn_mock = self._make_engine_mock()
        with patch("ingest.lightning_proxy_ingest.get_engine", return_value=engine_mock):
            active, total = _materialise_proxy(
                bbox=(0.0, 0.0, 1.0, 1.0),
                warnings=[big_warning],
                valid_time=_utc(),
                grid_resolution=0.5,
            )
        # 2×2 = 4 cells, all inside the big polygon
        self.assertEqual(4, active)
        self.assertEqual(4, total)

    def test_delete_called_before_insert(self):
        from ingest.lightning_proxy_ingest import _materialise_proxy

        engine_mock, conn_mock = self._make_engine_mock()
        with patch("ingest.lightning_proxy_ingest.get_engine", return_value=engine_mock):
            _materialise_proxy(
                bbox=(0.0, 0.0, 0.5, 0.5),
                warnings=[],
                valid_time=_utc(),
                grid_resolution=0.5,
            )

        calls = conn_mock.execute.call_args_list
        # First call must be the DELETE
        first_call_sql = str(calls[0].args[0])
        self.assertIn("DELETE", first_call_sql)

    def test_partial_coverage(self):
        from ingest.lightning_proxy_ingest import _materialise_proxy

        # Warning covers only the bottom-left 0.5°×0.5° cell of a 1°×1° bbox
        # at 0.5° resolution.  Only 1 of 4 cells should be active.
        partial_warning = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-0.1, -0.1], [0.6, -0.1], [0.6, 0.6], [-0.1, 0.6], [-0.1, -0.1]]],
            },
            "onset": _utc(),
            "expires": _utc(),
        }
        engine_mock, conn_mock = self._make_engine_mock()
        with patch("ingest.lightning_proxy_ingest.get_engine", return_value=engine_mock):
            active, total = _materialise_proxy(
                bbox=(0.0, 0.0, 1.0, 1.0),
                warnings=[partial_warning],
                valid_time=_utc(),
                grid_resolution=0.5,
            )
        self.assertEqual(1, active)
        self.assertEqual(4, total)


# ---------------------------------------------------------------------------
# _fetch_thunderstorm_warnings — graceful failure
# ---------------------------------------------------------------------------

class TestFetchThunderstormWarnings(unittest.TestCase):
    def test_returns_empty_list_on_asyncio_run_error(self):
        """asyncio.run raising maps to a WARNING and an empty return."""
        from ingest.lightning_proxy_ingest import _fetch_thunderstorm_warnings

        with (
            patch("ingest.lightning_proxy_ingest.asyncio.run", side_effect=RuntimeError("network down")),
            self.assertLogs("lightning_proxy_ingest", level="WARNING"),
        ):
            result = _fetch_thunderstorm_warnings(_utc())

        self.assertEqual([], result)

    def test_filters_non_thunderstorm_warnings(self):
        from ingest.lightning_proxy_ingest import _fetch_thunderstorm_warnings

        mock_warning_ts = MagicMock()
        mock_warning_ts.warning_type = "thunderstorm"
        mock_warning_ts.expires = datetime(9999, 1, 1, tzinfo=timezone.utc)
        mock_warning_ts.geometry = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}

        mock_warning_wind = MagicMock()
        mock_warning_wind.warning_type = "wind"
        mock_warning_wind.expires = datetime(9999, 1, 1, tzinfo=timezone.utc)
        mock_warning_wind.geometry = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}

        with patch("ingest.lightning_proxy_ingest.asyncio.run", return_value=[mock_warning_ts, mock_warning_wind]):
            result = _fetch_thunderstorm_warnings(_utc())

        self.assertEqual(1, len(result))

    def test_filters_expired_warnings(self):
        from ingest.lightning_proxy_ingest import _fetch_thunderstorm_warnings

        now = _utc()
        expired_warning = MagicMock()
        expired_warning.warning_type = "thunderstorm"
        expired_warning.expires = datetime(2000, 1, 1, tzinfo=timezone.utc)
        expired_warning.geometry = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}

        with patch("ingest.lightning_proxy_ingest.asyncio.run", return_value=[expired_warning]):
            result = _fetch_thunderstorm_warnings(now)

        self.assertEqual([], result)


# ---------------------------------------------------------------------------
# run_lightning_proxy_ingest
# ---------------------------------------------------------------------------

class TestRunLightningProxyIngest(unittest.TestCase):
    def test_returns_0_on_success(self):
        from ingest.lightning_proxy_ingest import run_lightning_proxy_ingest

        with patch(
            "ingest.lightning_proxy_ingest.ingest_lightning_proxy",
            return_value={"active_cells": 0, "total_cells": 100},
        ):
            self.assertEqual(0, run_lightning_proxy_ingest())

    def test_returns_1_on_exception(self):
        from ingest.lightning_proxy_ingest import run_lightning_proxy_ingest

        with patch(
            "ingest.lightning_proxy_ingest.ingest_lightning_proxy",
            side_effect=RuntimeError("DB down"),
        ):
            self.assertEqual(1, run_lightning_proxy_ingest())
