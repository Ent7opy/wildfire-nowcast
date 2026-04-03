"""Unit tests for ml.ignition.snapshot — feature extraction and label logic."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.ignition.snapshot import (
    _build_grid,
    _DAYS_SINCE_BURN_CAP,
    _GFS_GRID_DEG,
    compute_days_since_last_burn,
    _query_ignition_labels,
)


# ── Grid construction ─────────────────────────────────────────────────────────

class TestBuildGrid:
    def test_grid_resolution_matches_gfs(self):
        grid = _build_grid(-124.0, 32.0, -120.0, 36.0)
        assert not grid.empty
        # All lon steps should be 0.25°.
        lons = np.sort(grid["lon_grid"].unique())
        diffs = np.diff(lons)
        np.testing.assert_allclose(diffs, _GFS_GRID_DEG, atol=1e-6)

    def test_grid_coverage(self):
        grid = _build_grid(-122.0, 37.0, -121.0, 38.0)
        # At least one cell must fall within the bbox.
        assert (grid["lon_grid"] >= -122.0).all()
        assert (grid["lon_grid"] <= -121.0 + _GFS_GRID_DEG).all()

    def test_grid_small_bbox_has_cells(self):
        grid = _build_grid(-122.0, 37.0, -121.75, 37.25)
        assert len(grid) >= 1

    def test_grid_columns(self):
        grid = _build_grid(-122.0, 37.0, -120.0, 39.0)
        assert set(grid.columns) == {"lon_grid", "lat_grid"}


# ── Days-since-last-burn ──────────────────────────────────────────────────────

class TestDaysSinceLastBurn:
    """Tests for compute_days_since_last_burn using a mocked DB engine."""

    def _make_engine(self, result_rows: list[dict]) -> MagicMock:
        """Return a mock engine that yields result_rows from a SQL query."""
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result_df = pd.DataFrame(result_rows) if result_rows else pd.DataFrame(
            columns=["cell_idx", "days_since_last_burn"]
        )
        mock_conn.execute = MagicMock()

        # Patch pd.read_sql to return our controlled result.
        self._mock_read_sql = result_df
        return mock_engine

    def test_recently_burned_cell_returns_low_days(self):
        """Cells burned within 12 months should return a low days_since_last_burn."""
        grid = pd.DataFrame({"lon_grid": [-121.5], "lat_grid": [37.5]})
        ref_time = datetime(2025, 8, 1, tzinfo=timezone.utc)

        recent_days = 30.0  # burned 30 days ago
        expected_rows = [{"cell_idx": 0, "days_since_last_burn": recent_days}]

        with patch("ml.ignition.snapshot.pd.read_sql", return_value=pd.DataFrame(expected_rows)):
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            result = compute_days_since_last_burn(mock_engine, grid, ref_time)

        assert len(result) == 1
        assert float(result.iloc[0]) == pytest.approx(recent_days)

    def test_unburned_cell_returns_cap(self):
        """Cells with no fire history return the cap (3650 days)."""
        grid = pd.DataFrame({"lon_grid": [-121.5], "lat_grid": [37.5]})
        ref_time = datetime(2025, 8, 1, tzinfo=timezone.utc)

        # Query returns no matches (LEFT JOIN produces NULL → cap is used).
        empty_result = pd.DataFrame(columns=["cell_idx", "days_since_last_burn"])

        with patch("ml.ignition.snapshot.pd.read_sql", return_value=empty_result):
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            result = compute_days_since_last_burn(mock_engine, grid, ref_time)

        assert len(result) == 1
        assert float(result.iloc[0]) == pytest.approx(_DAYS_SINCE_BURN_CAP)

    def test_result_capped_at_cap_days(self):
        """Values returned by DB should be capped at cap_days."""
        grid = pd.DataFrame({"lon_grid": [-121.5], "lat_grid": [37.5]})
        ref_time = datetime(2025, 8, 1, tzinfo=timezone.utc)

        # DB returns a value above the cap (shouldn't happen, but test the cap).
        oversized_rows = [{"cell_idx": 0, "days_since_last_burn": 9999.0}]

        with patch("ml.ignition.snapshot.pd.read_sql", return_value=pd.DataFrame(oversized_rows)):
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            result = compute_days_since_last_burn(
                mock_engine, grid, ref_time, cap_days=_DAYS_SINCE_BURN_CAP
            )

        assert float(result.iloc[0]) <= _DAYS_SINCE_BURN_CAP

    def test_multiple_cells_ordered_correctly(self):
        """Results for multiple cells must align with the input grid order."""
        grid = pd.DataFrame({
            "lon_grid": [-121.5, -121.25, -121.0],
            "lat_grid": [37.5, 37.5, 37.5],
        })
        ref_time = datetime(2025, 8, 1, tzinfo=timezone.utc)
        db_rows = [
            {"cell_idx": 0, "days_since_last_burn": 10.0},
            {"cell_idx": 2, "days_since_last_burn": 500.0},
            # cell 1 has no match → should return cap.
        ]

        with patch("ml.ignition.snapshot.pd.read_sql", return_value=pd.DataFrame(db_rows)):
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            result = compute_days_since_last_burn(mock_engine, grid, ref_time)

        assert len(result) == 3
        assert float(result.iloc[0]) == pytest.approx(10.0)
        assert float(result.iloc[1]) == pytest.approx(_DAYS_SINCE_BURN_CAP)
        assert float(result.iloc[2]) == pytest.approx(500.0)

    def test_empty_grid_returns_empty_series(self):
        """An empty grid should return an empty Series without error."""
        grid = pd.DataFrame(columns=["lon_grid", "lat_grid"])
        ref_time = datetime(2025, 8, 1, tzinfo=timezone.utc)
        mock_engine = MagicMock()

        result = compute_days_since_last_burn(mock_engine, grid, ref_time)
        assert len(result) == 0


# ── Label construction ────────────────────────────────────────────────────────

class TestIgnitionLabels:
    """Verify new-ignition vs. spread-detection labeling logic."""

    def _make_engine_for_labels(
        self,
        future_rows: list[dict],
        prior_rows: list[dict],
    ) -> MagicMock:
        """Return a mock engine that cycles through future/prior results."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        return mock_engine

    def test_new_ignition_no_prior_is_positive(self):
        """Cell with future fire AND no prior activity → positive (1)."""
        grid = pd.DataFrame({"lon_grid": [-121.5], "lat_grid": [37.5]})
        ref_time = datetime(2025, 8, 1, tzinfo=timezone.utc)

        future_df = pd.DataFrame([{"lon_grid": -121.5, "lat_grid": 37.5, "det_count": 3}])
        prior_df = pd.DataFrame(columns=["lon_grid", "lat_grid", "det_count"])

        call_count = [0]

        def mock_read_sql(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return future_df
            return prior_df

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch("ml.ignition.snapshot.pd.read_sql", side_effect=mock_read_sql):
            labels = _query_ignition_labels(mock_engine, grid, ref_time)

        assert labels.loc[0, "ignition_label"] == 1

    def test_spread_detection_prior_activity_is_negative(self):
        """Cell with future fire AND prior activity → negative (0, it's spread)."""
        grid = pd.DataFrame({"lon_grid": [-121.5], "lat_grid": [37.5]})
        ref_time = datetime(2025, 8, 1, tzinfo=timezone.utc)

        # Both future and prior have detections → this is spread, not new ignition.
        future_df = pd.DataFrame([{"lon_grid": -121.5, "lat_grid": 37.5, "det_count": 3}])
        prior_df = pd.DataFrame([{"lon_grid": -121.5, "lat_grid": 37.5, "det_count": 2}])

        call_count = [0]

        def mock_read_sql(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return future_df
            return prior_df

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch("ml.ignition.snapshot.pd.read_sql", side_effect=mock_read_sql):
            labels = _query_ignition_labels(mock_engine, grid, ref_time)

        assert labels.loc[0, "ignition_label"] == 0

    def test_no_future_detections_is_negative(self):
        """Cell with no future detections → negative (0)."""
        grid = pd.DataFrame({"lon_grid": [-121.5], "lat_grid": [37.5]})
        ref_time = datetime(2025, 8, 1, tzinfo=timezone.utc)

        empty_df = pd.DataFrame(columns=["lon_grid", "lat_grid", "det_count"])

        with patch("ml.ignition.snapshot.pd.read_sql", return_value=empty_df):
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            labels = _query_ignition_labels(mock_engine, grid, ref_time)

        assert labels.loc[0, "ignition_label"] == 0

    def test_label_columns_present(self):
        """Output always has lon_grid, lat_grid, ignition_label."""
        grid = pd.DataFrame({"lon_grid": [-121.5, -121.25], "lat_grid": [37.5, 37.5]})
        ref_time = datetime(2025, 8, 1, tzinfo=timezone.utc)

        empty_df = pd.DataFrame(columns=["lon_grid", "lat_grid", "det_count"])

        with patch("ml.ignition.snapshot.pd.read_sql", return_value=empty_df):
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

            labels = _query_ignition_labels(mock_engine, grid, ref_time)

        assert set(labels.columns) >= {"lon_grid", "lat_grid", "ignition_label"}
        assert len(labels) == 2
