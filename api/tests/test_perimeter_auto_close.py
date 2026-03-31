"""Tests for auto_close_review_queue_by_perimeters."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from api.fires.repo import auto_close_review_queue_by_perimeters


def _make_mock_engine(rows: list[dict]) -> MagicMock:
    """Return a mock engine whose connection returns *rows* from execute()."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.mappings.return_value.all.return_value = rows

    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn
    return mock_engine


class TestAutoCloseReviewQueueByPerimeters(unittest.TestCase):
    def test_returns_empty_list_when_no_perimeter_matches(self):
        """Items whose centroid is outside all perimeters produce no auto-closures."""
        mock_engine = _make_mock_engine([])

        with patch("api.fires.repo.get_engine", return_value=mock_engine):
            result = auto_close_review_queue_by_perimeters()

        self.assertEqual(result, [])

    def test_returns_closed_items_when_centroid_inside_perimeter(self):
        """Items whose centroid falls within a fresh perimeter are auto-resolved."""
        fake_rows = [
            {
                "queue_id": 1,
                "event_id": "evt_inside_wfigs",
                "perimeter_ref": "42",
                "resolved_by": "auto:perimeter:wfigs",
            },
        ]
        mock_engine = _make_mock_engine(fake_rows)

        with patch("api.fires.repo.get_engine", return_value=mock_engine):
            result = auto_close_review_queue_by_perimeters()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event_id"], "evt_inside_wfigs")
        self.assertEqual(result[0]["resolved_by"], "auto:perimeter:wfigs")
        self.assertEqual(result[0]["perimeter_ref"], "42")

    def test_all_four_perimeter_sources_produce_correct_resolved_by(self):
        """resolved_by uses the canonical short name for every supported source."""
        fake_rows = [
            {"queue_id": 1, "event_id": "evt_wfigs", "perimeter_ref": "10", "resolved_by": "auto:perimeter:wfigs"},
            {"queue_id": 2, "event_id": "evt_cwfis", "perimeter_ref": "20", "resolved_by": "auto:perimeter:cwfis"},
            {"queue_id": 3, "event_id": "evt_copernicus", "perimeter_ref": "30", "resolved_by": "auto:perimeter:copernicus_ems"},
            {"queue_id": 4, "event_id": "evt_nifc", "perimeter_ref": "40", "resolved_by": "auto:perimeter:NIFC"},
        ]
        mock_engine = _make_mock_engine(fake_rows)

        with patch("api.fires.repo.get_engine", return_value=mock_engine):
            result = auto_close_review_queue_by_perimeters()

        resolved_bys = {r["event_id"]: r["resolved_by"] for r in result}
        self.assertEqual(resolved_bys["evt_wfigs"], "auto:perimeter:wfigs")
        self.assertEqual(resolved_bys["evt_cwfis"], "auto:perimeter:cwfis")
        self.assertEqual(resolved_bys["evt_copernicus"], "auto:perimeter:copernicus_ems")
        self.assertEqual(resolved_bys["evt_nifc"], "auto:perimeter:NIFC")

    def test_lookback_seconds_forwarded_to_sql(self):
        """The lookback_seconds parameter is passed through to the SQL execute call."""
        mock_engine = _make_mock_engine([])

        with patch("api.fires.repo.get_engine", return_value=mock_engine):
            auto_close_review_queue_by_perimeters(lookback_seconds=3600)

        mock_conn = mock_engine.begin.return_value
        _stmt, params = mock_conn.execute.call_args[0]
        self.assertEqual(params["lookback_seconds"], 3600)

    def test_multiple_open_items_all_resolved(self):
        """Multiple open items can be closed in a single call."""
        fake_rows = [
            {"queue_id": i, "event_id": f"evt_{i}", "perimeter_ref": str(i * 10), "resolved_by": "auto:perimeter:wfigs"}
            for i in range(1, 6)
        ]
        mock_engine = _make_mock_engine(fake_rows)

        with patch("api.fires.repo.get_engine", return_value=mock_engine):
            result = auto_close_review_queue_by_perimeters()

        self.assertEqual(len(result), 5)
