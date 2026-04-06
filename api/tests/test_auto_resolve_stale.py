"""Tests for auto_resolve_stale_review_queue."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from api.fires.repo import auto_resolve_stale_review_queue


def _make_mock_engine(rowcount: int) -> MagicMock:
    """Return a mock engine whose connection reports *rowcount* rows updated."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.rowcount = rowcount

    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn
    return mock_engine


class TestAutoResolveStaleReviewQueue(unittest.TestCase):
    def test_returns_zero_when_no_stale_items(self):
        """When no open items exceed the timeout, zero rows are resolved."""
        mock_engine = _make_mock_engine(rowcount=0)

        with patch("api.fires.repo.get_engine", return_value=mock_engine):
            result = auto_resolve_stale_review_queue(timeout_days=7)

        self.assertEqual(result, 0)

    def test_returns_count_of_resolved_items(self):
        """Resolved count matches the number of rows updated by the SQL statement."""
        mock_engine = _make_mock_engine(rowcount=42)

        with patch("api.fires.repo.get_engine", return_value=mock_engine):
            result = auto_resolve_stale_review_queue(timeout_days=7)

        self.assertEqual(result, 42)

    def test_timeout_days_forwarded_to_sql(self):
        """The timeout_days parameter is passed through to the SQL execute call."""
        mock_engine = _make_mock_engine(rowcount=0)

        with patch("api.fires.repo.get_engine", return_value=mock_engine):
            auto_resolve_stale_review_queue(timeout_days=14)

        mock_conn = mock_engine.begin.return_value
        _stmt, params = mock_conn.execute.call_args[0]
        self.assertEqual(params["timeout_days"], 14)

    def test_default_timeout_is_seven_days(self):
        """The default timeout_days should be 7."""
        mock_engine = _make_mock_engine(rowcount=0)

        with patch("api.fires.repo.get_engine", return_value=mock_engine):
            auto_resolve_stale_review_queue()

        mock_conn = mock_engine.begin.return_value
        _stmt, params = mock_conn.execute.call_args[0]
        self.assertEqual(params["timeout_days"], 7)

    def test_sql_sets_correct_resolution_fields(self):
        """The SQL must set resolved_by='auto:timeout' and resolved_notes='auto_resolved_timeout'."""
        import inspect
        from api.fires.repo import auto_resolve_stale_review_queue as fn

        src = inspect.getsource(fn)
        assert "auto:timeout" in src
        assert "auto_resolved_timeout" in src

    def test_logs_resolved_count(self):
        """Auto-resolution should log how many items were resolved."""
        mock_engine = _make_mock_engine(rowcount=5)
        import logging

        with (
            patch("api.fires.repo.get_engine", return_value=mock_engine),
            self.assertLogs("api.fires.repo", level=logging.INFO) as cm,
        ):
            auto_resolve_stale_review_queue(timeout_days=7)

        log_output = "\n".join(cm.output)
        self.assertIn("5", log_output)
        self.assertIn("Auto-resolved", log_output)
