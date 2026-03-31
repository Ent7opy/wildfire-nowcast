"""Unit tests for ingest/drought_ingest.py.

Tests focus on:
- Idempotency: rerunning does not duplicate rows when valid_time already exists.
- Stale WARNING: emitted when the latest completed run is older than threshold.
- CDSAPI_KEY absent: job skips gracefully without raising.
- run_drought_ingest: returns 0 on success and on a skip, 1 on failure.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(days_ago: float = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


# ---------------------------------------------------------------------------
# _check_staleness
# ---------------------------------------------------------------------------

class TestCheckStaleness(unittest.TestCase):
    def test_no_completed_run_logs_warning(self):
        from ingest.drought_ingest import _check_staleness

        with self.assertLogs("drought_ingest", level="WARNING") as cm:
            _check_staleness(None)

        self.assertTrue(any("No completed drought index" in line for line in cm.output))

    def test_fresh_run_does_not_warn(self):
        from ingest.drought_ingest import _check_staleness

        latest = {"valid_time": _utc(days_ago=1)}
        # Should produce no WARNING logs.
        with self.assertLogs("drought_ingest", level="INFO") as cm:
            import logging
            logging.getLogger("drought_ingest").info("probe")  # ensure logger fires
            _check_staleness(latest)

        warning_lines = [line for line in cm.output if "WARNING" in line and "stale" in line.lower()]
        self.assertEqual([], warning_lines)

    def test_stale_run_logs_warning(self):
        from ingest.drought_ingest import _check_staleness

        latest = {"valid_time": _utc(days_ago=11)}
        with self.assertLogs("drought_ingest", level="WARNING") as cm:
            _check_staleness(latest)

        self.assertTrue(any("stale" in line.lower() for line in cm.output))
        self.assertTrue(any("science_grade" in line for line in cm.output))

    def test_just_under_threshold_does_not_warn(self):
        from ingest.drought_ingest import _check_staleness

        # 9.9 days ago: clearly within the 10-day threshold, must not warn.
        latest = {"valid_time": _utc(days_ago=9.9)}
        with self.assertNoLogs("drought_ingest", level="WARNING"):
            _check_staleness(latest)


# ---------------------------------------------------------------------------
# _already_ingested (idempotency)
# ---------------------------------------------------------------------------

class TestAlreadyIngested(unittest.TestCase):
    def test_returns_false_when_db_empty(self):
        from ingest.drought_ingest import _already_ingested

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = None

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch("ingest.drought_ingest.get_engine", return_value=mock_engine):
            result = _already_ingested(_utc())

        self.assertFalse(result)

    def test_returns_true_when_row_exists(self):
        from ingest.drought_ingest import _already_ingested

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = (1,)

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch("ingest.drought_ingest.get_engine", return_value=mock_engine):
            result = _already_ingested(_utc())

        self.assertTrue(result)


# ---------------------------------------------------------------------------
# ingest_drought_index — key-absent skip
# ---------------------------------------------------------------------------

class TestIngestDroughtIndexNoKey(unittest.TestCase):
    def test_skips_gracefully_when_cdsapi_key_missing(self):
        from ingest.drought_ingest import ingest_drought_index

        with patch.dict("os.environ", {}, clear=False):
            # Ensure the key is absent regardless of host environment.
            import os
            os.environ.pop("CDSAPI_KEY", None)

            with self.assertLogs("drought_ingest", level="WARNING") as cm:
                result = ingest_drought_index()

        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "CDSAPI_KEY not configured")
        self.assertTrue(any("CDSAPI_KEY" in line for line in cm.output))


# ---------------------------------------------------------------------------
# ingest_drought_index — idempotency (already ingested)
# ---------------------------------------------------------------------------

class TestIngestDroughtIndexIdempotency(unittest.TestCase):
    def test_skips_when_valid_time_already_ingested(self):
        from ingest.drought_ingest import ingest_drought_index

        valid_time = _utc()

        with (
            patch.dict("os.environ", {"CDSAPI_KEY": "123:abc"}),
            patch("ingest.drought_ingest._latest_completed_ingest", return_value=None),
            patch("ingest.drought_ingest._check_staleness"),
            patch("ingest.drought_ingest._fetch_via_cdsapi", return_value={"valid_time": valid_time.isoformat()}),
            patch("ingest.drought_ingest._resolve_valid_time", return_value=valid_time),
            patch("ingest.drought_ingest._already_ingested", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.unlink"),
        ):
            result = ingest_drought_index()

        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "already_ingested")


# ---------------------------------------------------------------------------
# run_drought_ingest
# ---------------------------------------------------------------------------

class TestRunDroughtIngest(unittest.TestCase):
    def test_returns_0_on_success(self):
        from ingest.drought_ingest import run_drought_ingest

        with patch("ingest.drought_ingest.ingest_drought_index", return_value={"run_id": 1}):
            self.assertEqual(0, run_drought_ingest())

    def test_returns_0_on_skip(self):
        from ingest.drought_ingest import run_drought_ingest

        with patch(
            "ingest.drought_ingest.ingest_drought_index",
            return_value={"skipped": True, "reason": "CDSAPI_KEY not configured"},
        ):
            self.assertEqual(0, run_drought_ingest())

    def test_returns_1_on_exception(self):
        from ingest.drought_ingest import run_drought_ingest

        with patch(
            "ingest.drought_ingest.ingest_drought_index",
            side_effect=RuntimeError("CDS API down"),
        ):
            self.assertEqual(1, run_drought_ingest())


# ---------------------------------------------------------------------------
# _resolve_valid_time
# ---------------------------------------------------------------------------

class TestResolveValidTime(unittest.TestCase):
    def test_extracts_from_valid_time_key(self):
        from ingest.drought_ingest import _resolve_valid_time

        ts = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        meta = {"valid_time": "2026-03-01T00:00:00+00:00"}
        result = _resolve_valid_time(meta, fallback=_utc())
        self.assertEqual(ts, result)

    def test_falls_back_when_no_key(self):
        from ingest.drought_ingest import _resolve_valid_time

        fallback = _utc()
        result = _resolve_valid_time({}, fallback=fallback)
        self.assertEqual(fallback, result)

    def test_falls_back_on_unparseable_value(self):
        from ingest.drought_ingest import _resolve_valid_time

        fallback = _utc()
        result = _resolve_valid_time({"valid_time": "not-a-date"}, fallback=fallback)
        self.assertEqual(fallback, result)
