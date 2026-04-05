"""Tests for denoiser timeout circuit breaker logic (Issue #287)."""

import unittest
from unittest.mock import MagicMock, patch

from ingest import repository


class TestDenoiserTimeoutCircuitBreaker(unittest.TestCase):
    """Tests for tracking and handling consecutive denoiser timeouts."""

    @patch("ingest.repository.get_engine")
    def test_count_consecutive_denoiser_timeout_batches_none(self, mock_engine):
        """Test counting when there are no timeout batches."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("succeeded", None),
            ("no_data", None),
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        count = repository.count_consecutive_denoiser_timeout_batches(
            source="VIIRS_SNPP_NRT",
            area_key="-180.000000,-90.000000,180.000000,90.000000",
            threshold=3,
        )

        self.assertEqual(count, 0, "Should return 0 when most recent batch is not a timeout")

    @patch("ingest.repository.get_engine")
    def test_count_consecutive_denoiser_timeout_batches_one(self, mock_engine):
        """Test counting when there is exactly 1 timeout batch."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("failed", "true"),  # Most recent is timeout
            ("succeeded", None),
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        count = repository.count_consecutive_denoiser_timeout_batches(
            source="VIIRS_SNPP_NRT",
            area_key="-180.000000,-90.000000,180.000000,90.000000",
            threshold=3,
        )

        self.assertEqual(count, 1, "Should count 1 consecutive timeout")

    @patch("ingest.repository.get_engine")
    def test_count_consecutive_denoiser_timeout_batches_multiple(self, mock_engine):
        """Test counting when there are multiple consecutive timeout batches."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("failed", "true"),  # Timeout 3
            ("failed", "true"),  # Timeout 2
            ("failed", "true"),  # Timeout 1
            ("succeeded", None),  # Non-timeout, stops count
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        count = repository.count_consecutive_denoiser_timeout_batches(
            source="VIIRS_SNPP_NRT",
            area_key="-180.000000,-90.000000,180.000000,90.000000",
            threshold=5,
        )

        self.assertEqual(count, 3, "Should count 3 consecutive timeouts")

    @patch("ingest.repository.get_engine")
    def test_count_consecutive_denoiser_timeout_batches_non_timeout_failure(self, mock_engine):
        """Test that non-timeout failures don't count toward consecutive count."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("failed", None),  # Failed but not a timeout, stops count
            ("failed", "true"),  # Timeout batches below don't count
            ("failed", "true"),
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        count = repository.count_consecutive_denoiser_timeout_batches(
            source="VIIRS_SNPP_NRT",
            area_key="-180.000000,-90.000000,180.000000,90.000000",
            threshold=5,
        )

        self.assertEqual(count, 0, "Should return 0 when non-timeout failure breaks the chain")

    @patch("ingest.repository.get_engine")
    def test_mark_batch_detections_review_required(self, mock_engine):
        """Test marking detections as review_required."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 42
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        count = repository.mark_batch_detections_review_required(batch_id=123)

        self.assertEqual(count, 42, "Should return the number of updated rows")
        mock_conn.execute.assert_called_once()

    @patch("ingest.repository.get_engine")
    def test_mark_batch_detections_review_required_with_conn(self, mock_engine):
        """Test marking detections as review_required with provided connection."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 15
        mock_conn.execute.return_value = mock_result

        count = repository.mark_batch_detections_review_required(batch_id=456, conn=mock_conn)

        self.assertEqual(count, 15, "Should return the number of updated rows")
        mock_conn.execute.assert_called_once()
        # get_engine should not be called when conn is provided
        mock_engine.assert_not_called()


class TestDenoiserTimeoutErrorContext(unittest.TestCase):
    """Tests for recording denoiser timeout context in batch metadata."""

    @patch("ingest.repository.get_engine")
    def test_finalize_ingest_batch_with_error_context(self, mock_engine):
        """Test that error_context is properly recorded in batch metadata."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        error_context = {
            "denoiser_timeout": True,
            "timeout_seconds": 600,
            "consecutive_timeout_count": 2,
            "threshold": 3,
            "error_message": "Denoiser timed out after 600s",
        }

        repository.finalize_ingest_batch(
            batch_id=789,
            status="failed",
            fetched=100,
            inserted=0,
            skipped=0,
            conn=mock_conn,
            error_context=error_context,
        )

        # Verify execute was called
        mock_conn.execute.assert_called_once()

    @patch("ingest.repository.get_engine")
    def test_finalize_ingest_batch_without_error_context(self, mock_engine):
        """Test that finalize_ingest_batch works without error_context (backward compat)."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        repository.finalize_ingest_batch(
            batch_id=790,
            status="succeeded",
            fetched=100,
            inserted=95,
            skipped=5,
            conn=mock_conn,
        )

        mock_conn.execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
