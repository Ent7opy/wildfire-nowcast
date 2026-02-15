import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from ingest.firms_client import FirmsValidationSummary
from ingest.firms_ingest import _filter_detections_by_watermark, run_firms_ingest


class TestFirmsIncrementalWatermark(unittest.TestCase):
    def test_filter_detections_by_watermark_includes_late_arrivals_and_excludes_old_rows(self):
        watermark = datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc)
        detections = [
            SimpleNamespace(acq_time=datetime(2026, 2, 15, 10, 20, tzinfo=timezone.utc)),
            # Late arrival inside 90-minute grace window.
            SimpleNamespace(acq_time=datetime(2026, 2, 15, 10, 31, tzinfo=timezone.utc)),
            SimpleNamespace(acq_time=datetime(2026, 2, 15, 12, 10, tzinfo=timezone.utc)),
        ]

        filtered, max_seen = _filter_detections_by_watermark(
            detections,
            watermark_time_utc=watermark,
            grace_minutes=90,
        )

        self.assertEqual(2, len(filtered))
        self.assertEqual(datetime(2026, 2, 15, 12, 10, tzinfo=timezone.utc), max_seen)

    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows")
    @patch("ingest.firms_ingest.repository.create_ingest_batch")
    @patch("ingest.firms_ingest.repository.get_ingest_watermark")
    @patch("ingest.firms_ingest.ingest_settings")
    def test_watermark_advances_only_after_success(
        self,
        mock_settings,
        mock_get_watermark,
        mock_create_batch,
        mock_fetch_rows,
        mock_parse_rows,
        mock_insert,
        mock_finalize,
        mock_advance,
        _mock_scoring,
    ):
        mock_settings.map_key = "test-key"
        mock_settings.resolved_area = "20,40,21,41"
        mock_settings.day_range = 1
        mock_settings.sources = ["VIIRS_SNPP_NRT"]
        mock_settings.request_timeout_seconds = 30.0
        mock_settings.firms_watermark_grace_minutes = 90
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = False

        mock_get_watermark.return_value = {
            "last_acq_time_utc": datetime(2026, 2, 15, 11, 0, tzinfo=timezone.utc),
        }
        mock_create_batch.return_value = 321
        mock_fetch_rows.return_value = [{"id": "x"}]

        detection = SimpleNamespace(acq_time=datetime(2026, 2, 15, 11, 10, tzinfo=timezone.utc))
        mock_parse_rows.return_value = ([detection], FirmsValidationSummary(total_rows=1, parsed_rows=1))
        mock_insert.return_value = 1

        code = run_firms_ingest(day_range=None, area=None, sources=None)

        self.assertEqual(0, code)
        mock_finalize.assert_called_once()
        self.assertEqual("succeeded", mock_finalize.call_args.kwargs["status"])
        mock_advance.assert_called_once_with(
            source="VIIRS_SNPP_NRT",
            area_key="20.000000,40.000000,21.000000,41.000000",
            last_acq_time_utc=datetime(2026, 2, 15, 11, 10, tzinfo=timezone.utc),
            last_batch_id=321,
        )

    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows")
    @patch("ingest.firms_ingest.repository.create_ingest_batch")
    @patch("ingest.firms_ingest.repository.get_ingest_watermark")
    @patch("ingest.firms_ingest.ingest_settings")
    def test_watermark_not_advanced_on_failed_batch(
        self,
        mock_settings,
        mock_get_watermark,
        mock_create_batch,
        mock_fetch_rows,
        mock_parse_rows,
        mock_insert,
        mock_finalize,
        mock_advance,
    ):
        mock_settings.map_key = "test-key"
        mock_settings.resolved_area = "20,40,21,41"
        mock_settings.day_range = 1
        mock_settings.sources = ["VIIRS_SNPP_NRT"]
        mock_settings.request_timeout_seconds = 30.0
        mock_settings.firms_watermark_grace_minutes = 90
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = False

        mock_get_watermark.return_value = None
        mock_create_batch.return_value = 555
        mock_fetch_rows.return_value = [{"id": "x"}]
        detection = SimpleNamespace(acq_time=datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc))
        mock_parse_rows.return_value = ([detection], FirmsValidationSummary(total_rows=1, parsed_rows=1))
        mock_insert.side_effect = RuntimeError("boom")

        code = run_firms_ingest(day_range=None, area=None, sources=None)

        self.assertEqual(1, code)
        mock_finalize.assert_called_once()
        self.assertEqual("failed", mock_finalize.call_args.kwargs["status"])
        mock_advance.assert_not_called()

    @patch("ingest.firms_ingest._resolve_denoiser_model_run_dir", return_value=None)
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows")
    @patch("ingest.firms_ingest.repository.create_ingest_batch")
    @patch("ingest.firms_ingest.repository.get_ingest_watermark")
    @patch("ingest.firms_ingest.ingest_settings")
    def test_required_denoiser_fails_batch_without_resolved_model(
        self,
        mock_settings,
        _mock_get_watermark,
        mock_create_batch,
        mock_fetch_rows,
        mock_parse_rows,
        mock_insert,
        mock_finalize,
        mock_advance,
        _mock_resolve_model,
    ):
        mock_settings.map_key = "test-key"
        mock_settings.resolved_area = "20,40,21,41"
        mock_settings.day_range = 1
        mock_settings.sources = ["VIIRS_SNPP_NRT"]
        mock_settings.request_timeout_seconds = 30.0
        mock_settings.firms_watermark_grace_minutes = 90
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = True

        mock_create_batch.return_value = 777
        mock_fetch_rows.return_value = [{"id": "x"}]
        detection = SimpleNamespace(acq_time=datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc))
        mock_parse_rows.return_value = ([detection], FirmsValidationSummary(total_rows=1, parsed_rows=1))
        mock_insert.return_value = 1

        code = run_firms_ingest(day_range=None, area=None, sources=None)

        self.assertEqual(1, code)
        mock_finalize.assert_called_once()
        self.assertEqual("failed", mock_finalize.call_args.kwargs["status"])
        mock_advance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
