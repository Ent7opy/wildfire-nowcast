import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import ANY, patch

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

    def test_filter_detections_uses_hard_window_when_watermark_missing(self):
        detections = [
            SimpleNamespace(acq_time=datetime(2026, 2, 15, 8, 59, tzinfo=timezone.utc)),
            SimpleNamespace(acq_time=datetime(2026, 2, 15, 9, 1, tzinfo=timezone.utc)),
        ]
        filtered, max_seen = _filter_detections_by_watermark(
            detections,
            watermark_time_utc=None,
            grace_minutes=90,
            hard_window_start_utc=datetime(2026, 2, 15, 9, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(1, len(filtered))
        self.assertEqual(datetime(2026, 2, 15, 9, 1, tzinfo=timezone.utc), max_seen)

    def test_filter_detections_uses_max_of_watermark_and_hard_window(self):
        detections = [
            SimpleNamespace(acq_time=datetime(2026, 2, 15, 11, 20, tzinfo=timezone.utc)),
            SimpleNamespace(acq_time=datetime(2026, 2, 15, 11, 40, tzinfo=timezone.utc)),
        ]
        filtered, _max_seen = _filter_detections_by_watermark(
            detections,
            watermark_time_utc=datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc),
            grace_minutes=90,
            hard_window_start_utc=datetime(2026, 2, 15, 11, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(1, len(filtered))

    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", return_value=0)
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.repository.get_engine")
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
        mock_get_engine,
        mock_insert,
        _mock_incomplete,
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
        mock_settings.firms_initial_lookback_minutes = 100000
        mock_settings.firms_incremental_lookback_minutes = 100000
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = False
        mock_settings.firms_reconcile_unscored_batches = False
        mock_settings.firms_reconcile_max_batches = 5

        txn_conn = object()
        mock_get_engine.return_value.begin.return_value = nullcontext(txn_conn)

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
            conn=ANY,
        )

    @patch("ingest.firms_ingest._utc_now", return_value=datetime(2026, 3, 12, 15, 16, tzinfo=timezone.utc))
    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", return_value=0)
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.repository.get_engine")
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows")
    @patch("ingest.firms_ingest.repository.create_ingest_batch")
    @patch("ingest.firms_ingest.repository.get_ingest_watermark")
    @patch("ingest.firms_ingest.ingest_settings")
    def test_incremental_window_anchors_to_latest_feed_time_not_wall_clock(
        self,
        mock_settings,
        mock_get_watermark,
        mock_create_batch,
        mock_fetch_rows,
        mock_parse_rows,
        mock_get_engine,
        mock_insert,
        _mock_incomplete,
        mock_finalize,
        mock_advance,
        _mock_scoring,
        _mock_utc_now,
    ):
        mock_settings.map_key = "test-key"
        mock_settings.resolved_area = "20,40,21,41"
        mock_settings.day_range = 1
        mock_settings.sources = ["VIIRS_SNPP_NRT"]
        mock_settings.request_timeout_seconds = 30.0
        mock_settings.firms_watermark_grace_minutes = 90
        mock_settings.firms_initial_lookback_minutes = 360
        mock_settings.firms_incremental_lookback_minutes = 30
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = False
        mock_settings.firms_reconcile_unscored_batches = False
        mock_settings.firms_reconcile_max_batches = 5

        txn_conn = object()
        mock_get_engine.return_value.begin.return_value = nullcontext(txn_conn)

        # Existing watermark should force incremental mode.
        mock_get_watermark.return_value = {
            "last_acq_time_utc": datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc),
        }
        mock_create_batch.return_value = 654
        mock_fetch_rows.return_value = [{"id": "x"}]

        # Feed is delayed (latest detection around 13:10 while wall clock is 15:16).
        detection = SimpleNamespace(acq_time=datetime(2026, 3, 12, 13, 10, tzinfo=timezone.utc))
        mock_parse_rows.return_value = ([detection], FirmsValidationSummary(total_rows=1, parsed_rows=1))
        mock_insert.return_value = 1

        code = run_firms_ingest(day_range=None, area=None, sources=None)

        self.assertEqual(0, code)
        mock_finalize.assert_called_once()
        mock_advance.assert_called_once_with(
            source="VIIRS_SNPP_NRT",
            area_key="20.000000,40.000000,21.000000,41.000000",
            last_acq_time_utc=datetime(2026, 3, 12, 13, 10, tzinfo=timezone.utc),
            last_batch_id=654,
            conn=ANY,
        )

    @patch("ingest.firms_ingest._utc_now", return_value=datetime(2026, 3, 12, 15, 16, tzinfo=timezone.utc))
    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", return_value=0)
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.repository.get_engine")
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows")
    @patch("ingest.firms_ingest.repository.create_ingest_batch")
    @patch("ingest.firms_ingest.repository.get_ingest_watermark")
    @patch("ingest.firms_ingest.ingest_settings")
    def test_stale_watermark_switches_to_recovery_lookback_window(
        self,
        mock_settings,
        mock_get_watermark,
        mock_create_batch,
        mock_fetch_rows,
        mock_parse_rows,
        mock_get_engine,
        mock_insert,
        _mock_incomplete,
        mock_finalize,
        mock_advance,
        _mock_scoring,
        _mock_utc_now,
    ):
        mock_settings.map_key = "test-key"
        mock_settings.resolved_area = "20,40,21,41"
        mock_settings.day_range = 1
        mock_settings.sources = ["VIIRS_SNPP_NRT"]
        mock_settings.request_timeout_seconds = 30.0
        mock_settings.firms_watermark_grace_minutes = 90
        mock_settings.firms_initial_lookback_minutes = 360
        mock_settings.firms_incremental_lookback_minutes = 30
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = False
        mock_settings.firms_reconcile_unscored_batches = False
        mock_settings.firms_reconcile_max_batches = 5

        txn_conn = object()
        mock_get_engine.return_value.begin.return_value = nullcontext(txn_conn)

        # Watermark is older than the configured initial lookback window.
        mock_get_watermark.return_value = {
            "last_acq_time_utc": datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc),
        }
        mock_create_batch.return_value = 655
        mock_fetch_rows.return_value = [{"id": "x"}]

        # Include one detection older than incremental 30m but inside 6h recovery window.
        detections = [
            SimpleNamespace(acq_time=datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc)),
            SimpleNamespace(acq_time=datetime(2026, 3, 12, 13, 10, tzinfo=timezone.utc)),
        ]
        mock_parse_rows.return_value = (detections, FirmsValidationSummary(total_rows=2, parsed_rows=2))
        mock_insert.return_value = 2

        code = run_firms_ingest(day_range=None, area=None, sources=None)

        self.assertEqual(0, code)
        metadata_extra = mock_create_batch.call_args.kwargs["metadata_extra"]
        self.assertEqual("recovery", metadata_extra["lookback_mode"])
        self.assertEqual(360, metadata_extra["lookback_minutes"])
        self.assertEqual(2, len(mock_insert.call_args.args[0]))
        mock_finalize.assert_called_once()
        mock_advance.assert_called_once_with(
            source="VIIRS_SNPP_NRT",
            area_key="20.000000,40.000000,21.000000,41.000000",
            last_acq_time_utc=datetime(2026, 3, 12, 13, 10, tzinfo=timezone.utc),
            last_batch_id=655,
            conn=ANY,
        )

    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.repository.get_engine")
    @patch("ingest.firms_ingest.repository.count_detections_for_batch", return_value=0)
    @patch("ingest.firms_ingest.repository.delete_detections_for_batch", return_value=0)
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
        _mock_delete,
        _mock_count,
        mock_get_engine,
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
        mock_settings.firms_initial_lookback_minutes = 100000
        mock_settings.firms_incremental_lookback_minutes = 100000
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = False
        mock_settings.firms_reconcile_unscored_batches = False
        mock_settings.firms_reconcile_max_batches = 5

        txn_conn = object()
        mock_get_engine.return_value.begin.return_value = nullcontext(txn_conn)

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

    @patch("ingest.firms_ingest._resolve_denoiser_runtime_policy", return_value=None)
    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", return_value=0)
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.repository.get_engine")
    @patch("ingest.firms_ingest.repository.count_detections_for_batch", side_effect=[1, 0, 0])
    @patch("ingest.firms_ingest.repository.delete_detections_for_batch", return_value=1)
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
        _mock_delete,
        _mock_count,
        mock_get_engine,
        mock_insert,
        _mock_incomplete,
        mock_finalize,
        mock_advance,
        _mock_update_scoring,
        _mock_resolve_policy,
    ):
        mock_settings.map_key = "test-key"
        mock_settings.resolved_area = "20,40,21,41"
        mock_settings.day_range = 1
        mock_settings.sources = ["VIIRS_SNPP_NRT"]
        mock_settings.request_timeout_seconds = 30.0
        mock_settings.firms_watermark_grace_minutes = 90
        mock_settings.firms_initial_lookback_minutes = 100000
        mock_settings.firms_incremental_lookback_minutes = 100000
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = True
        mock_settings.firms_reconcile_unscored_batches = False
        mock_settings.firms_reconcile_max_batches = 5

        txn_conn = object()
        mock_get_engine.return_value.begin.return_value = nullcontext(txn_conn)

        mock_create_batch.return_value = 777
        mock_fetch_rows.return_value = [{"id": "x"}]
        detection = SimpleNamespace(acq_time=datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc))
        mock_parse_rows.return_value = ([detection], FirmsValidationSummary(total_rows=1, parsed_rows=1))
        mock_insert.return_value = 1

        code = run_firms_ingest(day_range=None, area=None, sources=None)

        self.assertEqual(2, code)
        mock_finalize.assert_not_called()
        mock_create_batch.assert_not_called()
        mock_advance.assert_not_called()

    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", return_value=2)
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.repository.get_engine")
    @patch("ingest.firms_ingest.repository.count_detections_for_batch", side_effect=[1, 0])
    @patch("ingest.firms_ingest.repository.delete_detections_for_batch", return_value=1)
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows")
    @patch("ingest.firms_ingest.repository.create_ingest_batch")
    @patch("ingest.firms_ingest.repository.get_ingest_watermark")
    @patch("ingest.firms_ingest.ingest_settings")
    def test_scoring_completeness_gate_fails_batch(
        self,
        mock_settings,
        _mock_get_watermark,
        mock_create_batch,
        mock_fetch_rows,
        mock_parse_rows,
        _mock_delete,
        _mock_count,
        mock_get_engine,
        mock_insert,
        _mock_incomplete,
        mock_finalize,
        mock_advance,
        _mock_update_scoring,
    ):
        mock_settings.map_key = "test-key"
        mock_settings.resolved_area = "20,40,21,41"
        mock_settings.day_range = 1
        mock_settings.sources = ["VIIRS_SNPP_NRT"]
        mock_settings.request_timeout_seconds = 30.0
        mock_settings.firms_watermark_grace_minutes = 90
        mock_settings.firms_initial_lookback_minutes = 100000
        mock_settings.firms_incremental_lookback_minutes = 100000
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = False
        mock_settings.firms_reconcile_unscored_batches = False
        mock_settings.firms_reconcile_max_batches = 5

        txn_conn = object()
        mock_get_engine.return_value.begin.return_value = nullcontext(txn_conn)
        mock_create_batch.return_value = 888
        mock_fetch_rows.return_value = [{"id": "x"}]
        detection = SimpleNamespace(acq_time=datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc))
        mock_parse_rows.return_value = ([detection], FirmsValidationSummary(total_rows=1, parsed_rows=1))
        mock_insert.return_value = 1

        code = run_firms_ingest(day_range=None, area=None, sources=None)

        self.assertEqual(1, code)
        mock_finalize.assert_called_once()
        self.assertEqual("failed", mock_finalize.call_args.kwargs["status"])
        mock_advance.assert_not_called()

    @patch("ingest.firms_ingest._run_denoiser_inference")
    @patch(
        "ingest.firms_ingest._resolve_denoiser_runtime_policy",
        return_value=SimpleNamespace(
            model_run_dir="/models/denoiser/run",
            using_promoted_model=True,
            model_id="denoiser-123",
            pipeline_version="v2",
            threshold_profile="strict_v1",
            threshold_source="registry_contract",
            thresholds={"strong_filter_threshold": 0.5},
        ),
    )
    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", side_effect=[0, 1])
    @patch("ingest.firms_ingest.repository.insert_detections")
    @patch("ingest.firms_ingest.repository.get_engine")
    @patch("ingest.firms_ingest.repository.count_detections_for_batch", side_effect=[1, 0])
    @patch("ingest.firms_ingest.repository.delete_detections_for_batch", return_value=1)
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows")
    @patch("ingest.firms_ingest.repository.create_ingest_batch")
    @patch("ingest.firms_ingest.repository.get_ingest_watermark")
    @patch("ingest.firms_ingest.ingest_settings")
    def test_required_denoiser_completeness_gate_fails_batch(
        self,
        mock_settings,
        _mock_get_watermark,
        mock_create_batch,
        mock_fetch_rows,
        mock_parse_rows,
        _mock_delete,
        _mock_count,
        mock_get_engine,
        mock_insert,
        _mock_incomplete,
        mock_finalize,
        mock_advance,
        _mock_update_scoring,
        _mock_resolve_policy,
        mock_run_denoiser,
    ):
        mock_settings.map_key = "test-key"
        mock_settings.resolved_area = "20,40,21,41"
        mock_settings.day_range = 1
        mock_settings.sources = ["VIIRS_SNPP_NRT"]
        mock_settings.request_timeout_seconds = 30.0
        mock_settings.firms_watermark_grace_minutes = 90
        mock_settings.firms_initial_lookback_minutes = 100000
        mock_settings.firms_incremental_lookback_minutes = 100000
        mock_settings.denoiser_enabled = False
        mock_settings.denoiser_required = True
        mock_settings.firms_reconcile_unscored_batches = False
        mock_settings.firms_reconcile_max_batches = 5

        _mock_get_watermark.return_value = None
        txn_conn = object()
        mock_get_engine.return_value.begin.return_value = nullcontext(txn_conn)
        mock_create_batch.return_value = 889
        mock_fetch_rows.return_value = [{"id": "x"}]
        detection = SimpleNamespace(acq_time=datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc))
        mock_parse_rows.return_value = ([detection], FirmsValidationSummary(total_rows=1, parsed_rows=1))
        mock_insert.return_value = 1

        code = run_firms_ingest(day_range=None, area=None, sources=None)

        self.assertEqual(1, code)
        mock_run_denoiser.assert_called_once()
        mock_finalize.assert_called_once()
        self.assertEqual("failed", mock_finalize.call_args.kwargs["status"])
        mock_advance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
