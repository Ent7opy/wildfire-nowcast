"""Verify that archive-ingested detections go through denoiser scoring.

Archive mode (archive_date=...) bypasses the watermark filter and does not
advance the watermark, but it must apply denoiser scoring identically to the
NRT path.  These tests exercise run_firms_ingest with archive_date set and
assert that _run_denoiser_inference is invoked and that every detection in the
batch is marked is_archive=True before insertion.
"""
import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ingest.firms_client import FirmsValidationSummary
from ingest.firms_ingest import run_firms_ingest


def _apply_settings(mock_settings, **overrides):
    """Configure a mock ingest_settings object with archive-test defaults."""
    attrs = {
        "map_key": "test-key",
        "resolved_area": "20,40,21,41",
        "day_range": 1,
        "sources": ["VIIRS_NOAA20_NRT"],
        "request_timeout_seconds": 30.0,
        "firms_watermark_grace_minutes": 90,
        "firms_initial_lookback_minutes": 100_000,
        "firms_incremental_lookback_minutes": 100_000,
        "denoiser_enabled": True,
        "denoiser_required": False,
        "denoiser_pipeline_version": "v2",
        "denoiser_shadow_mode": False,
        "firms_reconcile_unscored_batches": False,
        "firms_reconcile_max_batches": 5,
        **overrides,
    }
    for attr, val in attrs.items():
        setattr(mock_settings, attr, val)


_RESOLVED_POLICY = SimpleNamespace(
    model_run_dir="/models/denoiser/run",
    using_promoted_model=True,
    model_id="denoiser-abc",
    pipeline_version="v2",
    threshold_profile="strict_v1",
    threshold_source="registry_contract",
    thresholds={"strong_filter_threshold": 0.5},
)


class TestArchiveDenoiserScoring(unittest.TestCase):
    """Archive detections must receive denoiser scores identical to NRT."""

    @patch("ingest.firms_ingest._run_denoiser_inference")
    @patch("ingest.firms_ingest._resolve_denoiser_runtime_policy", return_value=_RESOLVED_POLICY)
    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    # count_rows_with_null_columns_for_batch: first call = scoring gate (0), second = denoiser gate (0)
    @patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", side_effect=[0, 0])
    @patch("ingest.firms_ingest.repository.insert_detections", return_value=2)
    @patch("ingest.firms_ingest.repository.get_engine")
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows", return_value=[{"id": "a"}, {"id": "b"}])
    @patch("ingest.firms_ingest.repository.create_ingest_batch", return_value=42)
    @patch("ingest.firms_ingest.repository.get_ingest_watermark", return_value=None)
    @patch("ingest.firms_ingest.ingest_settings")
    def test_archive_runs_denoiser(
        self,
        mock_settings,
        _mock_watermark,
        _mock_create_batch,
        _mock_fetch,
        mock_parse,
        mock_get_engine,
        _mock_insert,
        _mock_null_cols,
        _mock_finalize,
        mock_advance_watermark,
        _mock_scoring,
        _mock_resolve_policy,
        mock_run_denoiser,
    ):
        """_run_denoiser_inference is called for archive batches when denoiser is enabled."""
        _apply_settings(mock_settings)
        mock_get_engine.return_value.begin.return_value = nullcontext(object())

        det1 = SimpleNamespace(acq_time=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc))
        det2 = SimpleNamespace(acq_time=datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc))
        mock_parse.return_value = (
            [det1, det2],
            FirmsValidationSummary(total_rows=2, parsed_rows=2),
        )

        code = run_firms_ingest(
            day_range=None, area=None, sources=None, archive_date="2026-01-15"
        )

        self.assertEqual(0, code, "Expected exit code 0 (success)")
        mock_run_denoiser.assert_called_once_with(42, mock_settings, model_run_dir="/models/denoiser/run", runtime_policy=_RESOLVED_POLICY)

    @patch("ingest.firms_ingest._run_denoiser_inference")
    @patch("ingest.firms_ingest._resolve_denoiser_runtime_policy", return_value=_RESOLVED_POLICY)
    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", side_effect=[0, 0])
    @patch("ingest.firms_ingest.repository.insert_detections", return_value=2)
    @patch("ingest.firms_ingest.repository.get_engine")
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows", return_value=[{"id": "a"}, {"id": "b"}])
    @patch("ingest.firms_ingest.repository.create_ingest_batch", return_value=43)
    @patch("ingest.firms_ingest.repository.get_ingest_watermark", return_value=None)
    @patch("ingest.firms_ingest.ingest_settings")
    def test_archive_detections_marked_is_archive(
        self,
        mock_settings,
        _mock_watermark,
        _mock_create_batch,
        _mock_fetch,
        mock_parse,
        mock_get_engine,
        mock_insert,
        _mock_null_cols,
        _mock_finalize,
        mock_advance_watermark,
        _mock_scoring,
        _mock_resolve_policy,
        _mock_run_denoiser,
    ):
        """All detections inserted in archive mode must have is_archive=True."""
        _apply_settings(mock_settings)
        mock_get_engine.return_value.begin.return_value = nullcontext(object())

        det1 = SimpleNamespace(acq_time=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc))
        det2 = SimpleNamespace(acq_time=datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc))
        mock_parse.return_value = (
            [det1, det2],
            FirmsValidationSummary(total_rows=2, parsed_rows=2),
        )

        code = run_firms_ingest(
            day_range=None, area=None, sources=None, archive_date="2026-01-15"
        )

        self.assertEqual(0, code)
        inserted_detections = mock_insert.call_args.args[0]
        self.assertTrue(
            all(getattr(d, "is_archive", False) for d in inserted_detections),
            "All inserted archive detections must have is_archive=True",
        )

    @patch("ingest.firms_ingest._run_denoiser_inference")
    @patch("ingest.firms_ingest._resolve_denoiser_runtime_policy", return_value=_RESOLVED_POLICY)
    @patch("ingest.firms_ingest._update_all_scoring_atomic")
    @patch("ingest.firms_ingest.repository.advance_ingest_watermark")
    @patch("ingest.firms_ingest.repository.finalize_ingest_batch")
    @patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", side_effect=[0, 0])
    @patch("ingest.firms_ingest.repository.insert_detections", return_value=1)
    @patch("ingest.firms_ingest.repository.get_engine")
    @patch("ingest.firms_ingest.parse_detection_rows")
    @patch("ingest.firms_ingest.fetch_csv_rows", return_value=[{"id": "old"}])
    @patch("ingest.firms_ingest.repository.create_ingest_batch", return_value=44)
    @patch(
        "ingest.firms_ingest.repository.get_ingest_watermark",
        return_value={"last_acq_time_utc": datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)},
    )
    @patch("ingest.firms_ingest.ingest_settings")
    def test_archive_bypasses_watermark_and_does_not_advance(
        self,
        mock_settings,
        _mock_watermark,
        _mock_create_batch,
        _mock_fetch,
        mock_parse,
        mock_get_engine,
        mock_insert,
        _mock_null_cols,
        _mock_finalize,
        mock_advance_watermark,
        _mock_scoring,
        _mock_resolve_policy,
        _mock_run_denoiser,
    ):
        """Archive mode bypasses watermark filter (old detections pass through) and
        does NOT advance the watermark after a successful run."""
        _apply_settings(mock_settings)
        mock_get_engine.return_value.begin.return_value = nullcontext(object())

        # Detection well before the watermark (2026-01-15 vs watermark 2026-03-01).
        # NRT would drop this; archive must keep it.
        old_det = SimpleNamespace(acq_time=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc))
        mock_parse.return_value = (
            [old_det],
            FirmsValidationSummary(total_rows=1, parsed_rows=1),
        )

        code = run_firms_ingest(
            day_range=None, area=None, sources=None, archive_date="2026-01-15"
        )

        self.assertEqual(0, code)
        # Detection must have reached insert despite being older than watermark.
        self.assertEqual(1, len(mock_insert.call_args.args[0]))
        # Watermark must NOT be advanced for archive batches.
        mock_advance_watermark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
