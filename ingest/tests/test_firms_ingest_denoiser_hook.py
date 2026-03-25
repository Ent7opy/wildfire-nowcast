import subprocess
import unittest
from unittest.mock import MagicMock, patch

from ingest.firms_ingest import DenoiserTimeoutError, _run_denoiser_inference

class TestFirmsIngestDenoiserHook(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.denoiser_model_run_dir = "/models/v1"
        self.config.denoiser_pipeline_version = "v1"
        self.config.denoiser_invoke_method = "uv"
        self.config.denoiser_threshold = 0.7
        self.config.denoiser_batch_size = 100
        self.config.denoiser_region = "balkans"
        self.config.denoiser_strict_features = False
        self.config.denoiser_shadow_mode = False
        self.config.denoiser_strong_filter_threshold = 0.5
        self.config.denoiser_downweight_threshold = 0.7
        self.config.denoiser_uncertainty_band_low = 0.45
        self.config.denoiser_uncertainty_band_high = 0.55
        self.config.denoiser_event_front_radius_m = 2500.0
        self.config.denoiser_event_front_max_gap_minutes = 45
        self.config.denoiser_event_link_radius_m = 10000.0
        self.config.denoiser_event_link_max_gap_days = 11
        self.config.denoiser_event_static_persistence_threshold = 0.85
        self.config.denoiser_event_strict_static_split = True

    @patch("subprocess.run")
    @patch("ingest.firms_ingest.log_event")
    def test_run_denoiser_inference_success(self, mock_log, mock_run):
        # Setup mock subprocess response
        mock_result = MagicMock()
        mock_result.stdout = 'some logs\n{"batch_id": 1, "noise_percent": 12.5}'
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Execute
        _run_denoiser_inference(batch_id=1, config=self.config)

        # Verify command
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("ml.denoiser_inference", cmd)
        self.assertIn("--batch-id", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--threshold", cmd)
        self.assertIn("0.7", cmd)
        self.assertIn("--region", cmd)
        self.assertIn("balkans", cmd)

        # Verify logging
        mock_log.assert_called_once()
        log_args, log_kwargs = mock_log.call_args
        self.assertEqual(log_args[1], "firms.denoiser_inference")
        self.assertEqual(log_args[2], "Denoiser inference complete")
        self.assertEqual(log_kwargs["batch_id"], 1)
        self.assertEqual(log_kwargs["noise_percent"], 12.5)
        self.assertIn("pipeline_version", log_kwargs)
        self.assertIn("effective_thresholds", log_kwargs)

    @patch("subprocess.run")
    def test_run_denoiser_inference_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")

        with self.assertRaises(RuntimeError) as cm:
            _run_denoiser_inference(batch_id=1, config=self.config)

        self.assertIn("Denoiser inference failed for batch 1", str(cm.exception))

    @patch("subprocess.run")
    def test_run_denoiser_inference_timeout_raises_denoiser_timeout_error(self, mock_run):
        self.config.denoiser_subprocess_timeout_seconds = 30
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="uv run -m ml.denoiser_inference", timeout=30)

        with self.assertRaises(DenoiserTimeoutError) as cm:
            _run_denoiser_inference(batch_id=42, config=self.config)

        self.assertIn("timed out after 30s", str(cm.exception))
        self.assertIn("batch 42", str(cm.exception))

    @patch("subprocess.run")
    def test_run_denoiser_inference_timeout_is_not_generic_runtime_error(self, mock_run):
        """DenoiserTimeoutError must be distinguishable from generic RuntimeError."""
        self.config.denoiser_subprocess_timeout_seconds = 60
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="uv", timeout=60)

        raised = None
        try:
            _run_denoiser_inference(batch_id=5, config=self.config)
        except DenoiserTimeoutError as e:
            raised = e

        self.assertIsNotNone(raised, "Expected DenoiserTimeoutError to be raised")
        self.assertIsInstance(raised, RuntimeError, "DenoiserTimeoutError must be a RuntimeError subclass")

    def test_run_denoiser_inference_skipped_if_no_dir(self):
        self.config.denoiser_model_run_dir = None
        with patch("subprocess.run") as mock_run:
            _run_denoiser_inference(batch_id=1, config=self.config)
            mock_run.assert_not_called()

    @patch("subprocess.run")
    @patch("ingest.firms_ingest.log_event")
    def test_run_denoiser_inference_passes_strict_features_flag(self, _mock_log, mock_run):
        self.config.denoiser_strict_features = True
        mock_result = MagicMock()
        mock_result.stdout = '{"batch_id": 1, "noise_percent": 10.0}'
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        _run_denoiser_inference(batch_id=1, config=self.config)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--strict-features", cmd)

    @patch("subprocess.run")
    @patch("ingest.firms_ingest.log_event")
    def test_run_denoiser_inference_v2_shadow_mode_uses_v2_module(self, _mock_log, mock_run):
        self.config.denoiser_pipeline_version = "v2"
        self.config.denoiser_shadow_mode = True
        self.config.denoiser_model_run_dir = "/models/v2"
        mock_result = MagicMock()
        mock_result.stdout = '{"batch_id": 1, "events": 3}'
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        _run_denoiser_inference(batch_id=1, config=self.config)

        cmd = mock_run.call_args[0][0]
        self.assertIn("ml.denoiser_inference_v2", cmd)
        self.assertIn("--strong-filter-threshold", cmd)
        self.assertIn("--downweight-threshold", cmd)
        self.assertIn("--uncertainty-band-low", cmd)
        self.assertIn("--uncertainty-band-high", cmd)
        self.assertIn("--event-front-radius-m", cmd)
        self.assertIn("--event-front-max-gap-minutes", cmd)
        self.assertIn("--event-link-radius-m", cmd)
        self.assertIn("--event-link-max-gap-days", cmd)
        self.assertIn("--event-static-persistence-threshold", cmd)
        self.assertIn("--event-strict-static-split", cmd)
        self.assertIn("--shadow-mode", cmd)
        self.assertNotIn("--threshold", cmd)
        self.assertNotIn("--batch-size", cmd)

if __name__ == "__main__":
    unittest.main()
