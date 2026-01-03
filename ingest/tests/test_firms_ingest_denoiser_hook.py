import unittest
from unittest.mock import MagicMock, patch, ANY

from ingest.firms_ingest import _run_denoiser_inference

class TestFirmsIngestDenoiserHook(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.denoiser_model_run_dir = "/models/v1"
        self.config.denoiser_threshold = 0.7
        self.config.denoiser_batch_size = 100
        self.config.denoiser_region = "balkans"

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
        mock_log.assert_called_once_with(
            ANY,
            "firms.denoiser_inference",
            "Denoiser inference complete",
            batch_id=1,
            noise_percent=12.5
        )

    @patch("subprocess.run")
    def test_run_denoiser_inference_error(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")

        with self.assertRaises(RuntimeError) as cm:
            _run_denoiser_inference(batch_id=1, config=self.config)
        
        self.assertIn("Denoiser inference failed for batch 1", str(cm.exception))

    def test_run_denoiser_inference_skipped_if_no_dir(self):
        self.config.denoiser_model_run_dir = None
        with patch("subprocess.run") as mock_run:
            _run_denoiser_inference(batch_id=1, config=self.config)
            mock_run.assert_not_called()

if __name__ == "__main__":
    unittest.main()

