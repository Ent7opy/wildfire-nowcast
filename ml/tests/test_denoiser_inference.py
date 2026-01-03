import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from ml.denoiser_inference import build_features, load_model_artifacts, run_inference

class TestDenoiserInference(unittest.TestCase):
    def test_load_model_artifacts(self):
        with patch("os.path.exists", return_value=True), \
             patch("joblib.load", return_value=MagicMock()) as mock_load, \
             patch("builtins.open", unittest.mock.mock_open(read_data='["feat1", "feat2"]')):
            
            model, feature_list = load_model_artifacts("/tmp/model_run")
            self.assertEqual(feature_list, ["feat1", "feat2"])
            mock_load.assert_called_once()

    @patch("ml.denoiser_inference.add_firms_features")
    @patch("ml.denoiser_inference.add_time_features")
    @patch("ml.denoiser_inference.add_spatiotemporal_context_batch")
    def test_build_features(self, mock_context, mock_time, mock_firms):
        # Setup mocks
        mock_firms.side_effect = lambda df: df.assign(feat1=1.0)
        mock_time.side_effect = lambda df: df.assign(feat2=2.0)
        mock_context.side_effect = lambda df, engine: df.assign(feat3=3.0)

        df = pd.DataFrame({"id": [1], "lat": [42.0], "lon": [20.0], "acq_time": [pd.Timestamp("2025-01-01")]})
        engine = MagicMock()
        feature_list = ["feat1", "feat2", "feat3", "missing_feat"]

        X = build_features(df, engine, feature_list)

        self.assertEqual(len(X), 1)
        self.assertIn("feat1", X.columns)
        self.assertIn("feat2", X.columns)
        self.assertIn("feat3", X.columns)
        self.assertIn("missing_feat", X.columns)
        self.assertTrue(np.isnan(X["missing_feat"].iloc[0]))
        self.assertEqual(X.columns.tolist(), feature_list)

    @patch("ml.denoiser_inference.get_engine")
    @patch("ml.denoiser_inference.load_model_artifacts")
    @patch("ml.denoiser_inference.get_pending_detections")
    @patch("ml.denoiser_inference.build_features")
    @patch("ml.denoiser_inference.update_detections")
    def test_run_inference_flow(self, mock_update, mock_build, mock_pending, mock_load, mock_engine):
        # Setup
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        mock_load.return_value = (mock_model, ["feat1"])
        
        df_pending = pd.DataFrame({
            "id": [101, 102],
            "lat": [40.0, 41.0],
            "lon": [20.0, 21.0],
            "acq_time": [pd.Timestamp("2025-12-24"), pd.Timestamp("2025-12-24")]
        })
        mock_pending.return_value = df_pending
        mock_build.return_value = pd.DataFrame({"feat1": [1.0, 2.0]})

        # Execute
        with patch("builtins.print") as mock_print:
            run_inference(batch_id=1, model_run_dir="/tmp/run", threshold=0.5)

            # Verify
            mock_update.assert_called_once()
            args, _ = mock_update.call_args
            self.assertEqual(args[1], [101, 102]) # IDs
            self.assertEqual(args[2], [0.9, 0.2]) # Scores (probs of class 1)
            self.assertEqual(args[3], [False, True]) # is_noise (0.9 >= 0.5 is False noise, 0.2 < 0.5 is True noise)

            mock_print.assert_called_once()
            summary = json.loads(mock_print.call_args[0][0])
            self.assertEqual(summary["batch_id"], 1)
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["noise_count"], 1)

if __name__ == "__main__":
    unittest.main()

