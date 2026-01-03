import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
from datetime import datetime
from ml.spread.learned_v1 import LearnedSpreadModelV1
from ml.spread.contract import SpreadModelInput, SpreadForecast

class TestLearnedSpreadModelV1Integration(unittest.TestCase):
    @patch("ml.spread.learned_v1.joblib.load")
    @patch("ml.spread.learned_v1.os.path.exists")
    @patch("builtins.open")
    @patch("json.load")
    @patch("ml.spread.learned_v1.SpreadProbabilityCalibrator.load")
    def test_calibration_is_applied(self, mock_cal_load, mock_json, mock_open, mock_exists, mock_joblib):
        # Setup mocks
        mock_exists.return_value = True
        
        # Mock classifier
        mock_clf = MagicMock()
        # predict_proba returns [prob_class0, prob_class1]
        # We expect 1 pixel -> 1 sample -> shape (1, 2)
        mock_clf.predict_proba.return_value = np.array([[0.2, 0.8]], dtype=np.float32)
        mock_joblib.return_value = {24: mock_clf}
        
        mock_json.return_value = ["feature1"]
        
        # Mock calibrator
        mock_cal_instance = MagicMock()
        # The model flattens/reshapes, so verify we get the right values passed
        mock_cal_instance.calibrate_probs.return_value = np.array([0.55], dtype=np.float32)
        mock_cal_load.return_value = mock_cal_instance
        
        # Instantiate model with calibrator config
        model = LearnedSpreadModelV1(
            model_run_dir="/tmp/test",
            calibrator_run_dir="/tmp/calib"
        )
        
        # Create dummy input
        # 1x1 grid
        lat = np.array([40.0])
        lon = np.array([20.0])
        window_shape = (1, 1)
        
        mock_inputs = MagicMock(spec=SpreadModelInput)
        mock_inputs.horizons_hours = [24]
        mock_inputs.window.lat = lat
        mock_inputs.window.lon = lon
        mock_inputs.forecast_reference_time = datetime(2025, 1, 1)
        
        # Features
        mock_inputs.active_fires.heatmap = np.zeros(window_shape)
        mock_inputs.terrain.slope = np.zeros(window_shape)
        mock_inputs.terrain.aspect = np.zeros(window_shape)
        mock_inputs.terrain.elevation = np.zeros(window_shape)
        mock_inputs.terrain.valid_data_mask = None
        mock_inputs.terrain.aoi_mask = None
        
        # Weather mock
        mock_weather_slice = MagicMock()
        mock_weather_slice.__getitem__.side_effect = lambda key: {
            "u10": MagicMock(values=np.zeros(window_shape)),
            "v10": MagicMock(values=np.zeros(window_shape))
        }.get(key)
        # Also need .get for optional vars
        mock_weather_slice.get.return_value = None
        
        mock_weather = MagicMock()
        mock_weather.isel.return_value = mock_weather_slice
        mock_inputs.weather_cube = mock_weather
        
        # Act
        forecast = model.predict(mock_inputs)
        
        # Assert
        # 1. Check that calibrator was loaded
        mock_cal_load.assert_called_with("/tmp/calib")
        
        # 2. Check that calibrate_probs was called with raw probs (0.8) and horizon (24)
        args, _ = mock_cal_instance.calibrate_probs.call_args
        raw_probs = args[0]
        horizon = args[1]
        
        # raw_probs might be an array
        np.testing.assert_array_almost_equal(raw_probs, [0.8])
        assert horizon == 24
        
        # 3. Check output value is the calibrated one (0.55)
        out_val = forecast.probabilities.values[0, 0, 0]
        assert abs(out_val - 0.55) < 1e-6

