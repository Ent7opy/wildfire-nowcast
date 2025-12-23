"""Tests for FIRMS hotspot denoiser feature engineering."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml.denoiser.features import add_firms_features, add_time_features, add_spatiotemporal_context, add_terrain_features

class TestDenoiserFeatures(unittest.TestCase):
    def test_firms_features_basic(self):
        df = pd.DataFrame({
            "confidence": ["high", "nominal", "low", None],
            "brightness": [320, 310, 300, 290],
            "bright_t31": [300, 295, 290, 285],
            "instrument": ["VIIRS", "VIIRS", "MODIS", "MODIS"],
            "raw_properties": [{"daynight": "D"}, {"daynight": "N"}, {}, None]
        })
        
        df_feat = add_firms_features(df)
        
        # Check confidence normalization
        self.assertEqual(df_feat["confidence_norm"].iloc[0], 90.0)
        self.assertEqual(df_feat["confidence_norm"].iloc[1], 60.0)
        self.assertEqual(df_feat["confidence_norm"].iloc[2], 30.0)
        self.assertEqual(df_feat["confidence_norm"].iloc[3], 30.0)
        
        # Check brightness diff
        self.assertEqual(df_feat["brightness_minus_t31"].iloc[0], 20.0)
        
        # Check daynight extraction
        self.assertEqual(df_feat["is_day"].iloc[0], 1)
        self.assertEqual(df_feat["is_day"].iloc[1], 0)

    def test_time_features_cyclical(self):
        # 12:00 UTC
        dt1 = datetime(2024, 1, 1, 12, 0)
        # 00:00 UTC
        dt2 = datetime(2024, 1, 1, 0, 0)
        
        df = pd.DataFrame({"acq_time": [dt1, dt2]})
        df_feat = add_time_features(df)
        
        # 12:00 should have sin_hour near 0 (sin(2pi*12/24) = sin(pi) = 0) 
        # Actually sin(2pi * 12/24) is sin(pi) = 0.
        # cos(2pi * 12/24) is cos(pi) = -1.
        self.assertAlmostEqual(df_feat["sin_hour"].iloc[0], 0.0, places=5)
        self.assertAlmostEqual(df_feat["cos_hour"].iloc[0], -1.0, places=5)
        
        # 00:00 should have sin_hour 0, cos_hour 1
        self.assertAlmostEqual(df_feat["sin_hour"].iloc[1], 0.0, places=5)
        self.assertAlmostEqual(df_feat["cos_hour"].iloc[1], 1.0, places=5)

    @patch("sqlalchemy.Engine")
    def test_leakage_guard(self, mock_engine):
        # Create two detections: one at T, one at T+1 hour.
        # When computing features for T, T+1 should NOT be counted.
        t0 = datetime(2024, 1, 1, 10, 0)
        t1 = t0 + timedelta(hours=1)
        
        df = pd.DataFrame({
            "id": [1, 2],
            "lat": [40.0, 40.0],
            "lon": [-120.0, -120.0],
            "acq_time": [t0, t1]
        })
        
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        # Total queries per row: 4 (radii/windows) + 2 (grid) + 1 (NN) + 1 (seen) + 1 (days) = 9
        # For 2 rows, we need 18 values.
        mock_conn.execute.return_value.scalar.side_effect = [0] * 18
        
        df_feat = add_spatiotemporal_context(df, mock_engine)
        
        self.assertEqual(df_feat["n_2km_6h"].iloc[0], 0)
        self.assertEqual(len(df_feat), 2)

    @patch("ml.denoiser.features.load_dem_for_bbox")
    @patch("ml.denoiser.features.load_slope_aspect_for_bbox")
    def test_terrain_join(self, mock_load_slope, mock_load_dem):
        import xarray as xr
        # Mock xarray DataArrays
        coords = {"lat": [39.9, 40.0, 40.1], "lon": [-120.1, -120.0, -119.9]}
        mock_dem = xr.DataArray(np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]]), coords=coords, dims=("lat", "lon"))
        mock_slope = xr.DataArray(np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]]), coords=coords, dims=("lat", "lon"))
        mock_aspect = xr.DataArray(np.array([[0, 0, 0], [90, 90, 90], [180, 180, 180]]), coords=coords, dims=("lat", "lon"))
        
        mock_load_dem.return_value = mock_dem
        mock_load_slope.return_value = (mock_slope, mock_aspect)
        
        df = pd.DataFrame({
            "lat": [40.0],
            "lon": [-120.0]
        })
        
        df_feat = add_terrain_features(df, "test_region")
        
        self.assertEqual(df_feat["elevation_m"].iloc[0], 200.0)
        self.assertEqual(df_feat["slope_deg"].iloc[0], 20.0)
        self.assertAlmostEqual(df_feat["aspect_sin"].iloc[0], 1.0, places=5) # sin(90) = 1

if __name__ == "__main__":
    unittest.main()
