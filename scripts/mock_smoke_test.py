"""Mocked end-to-end smoke test for denoiser pipeline."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import os
from ml.denoiser.dataset import build_dataset

def run_mock_smoke_test():
    print("Running mocked smoke test...")
    
    # 1. Create dummy input data (as if loaded from DB)
    t0 = datetime.now() - timedelta(days=5)
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "lat": [40.0, 40.01, 30.0],
        "lon": [-120.0, -120.01, -110.0],
        "acq_time": [t0, t0 + timedelta(hours=1), t0],
        "confidence": [90.0, 80.0, 20.0],
        "frp": [100.0, 50.0, 10.0],
        "brightness": [320.0, 310.0, 290.0],
        "bright_t31": [300.0, 295.0, 280.0],
        "scan": [1.0, 1.0, 1.0],
        "track": [1.0, 1.0, 1.0],
        "sensor": ["VIIRS", "VIIRS", "MODIS"],
        "source": ["firms_viirs", "firms_viirs", "firms_modis"],
        "raw_properties": [{"daynight": "D"}, {"daynight": "D"}, {"daynight": "N"}],
        "label": ["POSITIVE", "POSITIVE", "NEGATIVE"]
    })
    
    # 2. Mock the engine and spatiotemporal context
    mock_engine = MagicMock()
    
    # We patch where they are used in dataset.py
    with patch("ml.denoiser.dataset.add_terrain_features") as mock_terrain, \
         patch("ml.denoiser.dataset.add_spatiotemporal_context") as mock_context:
        
        # Mock terrain returns
        def mock_terrain_fn(df, region_name):
            df = df.copy()
            df["elevation_m"] = 100.0
            df["slope_deg"] = 10.0
            df["aspect_sin"] = 0.0
            df["aspect_cos"] = 1.0
            return df
        mock_terrain.side_effect = mock_terrain_fn
        
        # Mock context returns
        def mock_context_fn(df, engine):
            df = df.copy()
            df["n_2km_6h"] = 0
            df["n_2km_24h"] = 0
            df["n_5km_24h"] = 0
            df["n_same_cell_24h"] = 0
            df["n_same_cell_7d"] = 0
            df["dist_nn_24h_km"] = 999.0
            df["seen_same_cell_past_3d"] = 0
            df["days_with_detection_past_30d_in_cell"] = 0
            return df
        mock_context.side_effect = mock_context_fn
        
        print("Building dataset...")
        X, y, meta = build_dataset(df, mock_engine, region_name="test_region")
        
        # 3. Verify output
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        assert not X.empty
        assert len(X) == 3
        assert "elevation_m" in X.columns
        assert "confidence_norm" in X.columns
        
        # 4. Write to parquet
        out_path = "data/denoiser/smoke_test_features.parquet"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        out_df = pd.concat([X, meta], axis=1)
        # Ensure numeric types for parquet
        for col in X.columns:
            out_df[col] = pd.to_numeric(out_df[col])

        print(f"Writing {len(out_df)} rows to {out_path}...")
        out_df.to_parquet(out_path, index=False)
        print(f"Wrote parquet to {out_path}")
        
        # 5. Verify parquet
        read_df = pd.read_parquet(out_path)
        print(f"Read parquet shape: {read_df.shape}")
        assert len(read_df) == 3
        
    print("Smoke test PASSED.")

if __name__ == "__main__":
    run_mock_smoke_test()
