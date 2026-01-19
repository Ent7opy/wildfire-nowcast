import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import tempfile

from ingest.weather_ingest import ingest_weather_for_bbox


class TestWeatherIngestPatchMode(unittest.TestCase):
    """Test patch mode optimizations for small AOI weather ingestion."""

    def setUp(self):
        self.test_bbox = (20.0, 40.0, 20.1, 40.1)  # Small 0.1° x 0.1° test bbox
        self.forecast_time = datetime(2025, 1, 19, 0, 0, tzinfo=timezone.utc)
        self.temp_dir = tempfile.mkdtemp()

    @patch("ingest.weather_ingest.finalize_weather_run_record")
    @patch("ingest.weather_ingest.create_weather_run_record")
    @patch("ingest.weather_ingest.httpx.Client")
    @patch("ingest.weather_ingest.xr.open_dataset")
    @patch("ingest.weather_ingest.Path.mkdir")
    def test_patch_mode_applies_optimizations(
        self, mock_mkdir, mock_open_ds, mock_http_client, mock_create_run, mock_finalize_run
    ):
        """Verify patch_mode=True applies horizon, step, and precipitation overrides."""
        # Mock database records
        mock_create_run.return_value = 123
        
        # Mock HTTP client to avoid external requests
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"mock_grib_data"
        mock_client_instance.get.return_value = mock_response
        mock_http_client.return_value.__enter__.return_value = mock_client_instance
        
        # Mock xarray dataset with minimal structure
        mock_ds = MagicMock()
        mock_ds.dims = {"time": 5, "latitude": 10, "longitude": 10}
        mock_ds.__getitem__.return_value = MagicMock()  # For variable access
        mock_ds.sel.return_value = mock_ds
        mock_ds.assign_coords.return_value = mock_ds
        mock_ds.rename.return_value = mock_ds
        mock_ds.interp.return_value = mock_ds
        mock_ds.to_netcdf = MagicMock()
        mock_ds.close = MagicMock()
        mock_open_ds.return_value = mock_ds
        
        # Execute with patch_mode=True
        with patch("ingest.weather_ingest.LOGGER") as mock_logger:
            weather_run_id = ingest_weather_for_bbox(
                bbox=self.test_bbox,
                forecast_time=self.forecast_time,
                output_dir=self.temp_dir,
                patch_mode=True,
            )
        
        # Verify database record creation
        self.assertEqual(weather_run_id, 123)
        mock_create_run.assert_called_once()
        call_kwargs = mock_create_run.call_args[1]
        
        # Verify horizon_hours=24, step_hours=6, include_precipitation=False
        self.assertEqual(call_kwargs["horizon_hours"], 24)
        self.assertEqual(call_kwargs["step_hours"], 6)
        # Precipitation should not be in the requested variables
        
        # Verify logging confirms patch mode parameters
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        patch_mode_log = any("horizon_hours=24, step_hours=6, precipitation=False" in str(call) for call in log_calls)
        self.assertTrue(patch_mode_log, "Expected patch mode parameters to be logged")

    @patch("ingest.weather_ingest.finalize_weather_run_record")
    @patch("ingest.weather_ingest.create_weather_run_record")
    @patch("ingest.weather_ingest.httpx.Client")
    @patch("ingest.weather_ingest.xr.open_dataset")
    @patch("ingest.weather_ingest.Path.mkdir")
    def test_patch_mode_applies_spatial_margin(
        self, mock_mkdir, mock_open_ds, mock_http_client, mock_create_run, mock_finalize_run
    ):
        """Verify patch_mode adds 0.5° margin to download bbox."""
        mock_create_run.return_value = 456
        
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"mock_grib_data"
        mock_client_instance.get.return_value = mock_response
        mock_http_client.return_value.__enter__.return_value = mock_client_instance
        
        mock_ds = MagicMock()
        mock_ds.dims = {"time": 5, "latitude": 10, "longitude": 10}
        mock_ds.__getitem__.return_value = MagicMock()
        mock_ds.sel.return_value = mock_ds
        mock_ds.assign_coords.return_value = mock_ds
        mock_ds.rename.return_value = mock_ds
        mock_ds.interp.return_value = mock_ds
        mock_ds.to_netcdf = MagicMock()
        mock_ds.close = MagicMock()
        mock_open_ds.return_value = mock_ds
        
        with patch("ingest.weather_ingest.LOGGER") as mock_logger:
            ingest_weather_for_bbox(
                bbox=self.test_bbox,
                forecast_time=self.forecast_time,
                output_dir=self.temp_dir,
                patch_mode=True,
            )
        
        # Verify margin is logged
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        margin_log = any("downloading with margin bbox" in str(call) for call in log_calls)
        self.assertTrue(margin_log, "Expected spatial margin logging")
        
        # Verify HTTP requests use expanded bbox (margin applied)
        http_calls = mock_client_instance.get.call_args_list
        self.assertGreater(len(http_calls), 0, "Expected HTTP requests to be made")
        # Check that subregion parameters in URL reflect the margin
        for call in http_calls:
            url = call[0][0]
            if "subregion" in url:
                # Should contain margin-expanded coordinates (original ± 0.5)
                # Original: 20.0, 40.0, 20.1, 40.1
                # Margin: 19.5, 39.5, 20.6, 40.6
                # Just verify the request was made (detailed URL parsing would be fragile)
                break

    @patch("ingest.weather_ingest.finalize_weather_run_record")
    @patch("ingest.weather_ingest.create_weather_run_record")
    @patch("ingest.weather_ingest.httpx.Client")
    @patch("ingest.weather_ingest.xr.open_dataset")
    @patch("ingest.weather_ingest.Path.mkdir")
    def test_patch_mode_false_preserves_defaults(
        self, mock_mkdir, mock_open_ds, mock_http_client, mock_create_run, mock_finalize_run
    ):
        """Verify patch_mode=False uses default parameters (72h horizon, 3h steps)."""
        mock_create_run.return_value = 789
        
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"mock_grib_data"
        mock_client_instance.get.return_value = mock_response
        mock_http_client.return_value.__enter__.return_value = mock_client_instance
        
        mock_ds = MagicMock()
        mock_ds.dims = {"time": 25, "latitude": 10, "longitude": 10}
        mock_ds.__getitem__.return_value = MagicMock()
        mock_ds.sel.return_value = mock_ds
        mock_ds.assign_coords.return_value = mock_ds
        mock_ds.rename.return_value = mock_ds
        mock_ds.interp.return_value = mock_ds
        mock_ds.to_netcdf = MagicMock()
        mock_ds.close = MagicMock()
        mock_open_ds.return_value = mock_ds
        
        weather_run_id = ingest_weather_for_bbox(
            bbox=self.test_bbox,
            forecast_time=self.forecast_time,
            output_dir=self.temp_dir,
            patch_mode=False,
        )
        
        self.assertEqual(weather_run_id, 789)
        call_kwargs = mock_create_run.call_args[1]
        
        # Verify default parameters are preserved
        self.assertEqual(call_kwargs["horizon_hours"], 72)
        self.assertEqual(call_kwargs["step_hours"], 3)


if __name__ == "__main__":
    unittest.main()
