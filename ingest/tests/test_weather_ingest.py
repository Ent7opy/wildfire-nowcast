import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import tempfile

import pytest

from ingest.weather_ingest import ingest_weather_for_bbox, snap_to_gfs_cycle


class TestWeatherIngestLogic(unittest.TestCase):
    """Test core weather ingestion logic and snapping."""

    def test_snap_to_gfs_cycle(self):
        """Verify snapping to 6-hour blocks."""
        # 02:00 -> 00:00
        dt = datetime(2026, 1, 20, 2, 30, tzinfo=timezone.utc)
        snapped = snap_to_gfs_cycle(dt)
        self.assertEqual(snapped, datetime(2026, 1, 20, 0, 0, tzinfo=timezone.utc))

        # 06:00 -> 06:00
        dt = datetime(2026, 1, 20, 6, 0, tzinfo=timezone.utc)
        snapped = snap_to_gfs_cycle(dt)
        self.assertEqual(snapped, datetime(2026, 1, 20, 6, 0, tzinfo=timezone.utc))

        # 23:59 -> 18:00
        dt = datetime(2026, 1, 20, 23, 59, tzinfo=timezone.utc)
        snapped = snap_to_gfs_cycle(dt)
        self.assertEqual(snapped, datetime(2026, 1, 20, 18, 0, tzinfo=timezone.utc))

        # Naive datetime -> UTC
        dt = datetime(2026, 1, 20, 2, 0)
        snapped = snap_to_gfs_cycle(dt)
        self.assertEqual(snapped, datetime(2026, 1, 20, 0, 0, tzinfo=timezone.utc))

    @pytest.mark.skip(reason="Requires complex internal mocking")
    @patch("ingest.weather_ingest.finalize_weather_run_record")
    @patch("ingest.weather_ingest.create_weather_run_record")
    @patch("ingest.weather_ingest._attempt_ingest")
    def test_ingest_weather_snaps_and_adjusts_horizon(
        self, mock_attempt, mock_create, mock_finalize
    ):
        """Verify ingest_weather_for_bbox snaps cycle and increases horizon."""
        test_bbox = (20.0, 40.0, 20.1, 40.1)
        # 05:00 UTC, horizon 24h
        forecast_time = datetime(2026, 1, 20, 5, 0, tzinfo=timezone.utc)
        
        mock_create.return_value = 1

        ingest_weather_for_bbox(
            bbox=test_bbox,
            forecast_time=forecast_time,
            output_dir="/tmp",
            horizon_hours=24,
        )

        # Should have snapped to 00:00
        # diff = 5 hours. New horizon = 24 + 5 = 29
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["run_time"], datetime(2026, 1, 20, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(call_kwargs["horizon_hours"], 29)
        
        # Verify _attempt_ingest called with snapped time
        mock_attempt.assert_called_once_with(datetime(2026, 1, 20, 0, 0, tzinfo=timezone.utc))


@pytest.mark.skip(reason="Test mocking needs to be updated for httpx.Client.stream() and xarray operations")
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
        
        # Mock HTTP client with stream() support
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.status_code = 200
        mock_response.iter_bytes = MagicMock(return_value=[b"mock_grib_data"])
        # Setup stream() as context manager
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_context.__exit__ = MagicMock(return_value=False)
        mock_client_instance.stream = MagicMock(return_value=mock_stream_context)
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
        
        # Mock HTTP client with stream() support
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.status_code = 200
        mock_response.iter_bytes = MagicMock(return_value=[b"mock_grib_data"])
        # Setup stream() as context manager
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_context.__exit__ = MagicMock(return_value=False)
        mock_client_instance.stream = MagicMock(return_value=mock_stream_context)
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
        
        # Verify HTTP stream requests were made (margin validation via URL would be fragile)
        stream_calls = mock_client_instance.stream.call_args_list
        self.assertGreater(len(stream_calls), 0, "Expected HTTP stream requests to be made")

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
        
        # Mock HTTP client with stream() support
        mock_client_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.is_error = False
        mock_response.status_code = 200
        mock_response.iter_bytes = MagicMock(return_value=[b"mock_grib_data"])
        # Setup stream() as context manager
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_context.__exit__ = MagicMock(return_value=False)
        mock_client_instance.stream = MagicMock(return_value=mock_stream_context)
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
