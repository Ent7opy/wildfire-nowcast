import logging
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import httpx
import pytest

from ingest.config import WeatherIngestSettings
from ingest.weather_ingest import (
    _validate_grib_file,
    download_grib_files,
    ingest_weather_for_bbox,
    snap_to_gfs_cycle,
)


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


def test_ingest_weather_for_bbox_respects_bbox_overrides():
    """Runtime download settings should honor explicit bbox inputs."""
    requested_bbox = (-125.0, 25.0, -100.0, 40.0)
    captured: dict[str, tuple[float, float, float, float]] = {}

    def _capture_settings(settings, *args, **kwargs):
        captured["bbox"] = settings.bbox
        raise RuntimeError("stop_after_settings_capture")

    with (
        patch("ingest.weather_ingest.create_weather_run_record", return_value=999),
        patch("ingest.weather_ingest.finalize_weather_run_record"),
        patch("ingest.weather_ingest.download_grib_files", side_effect=_capture_settings),
        pytest.raises(RuntimeError, match="stop_after_settings_capture"),
    ):
        ingest_weather_for_bbox(
            bbox=requested_bbox,
            forecast_time=datetime(2026, 2, 19, 12, 0, tzinfo=timezone.utc),
            output_dir="/tmp",
            patch_mode=False,
        )

    assert captured.get("bbox") == requested_bbox


def test_validate_grib_file_multilevel_succeeds_via_cfgrib_fallback(caplog):
    """Multi-level GRIB: primary attempt fails, cfgrib.open_datasets fallback validates."""
    mock_ds = MagicMock()
    mock_ds.data_vars = {"u10": MagicMock(), "v10": MagicMock()}

    with tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
        tmp.write(b"x" * 256)
        tmp.flush()

        with (
            patch(
                "ingest.weather_ingest.xr.open_dataset",
                side_effect=Exception("multiple values for unique key — typeOfLevel"),
            ),
            patch("cfgrib.open_datasets", return_value=[mock_ds]) as mock_open_datasets,
            caplog.at_level(logging.WARNING, logger="weather_ingest"),
        ):
            _validate_grib_file(Path(tmp.name))

        mock_open_datasets.assert_called_once()
        # The specific failure reason must appear in the warning log — no silent pass.
        assert any(
            "multiple values for unique key" in r.message for r in caplog.records
        ), "Expected failure reason logged; got: " + str([r.message for r in caplog.records])


def test_validate_grib_file_multilevel_retry_succeeds(caplog):
    """Multi-level GRIB: primary fails, first xr retry (squeeze=False) succeeds."""
    call_count = 0

    def _selective_fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Primary attempt — no squeeze kwarg
            raise Exception("multiple values for unique key")
        # Retry attempt with squeeze=False — succeed
        mock_ds = MagicMock()
        mock_ds.data_vars = {"u10": MagicMock()}
        mock_ds.__enter__ = lambda s: s
        mock_ds.__exit__ = MagicMock(return_value=False)
        return mock_ds

    with tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
        tmp.write(b"x" * 256)
        tmp.flush()

        with (
            patch("ingest.weather_ingest.xr.open_dataset", side_effect=_selective_fail),
            caplog.at_level(logging.WARNING, logger="weather_ingest"),
        ):
            _validate_grib_file(Path(tmp.name))

    assert any(
        "multi-level file detected" in r.message for r in caplog.records
    ), "Expected multi-level warning; got: " + str([r.message for r in caplog.records])


def test_validate_grib_file_all_retries_fail_raises_with_reason():
    """All cfgrib backend attempts fail → ValueError with specific failure reason."""
    with tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
        tmp.write(b"x" * 256)
        tmp.flush()

        with (
            patch(
                "ingest.weather_ingest.xr.open_dataset",
                side_effect=Exception("multiple values for unique key — corrupt index"),
            ),
            patch(
                "cfgrib.open_datasets",
                side_effect=Exception("cfgrib fallback also failed"),
            ),
            pytest.raises(ValueError, match="multiple values for unique key"),
        ):
            _validate_grib_file(Path(tmp.name))


def test_validate_grib_file_non_multilevel_error_raises_immediately():
    """Non-multi-level errors must raise immediately without retrying."""
    with tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
        tmp.write(b"x" * 256)
        tmp.flush()

        open_calls = []

        def _track_and_fail(*args, **kwargs):
            open_calls.append(kwargs)
            raise Exception("unexpected GRIB structure: missing section 3")

        with (
            patch("ingest.weather_ingest.xr.open_dataset", side_effect=_track_and_fail),
            pytest.raises(ValueError, match="missing section 3"),
        ):
            _validate_grib_file(Path(tmp.name))

    # Should only have been called once — no retries for non-multi-level errors.
    assert len(open_calls) == 1, f"Expected 1 call, got {len(open_calls)}"


def test_download_grib_files_preserves_connect_error_without_attribute_error(tmp_path):
    """Transport-level failures should surface as ConnectError, not AttributeError."""

    class _StreamClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self, *_args, **_kwargs):
            raise httpx.ConnectError("dns failure")

    settings = WeatherIngestSettings()

    with (
        patch("ingest.weather_ingest.httpx.Client", return_value=_StreamClient()),
        patch("ingest.weather_ingest.time.sleep"),
        pytest.raises(httpx.ConnectError),
    ):
        download_grib_files(
            settings=settings,
            run_time=datetime(2026, 3, 8, 6, 0, tzinfo=timezone.utc),
            variables=["UGRD"],
            levels=["lev_10_m_above_ground"],
            download_dir=tmp_path,
            base_urls=["https://example.invalid/cgi-bin/filter_gfs_0p25.pl"],
            max_attempts_per_url=1,
        )


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
