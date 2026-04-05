"""Tests for LFMC ecLand ingestion with remote job cancellation and orphan cleanup."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import httpx
import pytest
import sqlalchemy as sa
import xarray as xr
import numpy as np

from api.db import get_engine
from ingest.lfmc_ecland_ingest import (
    _cancel_job,
    _check_orphaned_jobs,
    _create_run_record,
    _finalize_run_record,
    _parse_run_time,
    _poll_job_until_ready,
    _submit_job,
    _update_run_record_remote_job_id,
    ingest_lfmc_ecland_for_bbox,
    LFMC_PROVIDER,
)


@pytest.fixture
def bbox() -> tuple[float, float, float, float]:
    """Standard test bounding box."""
    return (-120.0, 38.0, -119.0, 39.0)


@pytest.fixture
def run_time() -> datetime:
    """Standard test run time."""
    return datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for test files."""
    return tmp_path / "lfmc_output"


def create_dummy_lfmc_netcdf(path: Path) -> None:
    """Create a minimal valid LFMC NetCDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            "lfmc": (["lat", "lon"], np.random.rand(10, 10) * 200),
        },
        coords={
            "lat": np.linspace(38.0, 39.0, 10),
            "lon": np.linspace(-120.0, -119.0, 10),
        },
    )
    ds.to_netcdf(path)
    ds.close()


class TestParseRunTime:
    """Tests for _parse_run_time."""

    def test_parse_iso_with_z(self):
        result = _parse_run_time("2026-04-05T12:00:00Z")
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 5
        assert result.hour == 12
        assert result.tzinfo == timezone.utc

    def test_parse_iso_with_offset(self):
        result = _parse_run_time("2026-04-05T12:00:00+00:00")
        assert result.tzinfo == timezone.utc

    def test_parse_none_returns_current_hour(self):
        result = _parse_run_time(None)
        assert result.minute == 0
        assert result.second == 0
        assert result.tzinfo == timezone.utc


@pytest.mark.integration
class TestCreateRunRecord:
    """Tests for _create_run_record."""

    def test_create_run_record(self, bbox: tuple, run_time: datetime):
        run_id = _create_run_record(
            run_time=run_time,
            bbox=bbox,
            storage_path="s3://bucket/file.nc",
        )
        assert isinstance(run_id, int)
        assert run_id > 0

        # Verify record in database
        stmt = sa.text("SELECT status, provider, remote_job_id FROM fuel_moisture_runs WHERE id = :id")
        with get_engine().begin() as conn:
            row = conn.execute(stmt, {"id": run_id}).mappings().first()
        assert row is not None
        assert row["status"] == "running"
        assert row["provider"] == LFMC_PROVIDER
        assert row["remote_job_id"] is None  # Initially None


@pytest.mark.integration
class TestUpdateRunRecordRemoteJobId:
    """Tests for _update_run_record_remote_job_id."""

    def test_update_remote_job_id(self, bbox: tuple, run_time: datetime):
        run_id = _create_run_record(
            run_time=run_time,
            bbox=bbox,
            storage_path="pending://file.nc",
        )
        job_id = "ecmwf-job-12345"
        _update_run_record_remote_job_id(run_id=run_id, remote_job_id=job_id)

        # Verify update
        stmt = sa.text("SELECT remote_job_id FROM fuel_moisture_runs WHERE id = :id")
        with get_engine().begin() as conn:
            row = conn.execute(stmt, {"id": run_id}).mappings().first()
        assert row["remote_job_id"] == job_id


@pytest.mark.integration
class TestFinalizeRunRecord:
    """Tests for _finalize_run_record."""

    def test_finalize_completed(self, bbox: tuple, run_time: datetime):
        run_id = _create_run_record(
            run_time=run_time,
            bbox=bbox,
            storage_path="pending://file.nc",
        )
        _finalize_run_record(
            run_id=run_id,
            status="completed",
            storage_path="s3://bucket/file.nc",
            coverage_fraction=0.95,
        )

        # Verify update
        stmt = sa.text(
            "SELECT status, storage_path, coverage_fraction FROM fuel_moisture_runs WHERE id = :id"
        )
        with get_engine().begin() as conn:
            row = conn.execute(stmt, {"id": run_id}).mappings().first()
        assert row["status"] == "completed"
        assert row["storage_path"] == "s3://bucket/file.nc"
        assert row["coverage_fraction"] == pytest.approx(0.95)

    def test_finalize_failed(self, bbox: tuple, run_time: datetime):
        run_id = _create_run_record(
            run_time=run_time,
            bbox=bbox,
            storage_path="pending://file.nc",
        )
        _finalize_run_record(
            run_id=run_id,
            status="failed",
            storage_path="",
            coverage_fraction=None,
        )

        stmt = sa.text("SELECT status, coverage_fraction FROM fuel_moisture_runs WHERE id = :id")
        with get_engine().begin() as conn:
            row = conn.execute(stmt, {"id": run_id}).mappings().first()
        assert row["status"] == "failed"
        assert row["coverage_fraction"] is None


class TestCancelJob:
    """Tests for _cancel_job."""

    def test_cancel_job_via_delete(self):
        """Test successful job cancellation via DELETE."""
        with mock.patch("ingest.lfmc_ecland_ingest.httpx.Client"):
            mock_client = mock.MagicMock()
            mock_response = mock.MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_client.delete.return_value = mock_response

            _cancel_job(
                client=mock_client,
                api_url="https://api.example.com",
                job_id="job-123",
            )

            mock_client.delete.assert_called_once()
            call_args = mock_client.delete.call_args
            assert "job-123" in call_args[0][0]

    def test_cancel_job_fallback_to_post(self):
        """Test job cancellation fallback to POST /cancel."""
        with mock.patch("ingest.lfmc_ecland_ingest.httpx.Client"):
            mock_client = mock.MagicMock()
            delete_response = mock.MagicMock()
            delete_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404", request=mock.MagicMock(), response=mock.MagicMock()
            )
            post_response = mock.MagicMock()
            post_response.raise_for_status.return_value = None

            mock_client.delete.return_value = delete_response
            mock_client.post.return_value = post_response

            _cancel_job(
                client=mock_client,
                api_url="https://api.example.com",
                job_id="job-123",
            )

            mock_client.delete.assert_called_once()
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "job-123/cancel" in call_args[0][0]

    def test_cancel_job_both_methods_fail_does_not_raise(self):
        """Test that cancellation failure is logged but does not raise."""
        with mock.patch("ingest.lfmc_ecland_ingest.httpx.Client"):
            mock_client = mock.MagicMock()
            delete_response = mock.MagicMock()
            delete_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=mock.MagicMock(), response=mock.MagicMock()
            )
            post_response = mock.MagicMock()
            post_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=mock.MagicMock(), response=mock.MagicMock()
            )

            mock_client.delete.return_value = delete_response
            mock_client.post.return_value = post_response

            # Should not raise, just log warning
            _cancel_job(
                client=mock_client,
                api_url="https://api.example.com",
                job_id="job-123",
            )


@pytest.mark.integration
class TestCheckOrphanedJobs:
    """Tests for _check_orphaned_jobs."""

    def test_check_orphaned_jobs_finds_and_cancels(self, bbox: tuple, run_time: datetime):
        """Test that orphaned jobs are found, cancelled, and marked as failed."""
        # Create a stale running record with remote_job_id
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        run_id = _create_run_record(
            run_time=old_time,
            bbox=bbox,
            storage_path="pending://file.nc",
        )
        job_id = "orphan-job-999"
        _update_run_record_remote_job_id(run_id=run_id, remote_job_id=job_id)

        # Mock the HTTP client
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_client.delete.return_value = mock_response

        _check_orphaned_jobs(
            client=mock_client,
            api_url="https://api.example.com",
            timeout_seconds=1800,
        )

        # Verify the job was cancelled
        mock_client.delete.assert_called_once()

        # Verify the run record was marked as failed
        stmt = sa.text("SELECT status FROM fuel_moisture_runs WHERE id = :id")
        with get_engine().begin() as conn:
            row = conn.execute(stmt, {"id": run_id}).mappings().first()
        assert row["status"] == "failed"

    def test_check_orphaned_jobs_ignores_recent_running(self, bbox: tuple, run_time: datetime):
        """Test that recent running jobs are not marked as orphaned."""
        run_id = _create_run_record(
            run_time=run_time,
            bbox=bbox,
            storage_path="pending://file.nc",
        )
        job_id = "recent-job-123"
        _update_run_record_remote_job_id(run_id=run_id, remote_job_id=job_id)

        # Mock the HTTP client
        mock_client = mock.MagicMock()

        _check_orphaned_jobs(
            client=mock_client,
            api_url="https://api.example.com",
            timeout_seconds=1800,
        )

        # Verify the job was NOT cancelled (because it's recent)
        mock_client.delete.assert_not_called()

        # Verify the run record is still running
        stmt = sa.text("SELECT status FROM fuel_moisture_runs WHERE id = :id")
        with get_engine().begin() as conn:
            row = conn.execute(stmt, {"id": run_id}).mappings().first()
        assert row["status"] == "running"


class TestSubmitJob:
    """Tests for _submit_job."""

    def test_submit_job_extracts_job_id_from_id_field(self):
        """Test job submission with job_id in 'id' field."""
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"id": "job-abc-123"}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        job_id = _submit_job(
            client=mock_client,
            api_url="https://api.example.com",
            run_time=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
            bbox=(-120.0, 38.0, -119.0, 39.0),
        )

        assert job_id == "job-abc-123"

    def test_submit_job_extracts_job_id_from_job_id_field(self):
        """Test job submission with explicit job_id field."""
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"job_id": "job-xyz-789"}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        job_id = _submit_job(
            client=mock_client,
            api_url="https://api.example.com",
            run_time=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
            bbox=(-120.0, 38.0, -119.0, 39.0),
        )

        assert job_id == "job-xyz-789"


class TestPollJobUntilReady:
    """Tests for _poll_job_until_ready."""

    def test_poll_job_succeeds(self):
        """Test successful polling."""
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"status": "completed", "download_url": "https://example.com/file"}
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response

        body = _poll_job_until_ready(
            client=mock_client,
            api_url="https://api.example.com",
            job_id="job-123",
            poll_seconds=1,
            timeout_seconds=10,
        )

        assert body["status"] == "completed"

    def test_poll_job_timeout(self):
        """Test polling timeout."""
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"status": "running"}
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response

        with pytest.raises(TimeoutError):
            _poll_job_until_ready(
                client=mock_client,
                api_url="https://api.example.com",
                job_id="job-123",
                poll_seconds=1,
                timeout_seconds=1,
            )

    def test_poll_job_failed(self):
        """Test polling for failed job."""
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"status": "failed", "error": "Out of memory"}
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response

        with pytest.raises(RuntimeError, match="failed"):
            _poll_job_until_ready(
                client=mock_client,
                api_url="https://api.example.com",
                job_id="job-123",
                poll_seconds=1,
                timeout_seconds=10,
            )


@pytest.mark.integration
class TestIngestLfmcEclandForBbox:
    """Integration tests for ingest_lfmc_ecland_for_bbox."""

    def test_timeout_cancels_remote_job(
        self, bbox: tuple, run_time: datetime, temp_output_dir: Path, monkeypatch
    ):
        """Test that timeout triggers remote job cancellation."""
        monkeypatch.setenv("LFMC_ECLAND_API_URL", "https://api.example.com")
        monkeypatch.setenv("LFMC_ECLAND_API_TOKEN", "test-token")

        with mock.patch("ingest.lfmc_ecland_ingest._submit_job") as mock_submit:
            with mock.patch("ingest.lfmc_ecland_ingest._poll_job_until_ready") as mock_poll:
                with mock.patch("ingest.lfmc_ecland_ingest._cancel_job") as mock_cancel:
                    with mock.patch("ingest.lfmc_ecland_ingest._check_orphaned_jobs"):
                        mock_submit.return_value = "job-timeout-test"
                        mock_poll.side_effect = TimeoutError("Job timed out")

                        with pytest.raises(TimeoutError):
                            ingest_lfmc_ecland_for_bbox(
                                bbox=bbox,
                                run_time=run_time,
                                output_dir=temp_output_dir,
                                timeout_seconds=1,
                            )

                        # Verify cancellation was attempted
                        mock_cancel.assert_called_once()
                        assert mock_cancel.call_args[1]["job_id"] == "job-timeout-test"

    def test_successful_ingestion(
        self, bbox: tuple, run_time: datetime, temp_output_dir: Path, monkeypatch
    ):
        """Test successful end-to-end ingestion."""
        monkeypatch.setenv("LFMC_ECLAND_API_URL", "https://api.example.com")
        monkeypatch.setenv("LFMC_ECLAND_API_TOKEN", "test-token")

        # Create a dummy output file
        output_nc = (
            temp_output_dir
            / f"lfmc_ecland_{run_time:%Y%m%dT%HZ}_bbox_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}.nc"
        )
        create_dummy_lfmc_netcdf(output_nc)

        with mock.patch("ingest.lfmc_ecland_ingest._submit_job") as mock_submit:
            with mock.patch("ingest.lfmc_ecland_ingest._poll_job_until_ready") as mock_poll:
                with mock.patch("ingest.lfmc_ecland_ingest._download_result") as mock_download:
                    with mock.patch("ingest.lfmc_ecland_ingest._check_orphaned_jobs"):
                        mock_submit.return_value = "job-success-123"
                        mock_poll.return_value = {"status": "completed", "download_url": "https://example.com"}
                        mock_download.side_effect = lambda **kwargs: None  # Pretend file exists

                        result = ingest_lfmc_ecland_for_bbox(
                            bbox=bbox,
                            run_time=run_time,
                            output_dir=temp_output_dir,
                        )

                        assert result["run_id"] > 0
                        assert result["provider"] == LFMC_PROVIDER
                        assert "lfmc_ecland" in result["storage_path"]

                        # Verify DB record
                        stmt = sa.text("SELECT status, remote_job_id FROM fuel_moisture_runs WHERE id = :id")
                        with get_engine().begin() as conn:
                            row = conn.execute(stmt, {"id": result["run_id"]}).mappings().first()
                        assert row["status"] == "completed"
                        assert row["remote_job_id"] == "job-success-123"
