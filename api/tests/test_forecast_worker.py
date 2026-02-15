from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import patch

from api.forecast.worker import run_jit_forecast_pipeline


def test_run_jit_pipeline_releases_result_lock_on_cache_hit():
    job_id = uuid4()
    bbox = (20.0, 40.0, 21.0, 41.0)
    forecast_params = {
        "forecast_reference_time": datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc).isoformat(),
        "horizons_hours": [24, 48, 72],
        "use_result_cache": True,
    }
    lock = object()

    with patch("api.forecast.worker.acquire_forecast_result_lock", return_value=lock) as mock_acquire, \
         patch("api.forecast.worker.release_forecast_result_lock") as mock_release, \
         patch("api.forecast.worker.repo.find_cached_forecast_run", return_value={"id": 321}), \
         patch("api.forecast.worker.repo.list_rasters_for_run", return_value=[]), \
         patch("api.forecast.worker.repo.list_contours_for_run", return_value=[]), \
         patch("api.forecast.worker.repo.update_jit_job_status") as mock_update:
        run_jit_forecast_pipeline(job_id, bbox, forecast_params)

    mock_acquire.assert_called_once()
    mock_release.assert_called_once_with(lock)
    mock_update.assert_called_once()
    assert mock_update.call_args.args[1] == "completed"


def test_run_jit_pipeline_fails_fast_when_forecast_gate_blocks():
    job_id = uuid4()
    bbox = (20.0, 40.0, 21.0, 41.0)

    with patch("api.forecast.worker.build_data_status_snapshot", return_value={"as_of": "2026-02-11T00:00:00+00:00"}), \
         patch("api.forecast.worker.resolve_forecast_gate", return_value={
             "can_run": False,
             "reasons": ["weather_stale_or_missing"],
             "missing_or_stale_sources": ["weather"],
             "as_of": "2026-02-11T00:00:00+00:00",
             "retry_hint": "wait for weather refresh",
         }), \
         patch("api.forecast.worker.repo.update_jit_job_status") as mock_update:
        run_jit_forecast_pipeline(job_id, bbox, forecast_params={})

    mock_update.assert_called_once()
    assert mock_update.call_args.args[1] == "failed"
    assert "forecast_inputs_stale_or_missing" in (mock_update.call_args.kwargs.get("error") or "")
