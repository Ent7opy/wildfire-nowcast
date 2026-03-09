from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import patch

import numpy as np
import xarray as xr

from api.forecast.worker import run_jit_forecast_pipeline
from api.core.grid import GridSpec
from ml.spread.contract import SpreadForecast
from ml.spread.region_key import bbox_region_name


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


def test_run_jit_pipeline_propagates_confidence_and_shadow_metadata():
    job_id = uuid4()
    bbox = (20.0, 40.0, 21.0, 41.0)
    forecast_params = {
        "forecast_reference_time": datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc).isoformat(),
        "horizons_hours": [24],
        "use_result_cache": False,
    }

    forecast = SpreadForecast(
        probabilities=xr.DataArray(
            np.full((1, 2, 2), 0.2, dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [40.0, 40.1], "lon": [20.0, 20.1], "lead_time_hours": ("time", [24])},
            attrs={
                "confidence_level": "low",
                "staleness_hours": 18.0,
                "shadow_evaluated": True,
                "shadow_metrics_summary": {"latency_delta_ms": 12.3},
                "model_name": "HeuristicSpreadModelV0",
                "model_version": "v0",
            },
        ),
        forecast_reference_time=datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc),
        horizons_hours=[24],
        model_name="HeuristicSpreadModelV0",
        model_version="v0",
    )

    with (
        patch("api.forecast.worker.resolve_request_model_selection", return_value=("HeuristicSpreadModelV0", {}, None)),
        patch("api.forecast.worker.repo.find_cached_forecast_run", return_value=None),
        patch("api.forecast.worker.repo.find_cached_terrain", return_value={"id": 11}),
        patch("api.forecast.worker.repo.find_cached_weather", return_value={"id": 22}),
        patch("api.forecast.worker.repo.create_jit_job"),
        patch("api.forecast.worker.get_spread_model") as mock_get_model,
        patch("ml.spread.service.run_spread_forecast", return_value=forecast),
        patch("api.fires.service.get_region_grid_spec", return_value=GridSpec.from_bbox(bbox)),
        patch("ingest.spread_repository.create_spread_forecast_run", return_value=123),
        patch("ingest.spread_forecast.save_forecast_rasters", return_value=[]),
        patch("ingest.spread_forecast.build_contour_records", return_value=[]),
        patch("ingest.spread_repository.insert_spread_forecast_rasters"),
        patch("ingest.spread_repository.insert_spread_forecast_contours"),
        patch("ingest.spread_repository.finalize_spread_forecast_run"),
        patch("api.forecast.worker.repo.list_rasters_for_run", return_value=[]),
        patch("api.forecast.worker.repo.list_contours_for_run", return_value=[]),
        patch("api.forecast.worker.repo.update_jit_job_status") as mock_update,
    ):
        mock_get_model.return_value = object()
        run_jit_forecast_pipeline(job_id, bbox, forecast_params)

    # Last update call should be completed with enriched result payload.
    completed_calls = [c for c in mock_update.call_args_list if c.args[1] == "completed"]
    assert completed_calls
    result_payload = completed_calls[-1].kwargs.get("result", {})
    assert result_payload.get("confidence_level") == "low"
    assert result_payload.get("staleness_hours") == 18.0
    assert result_payload.get("shadow_evaluated") is True


def test_run_jit_pipeline_derives_region_key_for_bbox_requests():
    job_id = uuid4()
    bbox = (20.0, 40.0, 21.0, 41.0)
    expected_region = bbox_region_name(bbox)
    forecast_params = {
        "forecast_reference_time": datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc).isoformat(),
        "horizons_hours": [24],
        "use_result_cache": False,
    }

    forecast = SpreadForecast(
        probabilities=xr.DataArray(
            np.full((1, 2, 2), 0.2, dtype=np.float32),
            dims=("time", "lat", "lon"),
            coords={"time": [0], "lat": [40.0, 40.1], "lon": [20.0, 20.1], "lead_time_hours": ("time", [24])},
        ),
        forecast_reference_time=datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc),
        horizons_hours=[24],
        model_name="HeuristicSpreadModelV0",
        model_version="v0",
    )
    cache_lock = type("L", (), {"release": lambda self: None})()

    with (
        patch("api.forecast.worker.resolve_request_model_selection", return_value=("HeuristicSpreadModelV0", {}, None)),
        patch("api.forecast.worker._acquire_cache_lock", return_value=cache_lock),
        patch("api.forecast.worker.repo.find_cached_forecast_run", return_value=None),
        patch("api.forecast.worker.repo.find_cached_terrain", side_effect=[None, None]),
        patch("api.forecast.worker.repo.find_cached_weather", return_value={"id": 22}),
        patch("ingest.dem_preprocess.ingest_terrain_for_bbox", return_value=11) as mock_ingest_terrain,
        patch("api.forecast.worker.get_spread_model", return_value=object()),
        patch("ml.spread.service.run_spread_forecast", return_value=forecast),
        patch("api.fires.service.get_region_grid_spec", return_value=GridSpec.from_bbox(bbox)),
        patch("ingest.spread_repository.create_spread_forecast_run", return_value=123) as mock_create_run,
        patch("ingest.spread_forecast.save_forecast_rasters", return_value=[]),
        patch("ingest.spread_forecast.build_contour_records", return_value=[]),
        patch("ingest.spread_repository.insert_spread_forecast_rasters"),
        patch("ingest.spread_repository.insert_spread_forecast_contours"),
        patch("ingest.spread_repository.finalize_spread_forecast_run"),
        patch("api.forecast.worker.repo.update_jit_job_status"),
    ):
        run_jit_forecast_pipeline(job_id, bbox, forecast_params)

    assert mock_ingest_terrain.call_args.kwargs["region_name"] == expected_region
    assert mock_create_run.call_args.kwargs["region_name"] == expected_region
    assert mock_create_run.call_args.kwargs["metadata"]["effective_region_name"] == expected_region
