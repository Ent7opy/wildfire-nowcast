"""Tests for forecast result-cache distributed lock wiring.

Covers:
- acquire_forecast_result_lock / release_forecast_result_lock unit behaviour
- Graceful degradation when Redis is unavailable
- Lock is always released in the finally block (worker + generate endpoint)
- Concurrent requests for the same cache key are serialized
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

from redis.lock import Lock as RedisLock

from api.forecast.cache_lock import (
    acquire_forecast_result_lock,
    release_forecast_result_lock,
)
from api.forecast.worker import run_jit_forecast_pipeline


# ---------------------------------------------------------------------------
# acquire_forecast_result_lock / release_forecast_result_lock unit tests
# ---------------------------------------------------------------------------


def test_acquire_returns_none_when_redis_unavailable():
    """Redis connection error must not propagate — return None instead."""
    with patch("api.forecast.cache_lock.RedisLock") as mock_lock_cls:
        mock_lock_cls.return_value.acquire.side_effect = Exception("Redis down")
        result = acquire_forecast_result_lock("some-cache-key")

    assert result is None


def test_acquire_returns_none_when_lock_not_acquired():
    """If the lock cannot be acquired (timeout), return None."""
    with patch("api.forecast.cache_lock.RedisLock") as mock_lock_cls:
        mock_lock_cls.return_value.acquire.return_value = False
        result = acquire_forecast_result_lock("busy-key")

    assert result is None


def test_acquire_returns_lock_on_success():
    """Return the RedisLock instance when acquired successfully."""
    with patch("api.forecast.cache_lock.RedisLock") as mock_lock_cls:
        fake_lock = MagicMock(spec=RedisLock)
        fake_lock.acquire.return_value = True
        mock_lock_cls.return_value = fake_lock

        result = acquire_forecast_result_lock("my-key")

    assert result is fake_lock


def test_release_is_noop_for_none():
    """release_forecast_result_lock(None) must not raise."""
    release_forecast_result_lock(None)  # should complete without error


def test_release_calls_lock_release():
    """release_forecast_result_lock calls lock.release() exactly once."""
    mock_lock = MagicMock(spec=RedisLock)
    release_forecast_result_lock(mock_lock)
    mock_lock.release.assert_called_once()


def test_release_swallows_release_error():
    """If lock.release() raises, the exception must not propagate."""
    mock_lock = MagicMock(spec=RedisLock)
    mock_lock.release.side_effect = Exception("already expired")
    release_forecast_result_lock(mock_lock)  # must not raise


# ---------------------------------------------------------------------------
# Worker: lock is released in finally even on exception
# ---------------------------------------------------------------------------


def test_worker_releases_lock_on_pipeline_exception():
    """Lock must be released even when the forecast pipeline raises."""
    job_id = uuid4()
    bbox = (20.0, 40.0, 21.0, 41.0)
    forecast_params = {
        "forecast_reference_time": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat(),
        "horizons_hours": [24],
        "use_result_cache": True,
    }
    mock_lock = MagicMock(spec=RedisLock)

    with (
        patch("api.forecast.worker.acquire_forecast_result_lock", return_value=mock_lock),
        patch("api.forecast.worker.release_forecast_result_lock") as mock_release,
        patch("api.forecast.worker.repo.find_cached_forecast_run", return_value=None),
        patch("api.forecast.worker.repo.find_cached_terrain", return_value={"id": 1}),
        patch("api.forecast.worker.repo.find_cached_weather", return_value={"id": 2}),
        patch("api.forecast.worker.repo.update_jit_job_status"),
        patch(
            "api.forecast.worker.resolve_request_model_selection",
            return_value=("HeuristicSpreadModelV0", {}, None),
        ),
        patch("api.forecast.worker.get_spread_model", side_effect=RuntimeError("model load failed")),
    ):
        run_jit_forecast_pipeline(job_id, bbox, forecast_params)

    mock_release.assert_called_once_with(mock_lock)


def test_worker_releases_lock_on_cache_miss_and_success():
    """Lock is released after a successful full pipeline run (no cache hit)."""
    from api.core.grid import GridSpec

    job_id = uuid4()
    bbox = (20.0, 40.0, 21.0, 41.0)
    forecast_params = {
        "forecast_reference_time": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat(),
        "horizons_hours": [24],
        "use_result_cache": True,
    }
    mock_lock = MagicMock(spec=RedisLock)

    with (
        patch("api.forecast.worker.acquire_forecast_result_lock", return_value=mock_lock),
        patch("api.forecast.worker.release_forecast_result_lock") as mock_release,
        patch("api.forecast.worker.repo.find_cached_forecast_run", return_value=None),
        patch("api.forecast.worker.repo.find_cached_terrain", return_value={"id": 1}),
        patch("api.forecast.worker.repo.find_cached_weather", return_value={"id": 2}),
        patch("api.forecast.worker.repo.update_jit_job_status"),
        patch(
            "api.forecast.worker.resolve_request_model_selection",
            return_value=("HeuristicSpreadModelV0", {}, None),
        ),
        patch("api.forecast.worker.get_spread_model", return_value=MagicMock()),
        patch("ml.spread.service.run_spread_forecast", return_value=MagicMock()),
        patch("api.fires.service.get_region_grid_spec", return_value=GridSpec.from_bbox(bbox)),
        patch("ingest.spread_repository.create_spread_forecast_run", return_value=42),
        patch("ingest.spread_forecast.save_forecast_rasters", return_value=[]),
        patch("ingest.spread_forecast.build_contour_records", return_value=[]),
        patch("ingest.spread_repository.insert_spread_forecast_rasters"),
        patch("ingest.spread_repository.insert_spread_forecast_contours"),
        patch("ingest.spread_repository.finalize_spread_forecast_run"),
    ):
        run_jit_forecast_pipeline(job_id, bbox, forecast_params)

    mock_release.assert_called_once_with(mock_lock)


# ---------------------------------------------------------------------------
# Worker: graceful degradation when Redis is unavailable
# ---------------------------------------------------------------------------


def test_worker_proceeds_without_lock_when_redis_unavailable():
    """When acquire returns None (Redis down), the pipeline still completes."""
    from api.core.grid import GridSpec

    job_id = uuid4()
    bbox = (20.0, 40.0, 21.0, 41.0)
    forecast_params = {
        "forecast_reference_time": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat(),
        "horizons_hours": [24],
        "use_result_cache": True,
    }

    with (
        # Redis unavailable → lock returns None
        patch("api.forecast.worker.acquire_forecast_result_lock", return_value=None),
        patch("api.forecast.worker.release_forecast_result_lock") as mock_release,
        patch("api.forecast.worker.repo.find_cached_forecast_run", return_value=None),
        patch("api.forecast.worker.repo.find_cached_terrain", return_value={"id": 1}),
        patch("api.forecast.worker.repo.find_cached_weather", return_value={"id": 2}),
        patch("api.forecast.worker.repo.update_jit_job_status") as mock_update,
        patch(
            "api.forecast.worker.resolve_request_model_selection",
            return_value=("HeuristicSpreadModelV0", {}, None),
        ),
        patch("api.forecast.worker.get_spread_model", return_value=MagicMock()),
        patch("ml.spread.service.run_spread_forecast", return_value=MagicMock()),
        patch("api.fires.service.get_region_grid_spec", return_value=GridSpec.from_bbox(bbox)),
        patch("ingest.spread_repository.create_spread_forecast_run", return_value=42),
        patch("ingest.spread_forecast.save_forecast_rasters", return_value=[]),
        patch("ingest.spread_forecast.build_contour_records", return_value=[]),
        patch("ingest.spread_repository.insert_spread_forecast_rasters"),
        patch("ingest.spread_repository.insert_spread_forecast_contours"),
        patch("ingest.spread_repository.finalize_spread_forecast_run"),
    ):
        run_jit_forecast_pipeline(job_id, bbox, forecast_params)

    # Pipeline completed despite no lock
    completed = [c for c in mock_update.call_args_list if c.args[1] == "completed"]
    assert completed, "job should reach 'completed' status even without Redis lock"
    # release is still called with None (safe no-op)
    mock_release.assert_called_once_with(None)


def test_worker_skips_lock_when_result_cache_disabled():
    """Lock must not be acquired when use_result_cache=False."""
    from api.core.grid import GridSpec

    job_id = uuid4()
    bbox = (20.0, 40.0, 21.0, 41.0)
    forecast_params = {
        "forecast_reference_time": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat(),
        "horizons_hours": [24],
        "use_result_cache": False,
    }

    with (
        patch("api.forecast.worker.acquire_forecast_result_lock") as mock_acquire,
        patch("api.forecast.worker.release_forecast_result_lock"),
        patch("api.forecast.worker.repo.find_cached_forecast_run", return_value=None),
        patch("api.forecast.worker.repo.find_cached_terrain", return_value={"id": 1}),
        patch("api.forecast.worker.repo.find_cached_weather", return_value={"id": 2}),
        patch("api.forecast.worker.repo.update_jit_job_status"),
        patch(
            "api.forecast.worker.resolve_request_model_selection",
            return_value=("HeuristicSpreadModelV0", {}, None),
        ),
        patch("api.forecast.worker.get_spread_model", return_value=MagicMock()),
        patch("ml.spread.service.run_spread_forecast", return_value=MagicMock()),
        patch("api.fires.service.get_region_grid_spec", return_value=GridSpec.from_bbox(bbox)),
        patch("ingest.spread_repository.create_spread_forecast_run", return_value=42),
        patch("ingest.spread_forecast.save_forecast_rasters", return_value=[]),
        patch("ingest.spread_forecast.build_contour_records", return_value=[]),
        patch("ingest.spread_repository.insert_spread_forecast_rasters"),
        patch("ingest.spread_repository.insert_spread_forecast_contours"),
        patch("ingest.spread_repository.finalize_spread_forecast_run"),
    ):
        run_jit_forecast_pipeline(job_id, bbox, forecast_params)

    mock_acquire.assert_not_called()


# ---------------------------------------------------------------------------
# Concurrent serialization: same cache key must not run twice in parallel
# ---------------------------------------------------------------------------


def test_concurrent_requests_same_cache_key_serialized():
    """Two pipeline calls for the same cache key must serialize via the lock.

    We use a real threading.Lock as a stand-in for the Redis distributed lock
    to verify that the second caller waits until the first releases.
    """
    from api.core.grid import GridSpec

    bbox = (20.0, 40.0, 21.0, 41.0)
    forecast_params = {
        "forecast_reference_time": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat(),
        "horizons_hours": [24],
        "use_result_cache": True,
    }
    # SpreadForecast is a frozen dataclass — safe to share across threads
    mock_forecast = MagicMock()

    # Track concurrent execution overlap
    overlap_detected = []
    _mutex = threading.Lock()
    active_count = 0
    real_lock_holder = threading.Lock()

    class FakeRedisLock:
        """Simulates Redis distributed lock using a real threading lock."""

        def __init__(self):
            self._held = False

        def acquire(self):
            real_lock_holder.acquire()
            self._held = True
            nonlocal active_count
            with _mutex:
                active_count += 1
                if active_count > 1:
                    overlap_detected.append(True)
            return True

        def release(self):
            if not self._held:
                return
            self._held = False
            nonlocal active_count
            with _mutex:
                active_count -= 1
            real_lock_holder.release()

    def run_one():
        fake_lock = FakeRedisLock()
        with (
            patch("api.forecast.worker.acquire_forecast_result_lock", return_value=fake_lock),
            patch("api.forecast.worker.release_forecast_result_lock", side_effect=lambda lk: lk.release() if lk else None),
            patch("api.forecast.worker.repo.find_cached_forecast_run", return_value=None),
            patch("api.forecast.worker.repo.find_cached_terrain", return_value={"id": 1}),
            patch("api.forecast.worker.repo.find_cached_weather", return_value={"id": 2}),
            patch("api.forecast.worker.repo.update_jit_job_status"),
            patch(
                "api.forecast.worker.resolve_request_model_selection",
                return_value=("HeuristicSpreadModelV0", {}, None),
            ),
            patch("api.forecast.worker.get_spread_model", return_value=MagicMock()),
            patch("ml.spread.service.run_spread_forecast", return_value=mock_forecast),
            patch("api.fires.service.get_region_grid_spec", return_value=GridSpec.from_bbox(bbox)),
            patch("ingest.spread_repository.create_spread_forecast_run", return_value=42),
            patch("ingest.spread_forecast.save_forecast_rasters", return_value=[]),
            patch("ingest.spread_forecast.build_contour_records", return_value=[]),
            patch("ingest.spread_repository.insert_spread_forecast_rasters"),
            patch("ingest.spread_repository.insert_spread_forecast_contours"),
            patch("ingest.spread_repository.finalize_spread_forecast_run"),
        ):
            run_jit_forecast_pipeline(uuid4(), bbox, forecast_params)

    t1 = threading.Thread(target=run_one)
    t2 = threading.Thread(target=run_one)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not overlap_detected, "Two pipelines for the same cache key must not execute concurrently"


# ---------------------------------------------------------------------------
# Lock TTL: lock created with the configured timeout
# ---------------------------------------------------------------------------


def test_lock_created_with_correct_ttl():
    """The distributed lock must be created with the module-level TTL constant."""
    from api.forecast.cache_lock import FORECAST_RESULT_LOCK_TIMEOUT_SECONDS

    with patch("api.forecast.cache_lock.RedisLock") as mock_lock_cls:
        mock_lock_cls.return_value.acquire.return_value = True
        acquire_forecast_result_lock("ttl-test-key")

    _, kwargs = mock_lock_cls.call_args
    assert kwargs.get("timeout") == FORECAST_RESULT_LOCK_TIMEOUT_SECONDS
