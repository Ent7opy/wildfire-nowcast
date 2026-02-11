"""Distributed lock helpers for forecast result-cache workflows."""

from __future__ import annotations

import logging
import os
from typing import Optional

from redis import Redis
from redis.lock import Lock as RedisLock

LOGGER = logging.getLogger(__name__)

REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
FORECAST_RESULT_LOCK_TIMEOUT_SECONDS = 900
FORECAST_RESULT_LOCK_BLOCKING_SECONDS = 900

_redis_conn = Redis.from_url(REDIS_URL)


def acquire_forecast_result_lock(cache_key: str) -> Optional[RedisLock]:
    """Acquire a distributed lock for a forecast result-cache key."""
    lock_key = f"forecast:result:lock:{cache_key}"
    lock = RedisLock(
        _redis_conn,
        lock_key,
        timeout=FORECAST_RESULT_LOCK_TIMEOUT_SECONDS,
        blocking_timeout=FORECAST_RESULT_LOCK_BLOCKING_SECONDS,
    )
    try:
        if lock.acquire():
            return lock
    except Exception as e:  # pragma: no cover - defensive operational logging
        LOGGER.warning("Failed to acquire forecast-result lock for key=%s: %s", cache_key, e)
    return None


def release_forecast_result_lock(lock: RedisLock | None) -> None:
    """Release a previously-acquired forecast result lock."""
    if lock is None:
        return
    try:
        lock.release()
    except Exception as e:  # pragma: no cover - defensive operational logging
        LOGGER.warning("Failed to release forecast-result lock: %s", e)
