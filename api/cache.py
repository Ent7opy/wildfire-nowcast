"""Shared Redis connection pool singleton.

All sync Redis clients in the process share one ConnectionPool so the total
number of open sockets is bounded by REDIS_POOL_MAX_CONNECTIONS rather than
multiplying with each module that imports Redis.

Note: main.py uses redis.asyncio for FastAPILimiter — that async client is
separate from this sync pool by design.
"""

import redis

from api.config import settings

_pool: redis.ConnectionPool | None = None


def _get_redis_pool() -> redis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_pool_max_connections,
            decode_responses=True,
        )
    return _pool


def get_redis() -> redis.Redis:
    """Return a Redis client backed by the shared connection pool."""
    return redis.Redis(connection_pool=_get_redis_pool())
