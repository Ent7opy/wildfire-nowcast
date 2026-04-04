"""Shared FastAPI dependencies."""

import logging
from typing import Callable, Optional

from fastapi import Depends, Header, HTTPException, Response, status
from sqlalchemy.engine import Engine

from api.config import settings
from api.db import get_engine
from api.fires.repository import FireRepository

logger = logging.getLogger(__name__)


def cache_control(max_age: Optional[int] = None) -> Callable[[Response], None]:
    """Return a dependency that sets Cache-Control on the response.

    Pass max_age (seconds) for cacheable GETs, or omit for no-cache on POSTs.
    """
    header_value = "no-cache" if max_age is None else f"max-age={max_age}"

    def _dep(response: Response) -> None:
        response.headers["Cache-Control"] = header_value

    return _dep


# Named shortcuts used in route decorators
cache_60 = cache_control(60)
cache_300 = cache_control(300)
no_cache = cache_control()


def get_fire_repo(engine: Engine = Depends(get_engine)) -> FireRepository:
    """FastAPI dependency that provides a FireRepository bound to the current engine."""
    return FireRepository(engine)


def verify_internal_api_key(x_internal_api_key: str = Header(None)) -> None:
    """Verify X-Internal-API-Key header against INTERNAL_API_KEY setting.

    If INTERNAL_API_KEY is unset/empty in config, logs a WARNING and allows the request.
    This ensures dev setups don't break, but production deployments must set the key.

    If INTERNAL_API_KEY is set and the header is missing or doesn't match, raises 401.
    """
    if not settings.internal_api_key:
        # Key not configured — log warning and allow (dev/unprotected mode)
        logger.warning(
            "INTERNAL_API_KEY is not configured; internal endpoints are unprotected. "
            "Set INTERNAL_API_KEY in .env for production."
        )
        return

    # Key is configured — validate the header
    if not x_internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Internal-API-Key header",
        )

    if x_internal_api_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Internal-API-Key",
        )
