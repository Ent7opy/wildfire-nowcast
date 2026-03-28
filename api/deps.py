"""Shared FastAPI dependencies."""

from typing import Callable, Optional

from fastapi import Depends, Response
from sqlalchemy.engine import Engine

from api.db import get_engine
from api.fires.repository import FireRepository


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
