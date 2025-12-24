"""API route package for the FastAPI application."""

from .internal import internal_router
from .fires import fires_router

__all__ = ["internal_router", "fires_router"]

