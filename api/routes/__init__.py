"""API route package for the FastAPI application."""

from .internal import internal_router
from .fires import fires_router
from .forecast import forecast_router

__all__ = ["internal_router", "fires_router", "forecast_router"]

