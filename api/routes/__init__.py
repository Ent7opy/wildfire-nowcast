"""API route package for the FastAPI application."""

from .internal import internal_router
from .fires import fires_router
from .forecast import forecast_router
from .aois import aois_router
from .tiles import tiles_router
from .exports import exports_router
from .risk import risk_router

__all__ = ["internal_router", "fires_router", "forecast_router", "aois_router", "tiles_router", "exports_router", "risk_router"]

