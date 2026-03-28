"""Standardized error responses and domain exception hierarchy."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel
from fastapi import Request
from fastapi.responses import JSONResponse


class ErrorDetail(BaseModel):
    loc: Optional[list[str | int]] = None
    msg: str
    type: str


class ErrorResponse(BaseModel):
    """Standard error response model."""
    code: str
    message: str
    details: Optional[Any] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "validation_error",
                "message": "Invalid request parameters",
                "details": [
                    {"loc": ["query", "limit"], "msg": "value is not a valid integer", "type": "type_error.integer"}
                ]
            }
        }
    }



class WildfireError(Exception):
    """Base class for all domain errors."""


class FiresNotFoundError(WildfireError):
    """No fire detections matched the query."""


class InvalidBoundingBoxError(WildfireError):
    """Bounding box is malformed or too large."""


class StalenessError(WildfireError):
    """Requested data is older than the staleness threshold."""


class ModelNotReadyError(WildfireError):
    """No promoted model is available for inference."""


class ArchiveRangeError(WildfireError):
    """Requested archive range exceeds MAX_ARCHIVE_RANGE_DAYS or FIRMS 10-day limit."""


_STATUS_MAP: dict[type[WildfireError], int] = {
    FiresNotFoundError: 404,
    InvalidBoundingBoxError: 422,
    StalenessError: 503,
    ModelNotReadyError: 503,
    ArchiveRangeError: 400,
}


async def wildfire_error_handler(request: Request, exc: WildfireError) -> JSONResponse:
    """Map domain exceptions to HTTP responses with a typed error body."""
    status = _STATUS_MAP.get(type(exc), 500)
    return JSONResponse(
        status_code=status,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )
