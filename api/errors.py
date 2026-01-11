"""Standardized error responses."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel

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
