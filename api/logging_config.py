"""Structured JSON logging configuration for the API service.

Call ``setup_logging()`` once at application startup (before any log statements)
to replace the default text formatter on the root logger with a JSON formatter.
All existing ``logging.getLogger(__name__)`` calls continue to work unchanged.

When ``LOG_FORMAT=text`` is set (e.g. local development), the original
human-readable format is preserved.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextvars import ContextVar

from pythonjsonlogger.json import JsonFormatter

# ContextVar holding the current request_id; set by FastAPI middleware.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


class _RequestIdJsonFormatter(JsonFormatter):
    """JSON formatter that injects ``request_id`` from the context variable."""

    def add_fields(
        self,
        log_record: dict,
        record: logging.LogRecord,
        message_dict: dict,
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record.setdefault("service", "api")
        rid = request_id_ctx.get()
        if rid:
            log_record["request_id"] = rid


def setup_logging(*, level: str | None = None) -> None:
    """Configure the root logger with a JSON (or text) formatter.

    Parameters
    ----------
    level:
        Override log level.  Defaults to the ``LOG_LEVEL`` env-var, then INFO.
    """
    effective_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()

    handler = logging.StreamHandler()

    if log_format == "text":
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    else:
        handler.setFormatter(
            _RequestIdJsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
            )
        )

    logging.basicConfig(level=effective_level, handlers=[handler], force=True)
