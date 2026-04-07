"""Structured JSON logging configuration for the ingest service.

Call ``setup_logging()`` once at process startup (before any log statements)
to replace the default text formatter on the root logger with a JSON formatter.
All existing ``logging.getLogger(__name__)`` calls continue to work unchanged.

When ``LOG_FORMAT=text`` is set (e.g. local development), the original
human-readable format is preserved.
"""

from __future__ import annotations

import logging
import os

from pythonjsonlogger.json import JsonFormatter


class _ServiceJsonFormatter(JsonFormatter):
    """JSON formatter that tags every record with ``service: ingest``."""

    def add_fields(
        self,
        log_record: dict,
        record: logging.LogRecord,
        message_dict: dict,
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record.setdefault("service", "ingest")


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
            _ServiceJsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
            )
        )

    logging.basicConfig(level=effective_level, handlers=[handler], force=True)
