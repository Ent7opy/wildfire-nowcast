"""Shared helpers for structured/consistent ingest logging."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping


def _encode_context(context: Mapping[str, Any]) -> str:
    """Convert a context mapping to a JSON-ish string for log messages."""
    try:
        return json.dumps(context, default=str, sort_keys=True)
    except TypeError:
        safe_ctx = {k: str(v) for k, v in context.items()}
        return json.dumps(safe_ctx, sort_keys=True)


def log_event(
    logger: logging.Logger,
    event: str,
    message: str,
    *,
    level: str = "info",
    **fields: Any,
) -> None:
    """Emit a log message with a consistent event tag and structured context.

    Example:
        log_event(LOGGER, "firms.validation", "Dropped row", reason="bad_lat")
    """

    context = {k: v for k, v in fields.items() if v is not None}
    payload = f"[{event}] {message}"
    if context:
        payload = f"{payload} | {_encode_context(context)}"

    log_fn = getattr(logger, level, logger.info)
    log_fn(payload, extra={"event": event, **context})


