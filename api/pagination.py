"""Shared cursor-pagination utilities.

Cursors are opaque to callers — the internal representation (currently keyset
with ``t`` + ``id``) can change without breaking the API contract.  All
encoding/decoding is centralised here so that ``decode_cursor`` is the single
place to update if the format changes.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Callable


def encode_cursor(**fields: object) -> str:
    """Encode cursor fields as a URL-safe base64 JSON string.

    :class:`datetime` values are serialised as ISO-8601 strings; all other
    values are passed through as-is (must be JSON-serialisable).
    """
    serialized: dict[str, object] = {}
    for k, v in fields.items():
        serialized[k] = v.isoformat() if isinstance(v, datetime) else v
    return base64.urlsafe_b64encode(json.dumps(serialized).encode()).decode()


def decode_cursor(cursor: str) -> dict:
    """Decode a cursor string produced by :func:`encode_cursor`.

    Returns a dict with field ``t`` parsed as a timezone-aware
    :class:`datetime` (or ``None``) and all other fields left as-is.

    Raises :exc:`ValueError` on malformed input.
    """
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {exc}") from exc
    if data.get("t") is not None:
        try:
            t = datetime.fromisoformat(data["t"])
            data["t"] = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        except Exception as exc:
            raise ValueError(f"Invalid cursor timestamp: {exc}") from exc
    return data


def build_page(
    rows: list,
    limit: int,
    cursor_fn: Callable[[dict], str],
) -> dict:
    """Trim *rows* to *limit*, detect ``has_more``, encode ``next_cursor``.

    Returns a plain dict with keys ``data``, ``next_cursor``, ``has_more``,
    and ``limit`` suitable for use as a JSON response body.

    *cursor_fn* receives the last row (as a plain ``dict``) and must return an
    opaque cursor string via :func:`encode_cursor`.
    """
    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]
    next_cursor: str | None = None
    if has_more and rows:
        next_cursor = cursor_fn(dict(rows[-1]))
    return {
        "data": [dict(r) for r in rows],
        "next_cursor": next_cursor,
        "has_more": has_more,
        "limit": limit,
    }
