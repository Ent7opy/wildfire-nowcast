"""Shared AOI utility helpers.

Kept in a standalone module so both api.routes.aois and ingest modules can
import without circular dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _is_notifications_paused(aoi: dict[str, Any]) -> bool:
    """Return True if AOI notifications are currently paused.

    The column ``watch_notifications_paused_until`` is NULL when active, or set
    to a future TIMESTAMPTZ when paused.  A past timestamp means the pause has
    expired and notifications are treated as active.
    """
    paused_until = aoi.get("watch_notifications_paused_until")
    if paused_until is None:
        return False
    if isinstance(paused_until, str):
        paused_until = datetime.fromisoformat(paused_until)
    # Normalise to UTC-aware for safe comparison.
    if paused_until.tzinfo is None:
        paused_until = paused_until.replace(tzinfo=timezone.utc)
    return paused_until > datetime.now(timezone.utc)
