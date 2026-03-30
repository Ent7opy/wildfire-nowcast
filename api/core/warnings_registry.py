"""Module-level singleton for the MeteoAlarm warning cache.

``get_warning_cache()`` returns the shared ``WarningCache`` instance, or
``None`` when the MeteoAlarm integration is disabled via environment variable.

Disabling
---------
Set ``METEOALARM_ENABLED=false`` in the environment to skip warning fetches
entirely (useful for non-European deployments or dev environments without
external network access).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from api.core.meteoalarm_provider import MeteoAlarmProvider
from api.core.warning_cache import WarningCache

LOGGER = logging.getLogger(__name__)

_cache: WarningCache | None = None
_initialized = False


def get_warning_cache() -> WarningCache | None:
    """Return the global warning cache, initialising it on first call.

    Returns ``None`` when ``METEOALARM_ENABLED`` is set to ``"false"``.
    """
    global _cache, _initialized  # noqa: PLW0603
    if _initialized:
        return _cache

    enabled = os.environ.get("METEOALARM_ENABLED", "true").strip().lower()
    if enabled in ("false", "0", "no", "off"):
        LOGGER.info("MeteoAlarm integration disabled (METEOALARM_ENABLED=%s)", enabled)
        _cache = None
    else:
        _cache = WarningCache(MeteoAlarmProvider())
        LOGGER.info("MeteoAlarm warning cache initialised")

    _initialized = True
    return _cache


def get_brief_warnings_for_point(
    lat: float, lon: float, now: datetime | None = None
) -> list[dict] | None:
    """Return brief warning dicts for a lat/lon, or None when cache is disabled.

    Returns an empty list (not None) when the cache is enabled but no warnings
    cover the point.  Returns None when MeteoAlarm is disabled so callers can
    omit the field from API responses.
    """
    cache = get_warning_cache()
    if cache is None:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    briefs = [w.as_brief() for w in cache.warnings_for_point(lat, lon, now)]
    return briefs or None


def warnings_overlaps_bbox(
    warning_geom: dict,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> bool:
    """Return True if any ring of *warning_geom* overlaps the given bounding box."""
    from api.core.warning_cache import _iter_polygons
    try:
        for ring in _iter_polygons(warning_geom):
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            w_min_lon, w_max_lon = min(lons), max(lons)
            w_min_lat, w_max_lat = min(lats), max(lats)
            if (
                w_max_lon >= min_lon and w_min_lon <= max_lon
                and w_max_lat >= min_lat and w_min_lat <= max_lat
            ):
                return True
    except Exception:
        pass
    return False
