"""In-memory warning cache with 15-minute TTL.

``WarningCache`` wraps any ``WeatherWarningProvider`` and serves from a
cached result set until the TTL expires.  A background refresh is triggered
on the first request after TTL expiry so the refreshing fetch does not block
the caller — callers receive the stale data while the refresh is in flight.

Thread / coroutine safety
-------------------------
The cache uses an ``asyncio.Lock`` so it is safe to call from multiple
concurrent FastAPI request handlers.  The lock is held only during the
in-flight fetch, not during read access to the cached data.

Spatial filtering
-----------------
``warnings_for_point(lat, lon, now)`` does lightweight polygon containment
checks (bounding-box pre-filter + Shapely point-in-polygon) to narrow the
full warning list down to those covering a specific location.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from shapely.geometry import Point, shape

from api.core.weather_warnings import WeatherWarning, WeatherWarningProvider

LOGGER = logging.getLogger(__name__)

# Default cache TTL (MeteoAlarm refreshes every ~15 minutes)
_DEFAULT_TTL_SECONDS = 900


class WarningCache:
    """Caches ``WeatherWarningProvider`` results with a configurable TTL.

    Usage::

        cache = WarningCache(MeteoAlarmProvider(), ttl_seconds=900)
        warnings = await cache.get_all_warnings(now=datetime.now(timezone.utc))
        local = cache.warnings_for_point(lat, lon, now)
    """

    def __init__(
        self,
        provider: WeatherWarningProvider,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._provider = provider
        self._ttl = timedelta(seconds=ttl_seconds)
        self._cached: list[WeatherWarning] = []
        self._last_refresh: datetime | None = None
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[None] | None = None

    def _is_stale(self, now: datetime) -> bool:
        if self._last_refresh is None:
            return True
        return (now - self._last_refresh) > self._ttl

    async def _do_refresh(self) -> None:
        """Fetch fresh warnings and update the cache. Must NOT be called while
        holding ``self._lock`` — the update path acquires it internally."""
        try:
            fresh = await self._provider.get_warnings_for_region()
        except Exception as exc:
            LOGGER.warning("Warning cache refresh failed: %s", exc)
            return
        # Update cache under lock
        async with self._lock:
            self._cached = fresh
            self._last_refresh = datetime.now(timezone.utc)
        LOGGER.debug("Warning cache refreshed: %d warnings loaded", len(fresh))

    async def get_all_warnings(self, now: datetime | None = None) -> list[WeatherWarning]:
        """Return the full cached warning list, triggering a refresh if stale.

        If the cache is cold (never loaded), the refresh blocks and returns
        fresh data.  If stale, the refresh runs in the background and the
        caller receives the previous data immediately.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if not self._is_stale(now):
            return list(self._cached)

        if not self._cached:
            # Cold cache — block until first load completes.
            await self._do_refresh()
        else:
            # Stale but populated — refresh in background, serve stale data
            if self._refresh_task is None or self._refresh_task.done():
                self._refresh_task = asyncio.create_task(self._do_refresh())

        return list(self._cached)

    def warnings_for_point(
        self,
        lat: float,
        lon: float,
        now: datetime,
        active_only: bool = True,
    ) -> list[WeatherWarning]:
        """Return cached warnings whose geometry contains (lon, lat).

        Uses a bounding-box pre-filter before the more expensive
        point-in-polygon check.  Operates on the snapshot currently in
        ``self._cached`` without awaiting a refresh.
        """
        pt = Point(lon, lat)
        results: list[WeatherWarning] = []

        for warning in self._cached:
            if active_only and not warning.is_active(now):
                continue
            geom = warning.geometry
            if not geom:
                continue

            # Bounding-box pre-filter (coordinates are [lon, lat] pairs)
            try:
                polys = _iter_polygons(geom)
                for ring in polys:
                    lons = [c[0] for c in ring]
                    lats = [c[1] for c in ring]
                    if not (min(lons) <= lon <= max(lons) and min(lats) <= lat <= max(lats)):
                        continue
                    shapely_geom = shape(geom)
                    if shapely_geom.contains(pt):
                        results.append(warning)
                        break
            except Exception as exc:
                LOGGER.debug("Spatial check failed for warning %s: %s", warning.id, exc)

        return results

    def as_geojson_feature_collection(
        self,
        now: datetime,
        active_only: bool = True,
    ) -> dict[str, Any]:
        """Return active warnings as a GeoJSON FeatureCollection for the map layer."""
        features = [
            w.as_geojson_feature()
            for w in self._cached
            if not active_only or w.is_active(now)
        ]
        return {"type": "FeatureCollection", "features": features}


def _iter_polygons(geom: dict[str, Any]) -> list[list[list[float]]]:
    """Yield exterior rings from a GeoJSON Polygon or MultiPolygon."""
    geom_type = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if geom_type == "Polygon":
        return [coords[0]] if coords else []
    if geom_type == "MultiPolygon":
        return [poly[0] for poly in coords if poly]
    return []
