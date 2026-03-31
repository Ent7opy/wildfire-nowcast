"""Lightning / thunderstorm proximity proxy ingest.

Materialises a lightweight ``ignition_lightning_proxy`` staging table that
flags grid cells where active thunderstorm-type MeteoAlarm warnings intersect.
This is the ``mvp_operational`` proxy for lightning ignition signal; it will be
superseded by NOAA LIS/OTD or a proper lightning forecast product in a future
``science_grade`` iteration.

Design
------
- Query ``api.core.meteoalarm_provider.MeteoAlarmProvider`` (or the existing
  in-memory warning provider) for all active ``thunderstorm`` warnings.
- Project each warning polygon onto a regular 0.1° grid over the configured
  bbox and write one row per cell with a boolean ``thunderstorm_active`` flag.
- Each run is a complete refresh: delete the previous proxy rows and insert new
  ones.  This is safe because the proxy is a staging table used only by the
  ignition probability feature extraction; downstream reads always take the
  latest snapshot.
- The interface (table name ``ignition_lightning_proxy``, columns
  ``grid_lon``, ``grid_lat``, ``thunderstorm_active``, ``valid_time``) is
  intentionally stable so future upgrades to the lightning signal source do not
  require changes to the ML feature extraction layer.

Limitations (mvp_operational)
------------------------------
- MeteoAlarm covers Europe only; cells outside that coverage area will always
  have ``thunderstorm_active = false``.
- Polygon resolution is country/region-level, not point-level.
- No temporal interpolation: a warning active at any point during the next
  6 hours is treated as active for the entire current cycle.

Science-grade upgrade path
--------------------------
Replace ``_fetch_thunderstorm_warnings`` with a NOAA LIS/OTD or ENTLN adapter
that returns the same ``list[dict]`` schema (``geometry``, ``onset``,
``expires``).  The grid materialisation and DB write logic below is provider-
agnostic and does not need to change.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa

from api.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("lightning_proxy_ingest")

_DEFAULT_GRID_RESOLUTION_DEG = 0.1
_GRID_RESOLUTION_ENV = "LIGHTNING_PROXY_GRID_DEG"

# Default bbox: global (same as drought ingest default; narrows to MeteoAlarm
# coverage in practice since non-European cells will simply have no warnings).
_DEFAULT_BBOX: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _grid_resolution() -> float:
    raw = os.getenv(_GRID_RESOLUTION_ENV, "")
    try:
        val = float(raw)
        if val > 0:
            return val
    except (ValueError, TypeError):
        pass
    return _DEFAULT_GRID_RESOLUTION_DEG


def _generate_grid_cells(
    bbox: tuple[float, float, float, float],
    resolution: float,
) -> list[tuple[float, float]]:
    """Return (lon, lat) cell centre points for the given bbox and resolution."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lons: list[float] = []
    n_lon = max(1, math.ceil((max_lon - min_lon) / resolution))
    for i in range(n_lon):
        lons.append(round(min_lon + (i + 0.5) * resolution, 6))

    lats: list[float] = []
    n_lat = max(1, math.ceil((max_lat - min_lat) / resolution))
    for j in range(n_lat):
        lats.append(round(min_lat + (j + 0.5) * resolution, 6))

    return [(lon, lat) for lat in lats for lon in lons]


def _point_in_polygon(lon: float, lat: float, polygon_coords: list[list[list[float]]]) -> bool:
    """Ray-casting point-in-polygon test for a GeoJSON Polygon ring list."""
    for ring in polygon_coords:
        inside = False
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        if inside:
            return True
    return False


def _geometry_covers_point(geometry: dict[str, Any], lon: float, lat: float) -> bool:
    """Return True if a GeoJSON geometry (Polygon or MultiPolygon) covers the point."""
    geom_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        return _point_in_polygon(lon, lat, coords)
    if geom_type == "MultiPolygon":
        return any(_point_in_polygon(lon, lat, poly) for poly in coords)
    return False


def _fetch_thunderstorm_warnings(now: datetime) -> list[dict[str, Any]]:
    """Return active thunderstorm-type MeteoAlarm warnings as plain dicts.

    Each dict has keys: ``geometry`` (GeoJSON), ``onset`` (datetime), ``expires`` (datetime).
    Errors from individual country feeds are suppressed (MeteoAlarmProvider contract).
    """
    from api.core.meteoalarm_provider import MeteoAlarmProvider

    provider = MeteoAlarmProvider()
    try:
        warnings = asyncio.run(provider.get_warnings_for_region())
    except Exception as exc:
        LOGGER.warning(
            "MeteoAlarm fetch failed: %s — lightning proxy will be empty this cycle. "
            "Mitigation: check network connectivity; target: science_grade retry logic.",
            exc,
        )
        return []

    active: list[dict[str, Any]] = []
    for w in warnings:
        if w.warning_type != "thunderstorm":
            continue
        if w.expires < now:
            continue
        if w.geometry is None:
            continue
        active.append({"geometry": w.geometry, "onset": w.onset, "expires": w.expires})

    return active


def _materialise_proxy(
    *,
    bbox: tuple[float, float, float, float],
    warnings: list[dict[str, Any]],
    valid_time: datetime,
    grid_resolution: float,
) -> tuple[int, int]:
    """Refresh ``ignition_lightning_proxy`` with the current thunderstorm grid.

    Performs a delete-then-insert inside a single transaction so the table is
    never empty between cycles (the delete and insert are atomic).

    Returns ``(active_cells, total_cells)``.
    """
    cells = _generate_grid_cells(bbox, grid_resolution)
    total_cells = len(cells)

    lons: list[float] = []
    lats: list[float] = []
    active_flags: list[bool] = []
    active_count = 0
    for lon, lat in cells:
        active = any(_geometry_covers_point(w["geometry"], lon, lat) for w in warnings)
        if active:
            active_count += 1
        lons.append(lon)
        lats.append(lat)
        active_flags.append(active)

    with get_engine().begin() as conn:
        conn.execute(sa.text("DELETE FROM ignition_lightning_proxy"))
        if cells:
            conn.execute(
                sa.text(
                    """
                    INSERT INTO ignition_lightning_proxy (grid_lon, grid_lat, thunderstorm_active, valid_time)
                    SELECT unnest(:lons::double precision[]),
                           unnest(:lats::double precision[]),
                           unnest(:active::boolean[]),
                           :valid_time
                    """
                ),
                {
                    "lons": lons,
                    "lats": lats,
                    "active": active_flags,
                    "valid_time": valid_time,
                },
            )

    return active_count, total_cells


def ingest_lightning_proxy(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    grid_resolution: float | None = None,
) -> dict[str, Any]:
    """Materialise the thunderstorm-active grid from MeteoAlarm warnings.

    Parameters
    ----------
    bbox:
        WGS84 bounding box (min_lon, min_lat, max_lon, max_lat).
        Defaults to global coverage; MeteoAlarm data covers Europe only.
    grid_resolution:
        Grid cell size in decimal degrees.
        Defaults to ``LIGHTNING_PROXY_GRID_DEG`` env var or 0.1°.

    Returns a summary dict with ``valid_time``, ``total_cells``,
    ``active_cells``, and ``warning_count``.
    Raises RuntimeError on unrecoverable failures.
    """
    resolved_bbox = bbox or _DEFAULT_BBOX
    resolution = grid_resolution or _grid_resolution()
    now = _utc_now()

    warnings = _fetch_thunderstorm_warnings(now)
    LOGGER.info(
        "Lightning proxy: fetched %d active thunderstorm warning(s) from MeteoAlarm",
        len(warnings),
    )

    active_cells, total_cells = _materialise_proxy(
        bbox=resolved_bbox,
        warnings=warnings,
        valid_time=now,
        grid_resolution=resolution,
    )

    result = {
        "valid_time": now.isoformat(),
        "warning_count": len(warnings),
        "total_cells": total_cells,
        "active_cells": active_cells,
        "grid_resolution_deg": resolution,
        "bbox": list(resolved_bbox),
    }
    LOGGER.info(
        "Lightning proxy materialised: active_cells=%d / total_cells=%d warnings=%d valid_time=%s",
        active_cells,
        total_cells,
        len(warnings),
        now.isoformat(),
    )
    return result


def run_lightning_proxy_ingest() -> int:
    """Orchestrator-compatible entry point. Returns 0 on success, 1 on failure."""
    try:
        ingest_lightning_proxy()
        return 0
    except Exception as exc:
        LOGGER.error("Lightning proxy ingest failed: %s", exc)
        return 1
