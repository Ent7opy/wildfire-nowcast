"""Industrial source coverage queries for health monitoring and decision support.

Used by the internal health endpoint and spread service to detect forecast
regions with no industrial noise-filter coverage (blind spots).
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from api.db import get_engine

LOGGER = logging.getLogger(__name__)

# Fallback buffer (metres) when no active masking policy is found.
DEFAULT_BUFFER_M = 1000.0


def query_industrial_coverage(
    bbox: tuple[float, float, float, float],
    engine=None,
) -> dict:
    """Return industrial source coverage report for the given bounding box.

    Queries active industrial sources that intersect the bbox, then computes
    the fraction of bbox area covered by buffered source zones using the
    active masking policy's gold_buffer_m (or DEFAULT_BUFFER_M if no policy).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    engine : optional
        SQLAlchemy engine. Defaults to ``api.db.get_engine()``.

    Returns
    -------
    dict
        source_count : int
        types : list[str]  — distinct source type values present in bbox
        coverage_fraction : float  — 0.0–1.0, area fraction covered by buffers
        buffer_m : float  — buffer radius used for the computation
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    db_engine = engine or get_engine()

    with db_engine.begin() as conn:
        # Single CTE: resolve active policy buffer and compute coverage in one round-trip.
        row = conn.execute(
            text("""
                WITH active_policy AS (
                    SELECT COALESCE(gold_buffer_m, :default_buffer) AS buffer_m
                    FROM industrial_mask_policies
                    WHERE (active_to IS NULL OR active_to > NOW())
                    ORDER BY active_from DESC
                    LIMIT 1
                ),
                effective_policy AS (
                    SELECT COALESCE((SELECT buffer_m FROM active_policy), :default_buffer) AS buffer_m
                ),
                bbox AS (
                    SELECT ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326) AS geom
                ),
                srcs AS (
                    SELECT geom, type
                    FROM industrial_sources
                    WHERE is_active = true
                      AND ST_Intersects(geom, (SELECT geom FROM bbox))
                )
                SELECT
                    COUNT(*)::int AS source_count,
                    array_agg(DISTINCT type ORDER BY type)
                        FILTER (WHERE type IS NOT NULL) AS types,
                    CASE WHEN COUNT(*) > 0 THEN
                        ST_Area(
                            ST_Intersection(
                                ST_Union(ST_Buffer(geom::geography, (SELECT buffer_m FROM effective_policy))::geometry),
                                (SELECT geom FROM bbox)
                            )::geography
                        ) / NULLIF(ST_Area((SELECT geom FROM bbox)::geography), 0)
                    ELSE 0.0 END AS coverage_fraction,
                    (SELECT buffer_m FROM effective_policy) AS buffer_m
                FROM srcs
            """),
            {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "default_buffer": DEFAULT_BUFFER_M,
            },
        ).fetchone()

    return {
        "source_count": int(row["source_count"]),
        "types": sorted(row["types"] or []),
        "coverage_fraction": float(row["coverage_fraction"] or 0.0),
        "buffer_m": float(row["buffer_m"]),
    }
