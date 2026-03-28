"""DB queries for AOIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB

from api.db import get_engine

# Columns returned by every AOI SELECT (geometry computed via PostGIS functions).
_AOI_SELECT = """
    id,
    name,
    description,
    tags,
    owner_id,
    ST_AsGeoJSON(geom)::jsonb as geometry,
    ST_AsGeoJSON(bbox)::jsonb as bbox,
    area_km2,
    vertex_count,
    created_at,
    updated_at,
    watch_enabled,
    watch_interval_minutes,
    watch_alert_threshold,
    watch_last_checked_at,
    watch_last_alerted_at,
    watch_last_spread_prob
"""


def create_aoi(
    name: str,
    geom_geojson: dict[str, Any],
    description: Optional[str] = None,
    tags: Optional[dict[str, Any]] = None,
    owner_id: Optional[str] = None,
) -> dict[str, Any]:
    """Insert a new AOI and return the full record."""
    stmt = text(
        f"""
        WITH input_geom AS (
            SELECT ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(:geom_geojson), 4326)) AS geom
        ),
        processed_geom AS (
            SELECT
                CASE
                    WHEN ST_IsValid(geom) THEN geom
                    ELSE ST_MakeValid(geom)
                END AS geom
            FROM input_geom
        )
        INSERT INTO aois (
            name,
            description,
            tags,
            owner_id,
            geom,
            bbox,
            area_km2,
            vertex_count
        )
        SELECT
            :name,
            :description,
            :tags,
            :owner_id,
            geom,
            ST_Envelope(geom),
            ST_Area(geom::geography) / 1000000.0,
            ST_NPoints(geom)
        FROM processed_geom
        RETURNING
            {_AOI_SELECT}
        """
    ).bindparams(
        bindparam("tags", type_=JSONB),
        bindparam("geom_geojson", type_=JSONB)
    )

    with get_engine().begin() as conn:
        row = conn.execute(
            stmt,
            {
                "name": name,
                "description": description,
                "tags": tags,
                "owner_id": owner_id,
                "geom_geojson": geom_geojson,
            },
        ).mappings().one()

    return dict(row)


def get_aoi(aoi_id: UUID) -> Optional[dict[str, Any]]:
    """Fetch an AOI by ID."""
    stmt = text(
        f"""
        SELECT {_AOI_SELECT}
        FROM aois
        WHERE id = :aoi_id
        """
    )

    with get_engine().begin() as conn:
        row = conn.execute(stmt, {"aoi_id": aoi_id}).mappings().first()

    return dict(row) if row else None


def list_aois(
    limit: int = 50,
    offset: int = 0,
    bbox: Optional[tuple[float, float, float, float]] = None,
    name_search: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List AOIs with optional filtering."""

    where_clauses = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        where_clauses.append("bbox && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)")
        params.update({
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        })

    if name_search:
        where_clauses.append("name ILIKE :name_search")
        params["name_search"] = f"%{name_search}%"

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    stmt = text(
        f"""
        SELECT {_AOI_SELECT}
        FROM aois
        {where_sql}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
        """
    )

    with get_engine().begin() as conn:
        rows = conn.execute(stmt, params).mappings().all()

    return [dict(r) for r in rows]


def update_aoi(
    aoi_id: UUID,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[dict[str, Any]] = None,
    geom_geojson: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Update an AOI."""

    updates = ["updated_at = now()"]
    params: dict[str, Any] = {"aoi_id": aoi_id}

    if name is not None:
        updates.append("name = :name")
        params["name"] = name

    if description is not None:
        updates.append("description = :description")
        params["description"] = description

    if tags is not None:
        updates.append("tags = :tags")
        params["tags"] = tags

    if geom_geojson is not None:
        updates.append("geom = (SELECT geom FROM new_geom_cte)")
        updates.append("bbox = (SELECT ST_Envelope(geom) FROM new_geom_cte)")
        updates.append("area_km2 = (SELECT ST_Area(geom::geography) / 1000000.0 FROM new_geom_cte)")
        updates.append("vertex_count = (SELECT ST_NPoints(geom) FROM new_geom_cte)")
        params["geom_geojson"] = geom_geojson

    if len(updates) == 1:  # Only updated_at
        return get_aoi(aoi_id)

    # Build the query with optional CTE for geometry
    cte_part = ""
    if geom_geojson is not None:
        cte_part = """
        WITH new_geom_cte AS (
            SELECT
                CASE
                    WHEN ST_IsValid(g.geom) THEN g.geom
                    ELSE ST_MakeValid(g.geom)
                END as geom
            FROM (
                SELECT ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(:geom_geojson), 4326)) AS geom
            ) g
        ),
        updated AS (
        """
    else:
        cte_part = "WITH updated AS ("

    stmt_obj = text(
        f"""
        {cte_part}
            UPDATE aois
            SET {', '.join(updates)}
            WHERE id = :aoi_id
            RETURNING *
        )
        SELECT {_AOI_SELECT}
        FROM updated
        """
    )

    binds = []
    if tags is not None:
        binds.append(bindparam("tags", type_=JSONB))
    if geom_geojson is not None:
        binds.append(bindparam("geom_geojson", type_=JSONB))

    if binds:
        stmt_obj = stmt_obj.bindparams(*binds)

    with get_engine().begin() as conn:
        row = conn.execute(stmt_obj, params).mappings().first()

    return dict(row) if row else None


def delete_aoi(aoi_id: UUID) -> bool:
    """Delete an AOI."""
    stmt = text("DELETE FROM aois WHERE id = :aoi_id")
    with get_engine().begin() as conn:
        result = conn.execute(stmt, {"aoi_id": aoi_id})
        return result.rowcount > 0


def set_aoi_watch(
    aoi_id: UUID,
    enabled: bool,
    interval_minutes: Optional[int],
    alert_threshold: Optional[float],
) -> Optional[dict[str, Any]]:
    """Configure watchlist settings for an AOI."""
    stmt = text(
        f"""
        UPDATE aois
        SET
            watch_enabled = :enabled,
            watch_interval_minutes = :interval_minutes,
            watch_alert_threshold = :alert_threshold,
            updated_at = now()
        WHERE id = :aoi_id
        RETURNING {_AOI_SELECT}
        """
    )

    with get_engine().begin() as conn:
        row = conn.execute(
            stmt,
            {
                "aoi_id": aoi_id,
                "enabled": enabled,
                "interval_minutes": interval_minutes,
                "alert_threshold": alert_threshold,
            },
        ).mappings().first()

    return dict(row) if row else None


def list_watched_aois() -> list[dict[str, Any]]:
    """Return all AOIs with watch_enabled=True."""
    stmt = text(
        f"""
        SELECT {_AOI_SELECT}
        FROM aois
        WHERE watch_enabled = true
        ORDER BY name
        """
    )

    with get_engine().begin() as conn:
        rows = conn.execute(stmt).mappings().all()

    return [dict(r) for r in rows]


def list_watched_aois_due(now: datetime) -> list[dict[str, Any]]:
    """Return watched AOIs that are due for a forecast check.

    An AOI is due when:
      - It has never been checked (watch_last_checked_at IS NULL), OR
      - The time since last check >= watch_interval_minutes
    """
    stmt = text(
        f"""
        SELECT {_AOI_SELECT}
        FROM aois
        WHERE watch_enabled = true
          AND watch_interval_minutes IS NOT NULL
          AND watch_alert_threshold IS NOT NULL
          AND (
            watch_last_checked_at IS NULL
            OR watch_last_checked_at + (watch_interval_minutes * INTERVAL '1 minute') <= :now
          )
        ORDER BY COALESCE(watch_last_checked_at, '1970-01-01'::timestamptz) ASC
        """
    )

    with get_engine().begin() as conn:
        rows = conn.execute(stmt, {"now": now}).mappings().all()

    return [dict(r) for r in rows]


def update_aoi_watch_status(
    aoi_id: UUID,
    last_checked_at: datetime,
    last_spread_prob: Optional[float],
    last_alerted_at: Optional[datetime] = None,
) -> None:
    """Update watch status after a forecast check."""
    updates = [
        "watch_last_checked_at = :last_checked_at",
        "watch_last_spread_prob = :last_spread_prob",
    ]
    params: dict[str, Any] = {
        "aoi_id": aoi_id,
        "last_checked_at": last_checked_at,
        "last_spread_prob": last_spread_prob,
    }

    if last_alerted_at is not None:
        updates.append("watch_last_alerted_at = :last_alerted_at")
        params["last_alerted_at"] = last_alerted_at

    stmt = text(f"UPDATE aois SET {', '.join(updates)} WHERE id = :aoi_id")

    with get_engine().begin() as conn:
        conn.execute(stmt, params)
