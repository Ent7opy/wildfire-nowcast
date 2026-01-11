"""DB queries for AOIs."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB

from api.db import get_engine


def create_aoi(
    name: str,
    geom_geojson: dict[str, Any],
    description: Optional[str] = None,
    tags: Optional[dict[str, Any]] = None,
    owner_id: Optional[str] = None,
) -> dict[str, Any]:
    """Insert a new AOI and return the full record."""
    stmt = text(
        """
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
            updated_at
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
        """
        SELECT
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
            updated_at
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
    params = {"limit": limit, "offset": offset}

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
        SELECT
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
            updated_at
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
    params = {"aoi_id": aoi_id}
    
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

    if len(updates) == 1: # Only updated_at
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
        SELECT
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
            updated_at
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
