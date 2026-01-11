"""FastAPI routes for AOI management."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from api.aois import repo

aois_router = APIRouter(prefix="/aois", tags=["aois"])

# Limits
MAX_AOI_AREA_KM2 = 50000.0
MAX_AOI_VERTICES = 10000


class CreateAOIRequest(BaseModel):
    name: str
    geometry: dict[str, Any]  # GeoJSON
    description: Optional[str] = None
    tags: Optional[dict[str, Any]] = None
    owner_id: Optional[str] = None


class UpdateAOIRequest(BaseModel):
    name: Optional[str] = None
    geometry: Optional[dict[str, Any]] = None
    description: Optional[str] = None
    tags: Optional[dict[str, Any]] = None


class AOIResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    tags: Optional[dict[str, Any]]
    owner_id: Optional[str]
    geometry: dict[str, Any]
    bbox: dict[str, Any]
    area_km2: float
    vertex_count: int
    created_at: Any
    updated_at: Any


class AOIListResponse(BaseModel):
    items: list[AOIResponse]
    count: int


def _validate_geometry(geojson: dict[str, Any]) -> None:
    """Validate geometry constraints."""
    try:
        geom: BaseGeometry = shape(geojson)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid GeoJSON: {str(e)}",
        )

    # AOIs must be Polygons or MultiPolygons
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geometry must be a Polygon or MultiPolygon, not {geom.geom_type}",
        )

    if not geom.is_valid:
        # We could try to fix it, but let's encourage valid input
        pass

    if geom.is_empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Geometry is empty",
        )

    # Vertex count check
    if geom.geom_type == "MultiPolygon":
        # MultiPolygon has .geoms, each is a Polygon
        vertex_count = sum(len(g.exterior.coords) + sum(len(i.coords) for i in g.interiors) for g in geom.geoms)
    else:
        # Polygon has .exterior and .interiors
        vertex_count = len(geom.exterior.coords) + sum(len(i.coords) for i in geom.interiors)

    if vertex_count > MAX_AOI_VERTICES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Geometry too complex ({vertex_count} vertices, max {MAX_AOI_VERTICES})",
        )


@aois_router.post("", response_model=AOIResponse, status_code=status.HTTP_201_CREATED)
def create_aoi(request: CreateAOIRequest):
    """Create a new Area of Interest."""
    _validate_geometry(request.geometry)
    
    try:
        aoi = repo.create_aoi(
            name=request.name,
            geom_geojson=request.geometry,
            description=request.description,
            tags=request.tags,
            owner_id=request.owner_id,
        )
    except Exception as e:
        # In a real app, distinguish DB errors (integrity etc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # Post-creation validation (using DB-computed area)
    if aoi["area_km2"] > MAX_AOI_AREA_KM2:
        # Rollback? repo.create_aoi committed.
        # We should delete it or ideally checked before commit.
        # Since repo.create_aoi does one transaction, we can't rollback easily without context manager.
        # For MVP, we'll just delete it and error out, or just warn.
        # Let's delete it.
        repo.delete_aoi(aoi["id"])
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"AOI area ({aoi['area_km2']:.1f} km²) exceeds maximum ({MAX_AOI_AREA_KM2} km²)",
        )

    return aoi


@aois_router.get("", response_model=AOIListResponse)
def list_aois(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    min_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lon: Optional[float] = None,
    max_lat: Optional[float] = None,
    q: Optional[str] = Query(None, description="Name search"),
):
    """List AOIs."""
    bbox = None
    if all(x is not None for x in [min_lon, min_lat, max_lon, max_lat]):
        bbox = (min_lon, min_lat, max_lon, max_lat)
    
    items = repo.list_aois(limit=limit, offset=offset, bbox=bbox, name_search=q)
    return {"items": items, "count": len(items)}


@aois_router.get("/{aoi_id}", response_model=AOIResponse)
def get_aoi(aoi_id: UUID):
    """Get an AOI by ID."""
    aoi = repo.get_aoi(aoi_id)
    if not aoi:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found")
    return aoi


@aois_router.patch("/{aoi_id}", response_model=AOIResponse)
def update_aoi(aoi_id: UUID, request: UpdateAOIRequest):
    """Update an AOI."""
    old_aoi = repo.get_aoi(aoi_id)
    if not old_aoi:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found")

    if request.geometry:
        _validate_geometry(request.geometry)

    aoi = repo.update_aoi(
        aoi_id,
        name=request.name,
        description=request.description,
        tags=request.tags,
        geom_geojson=request.geometry,
    )
    if not aoi:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found")
    
    if request.geometry and aoi["area_km2"] > MAX_AOI_AREA_KM2:
        # Revert update
        repo.update_aoi(
            aoi_id,
            geom_geojson=old_aoi["geometry"],
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"New geometry area ({aoi['area_km2']:.1f} km²) exceeds maximum ({MAX_AOI_AREA_KM2} km²)",
        )

    return aoi


@aois_router.delete("/{aoi_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_aoi(aoi_id: UUID):
    """Delete an AOI."""
    if not repo.delete_aoi(aoi_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found")
