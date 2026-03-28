"""FastAPI routes for AOI management."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import no_cache
from pydantic import BaseModel, Field
from typing import Annotated
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from api.aois import repo

aois_router = APIRouter(prefix="/aois", tags=["aois"])

# Limits
MAX_AOI_AREA_KM2 = 50000.0
MAX_AOI_VERTICES = 10000
MIN_WATCH_INTERVAL_MINUTES = 5
MAX_WATCH_INTERVAL_MINUTES = 10080  # 1 week


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


class WatchConfigRequest(BaseModel):
    enabled: bool
    interval_minutes: Annotated[
        Optional[int],
        Field(None, ge=MIN_WATCH_INTERVAL_MINUTES, le=MAX_WATCH_INTERVAL_MINUTES),
    ] = None
    alert_threshold: Annotated[
        Optional[float],
        Field(None, gt=0.0, le=1.0),
    ] = None


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
    # Watch fields
    watch_enabled: bool = False
    watch_interval_minutes: Optional[int] = None
    watch_alert_threshold: Optional[float] = None
    watch_last_checked_at: Optional[Any] = None
    watch_last_alerted_at: Optional[Any] = None
    watch_last_spread_prob: Optional[float] = None


class WatchlistSummaryItem(BaseModel):
    id: UUID
    name: str
    watch_enabled: bool
    watch_interval_minutes: Optional[int]
    watch_alert_threshold: Optional[float]
    watch_last_checked_at: Optional[Any]
    watch_last_alerted_at: Optional[Any]
    watch_last_spread_prob: Optional[float]
    alert_active: bool  # True when last_spread_prob >= threshold


class WatchlistResponse(BaseModel):
    items: list[WatchlistSummaryItem]
    count: int


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


def _to_watchlist_item(aoi: dict[str, Any]) -> WatchlistSummaryItem:
    threshold = aoi.get("watch_alert_threshold")
    last_prob = aoi.get("watch_last_spread_prob")
    alert_active = (
        threshold is not None
        and last_prob is not None
        and last_prob >= threshold
    )
    return WatchlistSummaryItem(
        id=aoi["id"],
        name=aoi["name"],
        watch_enabled=aoi.get("watch_enabled", False),
        watch_interval_minutes=aoi.get("watch_interval_minutes"),
        watch_alert_threshold=threshold,
        watch_last_checked_at=aoi.get("watch_last_checked_at"),
        watch_last_alerted_at=aoi.get("watch_last_alerted_at"),
        watch_last_spread_prob=last_prob,
        alert_active=alert_active,
    )


@aois_router.post("", response_model=AOIResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(no_cache)])
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


@aois_router.get("/watchlist", response_model=WatchlistResponse)
def get_watchlist():
    """Return all watched AOIs with their latest forecast status."""
    aois = repo.list_watched_aois()
    items = [_to_watchlist_item(a) for a in aois]
    return {"items": items, "count": len(items)}


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


@aois_router.put("/{aoi_id}/watch", response_model=AOIResponse, dependencies=[Depends(no_cache)])
def configure_watch(aoi_id: UUID, request: WatchConfigRequest):
    """Configure watchlist settings for an AOI.

    When enabled=True, interval_minutes and alert_threshold are required.
    When enabled=False, the AOI is removed from the watchlist (other fields ignored).
    """
    if request.enabled:
        if request.interval_minutes is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="interval_minutes is required when enabling watch",
            )
        if request.alert_threshold is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="alert_threshold is required when enabling watch",
            )

    updated = repo.set_aoi_watch(
        aoi_id=aoi_id,
        enabled=request.enabled,
        interval_minutes=request.interval_minutes if request.enabled else None,
        alert_threshold=request.alert_threshold if request.enabled else None,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found")

    return updated


@aois_router.delete("/{aoi_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_aoi(aoi_id: UUID):
    """Delete an AOI."""
    if not repo.delete_aoi(aoi_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found")
