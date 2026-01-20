"""Grid-based fire risk computation."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from api.fires.landcover import LANDCOVER_SCORES, get_landcover_path
from api.fires.scoring import _get_weather_data_for_point

try:
    import rasterio
    from rasterio.transform import rowcol
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def _classify_risk_level(risk_score: float) -> str:
    """Classify numeric risk score into categorical level.
    
    Args:
        risk_score: Numeric risk score in range [0, 1]
        
    Returns:
        Risk level category: "low", "medium", or "high"
    """
    if risk_score < 0.3:
        return "low"
    elif risk_score < 0.6:
        return "medium"
    else:
        return "high"


def compute_risk_grid(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    *,
    cell_size_km: float = 10.0,
    ref_time: Optional[datetime] = None,
    include_weather: bool = True,
) -> dict:
    """Compute fire risk grid for a bounding box.
    
    Returns a GeoJSON FeatureCollection with grid cells containing risk scores
    based on static (land cover) and optionally dynamic (weather) factors.
    
    Args:
        min_lon: Minimum longitude (west boundary)
        min_lat: Minimum latitude (south boundary)
        max_lon: Maximum longitude (east boundary)
        max_lat: Maximum latitude (north boundary)
        cell_size_km: Grid cell size in kilometers (default 10km)
        ref_time: Reference time for weather data (default: now)
        include_weather: Whether to include weather scoring (default: True)
        
    Returns:
        GeoJSON FeatureCollection with grid cells and risk scores
        
    Notes:
        - Static factor: landcover-based flammability from LANDCOVER_SCORES
        - Dynamic factor (if include_weather=True): weather plausibility
        - Composite score: 0.6 * static + 0.4 * dynamic (if available)
        - Falls back to static-only if weather unavailable
    """
    if ref_time is None:
        ref_time = datetime.now(timezone.utc)
    
    # Approximate degrees per km at the center latitude
    # (rough conversion for grid sizing; good enough for MVP)
    center_lat = (min_lat + max_lat) / 2.0
    lat_per_km = 1.0 / 111.0  # ~111 km per degree latitude
    lon_per_km = 1.0 / (111.0 * math.cos(math.radians(center_lat)))
    
    cell_size_lat = cell_size_km * lat_per_km
    cell_size_lon = cell_size_km * lon_per_km
    
    # Limit grid size to prevent huge responses
    max_cells = 500
    n_lat = max(1, min(max_cells, int((max_lat - min_lat) / cell_size_lat)))
    n_lon = max(1, min(max_cells, int((max_lon - min_lon) / cell_size_lon)))
    
    if n_lat * n_lon > max_cells:
        # Adjust to keep total cells under limit
        scale_factor = math.sqrt(max_cells / (n_lat * n_lon))
        n_lat = max(1, int(n_lat * scale_factor))
        n_lon = max(1, int(n_lon * scale_factor))
    
    # Recompute actual cell sizes
    cell_size_lat = (max_lat - min_lat) / n_lat
    cell_size_lon = (max_lon - min_lon) / n_lon
    
    LOGGER.info(
        "Computing risk grid: bbox=(%.2f,%.2f,%.2f,%.2f), grid=%dx%d cells, cell_size=%.3f°x%.3f°",
        min_lon, min_lat, max_lon, max_lat, n_lon, n_lat, cell_size_lon, cell_size_lat
    )
    
    # Try to load landcover raster
    landcover_path = get_landcover_path()
    landcover_src = None
    if RASTERIO_AVAILABLE and landcover_path is not None:
        try:
            landcover_src = rasterio.open(landcover_path)
        except Exception as e:
            LOGGER.warning("Failed to open landcover raster: %s", e)
    
    features = []
    
    try:
        for i_lat in range(n_lat):
            for i_lon in range(n_lon):
                # Cell bounds
                cell_min_lon = min_lon + i_lon * cell_size_lon
                cell_max_lon = min_lon + (i_lon + 1) * cell_size_lon
                cell_min_lat = min_lat + i_lat * cell_size_lat
                cell_max_lat = min_lat + (i_lat + 1) * cell_size_lat
                
                # Cell center for sampling
                cell_center_lon = (cell_min_lon + cell_max_lon) / 2.0
                cell_center_lat = (cell_min_lat + cell_max_lat) / 2.0
                
                # Compute static score (landcover-based)
                static_score = _compute_static_risk(
                    cell_center_lat, cell_center_lon, landcover_src
                )
                
                # Compute dynamic score (weather-based) if requested
                dynamic_score = None
                if include_weather:
                    dynamic_score = _compute_dynamic_risk(
                        cell_center_lat, cell_center_lon, ref_time
                    )
                
                # Composite score: weighted combination
                if dynamic_score is not None:
                    # Both static and dynamic available
                    risk_score = 0.6 * static_score + 0.4 * dynamic_score
                    components = {
                        "static": round(static_score, 3),
                        "dynamic": round(dynamic_score, 3),
                    }
                else:
                    # Static only
                    risk_score = static_score
                    components = {
                        "static": round(static_score, 3),
                    }
                
                risk_level = _classify_risk_level(risk_score)
                
                # Create cell feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [cell_min_lon, cell_min_lat],
                            [cell_max_lon, cell_min_lat],
                            [cell_max_lon, cell_max_lat],
                            [cell_min_lon, cell_max_lat],
                            [cell_min_lon, cell_min_lat],
                        ]]
                    },
                    "properties": {
                        "risk_score": round(risk_score, 3),
                        "risk_level": risk_level,
                        "components": components,
                    }
                }
                
                features.append(feature)
    
    finally:
        if landcover_src is not None:
            landcover_src.close()
    
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _compute_static_risk(lat: float, lon: float, landcover_src) -> float:
    """Compute static risk score based on land cover.
    
    Args:
        lat: Latitude of the point
        lon: Longitude of the point
        landcover_src: Open rasterio dataset or None
        
    Returns:
        Static risk score in range [0, 1]
    """
    if landcover_src is None:
        # No landcover data: use neutral score
        return 0.5
    
    try:
        # Convert lat/lon to raster row/col
        row, col = rowcol(landcover_src.transform, lon, lat)
        
        # Check bounds
        if 0 <= row < landcover_src.height and 0 <= col < landcover_src.width:
            # Read land-cover class at this location
            landcover_class = int(landcover_src.read(1, window=((row, row+1), (col, col+1)))[0, 0])
            
            # Map to risk score (default 0.5 for unknown classes)
            # Use same scoring as fire likelihood landcover component
            return LANDCOVER_SCORES.get(landcover_class, 0.5)
        else:
            # Out of bounds
            return 0.5
    
    except Exception as e:
        LOGGER.debug("Failed to sample landcover at (%.3f, %.3f): %s", lat, lon, e)
        return 0.5


def _compute_dynamic_risk(lat: float, lon: float, ref_time: datetime) -> Optional[float]:
    """Compute dynamic risk score based on weather conditions.
    
    Args:
        lat: Latitude of the point
        lon: Longitude of the point
        ref_time: Reference time for weather data
        
    Returns:
        Dynamic risk score in range [0, 1] or None if weather unavailable
    """
    # Reuse weather plausibility scoring logic
    weather_data = _get_weather_data_for_point(
        lat=lat,
        lon=lon,
        ref_time=ref_time,
        time_tolerance_hours=6.0,
        precip_lookback_hours=72.0,
    )
    
    if weather_data is None:
        return None
    
    # Extract weather variables
    rh = weather_data.get("rh2m")
    precip_recent = weather_data.get("precip_recent_mm")
    wind_speed = weather_data.get("wind_speed_ms")
    
    # Base score: neutral
    score = 0.5
    
    # Apply penalties (wet conditions reduce fire risk)
    if rh is not None and rh > 70.0:
        score -= 0.3  # Very wet conditions suppress fires
    
    if precip_recent is not None and precip_recent > 10.0:
        score -= 0.2  # Recent heavy rain reduces fire risk
    
    # Apply bonuses (dry/windy conditions increase fire risk)
    if rh is not None and rh < 40.0:
        score += 0.2  # Dry conditions favor fires
    
    if wind_speed is not None and wind_speed > 3.0:
        score += 0.1  # Wind increases fire risk
    
    # Clamp to [0.1, 1.0] range
    return max(0.1, min(1.0, score))
