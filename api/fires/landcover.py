"""Land-cover plausibility scoring for fire detections."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import rasterio
from rasterio.transform import rowcol

from ingest.config import REPO_ROOT

LOGGER = logging.getLogger(__name__)


# Land-cover scoring rules based on fire plausibility
# Values follow ESA WorldCover or similar classification schemes
LANDCOVER_SCORES = {
    # High fire risk - vegetation
    10: 1.0,  # Tree cover / forest
    20: 1.0,  # Shrubland
    30: 1.0,  # Grassland
    # Moderate fire risk - agricultural
    40: 0.7,  # Cropland
    # Low fire risk - non-vegetated
    50: 0.1,  # Built-up / urban
    60: 0.1,  # Bare / sparse vegetation / desert
    70: 0.1,  # Snow and ice
    80: 0.1,  # Permanent water bodies
    90: 0.1,  # Herbaceous wetland
    95: 0.1,  # Mangroves
    100: 0.1,  # Moss and lichen
}


def get_landcover_path() -> Path | None:
    """Get path to land-cover raster if it exists.
    
    Returns:
        Path to landcover.tif if it exists, otherwise None
    """
    landcover_path = REPO_ROOT / "data" / "landcover.tif"
    return landcover_path if landcover_path.exists() else None


def compute_landcover_scores(
    detections: Iterable[dict],
    *,
    landcover_path: Path | None = None,
) -> dict[int, float]:
    """Compute land-cover plausibility scores for fire detections.
    
    Queries land-cover type at each detection location and applies scoring rules
    based on fire plausibility for different land-cover classes.
    
    Scoring rules (ESA WorldCover classification):
    - Forest/shrub/grassland (10, 20, 30): 1.0 (high fire plausibility)
    - Cropland (40): 0.7 (moderate plausibility)
    - Urban/water/desert/ice (50, 60, 70, 80+): 0.1 (low plausibility)
    
    Args:
        detections: Iterable of detection dicts with keys: id, lat, lon
        landcover_path: Optional path to landcover raster. If None, searches for
            data/landcover.tif. If not found, returns default score of 0.5 for all.
    
    Returns:
        Dict mapping detection_id → landcover_score in range [0, 1]
    
    Notes:
        - Expects landcover raster in EPSG:4326 (WGS84) coordinate system
        - Uses nearest-neighbor sampling for land-cover type lookup
        - Falls back to 0.5 (neutral) if no landcover data available
        - Unknown land-cover classes default to 0.5
    
    Example:
        >>> detections = [
        ...     {"id": 1, "lat": 42.5, "lon": 23.0},  # forest area
        ...     {"id": 2, "lat": 40.0, "lon": -74.0},  # urban area
        ... ]
        >>> scores = compute_landcover_scores(detections)
        >>> print(scores[1])  # forest → 1.0
        >>> print(scores[2])  # urban → 0.1
    """
    detection_list = list(detections)
    if not detection_list:
        return {}
    
    # Resolve landcover path
    if landcover_path is None:
        landcover_path = get_landcover_path()
    
    # If no landcover data available, return neutral scores
    if landcover_path is None:
        LOGGER.warning(
            "Landcover data file not found at %s. "
            "Land-cover scoring is disabled; returning neutral scores (0.5). "
            "To enable land-cover scoring, download ESA WorldCover or similar "
            "land-cover classification data and place it at data/landcover.tif.",
            REPO_ROOT / "data" / "landcover.tif"
        )
        return {d["id"]: 0.5 for d in detection_list}
    
    # Try to open the raster; if it fails, return neutral scores
    try:
        src_file = rasterio.open(landcover_path)
    except (FileNotFoundError, rasterio.errors.RasterioIOError):
        return {d["id"]: 0.5 for d in detection_list}
    
    scores: dict[int, float] = {}
    
    with src_file as src:
        for detection in detection_list:
            det_id = detection["id"]
            lat = detection["lat"]
            lon = detection["lon"]
            
            # Convert lat/lon to raster row/col
            row, col = rowcol(src.transform, lon, lat)
            
            # Check bounds
            if 0 <= row < src.height and 0 <= col < src.width:
                # Read land-cover class at this location
                landcover_class = int(src.read(1, window=((row, row+1), (col, col+1)))[0, 0])
                
                # Map to score (default 0.5 for unknown classes)
                score = LANDCOVER_SCORES.get(landcover_class, 0.5)
            else:
                # Out of bounds - neutral score
                score = 0.5
            
            scores[det_id] = score
    
    return scores
