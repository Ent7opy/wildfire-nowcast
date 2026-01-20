"""Server-side map rendering to PNG."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def _lon_lat_to_pixel(lon: float, lat: float, bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int]:
    """Convert lon/lat to pixel coordinates in Web Mercator projection.
    
    Args:
        lon: Longitude
        lat: Latitude
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        (x, y) pixel coordinates
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Simple linear mapping for MVP (good enough for small areas)
    # For more accuracy, use proper Web Mercator math
    x = int((lon - min_lon) / (max_lon - min_lon) * width)
    y = int((max_lat - lat) / (max_lat - min_lat) * height)  # Flip Y
    
    return (x, y)


def _risk_score_to_color(risk_score: float) -> tuple[int, int, int, int]:
    """Convert risk score to RGBA color.
    
    Args:
        risk_score: Risk score in range [0, 1]
        
    Returns:
        RGBA color tuple
    """
    if risk_score < 0.3:
        # Low risk: green
        return (34, 139, 34, 80)
    elif risk_score < 0.6:
        # Medium risk: yellow
        return (255, 215, 0, 100)
    else:
        # High risk: red
        return (220, 20, 60, 120)


def render_map_png(
    bbox: tuple[float, float, float, float],
    *,
    fires: list[dict] = None,
    risk_grid: dict = None,
    forecast_contours: list[dict] = None,
    width: int = 1600,
    height: int = 900,
    title: Optional[str] = None,
) -> bytes:
    """Render map layers to PNG image.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        fires: List of fire detection dicts with lat, lon, frp keys
        risk_grid: Risk grid GeoJSON FeatureCollection
        forecast_contours: List of forecast contour dicts
        width: Image width in pixels
        height: Image height in pixels
        title: Optional title to display on map
        
    Returns:
        PNG image bytes
        
    Raises:
        ImportError: If PIL/Pillow is not available
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required for PNG export. Install with: pip install pillow")
    
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Create blank image with light background
    img = Image.new("RGBA", (width, height), (245, 245, 245, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    
    # Try to load a font, fall back to default if unavailable
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_legend = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font_title = ImageFont.load_default()
        font_legend = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw grid for reference (lat/lon lines)
    _draw_graticule(draw, bbox, width, height, font_small)
    
    # Layer 1: Risk grid (if provided)
    if risk_grid and "features" in risk_grid:
        _draw_risk_grid(draw, risk_grid, bbox, width, height)
    
    # Layer 2: Forecast contours (if provided)
    if forecast_contours:
        _draw_forecast_contours(draw, forecast_contours, bbox, width, height)
    
    # Layer 3: Fires (if provided)
    if fires:
        _draw_fires(draw, fires, bbox, width, height)
    
    # Draw legend
    _draw_legend(draw, width, height, fires is not None, risk_grid is not None, forecast_contours is not None, font_legend)
    
    # Draw title
    if title:
        draw.text((10, 10), title, fill=(0, 0, 0, 255), font=font_title)
    
    # Draw timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    draw.text((10, height - 25), f"Generated: {timestamp}", fill=(80, 80, 80, 255), font=font_small)
    
    # Convert to PNG bytes
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_graticule(draw: ImageDraw.ImageDraw, bbox: tuple, width: int, height: int, font):
    """Draw lat/lon grid lines."""
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Determine grid spacing
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    # Use ~5 grid lines
    lon_step = max(0.1, round(lon_range / 5, 1))
    lat_step = max(0.1, round(lat_range / 5, 1))
    
    # Draw vertical lines (longitude)
    lon = math.ceil(min_lon / lon_step) * lon_step
    while lon <= max_lon:
        x = int((lon - min_lon) / lon_range * width)
        draw.line([(x, 0), (x, height)], fill=(200, 200, 200, 100), width=1)
        draw.text((x + 2, height - 15), f"{lon:.1f}°", fill=(120, 120, 120, 200), font=font)
        lon += lon_step
    
    # Draw horizontal lines (latitude)
    lat = math.ceil(min_lat / lat_step) * lat_step
    while lat <= max_lat:
        y = int((max_lat - lat) / lat_range * height)
        draw.line([(0, y), (width, y)], fill=(200, 200, 200, 100), width=1)
        draw.text((5, y + 2), f"{lat:.1f}°", fill=(120, 120, 120, 200), font=font)
        lat += lat_step


def _draw_risk_grid(draw: ImageDraw.ImageDraw, risk_grid: dict, bbox: tuple, width: int, height: int):
    """Draw risk grid cells."""
    for feature in risk_grid["features"]:
        risk_score = feature["properties"]["risk_score"]
        coords = feature["geometry"]["coordinates"][0]
        
        # Convert polygon coordinates to pixels
        pixels = [_lon_lat_to_pixel(lon, lat, bbox, width, height) for lon, lat in coords[:-1]]
        
        # Draw filled polygon
        color = _risk_score_to_color(risk_score)
        draw.polygon(pixels, fill=color, outline=color[:3] + (180,))


def _draw_forecast_contours(draw: ImageDraw.ImageDraw, contours: list, bbox: tuple, width: int, height: int):
    """Draw forecast contour outlines."""
    # Simplified: just draw bounding polygons with different colors per horizon
    horizon_colors = {
        24: (0, 100, 255, 100),  # Blue for T+24h
        48: (255, 165, 0, 100),  # Orange for T+48h
        72: (255, 50, 50, 100),  # Red for T+72h
    }
    
    for contour in contours:
        horizon = contour.get("horizon_hours", 24)
        color = horizon_colors.get(horizon, (128, 128, 128, 100))
        
        # Note: Real implementation would parse GeoJSON geometry
        # For MVP, skip complex polygon rendering
        pass


def _draw_fires(draw: ImageDraw.ImageDraw, fires: list, bbox: tuple, width: int, height: int):
    """Draw fire points."""
    for fire in fires:
        lat = fire.get("lat")
        lon = fire.get("lon")
        if lat is None or lon is None:
            continue
        
        x, y = _lon_lat_to_pixel(lon, lat, bbox, width, height)
        
        # Draw fire as a red circle
        radius = 4
        frp = fire.get("frp", 0)
        if frp and frp > 100:
            radius = 6  # Larger circles for high intensity fires
        
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=(255, 0, 0, 200),
            outline=(139, 0, 0, 255),
        )


def _draw_legend(draw: ImageDraw.ImageDraw, width: int, height: int, has_fires: bool, has_risk: bool, has_forecast: bool, font):
    """Draw map legend."""
    legend_x = width - 220
    legend_y = 50
    legend_width = 210
    legend_height = 10  # Will expand based on content
    
    items = []
    
    if has_fires:
        items.append(("🔴 Active fires", (255, 0, 0, 200)))
    
    if has_risk:
        items.append(("Risk: Low (green)", (34, 139, 34, 80)))
        items.append(("Risk: Medium (yellow)", (255, 215, 0, 100)))
        items.append(("Risk: High (red)", (220, 20, 60, 120)))
    
    if has_forecast:
        items.append(("Forecast: T+24h", (0, 100, 255, 100)))
        items.append(("Forecast: T+48h", (255, 165, 0, 100)))
        items.append(("Forecast: T+72h", (255, 50, 50, 100)))
    
    if not items:
        return
    
    legend_height = 15 + len(items) * 22 + 10
    
    # Draw legend background
    draw.rectangle(
        [(legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height)],
        fill=(255, 255, 255, 230),
        outline=(100, 100, 100, 255),
    )
    
    # Draw legend title
    draw.text((legend_x + 5, legend_y + 5), "Legend", fill=(0, 0, 0, 255), font=font)
    
    # Draw legend items
    y_offset = legend_y + 25
    for label, color in items:
        # Draw color swatch
        draw.rectangle(
            [(legend_x + 10, y_offset), (legend_x + 25, y_offset + 12)],
            fill=color,
            outline=(0, 0, 0, 180),
        )
        # Draw label
        draw.text((legend_x + 30, y_offset), label, fill=(0, 0, 0, 255), font=font)
        y_offset += 22
