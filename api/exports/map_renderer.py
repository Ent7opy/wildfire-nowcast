"""Server-side map rendering to PNG with a UI-like dark visual style."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Optional

import httpx

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

_TILE_SIZE = 256
_MAX_BASEMAP_ZOOM = 10
_MAX_TILE_COUNT = 80
_BASEMAP_URL = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
_TILE_CACHE: dict[tuple[int, int, int], bytes] = {}

# UI-aligned palette
_BG_PRIMARY = (10, 22, 40, 255)
_BG_CARD = (37, 41, 48, 235)
_BORDER_SUBTLE = (255, 255, 255, 30)
_TEXT_PRIMARY = (224, 224, 224, 255)
_TEXT_MUTED = (180, 180, 180, 220)
_ACCENT_EMBER = (255, 107, 53, 255)

_FIRE_VERY_HIGH_FILL = (220, 38, 38, 240)
_FIRE_HIGH_FILL = (239, 68, 68, 232)
_FIRE_MEDIUM_FILL = (255, 107, 53, 224)
_FIRE_LOW_FILL = (251, 191, 36, 212)
_FIRE_VERY_LOW_FILL = (253, 224, 71, 196)
_FIRE_UNSCORED_FILL = (128, 128, 128, 170)
_FIRE_OUTLINE_HIGH = (255, 107, 53, 220)
_FIRE_OUTLINE_DEFAULT = (255, 255, 255, 110)

_FIRE_THRESHOLD_VERY_HIGH = 0.8
_FIRE_THRESHOLD_HIGH = 0.6
_FIRE_THRESHOLD_MEDIUM = 0.4
_FIRE_THRESHOLD_LOW = 0.2


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _lon_to_mercator_x(lon: float) -> float:
    """Convert longitude to Web Mercator X coordinate."""
    return lon * 20037508.34 / 180.0


def _lat_to_mercator_y(lat: float) -> float:
    """Convert latitude to Web Mercator Y coordinate."""
    lat = max(-85.05112878, min(85.05112878, lat))
    lat_rad = math.radians(lat)
    y = math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return y * 20037508.34 / math.pi


def _lon_lat_to_world_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Convert lon/lat to global Web Mercator pixel coordinates at a zoom level."""
    lat = max(-85.05112878, min(85.05112878, lat))
    lat_rad = math.radians(lat)
    world_size = _TILE_SIZE * (2**zoom)
    x = (lon + 180.0) / 360.0 * world_size
    y = (1.0 - (math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi)) * 0.5 * world_size
    return x, y


def _lon_lat_to_pixel(
    lon: float,
    lat: float,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    """Convert lon/lat to output pixel coordinates using Web Mercator interpolation."""
    min_lon, min_lat, max_lon, max_lat = bbox

    x_merc = _lon_to_mercator_x(lon)
    y_merc = _lat_to_mercator_y(lat)
    x_min = _lon_to_mercator_x(min_lon)
    x_max = _lon_to_mercator_x(max_lon)
    y_min = _lat_to_mercator_y(min_lat)
    y_max = _lat_to_mercator_y(max_lat)

    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    x = int((x_merc - x_min) / x_range * width)
    y = int((y_max - y_merc) / y_range * height)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return (x, y)


def _estimate_zoom_for_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    max_zoom: int = _MAX_BASEMAP_ZOOM,
) -> int:
    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon = _clamp(min_lon, -180.0, 180.0)
    max_lon = _clamp(max_lon, -180.0, 180.0)
    min_lat = _clamp(min_lat, -85.05112878, 85.05112878)
    max_lat = _clamp(max_lat, -85.05112878, 85.05112878)

    x1, y1 = _lon_lat_to_world_pixel(min_lon, max_lat, 0)
    x2, y2 = _lon_lat_to_world_pixel(max_lon, min_lat, 0)
    bbox_width_px_0 = max(abs(x2 - x1), 1.0)
    bbox_height_px_0 = max(abs(y2 - y1), 1.0)

    zoom_x = math.log2(max(width, 1) / bbox_width_px_0)
    zoom_y = math.log2(max(height, 1) / bbox_height_px_0)
    zoom = int(math.floor(min(zoom_x, zoom_y)))
    return int(_clamp(zoom, 0, max_zoom))


def _tile_ranges_for_bbox(
    bbox: tuple[float, float, float, float],
    zoom: int,
) -> tuple[int, int, int, int, float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    x_left, y_top = _lon_lat_to_world_pixel(min_lon, max_lat, zoom)
    x_right, y_bottom = _lon_lat_to_world_pixel(max_lon, min_lat, zoom)
    x_min_px, x_max_px = sorted([x_left, x_right])
    y_min_px, y_max_px = sorted([y_top, y_bottom])

    tile_x_min = int(math.floor(x_min_px / _TILE_SIZE))
    tile_x_max = int(math.floor(x_max_px / _TILE_SIZE))
    tile_y_min = int(math.floor(y_min_px / _TILE_SIZE))
    tile_y_max = int(math.floor(y_max_px / _TILE_SIZE))

    n = 2**zoom
    tile_x_min = max(0, min(tile_x_min, n - 1))
    tile_x_max = max(0, min(tile_x_max, n - 1))
    tile_y_min = max(0, min(tile_y_min, n - 1))
    tile_y_max = max(0, min(tile_y_max, n - 1))

    return (
        tile_x_min,
        tile_x_max,
        tile_y_min,
        tile_y_max,
        x_min_px,
        x_max_px,
        y_min_px,
        y_max_px,
    )


def _download_tile(client: httpx.Client, zoom: int, x: int, y: int) -> Image.Image | None:
    key = (zoom, x, y)
    cached = _TILE_CACHE.get(key)
    if cached:
        return Image.open(BytesIO(cached)).convert("RGBA")

    url = _BASEMAP_URL.format(z=zoom, x=x, y=y)
    try:
        resp = client.get(url)
        if resp.status_code != 200 or not resp.content:
            return None
        _TILE_CACHE[key] = resp.content
        return Image.open(BytesIO(resp.content)).convert("RGBA")
    except Exception:
        return None


def _fallback_tile() -> Image.Image:
    tile = Image.new("RGBA", (_TILE_SIZE, _TILE_SIZE), (22, 28, 42, 255))
    draw = ImageDraw.Draw(tile, "RGBA")
    draw.rectangle([(0, 0), (_TILE_SIZE - 1, _TILE_SIZE - 1)], outline=(255, 255, 255, 10))
    return tile


def _get_resample_bilinear() -> int:
    try:
        return Image.Resampling.BILINEAR  # Pillow >= 9.1
    except AttributeError:
        return Image.BILINEAR


def _build_dark_basemap(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> Image.Image:
    """Fetch and stitch CARTO dark tiles for the requested bbox."""
    zoom = _estimate_zoom_for_bbox(bbox, width, height)
    tile_data = _tile_ranges_for_bbox(bbox, zoom)
    tile_x_min, tile_x_max, tile_y_min, tile_y_max, *_ = tile_data
    tile_count = (tile_x_max - tile_x_min + 1) * (tile_y_max - tile_y_min + 1)

    while tile_count > _MAX_TILE_COUNT and zoom > 0:
        zoom -= 1
        tile_data = _tile_ranges_for_bbox(bbox, zoom)
        tile_x_min, tile_x_max, tile_y_min, tile_y_max, *_ = tile_data
        tile_count = (tile_x_max - tile_x_min + 1) * (tile_y_max - tile_y_min + 1)

    (
        tile_x_min,
        tile_x_max,
        tile_y_min,
        tile_y_max,
        x_min_px,
        x_max_px,
        y_min_px,
        y_max_px,
    ) = _tile_ranges_for_bbox(bbox, zoom)

    mosaic_w = (tile_x_max - tile_x_min + 1) * _TILE_SIZE
    mosaic_h = (tile_y_max - tile_y_min + 1) * _TILE_SIZE
    mosaic = Image.new("RGBA", (mosaic_w, mosaic_h), _BG_PRIMARY)

    network_available = True
    timeout = httpx.Timeout(2.0, connect=0.6, read=1.0)
    headers = {"User-Agent": "wildfire-nowcast-api/1.0"}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        for tile_x in range(tile_x_min, tile_x_max + 1):
            for tile_y in range(tile_y_min, tile_y_max + 1):
                tile_img: Image.Image | None = None
                if network_available:
                    tile_img = _download_tile(client, zoom, tile_x, tile_y)
                    if tile_img is None:
                        # If the first network call fails, avoid repeated timeout costs.
                        network_available = False
                if tile_img is None:
                    tile_img = _fallback_tile()
                paste_x = (tile_x - tile_x_min) * _TILE_SIZE
                paste_y = (tile_y - tile_y_min) * _TILE_SIZE
                mosaic.paste(tile_img, (paste_x, paste_y))

    left = int(round(x_min_px - tile_x_min * _TILE_SIZE))
    top = int(round(y_min_px - tile_y_min * _TILE_SIZE))
    right = int(round(x_max_px - tile_x_min * _TILE_SIZE))
    bottom = int(round(y_max_px - tile_y_min * _TILE_SIZE))

    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    crop = mosaic.crop((left, top, right, bottom))
    base = crop.resize((width, height), resample=_get_resample_bilinear()).convert("RGBA")

    # Apply subtle navy tint to visually match the UI canvas.
    tint = Image.new("RGBA", (width, height), (10, 22, 40, 42))
    return Image.alpha_composite(base, tint)


def _severity_from_fire(fire: dict[str, Any]) -> float:
    fire_likelihood = _safe_float(fire.get("fire_likelihood"))
    if fire_likelihood is not None:
        return _clamp(fire_likelihood, 0.0, 1.0)

    confidence_score = _safe_float(fire.get("confidence_score"))
    if confidence_score is not None:
        return _clamp(confidence_score, 0.0, 1.0)

    confidence = _safe_float(fire.get("confidence"))
    if confidence is not None:
        return _clamp(confidence / 100.0, 0.0, 1.0)

    frp = _safe_float(fire.get("frp"))
    if frp is not None:
        if frp >= 80:
            return 0.8
        if frp >= 40:
            return 0.6
        if frp >= 20:
            return 0.4
        if frp >= 5:
            return 0.2
        return 0.1

    return -1.0


def _fire_style(fire: dict[str, Any]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    severity = _severity_from_fire(fire)
    if severity >= _FIRE_THRESHOLD_VERY_HIGH:
        fill = _FIRE_VERY_HIGH_FILL
    elif severity >= _FIRE_THRESHOLD_HIGH:
        fill = _FIRE_HIGH_FILL
    elif severity >= _FIRE_THRESHOLD_MEDIUM:
        fill = _FIRE_MEDIUM_FILL
    elif severity >= _FIRE_THRESHOLD_LOW:
        fill = _FIRE_LOW_FILL
    elif severity >= 0:
        fill = _FIRE_VERY_LOW_FILL
    else:
        fill = _FIRE_UNSCORED_FILL

    outline = _FIRE_OUTLINE_HIGH if severity >= _FIRE_THRESHOLD_HIGH else _FIRE_OUTLINE_DEFAULT
    return fill, outline


def _fire_radius(fire: dict[str, Any], width: int) -> int:
    frp = _safe_float(fire.get("frp")) or 0.0
    if frp > 100:
        base = 7
    elif frp > 50:
        base = 5
    elif frp > 20:
        base = 4
    else:
        base = 3
    scale = _clamp(width / 1600.0, 0.75, 1.6)
    return max(2, int(round(base * scale)))


def _risk_score_to_color(risk_score: float) -> tuple[int, int, int, int]:
    if risk_score < 0.3:
        return (34, 139, 34, 82)
    if risk_score < 0.6:
        return (255, 215, 0, 102)
    return (220, 20, 60, 122)


def render_map_png(
    bbox: tuple[float, float, float, float],
    *,
    fires: list[dict] | None = None,
    risk_grid: dict | None = None,
    forecast_contours: list[dict] | None = None,
    width: int = 1600,
    height: int = 900,
    title: Optional[str] = None,
) -> bytes:
    """Render map layers to a PNG image."""
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required for PNG export. Install with: pip install pillow")

    try:
        img = _build_dark_basemap(bbox, width, height)
    except Exception:
        LOGGER.warning("Failed to build tiled basemap for PNG export; using fallback background.")
        img = Image.new("RGBA", (width, height), _BG_PRIMARY)

    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font_title = ImageFont.truetype("arial.ttf", 21)
        font_legend = ImageFont.truetype("arial.ttf", 13)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_title = ImageFont.load_default()
        font_legend = ImageFont.load_default()
        font_small = ImageFont.load_default()

    if risk_grid and "features" in risk_grid:
        _draw_risk_grid(draw, risk_grid, bbox, width, height)

    if forecast_contours:
        _draw_forecast_contours(draw, forecast_contours, bbox, width, height)

    if fires:
        _draw_fires(draw, fires, bbox, width, height)

    _draw_legend(
        draw,
        width=width,
        height=height,
        has_fires=bool(fires),
        has_risk=bool(risk_grid),
        has_forecast=bool(forecast_contours),
        font=font_legend,
    )

    if title:
        draw.text((14, 14), title, fill=_TEXT_PRIMARY, font=font_title)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    draw.text((14, height - 22), f"Generated: {timestamp}", fill=_TEXT_MUTED, font=font_small)
    attribution = "© CARTO, © OpenStreetMap contributors"
    text_box = draw.textbbox((0, 0), attribution, font=font_small)
    text_w = text_box[2] - text_box[0]
    draw.text((width - text_w - 12, height - 22), attribution, fill=_TEXT_MUTED, font=font_small)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_risk_grid(
    draw: ImageDraw.ImageDraw,
    risk_grid: dict[str, Any],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> None:
    for feature in risk_grid.get("features", []):
        props = feature.get("properties", {}) or {}
        risk_score = _safe_float(props.get("risk_score"))
        if risk_score is None:
            continue
        geom = feature.get("geometry", {}) or {}
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        ring = coords[0]
        pixels = [_lon_lat_to_pixel(lon, lat, bbox, width, height) for lon, lat in ring[:-1]]
        if len(pixels) < 3:
            continue
        fill = _risk_score_to_color(risk_score)
        draw.polygon(pixels, fill=fill, outline=fill[:3] + (175,))


def _draw_forecast_contours(
    draw: ImageDraw.ImageDraw,
    contours: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> None:
    for contour in contours:
        raw_geom = contour.get("geom_geojson")
        if not raw_geom:
            continue
        try:
            geom = json.loads(raw_geom) if isinstance(raw_geom, str) else raw_geom
        except Exception:
            continue

        horizon = int(contour.get("horizon_hours") or 0)
        if horizon <= 24:
            stroke = (255, 165, 0, 210)
            fill = (255, 165, 0, 50)
        elif horizon <= 48:
            stroke = (255, 140, 0, 210)
            fill = (255, 140, 0, 44)
        else:
            stroke = (255, 110, 0, 210)
            fill = (255, 110, 0, 40)

        geom_type = geom.get("type")
        if geom_type == "Polygon":
            polygons = [geom.get("coordinates", [])]
        elif geom_type == "MultiPolygon":
            polygons = geom.get("coordinates", [])
        else:
            polygons = []

        for poly in polygons:
            if not poly:
                continue
            outer_ring = poly[0]
            points = [_lon_lat_to_pixel(lon, lat, bbox, width, height) for lon, lat in outer_ring]
            if len(points) < 3:
                continue
            draw.polygon(points, fill=fill, outline=stroke)


def _draw_fires(
    draw: ImageDraw.ImageDraw,
    fires: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> None:
    for fire in fires:
        lat = _safe_float(fire.get("lat"))
        lon = _safe_float(fire.get("lon"))
        if lat is None or lon is None:
            continue

        x, y = _lon_lat_to_pixel(lon, lat, bbox, width, height)
        radius = _fire_radius(fire, width)
        fill, outline = _fire_style(fire)

        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=fill,
            outline=outline,
            width=1,
        )


def _legend_items(has_fires: bool, has_risk: bool, has_forecast: bool) -> list[tuple[str, tuple[int, int, int, int]]]:
    items: list[tuple[str, tuple[int, int, int, int]]] = []
    if has_fires:
        items.extend(
            [
                ("Fires: Very high", _FIRE_VERY_HIGH_FILL),
                ("Fires: High", _FIRE_HIGH_FILL),
                ("Fires: Medium", _FIRE_MEDIUM_FILL),
                ("Fires: Low", _FIRE_LOW_FILL),
                ("Fires: Very low", _FIRE_VERY_LOW_FILL),
                ("Fires: Unscored", _FIRE_UNSCORED_FILL),
            ]
        )
    if has_risk:
        items.extend(
            [
                ("Risk: Low", (34, 139, 34, 160)),
                ("Risk: Medium", (255, 215, 0, 180)),
                ("Risk: High", (220, 20, 60, 190)),
            ]
        )
    if has_forecast:
        items.append(("Forecast contours", (255, 165, 0, 210)))
    return items


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    has_fires: bool,
    has_risk: bool,
    has_forecast: bool,
    font: ImageFont.ImageFont,
) -> None:
    items = _legend_items(has_fires, has_risk, has_forecast)
    if not items:
        return

    legend_width = 220
    line_h = 18
    legend_height = 14 + line_h * (len(items) + 1) + 10
    legend_x = 16
    legend_y = height - legend_height - 32

    draw.rounded_rectangle(
        [(legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height)],
        radius=8,
        fill=_BG_CARD,
        outline=_BORDER_SUBTLE,
        width=1,
    )
    draw.text((legend_x + 10, legend_y + 8), "Legend", fill=_TEXT_PRIMARY, font=font)

    y = legend_y + 28
    for label, color in items:
        draw.rectangle(
            [(legend_x + 10, y), (legend_x + 22, y + 10)],
            fill=color,
            outline=(0, 0, 0, 180),
            width=1,
        )
        draw.text((legend_x + 28, y - 2), label, fill=_TEXT_PRIMARY, font=font)
        y += line_h
