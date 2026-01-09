"""Map view component for wildfire dashboard."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import folium
import httpx
import streamlit as st
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster

from api_client import ApiError, ApiUnavailableError, generate_forecast, get_forecast, get_fires
from config.constants import (
    DEFAULT_MAP_CENTER,
    DEFAULT_ZOOM_LEVEL,
    MAP_HEIGHT,
    PLACEHOLDER_RISK_POLYGON,
)

FIRE_MARKER_SIMPLE_THRESHOLD = 300
FIRE_HEATMAP_THRESHOLD = 2000
FIRE_API_LIMIT = 5000


def _current_bbox() -> tuple[float, float, float, float]:
    """Return current bbox (min_lon, min_lat, max_lon, max_lat)."""
    if "map_bounds" in st.session_state and st.session_state.map_bounds:
        b = st.session_state.map_bounds
        # st-folium bounds format: {'_southWest': {'lat': 41.0, 'lng': 22.0}, ...}
        try:
            sw = b.get("_southWest", {})
            ne = b.get("_northEast", {})
            if sw.get("lng") is not None and sw.get("lat") is not None and ne.get("lng") is not None and ne.get("lat") is not None:
                return (
                    float(sw["lng"]),
                    float(sw["lat"]),
                    float(ne["lng"]),
                    float(ne["lat"]),
                )
        except (KeyError, TypeError, ValueError):
            # If map_bounds structure is invalid, fall back to default
            pass

    # Default bbox worldwide (so users can see fires immediately)
    return (-180.0, -90.0, 180.0, 90.0)


def _current_time_range() -> tuple[datetime, datetime]:
    """Return (start_time, end_time) based on selected time window.
    
    Time is rounded to the nearest minute to enable caching and prevent
    constant refetches due to second/microsecond changes.
    """
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    window = getattr(st.session_state, "time_window", "Last 24 hours")
    hours = 24
    if window == "Last 6 hours":
        hours = 6
    elif window == "Last 12 hours":
        hours = 12
    elif window == "Last 48 hours":
        hours = 48
    start = end - timedelta(hours=hours)
    return start, end


def _fire_tooltip(det: Dict[str, Any]) -> str:
    t = det.get("acq_time")
    conf = det.get("confidence")
    sensor = det.get("sensor")
    parts = []
    if t is not None:
        parts.append(str(t))
    if conf is not None:
        try:
            parts.append(f"conf={float(conf):.0f}")
        except (TypeError, ValueError):
            parts.append(f"conf={conf}")
    if sensor:
        parts.append(str(sensor))
    return " | ".join(parts) if parts else "Fire detection"


def _fire_popup(det: Dict[str, Any]) -> str:
    popup_parts = [
        f"id: {det.get('id')}",
        f"time: {det.get('acq_time')}",
        f"confidence: {det.get('confidence')}",
        f"frp: {det.get('frp')}",
        f"sensor: {det.get('sensor')}",
        f"source: {det.get('source')}",
    ]
    return "<br/>".join(popup_parts)


def add_fires_layer(map_obj: folium.Map, detections: List[Dict[str, Any]]) -> None:
    """Add fire detections with density handling (markers / clusters / heatmap)."""
    n = len(detections)
    if n == 0:
        return

    # Heatmap layer for very dense situations (keeps map usable).
    if n >= FIRE_HEATMAP_THRESHOLD:
        heat_points = []
        for det in detections:
            lat = det.get("lat")
            lon = det.get("lon")
            if lat is None or lon is None:
                continue
            heat_points.append([lat, lon, 1.0])
        if heat_points:
            HeatMap(heat_points, name="Fires (density)", radius=12, blur=16).add_to(map_obj)

        # Still add clustered markers so users can inspect individual detections.
        cluster = MarkerCluster(name="Fires (clusters)").add_to(map_obj)
        for det in detections:
            lat = det.get("lat")
            lon = det.get("lon")
            if lat is None or lon is None:
                continue
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                popup=_fire_popup(det),
                tooltip=_fire_tooltip(det),
                color="red",
                fill=True,
                fillColor="red",
                fillOpacity=0.6,
            ).add_to(cluster)
        return

    # Cluster markers once we get beyond a small number of points.
    if n > FIRE_MARKER_SIMPLE_THRESHOLD:
        cluster = MarkerCluster(name="Fires (clusters)").add_to(map_obj)
        for det in detections:
            lat = det.get("lat")
            lon = det.get("lon")
            if lat is None or lon is None:
                continue
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=_fire_popup(det),
                tooltip=_fire_tooltip(det),
                color="red",
                fill=True,
                fillColor="red",
                fillOpacity=0.7,
            ).add_to(cluster)
        return

    # Simple markers for small counts.
    for det in detections:
        lat = det.get("lat")
        lon = det.get("lon")
        if lat is None or lon is None:
            continue

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=_fire_popup(det),
            tooltip=_fire_tooltip(det),
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.7,
        ).add_to(map_obj)


def add_forecast_layers(map_obj: folium.Map) -> None:
    """Fetch and add real forecast layers to the map."""
    # Check if there's a location-based forecast from click_details
    if "last_forecast" in st.session_state and "last_forecast_bbox" in st.session_state:
        forecast_data = st.session_state.last_forecast
        bbox = st.session_state.last_forecast_bbox
        # Use the generated forecast
    else:
        # Use current map view
        bbox = _current_bbox()
        try:
            with st.spinner("Loading forecast overlay…"):
                forecast_data = get_forecast(bbox, horizons=[24, 48, 72], region_name=None)
        except ApiUnavailableError:
            st.error("API unavailable — please start the backend")
            return
        except ApiError as e:
            details = f"(status={e.status_code})" if e.status_code is not None else ""
            st.error(f"Forecast unavailable {details}".strip())
            if e.response_text:
                st.caption(str(e.response_text)[:300])
            return

    if not forecast_data.get("run") and not forecast_data.get("forecast"):
        st.info(
            "No forecast available for the current view. "
            "Click a fire and click 'Generate Spread Forecast' to create a location-based forecast."
        )
        return

    def _contour_color(feature: Dict[str, Any]) -> str:
        props = feature.get("properties") or {}
        h = props.get("horizon_hours")
        if h == 24:
            return "#2b83ba"  # blue
        if h == 48:
            return "#fdae61"  # orange
        if h == 72:
            return "#d7191c"  # red
        return "#984ea3"  # purple fallback

    # 1. Add contours
    if forecast_data.get("contours") and forecast_data["contours"].get("features"):
        folium.GeoJson(
            forecast_data["contours"],
            name="Forecast Contours",
            style_function=lambda feature: {
                "color": _contour_color(feature),
                "weight": 2,
                "fillOpacity": 0.05,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["horizon_hours", "threshold"], aliases=["Horizon (h)", "Probability"]
            ),
        ).add_to(map_obj)

    # 2. Add raster tiles (one per horizon if available)
    if forecast_data.get("rasters"):
        rasters = sorted(forecast_data["rasters"], key=lambda x: x.get("horizon_hours", 0))
        for r in rasters:
            tilejson_url = r.get("tilejson_url")
            if not tilejson_url:
                continue
            try:
                # We need to resolve TileJSON to get the actual tile template for Leaflet.
                # The URL from the API is external (e.g. localhost:8080), but we are inside Docker
                # so we need to use the internal service name.
                # TODO: Make these hostnames configurable via env vars
                internal_url = tilejson_url.replace("localhost:8080", "titiler:8000")

                resp = httpx.get(internal_url, timeout=5.0)
                if resp.status_code != 200:
                    continue
                tj = resp.json()
                tile_url = tj["tiles"][0]

                # Convert internal hostname back to external for the browser.
                tile_url_external = tile_url.replace("titiler:8000", "localhost:8080")
                horizon = r.get("horizon_hours")

                folium.TileLayer(
                    tiles=tile_url_external,
                    attr="TiTiler",
                    name=f"Spread Probability (T+{horizon}h)",
                    overlay=True,
                    opacity=0.45,
                    control=True,
                ).add_to(map_obj)
            except Exception as e:
                st.warning(f"Could not load raster tiles: {e}")


def add_risk_polygon(map_obj: folium.Map, polygon: List[List[float]]) -> None:
    """Add placeholder risk polygon to the map."""
    folium.Polygon(
        locations=polygon,
        popup="Fire risk index (placeholder)",
        tooltip="Risk index",
        color="purple",
        fill=True,
        fillColor="purple",
        fillOpacity=0.2,
        weight=2,
    ).add_to(map_obj)


def create_base_map() -> folium.Map:
    """Create a base Folium map with default settings."""
    return folium.Map(
        location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_ZOOM_LEVEL, tiles="OpenStreetMap"
    )


def render_map_view() -> Optional[Dict[str, float]]:
    """Render the map view and return click coordinates if any."""
    try:
        # Only recreate map if bounds changed significantly to reduce reruns
        m = create_base_map()

        # Add layers based on toggles
        if st.session_state.show_fires:
            bbox = _current_bbox()
            start_time, end_time = _current_time_range()
            include_noise = not bool(getattr(st.session_state, "fires_apply_denoiser", True))
            min_confidence = float(getattr(st.session_state, "fires_min_confidence", 0.0))
            
            # Create a cache key based on query parameters
            # Time is already rounded by _current_time_range(), so use as-is
            cache_key = (
                tuple(bbox),
                start_time.isoformat(),
                end_time.isoformat(),
                include_noise,
                min_confidence,
            )
            
            # Check if we have cached results for this query
            cached_result = st.session_state.get("fires_cache", {}).get(cache_key)
            
            if cached_result is None:
                # Need to fetch new data
                try:
                    with st.spinner("Loading fires…"):
                        fires_data = get_fires(
                            bbox=bbox,
                            time_range=(start_time, end_time),
                            filters={
                                "limit": FIRE_API_LIMIT,
                                "include_noise": include_noise,
                                "min_confidence": min_confidence,
                                "include_denoiser_fields": True,
                            },
                        )
                    detections = fires_data.get("detections", [])
                    # Cache the result (keep only last 10 queries to avoid memory issues)
                    if len(st.session_state.fires_cache) >= 10:
                        # Remove oldest entry
                        oldest_key = next(iter(st.session_state.fires_cache))
                        del st.session_state.fires_cache[oldest_key]
                    st.session_state.fires_cache[cache_key] = detections
                    st.session_state.fires_last_detections = detections
                except (ApiUnavailableError, ApiError, Exception) as e:
                    # Handle errors below
                    detections = []
                    st.session_state.fires_last_detections = []
                    if isinstance(e, ApiUnavailableError):
                        st.error("API unavailable — please start the backend")
                    elif isinstance(e, ApiError):
                        details = f"(status={e.status_code})" if e.status_code is not None else ""
                        st.error(f"Fires unavailable {details}".strip())
                        if e.response_text:
                            st.caption(str(e.response_text)[:300])
                    else:
                        st.error(f"Unexpected error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                # Use cached result
                detections = cached_result
                st.session_state.fires_last_detections = detections
            
            # Display fires if we have them
            if detections:
                if len(detections) >= FIRE_API_LIMIT:
                    st.warning(
                        f"Too many detections for the current view/time window. "
                        f"Showing the first {FIRE_API_LIMIT}. Zoom in or narrow the time window."
                    )
                add_fires_layer(m, detections)
            elif cached_result is not None:  # Only show message if we actually queried (not from cache)
                st.info(
                    "No fires found for the current filters. "
                    "Try a wider time window, lower minimum confidence, or zoom out."
                )
        else:
            # Clear detections if the layer is disabled, to avoid stale inspection data.
            st.session_state.fires_last_detections = []

        if st.session_state.show_forecast:
            add_forecast_layers(m)

        if st.session_state.show_risk:
            add_risk_polygon(m, PLACEHOLDER_RISK_POLYGON)

        # Add layer control to toggle between contours/rasters if both present
        folium.LayerControl().add_to(m)

        # Render map and capture interactions
        # Only request bounds if refresh was explicitly requested (prevents automatic reruns)
        # This prevents continuous rerun loops while still allowing manual refresh
        map_refresh_requested = st.session_state.get("map_refresh_requested", False)
        requested_objects = ["last_clicked"]
        if map_refresh_requested:
            # Only request bounds when refresh button was clicked
            requested_objects.append("bounds")
        
        map_data = st_folium(
            m, 
            width=None, 
            height=MAP_HEIGHT, 
            returned_objects=requested_objects,
            key="main_map",
        )

        # Update bounds only if refresh was requested and bounds were captured
        if map_refresh_requested and map_data.get("bounds"):
            new_bounds = map_data["bounds"]
            st.session_state.map_bounds = new_bounds
            # Clear the refresh flag after successfully capturing bounds
            st.session_state.map_refresh_requested = False
            # Cache will be cleared by the button handler, so fires will refetch automatically

        # Handle map click safely
        last_clicked = map_data.get("last_clicked")
        if last_clicked is not None:
            clicked_lat = last_clicked.get("lat")
            clicked_lng = last_clicked.get("lng")
            if clicked_lat is not None and clicked_lng is not None:
                return {"lat": clicked_lat, "lng": clicked_lng}

    except Exception as e:
        st.error(f"Error rendering map: {e}")
        st.info(
            "Please refresh the page. If the problem persists, check your connection and try again."
        )
        st.stop()

    return None
