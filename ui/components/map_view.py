"""Map view component for wildfire dashboard."""

import folium
import streamlit as st
from streamlit_folium import st_folium
from typing import Optional, Dict, Any, List
from config.constants import (
    DEFAULT_MAP_CENTER, DEFAULT_ZOOM_LEVEL, MAP_HEIGHT,
    PLACEHOLDER_FIRE_LOCATIONS, PLACEHOLDER_FORECAST_POLYGON, PLACEHOLDER_RISK_POLYGON
)

def add_fire_markers(map_obj: folium.Map, locations: List[List[float]]) -> None:
    """Add placeholder fire markers to the map."""
    for loc in locations:
        folium.CircleMarker(
            location=loc,
            radius=8,
            popup="Placeholder fire detection",
            tooltip="Active fire (placeholder)",
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.7,
        ).add_to(map_obj)

def add_forecast_polygon(map_obj: folium.Map, polygon: List[List[float]]) -> None:
    """Add placeholder forecast polygon to the map."""
    folium.Polygon(
        locations=polygon,
        popup="Forecast spread area (24-72h placeholder)",
        tooltip="Forecast (24-72h)",
        color="orange",
        fill=True,
        fillColor="orange",
        fillOpacity=0.3,
        weight=2,
    ).add_to(map_obj)

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
        location=DEFAULT_MAP_CENTER,
        zoom_start=DEFAULT_ZOOM_LEVEL,
        tiles="OpenStreetMap"
    )

def render_map_view() -> Optional[Dict[str, float]]:
    """Render the map view and return click coordinates if any."""
    try:
        m = create_base_map()

        # Add placeholder layers based on toggles
        if st.session_state.show_fires:
            add_fire_markers(m, PLACEHOLDER_FIRE_LOCATIONS)

        if st.session_state.show_forecast:
            add_forecast_polygon(m, PLACEHOLDER_FORECAST_POLYGON)

        if st.session_state.show_risk:
            add_risk_polygon(m, PLACEHOLDER_RISK_POLYGON)

        # Render map and capture interactions
        map_data = st_folium(m, width=None, height=MAP_HEIGHT, returned_objects=["last_clicked"])

        # Handle map click safely
        last_clicked = map_data.get("last_clicked")
        if last_clicked is not None:
            clicked_lat = last_clicked.get("lat")
            clicked_lng = last_clicked.get("lng")
            if clicked_lat is not None and clicked_lng is not None:
                return {"lat": clicked_lat, "lng": clicked_lng}

    except Exception as e:
        st.error(f"Error rendering map: {e}")
        st.info("Please refresh the page. If the problem persists, check your connection and try again.")
        st.stop()

    return None
