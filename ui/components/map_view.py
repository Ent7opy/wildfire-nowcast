"""Map view component for wildfire dashboard."""

from typing import Dict, List, Optional

import folium
import httpx
import streamlit as st
from streamlit_folium import st_folium

from ui.config.constants import (
    DEFAULT_MAP_CENTER,
    DEFAULT_ZOOM_LEVEL,
    MAP_HEIGHT,
    PLACEHOLDER_FIRE_LOCATIONS,
    PLACEHOLDER_RISK_POLYGON,
)
from ui.services.api import get_latest_forecast


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


def add_forecast_layers(map_obj: folium.Map) -> None:
    """Fetch and add real forecast layers to the map."""
    # Use bounds from session state if available, otherwise default
    if "map_bounds" in st.session_state and st.session_state.map_bounds:
        b = st.session_state.map_bounds
        # st-folium bounds format: {'_southWest': {'lat': 41.0, 'lng': 22.0}, ...}
        bbox = (
            b["_southWest"]["lng"],
            b["_southWest"]["lat"],
            b["_northEast"]["lng"],
            b["_northEast"]["lat"],
        )
    else:
        # Default bbox around Bulgaria area
        bbox = (22.0, 41.0, 29.0, 44.5)

    # For now we use a hardcoded region name. In the future this should be dynamic.
    forecast_data = get_latest_forecast("smoke_grid", bbox)
    if not forecast_data:
        return

    # 1. Add contours
    if forecast_data.get("contours") and forecast_data["contours"].get("features"):
        folium.GeoJson(
            forecast_data["contours"],
            name="Forecast Contours",
            style_function=lambda x: {
                "color": "orange",
                "weight": 2,
                "fillOpacity": 0.1,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["horizon_hours", "threshold"], aliases=["Horizon (h)", "Probability"]
            ),
        ).add_to(map_obj)

    # 2. Add raster tiles (latest horizon)
    if forecast_data.get("rasters"):
        # Show the longest horizon by default for visual impact
        r = sorted(forecast_data["rasters"], key=lambda x: x["horizon_hours"])[-1]
        tilejson_url = r["tilejson_url"]
        try:
            # We need to resolve TileJSON to get the actual tile template for Leaflet
            resp = httpx.get(tilejson_url, timeout=5.0)
            if resp.status_code == 200:
                tj = resp.json()
                tile_url = tj["tiles"][0]
                folium.TileLayer(
                    tiles=tile_url,
                    attr="TiTiler",
                    name=f"Spread Probability (T+{r['horizon_hours']}h)",
                    overlay=True,
                    opacity=0.5,
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
        m = create_base_map()

        # Add layers based on toggles
        if st.session_state.show_fires:
            add_fire_markers(m, PLACEHOLDER_FIRE_LOCATIONS)

        if st.session_state.show_forecast:
            add_forecast_layers(m)

        if st.session_state.show_risk:
            add_risk_polygon(m, PLACEHOLDER_RISK_POLYGON)

        # Add layer control to toggle between contours/rasters if both present
        folium.LayerControl().add_to(m)

        # Render map and capture interactions
        map_data = st_folium(
            m, width=None, height=MAP_HEIGHT, returned_objects=["last_clicked", "bounds"]
        )

        # Update bounds in session state for next rerun (for API querying)
        if map_data.get("bounds"):
            st.session_state.map_bounds = map_data["bounds"]

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
