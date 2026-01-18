"""Map view component for wildfire dashboard using PyDeck."""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pydeck as pdk
import streamlit as st

from runtime_config import api_public_base_url
from config.constants import (
    DEFAULT_MAP_CENTER,
    DEFAULT_ZOOM_LEVEL,
    MAP_HEIGHT,
)

# Constants for defaults if not in session state
INITIAL_LAT = DEFAULT_MAP_CENTER[0]
INITIAL_LON = DEFAULT_MAP_CENTER[1]
INITIAL_ZOOM = DEFAULT_ZOOM_LEVEL

def _isoformat(dt: datetime) -> str:
    """Format datetime for API query parameters."""
    offset = dt.utcoffset() if dt.tzinfo is not None else None
    if offset is not None and offset.total_seconds() == 0:
        dt_no_microseconds = dt.replace(microsecond=0)
        return dt_no_microseconds.replace(tzinfo=None).isoformat() + "Z"
    dt_no_microseconds = dt.replace(microsecond=0)
    return dt_no_microseconds.isoformat()

def _current_time_range() -> tuple[datetime, datetime]:
    """Return (start_time, end_time) based on selected time window."""
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

def render_map_view() -> Optional[Dict[str, float]]:
    """Render the PyDeck map view and return click coordinates if any."""
    
    # Initialize view state in session state to persist across reruns
    if "map_view_state" not in st.session_state:
        st.session_state.map_view_state = pdk.ViewState(
            latitude=INITIAL_LAT,
            longitude=INITIAL_LON,
            zoom=INITIAL_ZOOM,
            pitch=0,
            bearing=0,
        )

    layers = []

    # 1. Fires Layer (MVT)
    if st.session_state.show_fires:
        start_time, end_time = _current_time_range()
        include_noise = not bool(getattr(st.session_state, "fires_apply_denoiser", True))
        min_confidence = float(getattr(st.session_state, "fires_min_confidence", 0.0))
        
        # Build query params for the tile URL
        params = {
            "start_time": _isoformat(start_time),
            "end_time": _isoformat(end_time),
            "min_confidence": min_confidence,
            "include_noise": str(include_noise).lower()
        }
        query_str = "&".join([f"{k}={v}" for k, v in params.items()])
        
        # Point to the API proxy for vector tiles
        tile_url = f"{api_public_base_url()}/tiles/fires/{{z}}/{{x}}/{{y}}.pbf?{query_str}"
        
        layers.append(pdk.Layer(
            "MVTLayer",
            data=tile_url,
            id="fires",
            pickable=True,
            auto_highlight=True,
            get_fill_color=[255, 0, 0, 200],
            point_radius_min_pixels=4,
            point_radius_max_pixels=10,
        ))

    # 2. Forecast Contours (MVT - using same proxy if available)
    if st.session_state.show_forecast:
        # For now, we only show fires in this migration, but we can add a placeholder
        # MVTLayer for contours if the backend supports it.
        # Based on api/routes/tiles.py, "forecast_contours" is a valid layer.
        run_id = st.session_state.get("last_forecast", {}).get("run", {}).get("id")
        contour_url = f"{api_public_base_url()}/tiles/forecast_contours/{{z}}/{{x}}/{{y}}.pbf"
        if run_id:
            contour_url += f"?run_id={run_id}"
            
        layers.append(pdk.Layer(
            "MVTLayer",
            data=contour_url,
            id="forecast_contours",
            pickable=True,
            get_fill_color=[255, 165, 0, 40], # semi-transparent orange
            get_line_color=[255, 165, 0, 200],
            get_line_width=2,
            line_width_min_pixels=1,
        ))

    # Create the Deck
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=st.session_state.map_view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": "<b>ID:</b> {id}<br/>"
                    "<b>Time:</b> {acq_time}<br/>"
                    "<b>Confidence:</b> {confidence}<br/>"
                    "<b>Intensity (FRP):</b> {frp}",
            "style": {"color": "white", "backgroundColor": "#333"}
        },
    )

    # Render with selection support
    # Note: st.pydeck_chart selection support requires streamlit >= 1.35.0
    event = st.pydeck_chart(
        deck, 
        height=MAP_HEIGHT, 
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-object",
        key="main_map"
    )

    # Handle interactions
    if event and event.selection:
        # Check fires layer
        selected_fires = event.selection.objects.get("fires", [])
        if selected_fires:
            props = selected_fires[0]
            # Save props for the details panel
            st.session_state.selected_fire = props
            # Return coordinates for app level click tracking
            return {"lat": props.get("lat"), "lng": props.get("lon")}
            
    return None
