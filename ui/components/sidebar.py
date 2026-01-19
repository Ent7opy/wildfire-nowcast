"""Sidebar component for wildfire dashboard controls."""

from datetime import datetime, timedelta, timezone
import streamlit as st
from config.constants import TIME_WINDOW_OPTIONS
from runtime_config import api_public_base_url


def get_time_window_index(time_window: str) -> int:
    """Get the index of the time window in the options list, defaulting to 0 if not found."""
    try:
        return TIME_WINDOW_OPTIONS.index(time_window)
    except ValueError:
        return 0


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


def _calculate_viewport_bbox() -> tuple[float, float, float, float]:
    """Calculate bbox from current map viewport state."""
    view_state = st.session_state.get("map_view_state")
    if view_state is None:
        # Default bbox (rough global extent)
        return (-180.0, -85.0, 180.0, 85.0)
    
    lat = view_state.latitude
    lon = view_state.longitude
    zoom = view_state.zoom
    
    # Approximate visible area in degrees based on zoom level
    # At zoom 0, the map shows ~360 degrees. Each zoom level halves the visible area.
    degrees_per_tile = 360.0 / (2 ** zoom)
    
    # Adjust for latitude (Mercator projection distortion)
    lat_extent = degrees_per_tile * 0.5
    lon_extent = degrees_per_tile * 0.5
    
    min_lon = max(lon - lon_extent, -180.0)
    max_lon = min(lon + lon_extent, 180.0)
    min_lat = max(lat - lat_extent, -85.0)
    max_lat = min(lat + lat_extent, 85.0)
    
    return (min_lon, min_lat, max_lon, max_lat)


def render_sidebar() -> str:
    """Render the sidebar controls and return selected time window."""
    st.header("Filters & Controls")

    # Fires filters (debounced via Apply button)
    st.subheader("Filters")
    st.caption("Adjust fires filters, then click **Apply filters**.")
    with st.form("fires_filters", clear_on_submit=False):
        pending_time_window = st.selectbox(
            "Time window",
            options=TIME_WINDOW_OPTIONS,
            index=get_time_window_index(st.session_state.time_window),
            key="pending_time_window",
        )

        pending_min_likelihood = st.slider(
            "Minimum fire likelihood",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.fires_min_likelihood),
            step=0.05,
            key="pending_min_likelihood",
            help="Composite score combining FIRMS confidence (20%), persistence (30%), land-cover plausibility (25%), and weather conditions (25%). Values <0.3 indicate low confidence, 0.3-0.6 uncertain, >0.6 likely real fire.",
        )

        pending_apply_denoiser = st.checkbox(
            "Apply noise filter (exclude false alarms)",
            value=bool(st.session_state.fires_apply_denoiser),
            key="pending_apply_denoiser",
        )

        applied = st.form_submit_button("Apply filters", type="primary")
        if applied:
            st.session_state.time_window = pending_time_window
            st.session_state.fires_min_likelihood = float(pending_min_likelihood)
            st.session_state.fires_apply_denoiser = bool(pending_apply_denoiser)

    # Export fires button
    st.caption("Export current view as CSV or GeoJSON.")
    bbox = _calculate_viewport_bbox()
    start_time, end_time = _current_time_range()
    min_lon, min_lat, max_lon, max_lat = bbox
    
    export_url = (
        f"{api_public_base_url()}/fires/export?"
        f"min_lon={min_lon}&min_lat={min_lat}&max_lon={max_lon}&max_lat={max_lat}&"
        f"start_time={_isoformat(start_time)}&end_time={_isoformat(end_time)}&"
        f"format=csv&limit=1000"
    )
    
    st.link_button(
        "📥 Export fires (CSV)",
        export_url,
        use_container_width=True,
    )

    st.divider()

    # Layers section
    st.subheader("Layers")
    st.session_state.show_fires = st.checkbox(
        "Active fires",
        value=st.session_state.show_fires,
        key="fires_checkbox"
    )
    st.session_state.show_forecast = st.checkbox(
        "Show forecast overlay",
        value=st.session_state.show_forecast,
        key="forecast_checkbox"
    )
    st.session_state.show_risk = st.checkbox(
        "Risk index",
        value=st.session_state.show_risk,
        key="risk_checkbox"
    )

    # Count active layers
    active_layers = sum([
        st.session_state.show_fires,
        st.session_state.show_forecast,
        st.session_state.show_risk
    ])
    st.caption(f"Layers active: {active_layers}")

    st.divider()

    # Map controls
    st.subheader("Map Controls")
    st.caption("Pan and zoom to explore. The map updates as you move.")
    if st.button("🗑️ Clear selection", use_container_width=True):
        st.session_state.selected_fire = None
        st.session_state.last_click = None
        st.rerun()
    
    st.divider()

    # AOI behavior (MVP)
    st.subheader("Forecast area")
    st.caption("Forecast uses the area currently shown on the map.")

    st.divider()

    # About / Data notice
    st.subheader("About")
    st.caption(
        "**Data sources**\n\n"
        "- Fires and forecast layers are updated automatically from our data service.\n"
        "- If data can’t be reached, you’ll see an error and can retry.\n"
        "- The risk layer is still a placeholder."
    )

    return st.session_state.time_window
