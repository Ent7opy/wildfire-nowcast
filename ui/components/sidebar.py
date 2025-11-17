"""Sidebar component for wildfire dashboard controls."""

import streamlit as st
from config.constants import TIME_WINDOW_OPTIONS

def get_time_window_index(time_window: str) -> int:
    """Get the index of the time window in the options list, defaulting to 0 if not found."""
    try:
        return TIME_WINDOW_OPTIONS.index(time_window)
    except ValueError:
        return 0

def render_sidebar() -> str:
    """Render the sidebar controls and return selected time window."""
    st.header("Controls")

    # Time window section
    st.subheader("Time window")
    selected_time = st.selectbox(
        "Select time range",
        options=TIME_WINDOW_OPTIONS,
        index=get_time_window_index(st.session_state.time_window),
        key="time_window_select"
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
        "Forecast (24–72h)",
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

    # Region / AOI placeholder
    st.subheader("Region / AOI")
    st.info(
        "Region selection will be available in a future version. "
        "This will allow you to focus on specific areas of interest."
    )

    st.divider()

    # About / Data notice
    st.subheader("About / Data")
    st.caption(
        "⚠️ **Placeholder Data Notice**\n\n"
        "All map data, fire detections, forecasts, and risk indices shown here are "
        "placeholder demonstrations. Real satellite data (NASA FIRMS), weather forecasts, "
        "and terrain analysis will be integrated in future versions."
    )

    return selected_time
