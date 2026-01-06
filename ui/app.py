"""Main Streamlit application for Wildfire Nowcast & Forecast."""

import streamlit as st
from config.constants import TIME_WINDOW_OPTIONS

# Import components
from components.sidebar import render_sidebar
from components.map_view import render_map_view
from components.legend import render_legend
from components.click_details import render_click_details

def main() -> None:
    """Main application entry point."""
    # Page configuration - must be the first Streamlit command
    st.set_page_config(page_title="Wildfire Nowcast & Forecast", layout="wide")

    # Initialize session state after set_page_config
    if "time_window" not in st.session_state:
        st.session_state.time_window = TIME_WINDOW_OPTIONS[0]
    if "fires_min_confidence" not in st.session_state:
        st.session_state.fires_min_confidence = 0.0
    if "fires_apply_denoiser" not in st.session_state:
        st.session_state.fires_apply_denoiser = True
    if "show_fires" not in st.session_state:
        st.session_state.show_fires = True
    if "show_forecast" not in st.session_state:
        st.session_state.show_forecast = False
    if "show_risk" not in st.session_state:
        st.session_state.show_risk = False
    if "last_click" not in st.session_state:
        st.session_state.last_click = None
    if "fires_last_detections" not in st.session_state:
        st.session_state.fires_last_detections = []
    if "map_bounds" not in st.session_state:
        st.session_state.map_bounds = None

    # App identity - Title and subtitle
    st.title("Wildfire Nowcast & Forecast")
    st.caption(
        "Monitor active wildfires and view short-term spread forecasts (24-72 hours) "
        "using satellite data, weather, and terrain analysis. "
    )

    # Sidebar controls
    with st.sidebar:
        render_sidebar()

    # Main content area - Map and indicators
    # Active filters indicator
    denoiser_state = "on" if st.session_state.fires_apply_denoiser else "off"
    st.caption(
        f"**Fires filters:** {st.session_state.time_window}, "
        f"min confidence ≥ {st.session_state.fires_min_confidence:.0f}, "
        f"denoiser {denoiser_state}"
    )

    # Render map + details side-by-side
    col_map, col_details = st.columns([3, 1], gap="large")
    with col_map:
        click_coords = render_map_view()
        if click_coords is not None:
            st.session_state.last_click = click_coords

    with col_details:
        render_click_details(st.session_state.last_click)
        render_legend()

if __name__ == "__main__":
    main()
