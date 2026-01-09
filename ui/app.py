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

    # Minimal styling (Streamlit-supported CSS injection)
    st.markdown(
        """
        <style>
          /* Tighten overall top spacing a bit */
          .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }

          /* Make sidebars feel less “boilerplate” */
          section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }

          /* Slightly tighten caption spacing */
          div[data-testid="stCaptionContainer"] p { margin-bottom: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
    if "fires_cache" not in st.session_state:
        st.session_state.fires_cache = {}
    if "map_refresh_requested" not in st.session_state:
        st.session_state.map_refresh_requested = False

    # App identity
    st.title("Wildfire Nowcast & Forecast")
    st.caption(
        "Live satellite fire detections (FIRMS) with optional probabilistic spread overlays."
    )
    st.info(
        "Forecast overlays are **experimental** and **probabilistic** (not deterministic). "
        "Use them as situational awareness, not as operational guidance.",
        icon="ℹ️",
    )

    # Sidebar controls
    with st.sidebar:
        render_sidebar()

    # Main content area - Map and details
    st.subheader("Map")

    # Active filters summary
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
        # Only update last_click if click actually happened
        if click_coords is not None:
            # Check if click is actually different to avoid unnecessary updates
            current_click = st.session_state.get("last_click")
            if (current_click is None or 
                current_click.get("lat") != click_coords.get("lat") or 
                current_click.get("lng") != click_coords.get("lng")):
                st.session_state.last_click = click_coords

    with col_details:
        st.subheader("Details")
        render_click_details(st.session_state.last_click)
        render_legend()

if __name__ == "__main__":
    main()
