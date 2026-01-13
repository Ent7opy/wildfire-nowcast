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

        pending_min_confidence = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.fires_min_confidence),
            step=5.0,
            key="pending_min_confidence",
        )

        pending_apply_denoiser = st.checkbox(
            "Apply noise filter (exclude false alarms)",
            value=bool(st.session_state.fires_apply_denoiser),
            key="pending_apply_denoiser",
        )

        applied = st.form_submit_button("Apply filters", type="primary")
        if applied:
            st.session_state.time_window = pending_time_window
            st.session_state.fires_min_confidence = float(pending_min_confidence)
            st.session_state.fires_apply_denoiser = bool(pending_apply_denoiser)

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
