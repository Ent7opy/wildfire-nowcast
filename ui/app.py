"""Main Streamlit application for Wildfire Nowcast & Forecast."""

import streamlit as st
from config.constants import TIME_WINDOW_OPTIONS

# Import components
from components.sidebar import render_sidebar
from components.map_view import render_map_view
from components.click_details import render_click_details
from components.forecast_status import render_forecast_status_polling


def _get_matching_preset() -> str | None:
    """Return the name of a preset that matches current filter state, or None."""
    from config.theme import FilterPresets

    current_start = st.session_state.get("time_range_hours_start", 24)
    current_end = st.session_state.get("time_range_hours_end", 0)
    current_likelihood = st.session_state.get("fires_min_likelihood", 0.0)
    current_denoiser = st.session_state.get("fires_apply_denoiser", True)

    for name, hours_start, hours_end, likelihood, denoiser in FilterPresets.all_presets():
        if (hours_start == current_start and
            hours_end == current_end and
            abs(likelihood - current_likelihood) < 0.01 and
            denoiser == current_denoiser):
            return name
    return None


def _load_filters_from_url():
    """Load filter state from URL query parameters on first load."""
    params = st.query_params

    # Load individual filter values from URL
    if "start" in params:
        try:
            st.session_state.time_range_hours_start = int(params["start"])
        except ValueError:
            pass
    if "end" in params:
        try:
            st.session_state.time_range_hours_end = int(params["end"])
        except ValueError:
            pass
    if "likelihood" in params:
        try:
            st.session_state.fires_min_likelihood = float(params["likelihood"])
        except ValueError:
            pass
    if "denoiser" in params:
        st.session_state.fires_apply_denoiser = params["denoiser"].lower() == "true"

    # Compute time_window string from loaded values
    start_hours = st.session_state.get("time_range_hours_start", 24)
    end_hours = st.session_state.get("time_range_hours_end", 0)
    hours_window = start_hours - end_hours
    if hours_window <= 6:
        st.session_state.time_window = "Last 6 hours"
    elif hours_window <= 12:
        st.session_state.time_window = "Last 12 hours"
    elif hours_window <= 24:
        st.session_state.time_window = "Last 24 hours"
    else:
        st.session_state.time_window = "Last 48 hours"

    # Determine active preset based on loaded filter values
    # This ensures preset button highlights correctly when loading from URL
    matching_preset = _get_matching_preset()
    if matching_preset:
        st.session_state.active_preset = matching_preset
    elif any(key in params for key in ["start", "end", "likelihood", "denoiser"]):
        # If URL has filter params but they don't match any preset, mark as Custom
        st.session_state.active_preset = "Custom"

def main() -> None:
    """Main application entry point."""
    # Page configuration - must be the first Streamlit command
    st.set_page_config(page_title="Wildfire Nowcast & Forecast", layout="wide")

    # Minimal styling (Streamlit-supported CSS injection)
    st.markdown(
        """
        <style>
          /* CSS Custom Properties (for future theming capabilities) */
          :root {
            --fire-high: rgb(255, 0, 0);
            --fire-medium: rgb(255, 165, 0);
            --fire-low: rgb(255, 255, 0);
            --risk-low: rgb(34, 139, 34);
            --risk-medium: rgb(255, 215, 0);
            --risk-high: rgb(220, 20, 60);
            --tooltip-bg: #333;
            --tooltip-text: white;
          }

          /* Floating Legend */
          #floating-legend {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 13px;
            line-height: 1.5;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
            z-index: 1000;
            max-width: 280px;
          }

          /* Tighten overall top spacing a bit */
          .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }

          /* Make sidebars feel less "boilerplate" */
          section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }

          /* Slightly tighten caption spacing */
          div[data-testid="stCaptionContainer"] p { margin-bottom: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load filters from URL on first load (before setting defaults)
    if "time_window" not in st.session_state:
        _load_filters_from_url()

    # Initialize session state with defaults for any missing values
    if "time_window" not in st.session_state:
        st.session_state.time_window = TIME_WINDOW_OPTIONS[0]
    if "fires_min_likelihood" not in st.session_state:
        st.session_state.fires_min_likelihood = 0.0
    if "fires_apply_denoiser" not in st.session_state:
        st.session_state.fires_apply_denoiser = True
    if "time_range_hours_start" not in st.session_state:
        st.session_state.time_range_hours_start = 24  # 24 hours ago
    if "time_range_hours_end" not in st.session_state:
        st.session_state.time_range_hours_end = 0  # Now
    if "active_preset" not in st.session_state:
        st.session_state.active_preset = None
    if "show_fires" not in st.session_state:
        st.session_state.show_fires = True
    if "show_forecast" not in st.session_state:
        st.session_state.show_forecast = False
    if "show_risk" not in st.session_state:
        st.session_state.show_risk = False
    if "last_click" not in st.session_state:
        st.session_state.last_click = None
    if "selected_fire" not in st.session_state:
        st.session_state.selected_fire = None

    # App identity
    st.title("Wildfire Nowcast & Forecast")
    st.caption(
        "Live satellite fire detections with optional spread overlays."
    )
    st.info(
        "Forecast overlays are **experimental** and **probabilistic** (not deterministic). "
        "Use them as situational awareness, not as operational guidance.",
        icon="ℹ️",
    )

    # Sidebar controls
    with st.sidebar:
        render_sidebar()

    # Check for ongoing JIT forecast polling - display as status banner
    if st.session_state.get("jit_job_id"):
        with st.container():
            render_forecast_status_polling(st.session_state.jit_job_id)

    # Main content area - Map and details
    st.subheader("Map")

    # Active filters summary
    filter_state = "on" if st.session_state.fires_apply_denoiser else "off"
    st.caption(
        f"**Fires filters:** {st.session_state.time_window}, "
        f"likelihood at least {st.session_state.fires_min_likelihood:.2f}, "
        f"noise filter {filter_state}"
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

        # Render floating legend overlay
        from components.legend import get_legend_html
        legend_html = get_legend_html()
        if legend_html:
            st.markdown(legend_html, unsafe_allow_html=True)

    with col_details:
        st.subheader("Details")
        render_click_details(st.session_state.last_click)

if __name__ == "__main__":
    main()
