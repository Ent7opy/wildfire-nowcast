"""Main Streamlit application for Wildfire Nowcast & Forecast."""

import streamlit as st

from state import app_state

# Import components
from components.sidebar import render_sidebar
from components.map_view import render_map_view
from components.click_details import render_click_details
from components.forecast_status import render_forecast_status_polling


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

    # Initialize centralized state (loads URL params on first run, restores on reruns)
    app_state.initialize()

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
    if app_state.forecast_job.job_id:
        with st.container():
            render_forecast_status_polling(app_state.forecast_job.job_id)

    # Main content area - Map and details
    st.subheader("Map")

    # Active filters summary
    filter_state = "on" if app_state.filters.apply_denoiser else "off"
    st.caption(
        f"**Fires filters:** {app_state.time_window}, "
        f"likelihood at least {app_state.filters.min_likelihood:.2f}, "
        f"noise filter {filter_state}"
    )

    # Render map + details side-by-side
    col_map, col_details = st.columns([3, 1], gap="large")
    with col_map:
        click_coords = render_map_view()
        app_state.selection.update_click(click_coords)
        app_state._persist()

        # Render floating legend overlay
        from components.legend import get_legend_html
        legend_html = get_legend_html()
        if legend_html:
            st.markdown(legend_html, unsafe_allow_html=True)

    with col_details:
        st.subheader("Details")
        render_click_details(app_state.selection.last_click)

if __name__ == "__main__":
    main()
