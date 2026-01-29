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

    # Comprehensive dark theme + typography styling
    st.markdown(
        """
        <style>
          /* ── Design Tokens ──────────────────────────────────────────── */
          :root {
            --bg-primary: #0a1628;
            --bg-secondary: #1a1d29;
            --bg-card: #252930;
            --accent-ember: #ff6b35;
            --accent-warning: #e63946;
            --accent-amber: #fbbf24;
            --text-primary: #e0e0e0;
            --text-secondary: rgba(255,255,255,0.7);
            --border-subtle: rgba(255,255,255,0.08);
            --radius-card: 8px;
            --radius-pill: 20px;
            --font-stack: 'Inter', -apple-system, 'Segoe UI', 'Roboto', sans-serif;

            /* Fire gradient */
            --fire-very-high: rgb(220, 38, 38);
            --fire-high: rgb(239, 68, 68);
            --fire-medium: rgb(255, 107, 53);
            --fire-low: rgb(251, 191, 36);
            --fire-very-low: rgb(253, 224, 71);

            /* Risk */
            --risk-low: rgb(34, 139, 34);
            --risk-medium: rgb(255, 215, 0);
            --risk-high: rgb(220, 20, 60);
          }

          /* ── Typography ─────────────────────────────────────────────── */
          html, body, [data-testid="stAppViewContainer"],
          .stMarkdown, .stText, p, label {
            font-family: var(--font-stack) !important;
          }

          /* Apply font to span/div but exclude Material icon elements */
          span:not([class*="material"]):not([data-testid*="Icon"]),
          div:not([class*="material"]) {
            font-family: var(--font-stack);
          }

          h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: var(--font-stack) !important;
            font-weight: 600 !important;
            font-size: 18px !important;
            line-height: 1.3 !important;
          }

          h1 { font-size: 24px !important; }

          p, .stMarkdown p, [data-testid="stText"] {
            font-size: 14px !important;
            line-height: 16px !important;
          }

          /* ── Layout spacing ─────────────────────────────────────────── */
          .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            transition: opacity 0.15s ease;
          }

          section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
          }

          div[data-testid="stCaptionContainer"] p {
            margin-bottom: 0.35rem;
          }

          /* ── Sidebar card containers ────────────────────────────────── */
          section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--bg-card) !important;
            border-radius: var(--radius-card) !important;
            border: 1px solid var(--border-subtle) !important;
            padding: 12px !important;
            margin-bottom: 8px !important;
          }

          /* ── Pill-shaped sidebar buttons ─────────────────────────────── */
          section[data-testid="stSidebar"] .stButton > button {
            border-radius: var(--radius-pill) !important;
            font-size: 13px !important;
            padding: 6px 16px !important;
            transition: all 0.2s ease !important;
          }

          /* Active preset glow */
          section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            box-shadow: 0 0 12px rgba(255, 107, 53, 0.4) !important;
            background-color: var(--accent-ember) !important;
            border-color: var(--accent-ember) !important;
          }

          section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-subtle) !important;
            color: var(--text-primary) !important;
          }

          section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background-color: rgba(255, 107, 53, 0.1) !important;
            border-color: var(--accent-ember) !important;
          }

          /* ── Slider styling ─────────────────────────────────────────── */
          [data-testid="stSlider"] [role="slider"] {
            background-color: var(--accent-ember) !important;
          }

          [data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: var(--accent-ember) !important;
          }

          /* ── Export / link button styling ────────────────────────────── */
          section[data-testid="stSidebar"] .stLinkButton a {
            border-radius: var(--radius-card) !important;
            transition: all 0.2s ease !important;
            background: var(--bg-card) !important;
            border: 1px solid var(--border-subtle) !important;
            color: var(--text-primary) !important;
          }

          section[data-testid="stSidebar"] .stLinkButton a:hover {
            background: rgba(255, 107, 53, 0.15) !important;
            border-color: var(--accent-ember) !important;
            box-shadow: 0 2px 8px rgba(255, 107, 53, 0.2) !important;
          }

          /* ── Prominent forecast button ──────────────────────────────── */
          button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #ff6b35, #e63946) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3) !important;
            font-size: 15px !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            transition: all 0.2s ease !important;
          }

          button[data-testid="stBaseButton-primary"]:hover {
            box-shadow: 0 6px 16px rgba(255, 107, 53, 0.5) !important;
            transform: translateY(-1px) !important;
          }

          /* ── Floating Legend (dark) ──────────────────────────────────── */
          #floating-legend {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(26, 29, 41, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: var(--radius-card);
            padding: 12px 16px;
            font-size: 13px;
            line-height: 1.5;
            color: var(--text-primary);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            z-index: 1000;
            max-width: 280px;
          }

          /* ── Slide-in animation for details panel ───────────────────── */
          @keyframes slideIn {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
          }

          [data-testid="column"]:last-child {
            animation: slideIn 0.3s ease-out;
          }

          /* ── Toggle styling ─────────────────────────────────────────── */
          [data-testid="stToggle"] label span {
            color: var(--text-primary) !important;
          }
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
        render_click_details(app_state.selection.last_click)

if __name__ == "__main__":
    main()
