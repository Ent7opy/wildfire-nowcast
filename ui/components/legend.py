"""Legend component for map layers."""

import streamlit as st

def render_legend() -> None:
    """Render the map legend based on active layers."""
    with st.expander("Legend", expanded=False):
        legend_items = []
        if st.session_state.show_fires:
            legend_items.append("🔴 **Active fires** (red markers/clusters) - Live detections")
        if st.session_state.show_forecast:
            legend_items.append("🟠 **Forecast overlay** (viewport AOI)")
            legend_items.append(
                "**Contours by horizon**: "
                "T+24h (blue), T+48h (orange), T+72h (red)"
            )
            legend_items.append("**Raster**: probability tiles (semi-transparent)")
        if st.session_state.show_risk:
            legend_items.append("🟣 **Risk index** (purple polygon) - Placeholder fire risk")

        if legend_items:
            for item in legend_items:
                st.markdown(f"- {item}")
        else:
            st.caption("No layers currently visible. Enable layers in the sidebar to see placeholders.")
