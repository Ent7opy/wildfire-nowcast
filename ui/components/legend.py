"""Legend component for map layers."""

import streamlit as st

def render_legend() -> None:
    """Render the map legend based on active layers."""
    with st.expander("Legend", expanded=False):
        legend_items = []
        if st.session_state.show_fires:
            legend_items.append("🔥 **Active fires** — size and color indicate intensity (FRP)")
        if st.session_state.show_forecast:
            legend_items.append("🟠 **Forecast overlay** — spread outlook")
            legend_items.append(
                "**Contours by horizon**: "
                "T+24h (blue), T+48h (orange), T+72h (red)"
            )
            legend_items.append("Shaded layer: higher = more likely spread")
        if st.session_state.show_risk:
            legend_items.append("**🔥 Risk index** — fire risk heatmap:")
            legend_items.append("  • 🟢 Low (0.0-0.3): minimal fire risk")
            legend_items.append("  • 🟡 Medium (0.3-0.6): moderate fire risk")
            legend_items.append("  • 🔴 High (0.6-1.0): elevated fire risk")
            legend_items.append("  Based on land cover + weather conditions")

        if legend_items:
            for item in legend_items:
                st.markdown(f"- {item}")
        else:
            st.caption("No layers currently visible. Enable layers in the sidebar to see placeholders.")
