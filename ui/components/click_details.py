"""Click details component for map interactions."""

import streamlit as st
from typing import Optional, Dict

def render_click_details(last_click: Optional[Dict[str, float]]) -> None:
    """Render the click details panel."""
    st.divider()
    if last_click is not None:
        st.subheader("Map Click Details")
        st.write(
            f"**Coordinates:** {last_click['lat']:.4f}, "
            f"{last_click['lng']:.4f}"
        )
        st.info("Fire details will appear here in a future version.")
    else:
        st.caption("Click on the map to see location details (placeholder)")
