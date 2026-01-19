"""Click-to-inspect details panel for fire detections using PyDeck selection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st

from api_client import (
    ApiError,
    ApiUnavailableError,
    create_jit_forecast,
)

def _parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None

def render_click_details(last_click: Optional[Dict[str, float]]) -> None:
    """Render details for the selected fire based on PyDeck selection."""
    st.subheader("Fire details")

    # Use the selected fire from session state (set by map_view)
    det = st.session_state.get("selected_fire")

    if not det:
        if last_click is None:
            st.caption("Click a fire on the map to inspect details.")
        else:
            st.info("No fire data selected. Try clicking exactly on a fire marker.")
        return

    # Display fire metadata
    st.write("**Selection:** Fire detection")
    
    # We now get lat/lon directly from the MVT properties (thanks to backend update)
    lat = det.get("lat")
    lon = det.get("lon")
    if lat is not None and lon is not None:
        st.caption(f"Location: {lat:.4f}, {lon:.4f}")
    
    acq_time = det.get("acq_time")
    st.write(f"**Timestamp:** {acq_time}")
    st.write(f"**Satellite:** {det.get('sensor')}")
    st.write(f"**Confidence:** {det.get('confidence')}")
    st.write(f"**Fire intensity (FRP):** {det.get('frp')}")
    st.write(f"**Source:** {det.get('source')}")

    if "denoised_score" in det or "is_noise" in det:
        st.divider()
        st.write("**Noise Filter**")
        
        denoised_score = det.get("denoised_score")
        if denoised_score is not None:
            try:
                st.write(f"**Denoised score:** {float(denoised_score):.4f}")
            except (ValueError, TypeError):
                st.write(f"**Denoised score:** {denoised_score}")

        is_noise = det.get("is_noise")
        if is_noise is not None:
            # Ensure boolean treatment
            if isinstance(is_noise, str):
                is_noise_bool = is_noise.lower() == "true"
            else:
                is_noise_bool = bool(is_noise)
            st.write(f"**Is noise:** {is_noise_bool}")
    
    # Generate forecast button
    st.divider()
    st.write("**Forecast**")
    
    # Create a bbox around the fire (50km radius)
    if lat is not None and lon is not None:
        radius_deg = 50.0 / 111.0  # Approximate: 1 degree ≈ 111 km
        fire_lat = float(lat)
        fire_lon = float(lon)
        forecast_bbox = (
            fire_lon - radius_deg,  # min_lon
            fire_lat - radius_deg,  # min_lat
            fire_lon + radius_deg,  # max_lon
            fire_lat + radius_deg,  # max_lat
        )
        
        # Disable button if a JIT forecast is currently running
        is_forecast_running = st.session_state.get("jit_job_id") is not None
        
        if st.button(
            "Generate Spread Forecast",
            key="generate_forecast_btn",
            disabled=is_forecast_running,
        ):
            try:
                ref_time = _parse_time(acq_time)
                if ref_time is None:
                    ref_time = datetime.now(timezone.utc)
                elif ref_time.tzinfo is None:
                    ref_time = ref_time.replace(tzinfo=timezone.utc)
                
                job_data = create_jit_forecast(
                    bbox=forecast_bbox,
                    horizons=[24, 48, 72],
                    forecast_reference_time=ref_time,
                )
                
                job_id = job_data.get("job_id")
                if job_id:
                    st.session_state.jit_job_id = job_id
                    st.success("Forecast job queued successfully!")
                    st.rerun()
                else:
                    st.error("Failed to start forecast: no job ID returned")
            except ApiUnavailableError:
                st.error("Data service is unavailable right now. Please try again in a moment.")
            except ApiError as e:
                details = f"(status={e.status_code})" if e.status_code is not None else ""
                st.error(f"Forecast generation failed {details}".strip())
                if e.response_text:
                    st.caption(str(e.response_text)[:300])
        
        if is_forecast_running:
            st.caption("Forecast in progress...")
    else:
        st.warning("Selected fire is missing coordinates. Cannot generate forecast.")
