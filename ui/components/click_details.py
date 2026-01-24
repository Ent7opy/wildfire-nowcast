"""Click-to-inspect details panel for fire detections using PyDeck selection."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st

from api_client import (
    ApiError,
    ApiUnavailableError,
    create_jit_forecast,
)

logger = logging.getLogger(__name__)


def _render_progress_bar(label: str, value: float, max_value: float = 1.0) -> str:
    """Generate HTML for a horizontal progress bar.

    Args:
        label: Text label for the bar
        value: Current value (0.0-1.0)
        max_value: Maximum value (default 1.0)

    Returns:
        HTML string for the progress bar
    """
    from config.theme import FireThresholds

    percentage = (value / max_value) * 100

    # Color based on value (using fire likelihood thresholds)
    if value >= FireThresholds.HIGH:
        color = "#DC143C"  # Crimson (high)
    elif value >= FireThresholds.MEDIUM:
        color = "#FFA500"  # Orange (medium)
    else:
        color = "#FFD700"  # Gold (low)

    return f"""
    <div style="margin: 6px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
            <span style="font-size: 12px; color: #555;">{label}</span>
            <span style="font-size: 12px; font-weight: bold;">{value:.3f}</span>
        </div>
        <div style="width: 100%; background: #e0e0e0; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="width: {percentage}%; background: {color}; height: 100%; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """


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

    # Display fire likelihood and component scores
    fire_likelihood = det.get("fire_likelihood")
    if fire_likelihood is not None:
        st.divider()
        st.write("**Fire Likelihood**")
        try:
            likelihood_val = float(fire_likelihood)

            # Composite score with progress bar
            composite_html = _render_progress_bar("Composite Score", likelihood_val)
            st.markdown(composite_html, unsafe_allow_html=True)

            # Component scores with smaller progress bars
            st.caption("**Component Breakdown:**")

            component_bars = []

            confidence_score = det.get("confidence_score")
            if confidence_score is not None:
                component_bars.append(_render_progress_bar("Confidence (20%)", float(confidence_score)))

            persistence_score = det.get("persistence_score")
            if persistence_score is not None:
                component_bars.append(_render_progress_bar("Persistence (30%)", float(persistence_score)))

            landcover_score = det.get("landcover_score")
            if landcover_score is not None:
                component_bars.append(_render_progress_bar("Land Cover (25%)", float(landcover_score)))

            weather_score = det.get("weather_score")
            if weather_score is not None:
                component_bars.append(_render_progress_bar("Weather (25%)", float(weather_score)))

            if component_bars:
                components_html = f"""
                <div style="margin-top: 8px; padding: 8px; background: #f8f9fa; border-radius: 4px;">
                    {''.join(component_bars)}
                </div>
                """
                st.markdown(components_html, unsafe_allow_html=True)

            false_source_masked = det.get("false_source_masked")
            if false_source_masked is not None:
                masked_str = "Yes" if false_source_masked else "No"
                st.caption(f"Industrial Source Masked: {masked_str}")

        except (ValueError, TypeError):
            st.write(f"**Composite Score:** {fire_likelihood}")

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
    
    # Validate and create a bbox around the fire (50km radius)
    if lat is not None and lon is not None:
        # Validate coordinate ranges
        try:
            fire_lat = float(lat)
            fire_lon = float(lon)

            if not (-90 <= fire_lat <= 90):
                st.error(f"Invalid latitude: {fire_lat} (must be between -90 and 90)")
                return
            if not (-180 <= fire_lon <= 180):
                st.error(f"Invalid longitude: {fire_lon} (must be between -180 and 180)")
                return
        except (ValueError, TypeError):
            st.error(f"Invalid coordinates: lat={lat}, lon={lon}")
            return

        radius_deg = 50.0 / 111.0  # Approximate: 1 degree ≈ 111 km
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

                logger.info(
                    "Generating forecast for fire: lat=%.4f, lon=%.4f, bbox=%s",
                    fire_lat, fire_lon, forecast_bbox
                )

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
                    logger.error("Forecast job creation returned no job_id")
                    st.error("Failed to start forecast: no job ID returned")
            except ApiUnavailableError:
                st.error("Data service is unavailable right now. Please try again in a moment.")
            except ApiError as e:
                logger.error(
                    "Forecast generation failed: status=%s, response=%s, bbox=%s",
                    e.status_code, e.response_text, forecast_bbox
                )
                details = f"(status={e.status_code})" if e.status_code is not None else ""
                st.error(f"Forecast generation failed {details}".strip())
                if e.response_text:
                    st.caption(str(e.response_text)[:300])
        
        if is_forecast_running:
            st.caption("Forecast in progress...")
    else:
        st.warning("Selected fire is missing coordinates. Cannot generate forecast.")
