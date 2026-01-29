"""Click-to-inspect details panel for fire detections using PyDeck selection."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import streamlit as st

from state import app_state, isoformat
from api_client import (
    ApiError,
    ApiUnavailableError,
    create_jit_forecast,
)

logger = logging.getLogger(__name__)


def _render_progress_bar(label: str, value: float, max_value: float = 1.0) -> str:
    """Generate HTML for a horizontal progress bar."""
    from config.theme import DarkTheme

    percentage = (value / max_value) * 100

    if value >= 0.6:
        color = DarkTheme.ACCENT_WARNING   # Red for high
    elif value >= 0.4:
        color = DarkTheme.ACCENT_EMBER     # Orange for medium
    else:
        color = DarkTheme.ACCENT_AMBER     # Amber for low

    return (
        f'<div style="margin:6px 0;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:2px;">'
        f'<span style="font-size:12px;color:rgba(255,255,255,0.7);">{label}</span>'
        f'<span style="font-size:12px;font-weight:bold;color:rgba(255,255,255,0.9);">{value:.3f}</span>'
        f'</div>'
        f'<div style="width:100%;background:rgba(255,255,255,0.15);border-radius:4px;height:8px;overflow:hidden;">'
        f'<div style="width:{percentage}%;background:{color};height:100%;border-radius:4px;"></div>'
        f'</div></div>'
    )


def _render_confidence_gauge(value: float) -> str:
    """Generate an SVG semicircular gauge for confidence value."""
    from config.theme import DarkTheme

    if value >= 0.6:
        color = DarkTheme.ACCENT_WARNING
    elif value >= 0.4:
        color = DarkTheme.ACCENT_EMBER
    else:
        color = DarkTheme.ACCENT_AMBER

    percentage = value * 100
    # SVG arc: semicircle from left to right
    # Center at (60, 55), radius 40, arc from 180deg to 0deg
    radius = 40
    cx, cy = 60, 55

    # Full arc background (gray)
    bg_start_x = cx - radius
    bg_start_y = cy
    bg_end_x = cx + radius
    bg_end_y = cy

    # Value arc: angle from 180deg (left) sweeping clockwise by value * 180deg
    angle_deg = value * 180.0
    angle_rad = math.radians(180.0 - angle_deg)
    val_end_x = cx + radius * math.cos(angle_rad)
    val_end_y = cy - radius * math.sin(angle_rad)
    large_arc = 1 if angle_deg > 180 else 0

    return (
        f'<div style="text-align:center;margin:8px 0;">'
        f'<svg width="120" height="72" viewBox="0 0 120 72">'
        # Background arc (dark)
        f'<path d="M {bg_start_x} {bg_start_y} A {radius} {radius} 0 0 1 {bg_end_x} {bg_end_y}" '
        f'stroke="rgba(255,255,255,0.1)" stroke-width="10" fill="none" stroke-linecap="round"/>'
        # Value arc
        f'<path d="M {bg_start_x} {bg_start_y} A {radius} {radius} 0 {large_arc} 1 {val_end_x:.1f} {val_end_y:.1f}" '
        f'stroke="{color}" stroke-width="10" fill="none" stroke-linecap="round"/>'
        # Center text
        f'<text x="60" y="58" text-anchor="middle" fill="#e0e0e0" '
        f'font-size="16" font-weight="bold" font-family="Inter,sans-serif">'
        f'{percentage:.0f}%</text>'
        f'<text x="60" y="70" text-anchor="middle" fill="rgba(255,255,255,0.5)" '
        f'font-size="9" font-family="Inter,sans-serif">likelihood</text>'
        f'</svg></div>'
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


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_aggregate_stats(
    bbox: Tuple[float, float, float, float],
    start_iso: str,
    end_iso: str,
    min_likelihood: float,
    include_noise: str,
) -> Dict[str, Any]:
    """Fetch aggregate fire stats from the API with caching.

    Returns ``{"ok": True, "data": {...}}`` on success or
    ``{"ok": False, "error": "..."}`` on failure.
    """
    from api_client import get_fires

    try:
        data = get_fires(
            bbox=bbox,
            time_range=(
                datetime.fromisoformat(start_iso),
                datetime.fromisoformat(end_iso),
            ),
            filters={
                "min_fire_likelihood": min_likelihood,
                "include_noise": include_noise,
                "limit": 1000,
            },
        )
        return {"ok": True, "data": data}
    except Exception as exc:
        logger.warning("Failed to fetch aggregate stats: %s", exc, exc_info=True)
        return {"ok": False, "error": str(exc)}


def _render_aggregate_stats() -> None:
    """Render aggregate statistics when no fire is selected."""
    bbox = app_state.viewport_bbox
    start_time, end_time = app_state.time_range

    result = _fetch_aggregate_stats(
        bbox=bbox,
        start_iso=isoformat(start_time),
        end_iso=isoformat(end_time),
        min_likelihood=app_state.filters.min_likelihood,
        include_noise=str(not app_state.filters.apply_denoiser).lower(),
    )

    if not result.get("ok"):
        error_msg = result.get("error", "Unknown error")
        st.caption(f"\u26a0\ufe0f Could not load stats: {error_msg}")
        st.caption("Click a fire on the map to inspect details.")
        return

    data = result["data"]
    count = data.get("count", 0)
    detections = data.get("detections", [])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active fires", count)
    with col2:
        if detections:
            max_lh = max(
                (float(d.get("fire_likelihood") or 0) for d in detections),
                default=0,
            )
            st.metric("Highest confidence", f"{max_lh:.2f}")
        else:
            st.metric("Highest confidence", "N/A")

    col3, col4 = st.columns(2)
    with col3:
        if detections:
            times = [_parse_time(d.get("acq_time")) for d in detections]
            valid_times = [t for t in times if t is not None]
            if valid_times:
                most_recent = max(valid_times)
                st.metric("Most recent", most_recent.strftime("%H:%M UTC"))
            else:
                st.metric("Most recent", "N/A")
        else:
            st.metric("Most recent", "N/A")
    with col4:
        st.metric("Time window", app_state.time_window)


def render_click_details(last_click: Optional[Dict[str, float]]) -> None:
    """Render details for the selected fire based on PyDeck selection."""
    # Use the selected fire from state manager (set by map_view)
    det = app_state.selection.selected_fire

    if not det:
        # Show aggregate stats when no fire is selected
        st.subheader("Overview")
        if last_click is None:
            _render_aggregate_stats()
        else:
            st.info("No fire data selected. Try clicking exactly on a fire marker.")
        return

    # ── Prominent forecast button at the top ──────────────────────────
    lat = det.get("lat")
    lon = det.get("lon")
    _render_forecast_section(lat, lon, det.get("acq_time"))

    st.subheader("Fire details")

    # Display fire metadata
    st.write("**Selection:** Fire detection")

    if lat is not None and lon is not None:
        st.caption(f"Location: {lat:.4f}, {lon:.4f}")

    acq_time = det.get("acq_time")
    st.write(f"**Timestamp:** {acq_time}")
    st.write(f"**Satellite:** {det.get('sensor')}")
    st.write(f"**Confidence:** {det.get('confidence')}")
    st.write(f"**Fire intensity (FRP):** {det.get('frp')}")
    st.write(f"**Source:** {det.get('source')}")

    # Display fire likelihood with visual gauge and component scores
    fire_likelihood = det.get("fire_likelihood")
    if fire_likelihood is not None:
        st.divider()
        st.write("**Fire Likelihood**")
        try:
            likelihood_val = float(fire_likelihood)

            # SVG confidence gauge
            gauge_html = _render_confidence_gauge(likelihood_val)
            st.markdown(gauge_html, unsafe_allow_html=True)

            # Component scores with progress bars
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
                components_html = (
                    '<div style="margin-top:8px;padding:8px;'
                    'background:rgba(255,255,255,0.05);border-radius:8px;'
                    'border:1px solid rgba(255,255,255,0.08);">'
                    + ''.join(component_bars) +
                    '</div>'
                )
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
            if isinstance(is_noise, str):
                is_noise_bool = is_noise.lower() == "true"
            else:
                is_noise_bool = bool(is_noise)
            st.write(f"**Is noise:** {is_noise_bool}")


def _render_forecast_section(
    lat: Any, lon: Any, acq_time: Any
) -> None:
    """Render the prominent forecast button at the top of the details panel."""
    if lat is None or lon is None:
        st.warning("Selected fire is missing coordinates. Cannot generate forecast.")
        return

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

    radius_deg = 50.0 / 111.0  # Approximate: 1 degree ~ 111 km
    forecast_bbox = (
        fire_lon - radius_deg,
        fire_lat - radius_deg,
        fire_lon + radius_deg,
        fire_lat + radius_deg,
    )

    is_forecast_running = app_state.forecast_job.job_id is not None

    if st.button(
        "Generate Spread Forecast",
        key="generate_forecast_btn",
        disabled=is_forecast_running,
        type="primary",
        use_container_width=True,
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
                app_state.forecast_job.start(job_id)
                app_state._persist()
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
