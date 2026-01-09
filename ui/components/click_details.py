"""Click-to-inspect details panel for fire detections."""

from __future__ import annotations

from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional

import streamlit as st

from api_client import ApiError, ApiUnavailableError, generate_forecast


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        s = value.strip()
        # Backend commonly returns ISO strings with "Z"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _nearby_detections(
    detections: List[Dict[str, Any]],
    click_lat: float,
    click_lng: float,
    radius_km: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for det in detections:
        lat = det.get("lat")
        lon = det.get("lon")
        if lat is None or lon is None:
            continue
        try:
            d_km = _haversine_km(float(click_lat), float(click_lng), float(lat), float(lon))
        except (TypeError, ValueError):
            continue
        if d_km <= radius_km:
            out.append(det)
    return out


def _nearest_detection(
    detections: List[Dict[str, Any]], click_lat: float, click_lng: float
) -> Optional[Dict[str, Any]]:
    best = None
    best_km = None
    for det in detections:
        lat = det.get("lat")
        lon = det.get("lon")
        if lat is None or lon is None:
            continue
        try:
            d_km = _haversine_km(float(click_lat), float(click_lng), float(lat), float(lon))
        except (TypeError, ValueError):
            continue
        if best_km is None or d_km < best_km:
            best = det
            best_km = d_km
    return best


def render_click_details(last_click: Optional[Dict[str, float]]) -> None:
    """Render details for the selected fire (or nearby cluster)."""
    st.subheader("Fire details")

    if last_click is None:
        st.caption("Click a fire on the map to inspect details.")
        return

    click_lat = float(last_click["lat"])
    click_lng = float(last_click["lng"])
    st.caption(f"Clicked: {click_lat:.4f}, {click_lng:.4f}")

    detections: List[Dict[str, Any]] = st.session_state.get("fires_last_detections", [])
    if not detections:
        st.info("No fire data available for inspection (try enabling ‘Active fires’).")
        return

    # Treat "cluster" as multiple detections near the click point.
    cluster_radius_km = 5.0
    select_radius_km = 10.0

    nearby = _nearby_detections(detections, click_lat, click_lng, radius_km=cluster_radius_km)
    if len(nearby) >= 2:
        times = sorted(t for t in (_parse_time(d.get("acq_time")) for d in nearby) if t is not None)
        first_t = times[0] if times else None
        last_t = times[-1] if times else None

        st.write("**Selection:** Cluster")
        st.metric("Detections", len(nearby))
        if first_t and last_t:
            st.write(f"**Time span:** {first_t.isoformat()} → {last_t.isoformat()}")
        elif last_t:
            st.write(f"**Time:** {last_t.isoformat()}")

        # Small, cheap “timeline”: just show first/last + a few sample times
        if times:
            if len(times) <= 6:
                preview = [t.isoformat() for t in times]
            else:
                preview = [t.isoformat() for t in times[:3]]
                preview.append("…")
                preview.extend(t.isoformat() for t in times[-3:])
            st.caption("Times (preview): " + ", ".join(preview))
        return

    # Otherwise, try to pick the nearest detection.
    det = _nearest_detection(detections, click_lat, click_lng)
    if not det:
        st.info("No fire detected near this location. Zoom in and click closer to a marker/cluster.")
        return

    # Guard against clicks far away from any detection.
    lat = det.get("lat")
    lon = det.get("lon")
    if lat is None or lon is None:
        st.info("No fire detected near this location.")
        return
    d_km = _haversine_km(click_lat, click_lng, float(lat), float(lon))
    if d_km > select_radius_km:
        st.info("No fire detected near this location. Zoom in and click closer to a marker/cluster.")
        return

    st.write("**Selection:** Fire detection")
    st.caption(f"Match distance: {d_km:.1f} km")

    acq_time = det.get("acq_time")
    st.write(f"**Timestamp:** {acq_time}")
    st.write(f"**Sensor:** {det.get('sensor')}")
    st.write(f"**Confidence:** {det.get('confidence')}")
    st.write(f"**Brightness:** {det.get('brightness')}")
    st.write(f"**Brightness (T31):** {det.get('bright_t31')}")

    if "denoised_score" in det or "is_noise" in det:
        st.divider()
        st.write("**Denoiser**")
        st.write(f"**Denoised score:** {det.get('denoised_score')}")
        st.write(f"**Is noise:** {det.get('is_noise')}")
    
    # Generate forecast button
    st.divider()
    st.write("**Forecast**")
    
    # Create a bbox around the fire (50km radius)
    radius_deg = 50.0 / 111.0  # Approximate: 1 degree ≈ 111 km
    fire_lat = float(lat)
    fire_lon = float(lon)
    forecast_bbox = (
        fire_lon - radius_deg,  # min_lon
        fire_lat - radius_deg,  # min_lat
        fire_lon + radius_deg,  # max_lon
        fire_lat + radius_deg,  # max_lat
    )
    
    if st.button("Generate Spread Forecast", key="generate_forecast_btn"):
        try:
            with st.spinner("Generating forecast (this may take a moment)…"):
                # Use the fire's acquisition time as reference, or current time
                ref_time = _parse_time(acq_time)
                if ref_time is None:
                    ref_time = datetime.now(timezone.utc)
                elif ref_time.tzinfo is None:
                    ref_time = ref_time.replace(tzinfo=timezone.utc)
                
                forecast_data = generate_forecast(
                    bbox=forecast_bbox,
                    horizons=[24, 48, 72],
                    region_name=None,  # Location-based (no region)
                    forecast_reference_time=ref_time,
                )
                
                st.success("Forecast generated successfully!")
                st.json(forecast_data.get("forecast", {}))
                
                # Store forecast in session state so map_view can display it
                st.session_state.last_forecast = forecast_data
                st.session_state.last_forecast_bbox = forecast_bbox
        except ApiUnavailableError:
            st.error("API unavailable — please start the backend")
        except ApiError as e:
            details = f"(status={e.status_code})" if e.status_code is not None else ""
            st.error(f"Forecast generation failed {details}".strip())
            if e.response_text:
                st.caption(str(e.response_text)[:300])
