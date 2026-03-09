"""Click-to-inspect details panel for fire events using PyDeck selection."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pydeck as pdk
import requests
import streamlit as st

from state import app_state, isoformat
from api_client import (
    ApiError,
    ApiUnavailableError,
    create_jit_forecast,
    create_jit_forecast_from_front,
    get_active_spread_model_id,
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


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None on invalid inputs."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


@st.cache_data(ttl=15 * 60, show_spinner=False)
def _fetch_open_wildfire_events() -> list[dict[str, Any]]:
    """Fetch open wildfire events from NASA EONET (external context)."""
    try:
        response = requests.get(
            "https://eonet.gsfc.nasa.gov/api/v3/events",
            params={
                "status": "open",
                "category": "wildfires",
                "limit": 200,
            },
            headers={"Accept": "application/json"},
            timeout=(2.0, 5.0),
        )
        if response.status_code != 200:
            return []
        payload = response.json()
        events = payload.get("events", []) if isinstance(payload, dict) else []
        return events if isinstance(events, list) else []
    except Exception:
        return []


def _nearest_open_wildfire_event(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Return nearest open EONET wildfire event to a point."""
    events = _fetch_open_wildfire_events()
    best: Optional[Dict[str, Any]] = None
    for event in events:
        geometries = event.get("geometry", [])
        if not isinstance(geometries, list):
            continue
        for g in geometries:
            coords = g.get("coordinates") if isinstance(g, dict) else None
            if not (isinstance(coords, list) and len(coords) >= 2):
                continue
            ev_lon = _safe_float(coords[0])
            ev_lat = _safe_float(coords[1])
            if ev_lon is None or ev_lat is None:
                continue
            dist_km = _haversine_km(lat, lon, ev_lat, ev_lon)
            if best is None or dist_km < float(best["distance_km"]):
                best = {
                    "title": event.get("title", "Unnamed wildfire event"),
                    "id": event.get("id"),
                    "date": g.get("date"),
                    "distance_km": dist_km,
                    "source": "NASA EONET",
                }
    return best


def _clamp_01(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _build_event_key(event: Dict[str, Any], lat: float, lon: float) -> str:
    event_id = event.get("event_id")
    if event_id is not None and str(event_id).strip():
        return f"event_id:{event_id}"
    end_time = str(event.get("end_time") or "")
    return f"point:{lat:.4f}:{lon:.4f}:{end_time}"


def _location_label_for_event(event: Dict[str, Any], lat: float, lon: float) -> str:
    for key in ("country", "admin0_name", "admin1_name", "region_name", "location_name"):
        val = event.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    country = _lookup_country_for_coordinates(lat, lon)
    if country:
        return country
    return f"{lat:.2f}, {lon:.2f}"


def _compute_event_severity(event: Dict[str, Any]) -> float:
    """Compute a strict 0-1 severity score from event_score."""
    event_score = _safe_float(event.get("event_score"))
    if event_score is None:
        return 0.0
    return _clamp_01(event_score)


def _compute_significance(event: Dict[str, Any], *, now_utc: datetime) -> float:
    """Composite ranking for 'major fires' cards in the overview panel."""
    severity = _compute_event_severity(event)
    detection_count = _safe_float(event.get("detection_count")) or 0.0
    detections_component = _clamp_01(math.log1p(max(detection_count, 0.0)) / math.log1p(100.0))

    recency_component = 0.0
    end_time = _parse_time(event.get("end_time"))
    if end_time is not None:
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        age_hours = max((now_utc - end_time).total_seconds() / 3600.0, 0.0)
        recency_component = _clamp_01(1.0 - min(age_hours, 24.0) / 24.0)

    return 0.65 * severity + 0.25 * detections_component + 0.10 * recency_component


def _render_major_fires(events: list[Dict[str, Any]], *, now_utc: datetime, limit: int = 5) -> None:
    """Render top significant fires from currently loaded events."""
    ranked: list[tuple[float, Dict[str, Any]]] = []
    for event in events:
        ranked.append((_compute_significance(event, now_utc=now_utc), event))

    top = sorted(ranked, key=lambda item: item[0], reverse=True)[: max(0, limit)]
    if not top:
        return

    st.markdown(
        '<div style="margin-top:14px;font-size:12px;color:rgba(255,255,255,0.5);'
        'text-transform:uppercase;letter-spacing:0.5px;">Major fires in view</div>',
        unsafe_allow_html=True,
    )

    for idx, (score, event) in enumerate(top, start=1):
        lat = _safe_float(event.get("lat"))
        lon = _safe_float(event.get("lon"))
        event_id = str(event.get("event_id") or "unknown")
        detection_count = int(_safe_float(event.get("detection_count")) or 0)
        event_score = _compute_event_severity(event)
        end_time = _parse_time(event.get("end_time"))

        country = None
        if lat is not None and lon is not None:
            country = _lookup_country_for_coordinates(lat, lon)
        if lat is not None and lon is not None:
            coords = f"{lat:.2f}, {lon:.2f}"
            location = f"{country} ({coords})" if country else coords
        else:
            location = "Unknown location"
        end_str = end_time.astimezone(timezone.utc).strftime("%H:%M UTC") if end_time is not None else "Unknown"
        button_label = (
            f"#{idx} {location} | {event_id} | "
            f"detections {detection_count} | event {event_score:.2f}"
        )
        button_key = (
            f"major_fire_select_{event_id}_{idx}_"
            f"{int((lat or 0.0) * 100)}_{int((lon or 0.0) * 100)}"
        )

        if st.button(button_label, key=button_key, use_container_width=True):
            _select_fire_from_overview(event, lat=lat, lon=lon)
            st.rerun()

        st.caption(f"Last seen {end_str} | rank {score:.2f}")


def _select_fire_from_overview(event: Dict[str, Any], *, lat: Optional[float], lon: Optional[float]) -> None:
    """Select an event from the overview list and re-center the map."""
    if lat is None or lon is None:
        return

    selected = dict(event)
    selected["lat"] = lat
    selected["lon"] = lon

    app_state.selection.selected_fire = selected
    app_state.selection.last_click = {"lat": lat, "lng": lon}

    current_view = st.session_state.get("map_view_state")
    current_zoom = float(getattr(current_view, "zoom", 2.0)) if current_view is not None else 2.0
    pitch = float(getattr(current_view, "pitch", 0.0)) if current_view is not None else 0.0
    bearing = float(getattr(current_view, "bearing", 0.0)) if current_view is not None else 0.0
    target_zoom = max(current_zoom, 5.5)

    st.session_state.map_view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=target_zoom,
        pitch=pitch,
        bearing=bearing,
    )
    app_state._persist()


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _lookup_country_for_coordinates(lat: float, lon: float) -> Optional[str]:
    """Best-effort reverse geocode to country name for display purposes."""
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "format": "jsonv2",
                "lat": f"{lat:.4f}",
                "lon": f"{lon:.4f}",
                "zoom": 3,
                "addressdetails": 1,
                "accept-language": "en",
            },
            headers={
                "User-Agent": "wildfire-nowcast-ui/1.0",
                "Accept": "application/json",
                "Accept-Language": "en",
            },
            timeout=(1.5, 3.0),
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        address = payload.get("address", {}) if isinstance(payload, dict) else {}
        country = address.get("country")
        if isinstance(country, str) and country.strip():
            return country.strip()
        return None
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_aggregate_stats(
    bbox: Tuple[float, float, float, float],
    start_iso: str,
    end_iso: str,
    min_likelihood: float,
) -> Dict[str, Any]:
    """Fetch aggregate fire stats from the API with caching.

    Returns ``{"ok": True, "data": {...}}`` on success or
    ``{"ok": False, "error": "..."}`` on failure.
    """
    from api_client import get_fire_events

    try:
        data = get_fire_events(
            bbox=bbox,
            time_range=(
                datetime.fromisoformat(start_iso),
                datetime.fromisoformat(end_iso),
            ),
            filters={
                "min_event_score": min_likelihood,
                "include_review_required": True,
                "limit": 1000,
            },
        )
        return {"ok": True, "data": data}
    except Exception as exc:
        logger.warning("Failed to fetch aggregate stats: %s", exc, exc_info=True)
        return {"ok": False, "error": str(exc)}


def _render_aggregate_stats() -> None:
    """Render aggregate statistics when no fire is selected."""
    from config.theme import DarkTheme

    bbox = app_state.viewport_bbox
    start_time, end_time = app_state.time_range

    result = _fetch_aggregate_stats(
        bbox=bbox,
        start_iso=isoformat(start_time),
        end_iso=isoformat(end_time),
        min_likelihood=app_state.filters.min_likelihood,
    )

    if result.get("ok"):
        st.session_state.aggregate_stats_last_success = result["data"]
        data = result["data"]
    else:
        error_msg = result.get("error", "Unknown error")
        cached_data = st.session_state.get("aggregate_stats_last_success")
        if isinstance(cached_data, dict):
            data = cached_data
            st.caption(f"\u26a0\ufe0f Live stats unavailable ({error_msg}); showing last successful snapshot.")
        else:
            st.caption(f"\u26a0\ufe0f Could not load stats: {error_msg}")
            st.caption("Click an event on the map to inspect details.")
            return

    count = data.get("count", 0)
    events = data.get("events", [])

    # Compute derived values
    max_lh: float | None = None
    most_recent_str = "N/A"
    if events:
        max_lh = max(
            (_compute_event_severity(event) for event in events),
            default=0,
        )
        times = [_parse_time(event.get("end_time")) for event in events]
        valid_times = [t for t in times if t is not None]
        if valid_times:
            most_recent_str = max(valid_times).strftime("%H:%M UTC")

    # ── Hero: Active event count ─────────────────────────────────
    st.markdown(
        f'<div style="text-align:center;padding:16px 0 8px;">'
        f'<div style="font-size:42px;font-weight:700;color:{DarkTheme.ACCENT_EMBER};'
        f'line-height:1;">{count}</div>'
        f'<div style="font-size:12px;color:rgba(255,255,255,0.5);'
        f'margin-top:4px;text-transform:uppercase;letter-spacing:0.5px;">'
        f'Active events</div></div>',
        unsafe_allow_html=True,
    )

    # ── Confidence gauge ─────────────────────────────────────────
    if max_lh is not None and max_lh > 0:
        st.markdown(
            '<div style="font-size:12px;color:rgba(255,255,255,0.5);'
            'text-transform:uppercase;letter-spacing:0.5px;'
            'text-align:center;margin-top:8px;">Highest severity</div>',
            unsafe_allow_html=True,
        )
        gauge_html = _render_confidence_gauge(max_lh)
        st.markdown(gauge_html, unsafe_allow_html=True)
    else:
        _render_stat_row("Highest severity", "N/A")

    # ── Stat rows ────────────────────────────────────────────────
    _render_stat_row("Most recent", most_recent_str)
    _render_stat_row("Time window", app_state.time_window)
    _render_major_fires(events, now_utc=end_time, limit=5)


def _render_stat_row(label: str, value: str) -> None:
    """Render a single label + value row."""
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;padding:10px 0;'
        f'border-top:1px solid rgba(255,255,255,0.06);">'
        f'<span style="font-size:12px;color:rgba(255,255,255,0.5);'
        f'text-transform:uppercase;letter-spacing:0.5px;">{label}</span>'
        f'<span style="font-size:15px;font-weight:600;'
        f'color:#e0e0e0;">{value}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_fire_summary(event: Dict[str, Any]) -> None:
    """Render compact analyst-oriented summary with optional external context."""
    lat = _safe_float(event.get("lat"))
    lon = _safe_float(event.get("lon"))
    severity = _compute_event_severity(event)
    detection_count = int(_safe_float(event.get("detection_count")) or 0)
    front_count = int(_safe_float(event.get("front_count")) or 0)
    decision = str(event.get("denoiser_decision") or "").strip().lower()
    review_required = bool(event.get("review_required"))

    signals: list[str] = []
    if severity >= 0.6:
        signals.append("high event score")
    if detection_count >= 10:
        signals.append("multi-detection cluster")
    if front_count >= 2:
        signals.append("multi-front continuity")
    if decision in {"pass", "downweight"}:
        signals.append(f"decision={decision}")
    if review_required:
        signals.append("review-required")

    if not signals:
        summary = "Low-signal event profile; monitor before action."
    else:
        summary = f"Likely active fire signal based on {', '.join(signals)}."
    st.info(summary)

    if lat is None or lon is None:
        return

    nearest = _nearest_open_wildfire_event(lat, lon)
    if nearest is None:
        st.caption("External context: no open wildfire events returned from NASA EONET.")
        return

    dist_km = float(nearest["distance_km"])
    if dist_km <= 250.0:
        st.success(
            f"External context ({nearest['source']}): "
            f"possible match '{nearest['title']}' ({dist_km:.0f} km away)."
        )
    else:
        st.caption(
            f"External context ({nearest['source']}): nearest open event is "
            f"'{nearest['title']}' at {dist_km:.0f} km."
        )


def render_click_details(last_click: Optional[Dict[str, float]]) -> None:
    """Render details for the selected fire event based on PyDeck selection."""
    # Use the selected fire from state manager (set by map_view)
    selected_event = app_state.selection.selected_fire

    if not isinstance(selected_event, dict) or not selected_event:
        # Show aggregate stats when no fire is selected
        st.subheader("Overview")
        if last_click is None:
            _render_aggregate_stats()
            return
        st.info("No event selected. Try clicking exactly on a fire event marker.")
        return

    # ── Prominent forecast button at the top ──────────────────────────
    lat = selected_event.get("lat")
    lon = selected_event.get("lon")
    _render_forecast_section(lat, lon, selected_event.get("end_time"), selected_event)

    st.subheader("Fire details")

    st.write("**Selection:** Fire event")

    if lat is not None and lon is not None:
        st.caption(f"Location: {lat:.4f}, {lon:.4f}")

    st.write(f"**Event ID:** {selected_event.get('event_id')}")
    st.write(f"**Source:** {selected_event.get('source')}")
    st.write(f"**Satellite:** {selected_event.get('sensor')}")
    st.write(
        f"**Window:** {selected_event.get('start_time')} → {selected_event.get('end_time')}"
    )
    st.write(f"**Detections:** {selected_event.get('detection_count')}")
    st.write(f"**Fronts:** {selected_event.get('front_count')}")
    st.write(f"**Decision:** {selected_event.get('denoiser_decision')}")
    st.write(f"**Review required:** {selected_event.get('review_required')}")
    _render_fire_summary(selected_event)

    event_score = _safe_float(selected_event.get("event_score"))
    if event_score is not None:
        st.divider()
        st.write("**Event score**")
        gauge_html = _render_confidence_gauge(_clamp_01(event_score))
        st.markdown(gauge_html, unsafe_allow_html=True)
        st.write(f"**Score:** {event_score:.4f}")


def _render_forecast_section(
    lat: Any, lon: Any, acq_time: Any, selected_event: Dict[str, Any]
) -> None:
    """Render the prominent forecast button at the top of the details panel."""
    if lat is None or lon is None:
        st.warning("Selected event is missing coordinates. Cannot generate forecast.")
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

    radius_km = 20.0
    lat_delta = float(radius_km) / 111.0
    lon_scale = max(math.cos(math.radians(fire_lat)), 0.1)
    lon_delta = float(radius_km) / (111.0 * lon_scale)
    forecast_bbox = (
        fire_lon - lon_delta,
        fire_lat - lat_delta,
        fire_lon + lon_delta,
        fire_lat + lat_delta,
    )

    event_key = _build_event_key(selected_event, fire_lat, fire_lon)
    is_forecast_running = app_state.forecast_job.job_id is not None
    last_forecast = app_state.forecast_job.last_forecast or {}
    same_event_completed = bool(
        (last_forecast.get("run", {}) or {}).get("id")
        and str(last_forecast.get("event_key") or "") == event_key
    )
    disable_reason = ""
    if is_forecast_running:
        disable_reason = "A forecast is already running."
    elif same_event_completed:
        disable_reason = "A forecast already exists for this fire event."
    button_label = "Generate Spread Forecast"
    if is_forecast_running:
        button_label = "Generating Spread Forecast..."
    elif same_event_completed:
        button_label = "Spread Forecast Already Generated"

    if st.button(
        button_label,
        key="generate_forecast_btn",
        disabled=bool(disable_reason),
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

            selected = app_state.selection.selected_fire or {}
            event_id = selected.get("event_id")
            front_lookup = app_state.selection.front_index_by_event or {}
            matched_front = front_lookup.get(str(event_id)) if event_id is not None else None
            front_id = matched_front.get("front_id") if isinstance(matched_front, dict) else None
            event_snapshot = dict(selected_event)
            event_snapshot["lat"] = fire_lat
            event_snapshot["lon"] = fire_lon
            request_context = {
                "event_id": event_id,
                "event_key": event_key,
                "front_id": front_id,
                "lat": fire_lat,
                "lon": fire_lon,
                "location_label": _location_label_for_event(selected_event, fire_lat, fire_lon),
                "event_snapshot": event_snapshot,
            }
            model_id = get_active_spread_model_id()

            if front_id:
                job_data = create_jit_forecast_from_front(
                    str(front_id),
                    buffer_km=3.0,
                    horizons=[24, 48, 72],
                    forecast_reference_time=ref_time,
                    model_id=model_id,
                )
                logger.info(
                    "Using front-driven forecast trigger: front_id=%s model_id=%s",
                    front_id,
                    model_id,
                )
            else:
                job_data = create_jit_forecast(
                    bbox=forecast_bbox,
                    horizons=[24, 48, 72],
                    forecast_reference_time=ref_time,
                    model_id=model_id,
                )
                logger.info(
                    "Using point-based forecast trigger: event_id=%s model_id=%s",
                    event_id,
                    model_id,
                )

            job_id = job_data.get("job_id")
            if job_id:
                app_state.forecast_job.start(job_id, request_context=request_context)
                app_state.forecast_job.notification = {
                    "kind": "info",
                    "message": "Spread forecast is being generated and may take around a minute.",
                    "created_at": time.time(),
                    "ttl_seconds": 20.0,
                }
                app_state._persist()
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
            msg = (e.message or "Forecast generation failed").strip()
            st.error(f"{msg} {details}".strip())
            if e.response_text:
                st.caption(str(e.response_text)[:300])

    if disable_reason:
        st.caption(disable_reason)
