"""Sidebar component for wildfire dashboard controls."""

from datetime import datetime, timedelta, timezone
import streamlit as st
from runtime_config import api_public_base_url


def _isoformat(dt: datetime) -> str:
    """Format datetime for API query parameters."""
    offset = dt.utcoffset() if dt.tzinfo is not None else None
    if offset is not None and offset.total_seconds() == 0:
        dt_no_microseconds = dt.replace(microsecond=0)
        return dt_no_microseconds.replace(tzinfo=None).isoformat() + "Z"
    dt_no_microseconds = dt.replace(microsecond=0)
    return dt_no_microseconds.isoformat()


def _current_time_range() -> tuple[datetime, datetime]:
    """Return (start_time, end_time) based on timeline scrubber values."""
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    # Calculate from scrubber values
    start_hours = st.session_state.get("time_range_hours_start", 24)
    end_hours = st.session_state.get("time_range_hours_end", 0)

    end = end - timedelta(hours=end_hours)
    start = end - timedelta(hours=(start_hours - end_hours))

    return start, end


def _get_matching_preset() -> str | None:
    """Return the name of a preset that matches current filter state, or None."""
    from config.theme import FilterPresets

    current_start = st.session_state.get("time_range_hours_start", 24)
    current_end = st.session_state.get("time_range_hours_end", 0)
    current_likelihood = st.session_state.get("fires_min_likelihood", 0.0)
    current_denoiser = st.session_state.get("fires_apply_denoiser", True)

    for name, hours_start, hours_end, likelihood, denoiser in FilterPresets.all_presets():
        if (hours_start == current_start and
            hours_end == current_end and
            abs(likelihood - current_likelihood) < 0.01 and
            denoiser == current_denoiser):
            return name
    return None


def _apply_preset(name: str, hours_start: int, hours_end: int, likelihood: float, denoiser: bool):
    """Apply a preset by updating all filter state values."""
    st.session_state.active_preset = name
    st.session_state.time_range_hours_start = hours_start
    st.session_state.time_range_hours_end = hours_end
    st.session_state.fires_min_likelihood = likelihood
    st.session_state.fires_apply_denoiser = denoiser

    # Mark that we just applied a preset so change detection doesn't override it
    st.session_state._just_applied_preset = True

    # Set widget keys directly to new values (Streamlit uses these as source of truth)
    # Slider widget stores (end, start) since Streamlit needs (low, high) tuple
    st.session_state.timeline_scrubber = (hours_end, hours_start)
    st.session_state.min_likelihood = likelihood
    st.session_state.apply_denoiser = denoiser

    # Update time_window string for backward compatibility
    hours_window = hours_start - hours_end
    if hours_window <= 6:
        st.session_state.time_window = "Last 6 hours"
    elif hours_window <= 12:
        st.session_state.time_window = "Last 12 hours"
    elif hours_window <= 24:
        st.session_state.time_window = "Last 24 hours"
    else:
        st.session_state.time_window = "Last 48 hours"


def _sync_url_params():
    """Sync current filter state to URL query parameters."""
    st.query_params["start"] = str(st.session_state.get("time_range_hours_start", 24))
    st.query_params["end"] = str(st.session_state.get("time_range_hours_end", 0))
    st.query_params["likelihood"] = f"{st.session_state.get('fires_min_likelihood', 0.0):.2f}"
    st.query_params["denoiser"] = str(st.session_state.get("fires_apply_denoiser", True)).lower()

    active_preset = st.session_state.get("active_preset")
    if active_preset:
        st.query_params["preset"] = active_preset
    elif "preset" in st.query_params:
        del st.query_params["preset"]


def _calculate_viewport_bbox() -> tuple[float, float, float, float]:
    """Calculate bbox from current map viewport state."""
    view_state = st.session_state.get("map_view_state")
    if view_state is None:
        # Default bbox (rough global extent)
        return (-180.0, -85.0, 180.0, 85.0)

    lat = view_state.latitude
    lon = view_state.longitude
    zoom = view_state.zoom

    # Approximate visible area in degrees based on zoom level
    # At zoom 0, the map shows ~360 degrees. Each zoom level halves the visible area.
    degrees_per_tile = 360.0 / (2 ** zoom)

    # Adjust for latitude (Mercator projection distortion)
    lat_extent = degrees_per_tile * 0.5
    lon_extent = degrees_per_tile * 0.5

    min_lon = max(lon - lon_extent, -180.0)
    max_lon = min(lon + lon_extent, 180.0)
    min_lat = max(lat - lat_extent, -85.0)
    max_lat = min(lat + lat_extent, 85.0)

    return (min_lon, min_lat, max_lon, max_lat)


def render_sidebar() -> str:
    """Render the sidebar controls and return selected time window."""
    st.header("Filters & Controls")

    # Fires filters section
    st.subheader("Filters")

    # Filter presets
    from config.theme import FilterPresets
    st.caption("**Quick presets:**")

    # Initialize active preset tracker if not exists
    if "active_preset" not in st.session_state:
        st.session_state.active_preset = None

    # Get all presets including Custom for display
    all_presets = FilterPresets.all_presets_with_custom()

    # Render preset buttons in a 2-column grid
    # We have 5 presets, so we need 3 rows
    preset_cols = st.columns(2)

    for idx, preset in enumerate(all_presets):
        name = preset[0]
        is_custom = name == "Custom"

        col = preset_cols[idx % 2]
        with col:
            is_active = st.session_state.active_preset == name

            # Create help text based on preset type
            if is_custom:
                help_text = "Manually adjusted filters"
            else:
                _, hours_start, hours_end, likelihood, denoiser = preset
                help_text = f"Time: {hours_start}h, Likelihood: {likelihood}, Denoiser: {'On' if denoiser else 'Off'}"

            if st.button(
                name,
                key=f"preset_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                disabled=is_custom,  # Custom button is just an indicator
                help=help_text
            ):
                if not is_custom:
                    _, hours_start, hours_end, likelihood, denoiser = preset
                    _apply_preset(name, hours_start, hours_end, likelihood, denoiser)
                    st.rerun()

    # Timeline scrubber - dual-handle slider
    st.caption("**Time window:**")

    # Track if we just applied a preset (to avoid change detection overwriting it)
    just_applied_preset = st.session_state.get("_just_applied_preset", False)
    if just_applied_preset:
        st.session_state._just_applied_preset = False

    # Get previous values for change detection
    # start_hours = how far back in the past (e.g., 24 = 24 hours ago)
    # end_hours = how far back the range ends (e.g., 0 = now)
    prev_start = st.session_state.get("time_range_hours_start", 24)
    prev_end = st.session_state.get("time_range_hours_end", 0)

    # Streamlit range slider expects (low, high) tuple
    # We store as (end_hours, start_hours) in the widget since end <= start
    # e.g., for "24h ago to now": end=0, start=24, widget stores (0, 24)
    if "timeline_scrubber" not in st.session_state:
        st.session_state.timeline_scrubber = (prev_end, prev_start)

    time_range = st.slider(
        "Time range",
        min_value=0,
        max_value=48,
        value=(prev_end, prev_start),
        step=1,
        key="timeline_scrubber",
        help="Select time range in hours ago. Left = end time (0=now), Right = start time (further back)",
        format="%dh ago"
    )

    # Unpack: slider returns (low, high) = (end_hours, start_hours)
    end_hours, start_hours = time_range

    # Validate that start > end (start is further in the past)
    if start_hours <= end_hours:
        start_hours = end_hours + 1

    # Check if slider values changed (user dragged the slider)
    slider_changed = (start_hours != prev_start or end_hours != prev_end)

    # Sync our state keys with widget values
    st.session_state.time_range_hours_start = start_hours
    st.session_state.time_range_hours_end = end_hours

    # Display selected range in human-readable format
    if end_hours == 0:
        end_str = "now"
    else:
        end_str = f"{end_hours}h ago"

    st.caption(f"Selected: {start_hours}h ago to {end_str} ({start_hours - end_hours}h window)")

    # Convert to time window string for backward compatibility
    hours_window = start_hours - end_hours
    if hours_window <= 6:
        st.session_state.time_window = "Last 6 hours"
    elif hours_window <= 12:
        st.session_state.time_window = "Last 12 hours"
    elif hours_window <= 24:
        st.session_state.time_window = "Last 24 hours"
    else:
        st.session_state.time_window = "Last 48 hours"

    # Likelihood slider
    prev_likelihood = st.session_state.get("fires_min_likelihood", 0.0)

    # Initialize widget key if not present
    if "min_likelihood" not in st.session_state:
        st.session_state.min_likelihood = prev_likelihood

    pending_min_likelihood = st.slider(
        "Minimum fire likelihood",
        min_value=0.0,
        max_value=1.0,
        value=float(prev_likelihood),
        step=0.05,
        key="min_likelihood",
        help="Composite score combining FIRMS confidence (20%), persistence (30%), land-cover plausibility (25%), and weather conditions (25%). Values <0.3 indicate low confidence, 0.3-0.6 uncertain, >0.6 likely real fire.",
    )

    likelihood_changed = abs(pending_min_likelihood - prev_likelihood) > 0.001
    st.session_state.fires_min_likelihood = pending_min_likelihood

    # Denoiser checkbox
    prev_denoiser = st.session_state.get("fires_apply_denoiser", True)

    # Initialize widget key if not present
    if "apply_denoiser" not in st.session_state:
        st.session_state.apply_denoiser = prev_denoiser

    pending_apply_denoiser = st.checkbox(
        "Apply noise filter (exclude false alarms)",
        value=bool(prev_denoiser),
        key="apply_denoiser",
    )

    denoiser_changed = pending_apply_denoiser != prev_denoiser
    st.session_state.fires_apply_denoiser = pending_apply_denoiser

    # If any filter changed manually (not from preset), update active_preset
    # Skip if we just applied a preset to avoid overwriting it
    if not just_applied_preset and (slider_changed or likelihood_changed or denoiser_changed):
        # Check if current values match any preset
        matching_preset = _get_matching_preset()
        st.session_state.active_preset = matching_preset if matching_preset else "Custom"

    # Sync URL params at the end of render (always, to keep URL in sync)
    _sync_url_params()

    # Export fires button
    st.caption("Export current view as CSV or GeoJSON.")
    bbox = _calculate_viewport_bbox()
    start_time, end_time = _current_time_range()
    min_lon, min_lat, max_lon, max_lat = bbox

    export_url = (
        f"{api_public_base_url()}/fires/export?"
        f"min_lon={min_lon}&min_lat={min_lat}&max_lon={max_lon}&max_lat={max_lat}&"
        f"start_time={_isoformat(start_time)}&end_time={_isoformat(end_time)}&"
        f"format=csv&limit=1000"
    )

    st.link_button(
        "📥 Export fires (CSV)",
        export_url,
        use_container_width=True,
    )

    # Export map as PNG
    png_export_url = (
        f"{api_public_base_url()}/exports/map.png?"
        f"min_lon={min_lon}&min_lat={min_lat}&max_lon={max_lon}&max_lat={max_lat}&"
        f"start_time={_isoformat(start_time)}&end_time={_isoformat(end_time)}&"
        f"include_fires={'true' if st.session_state.show_fires else 'false'}&"
        f"include_risk={'true' if st.session_state.show_risk else 'false'}&"
        f"include_forecast={'true' if st.session_state.show_forecast else 'false'}"
    )

    # Add run_id if forecast is enabled and available
    if st.session_state.show_forecast:
        run_id = st.session_state.get("last_forecast", {}).get("run", {}).get("id")
        if run_id:
            png_export_url += f"&run_id={run_id}"

    st.link_button(
        "🖼️ Export map (PNG)",
        png_export_url,
        use_container_width=True,
    )

    st.divider()

    # Layers section
    st.subheader("Layers")
    st.session_state.show_fires = st.checkbox(
        "Active fires",
        value=st.session_state.show_fires,
        key="fires_checkbox"
    )
    st.session_state.show_forecast = st.checkbox(
        "Show forecast overlay",
        value=st.session_state.show_forecast,
        key="forecast_checkbox"
    )
    st.session_state.show_risk = st.checkbox(
        "Risk index",
        value=st.session_state.show_risk,
        key="risk_checkbox"
    )

    # Count active layers
    active_layers = sum([
        st.session_state.show_fires,
        st.session_state.show_forecast,
        st.session_state.show_risk
    ])
    st.caption(f"Layers active: {active_layers}")

    st.divider()

    # Map controls
    st.subheader("Map Controls")
    st.caption("Pan and zoom to explore. The map updates as you move.")
    if st.button("🗑️ Clear selection", use_container_width=True):
        st.session_state.selected_fire = None
        st.session_state.last_click = None
        st.rerun()

    st.divider()

    # AOI behavior (MVP)
    st.subheader("Forecast area")
    st.caption("Forecast uses the area currently shown on the map.")

    st.divider()

    # About / Data notice
    st.subheader("About")
    st.caption(
        "**Data sources**\n\n"
        "- Fires and forecast layers are updated automatically from our data service.\n"
        "- If data can't be reached, you'll see an error and can retry.\n"
        "- The risk layer is still a placeholder."
    )

    return st.session_state.time_window
