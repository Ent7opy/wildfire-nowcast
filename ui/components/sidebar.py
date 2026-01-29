"""Sidebar component for wildfire dashboard controls."""

import streamlit as st

from state import app_state, isoformat
from runtime_config import api_public_base_url


def render_sidebar() -> None:
    """Render the sidebar controls."""
    st.header("Filters & Controls")

    # ── Quick presets ─────────────────────────────────────────────────
    with st.container(border=True):
        from config.theme import FilterPresets
        st.caption("**Quick presets**")

        all_presets = FilterPresets.all_presets_with_custom()

        preset_cols = st.columns(2)
        for idx, preset in enumerate(all_presets):
            name = preset[0]
            is_custom = name == "Custom"

            col = preset_cols[idx % 2]
            with col:
                is_active = app_state.active_preset == name

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
                    disabled=is_custom,
                    help=help_text,
                ):
                    if not is_custom:
                        _, hours_start, hours_end, likelihood, denoiser = preset
                        app_state.apply_preset(name, hours_start, hours_end, likelihood, denoiser)
                        st.rerun()

    # ── Widget sync: push canonical -> widget keys ────────────────────
    app_state.sync_widgets_before_render()

    # ── Time & likelihood filters ─────────────────────────────────────
    with st.container(border=True):
        st.caption("**Time window**")

        time_range = st.slider(
            "Time range",
            min_value=0,
            max_value=48,
            step=1,
            key="timeline_scrubber",
            help="Select time range in hours ago. Left = end time (0=now), Right = start time (further back)",
            format="%dh ago",
        )

        end_hours, start_hours = time_range
        end_str = "now" if end_hours == 0 else f"{end_hours}h ago"
        st.caption(f"Selected: {start_hours}h ago to {end_str} ({start_hours - end_hours}h window)")

        st.slider(
            "Minimum fire likelihood",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="min_likelihood",
            help="Composite score combining FIRMS confidence (20%), persistence (30%), land-cover plausibility (25%), and weather conditions (25%). Values <0.3 indicate low confidence, 0.3-0.6 uncertain, >0.6 likely real fire.",
        )

        # Dynamic likelihood intensity label (best-effort for fire icon feedback)
        likelihood_val = st.session_state.get("min_likelihood", 0.0)
        if likelihood_val >= 0.6:
            st.caption("Threshold: **High** \u2014 likely real fires only")
        elif likelihood_val >= 0.3:
            st.caption("Threshold: **Medium** \u2014 filtering uncertain detections")
        else:
            st.caption("Threshold: **Low** \u2014 showing all detections")

        st.toggle(
            "Noise filter",
            key="apply_denoiser",
            help="Exclude false alarms",
        )

    # ── Widget sync: pull widget keys -> canonical state ──────────────
    app_state.read_widgets_after_render()

    # ── URL sync ─────────────────────────────────────────────────────
    app_state.sync_to_url()

    # ── Export ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.caption("**Export current view**")

        bbox = app_state.viewport_bbox
        start_time, end_time = app_state.time_range
        min_lon, min_lat, max_lon, max_lat = bbox

        export_url = (
            f"{api_public_base_url()}/fires/export?"
            f"min_lon={min_lon}&min_lat={min_lat}&max_lon={max_lon}&max_lat={max_lat}&"
            f"start_time={isoformat(start_time)}&end_time={isoformat(end_time)}&"
            f"format=csv&limit=1000"
        )

        st.link_button(
            "Export fires (CSV)",
            export_url,
            use_container_width=True,
            icon=":material/download:",
        )

        png_export_url = (
            f"{api_public_base_url()}/exports/map.png?"
            f"min_lon={min_lon}&min_lat={min_lat}&max_lon={max_lon}&max_lat={max_lat}&"
            f"start_time={isoformat(start_time)}&end_time={isoformat(end_time)}&"
            f"include_fires={'true' if app_state.layers.show_fires else 'false'}&"
            f"include_risk={'true' if app_state.layers.show_risk else 'false'}&"
            f"include_forecast={'true' if app_state.layers.show_forecast else 'false'}"
        )

        if app_state.layers.show_forecast:
            run_id = (app_state.forecast_job.last_forecast or {}).get("run", {}).get("id")
            if run_id:
                png_export_url += f"&run_id={run_id}"

        st.link_button(
            "Export map (PNG)",
            png_export_url,
            use_container_width=True,
            icon=":material/image:",
        )

    # ── Layers ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.caption("**Layers**")

        st.toggle("Active fires", key="fires_checkbox")
        st.toggle("Forecast overlay", key="forecast_checkbox")
        st.toggle("Risk index", key="risk_checkbox")

        # Read layer toggle values now that they've rendered
        app_state.layers.show_fires = st.session_state.get("fires_checkbox", app_state.layers.show_fires)
        app_state.layers.show_forecast = st.session_state.get("forecast_checkbox", app_state.layers.show_forecast)
        app_state.layers.show_risk = st.session_state.get("risk_checkbox", app_state.layers.show_risk)
        app_state._persist()

        active_layers = sum([
            app_state.layers.show_fires,
            app_state.layers.show_forecast,
            app_state.layers.show_risk,
        ])
        st.caption(f"Layers active: {active_layers}")

    # ── Map controls ──────────────────────────────────────────────────
    with st.container(border=True):
        st.caption("**Map Controls**")
        st.caption("Pan and zoom to explore. The map updates as you move.")
        if st.button("Clear selection", use_container_width=True, icon=":material/delete:"):
            app_state.selection.selected_fire = None
            app_state.selection.last_click = None
            app_state._persist()
            st.rerun()

    # ── About ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.caption("**About**")
        st.caption(
            "**Data sources**\n\n"
            "- Fires and forecast layers are updated automatically from our data service.\n"
            "- If data can't be reached, you'll see an error and can retry.\n"
            "- The risk layer is still a placeholder."
        )
