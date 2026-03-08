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
                    _, hours_start, hours_end, likelihood = preset
                    help_text = f"Time: {hours_start}h, Likelihood: {likelihood}"

                if st.button(
                    name,
                    key=f"preset_{idx}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    disabled=is_custom,
                    help=help_text,
                ):
                    if not is_custom:
                        _, hours_start, hours_end, likelihood = preset
                        app_state.apply_preset(name, hours_start, hours_end, likelihood)
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
            "Minimum event score",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="min_likelihood",
            help="Event-level denoiser score. Values <0.3 are low-signal events, 0.3-0.6 uncertain, >0.6 likely real fire events.",
        )

        # Dynamic likelihood intensity label (best-effort for fire icon feedback)
        likelihood_val = st.session_state.get("min_likelihood", 0.0)
        if likelihood_val >= 0.6:
            st.caption("Threshold: **High** \u2014 likely real events only")
        elif likelihood_val >= 0.3:
            st.caption("Threshold: **Medium** \u2014 filtering uncertain events")
        else:
            st.caption("Threshold: **Low** \u2014 showing all events")

        st.toggle(
            "Active incidents only",
            key="active_only",
            help=(
                "Hide lower-signal events using event-level decisioning and score thresholds."
            ),
        )
        st.toggle(
            "Cluster nearby points",
            key="cluster_points",
            help="Aggregate nearby events into incident bubbles to reduce visual clutter.",
        )
        cluster_enabled = bool(st.session_state.get("cluster_points", True))
        st.toggle(
            "Include risk index overlay",
            key="risk_checkbox",
            disabled=not cluster_enabled,
            help=(
                "Show coarse risk cells around the current viewport. "
                "Best used with clustering at regional zoom levels."
            ),
        )
        if not cluster_enabled:
            st.session_state.risk_checkbox = False

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
            f"{api_public_base_url()}/map.png?"
            f"min_lon={min_lon}&min_lat={min_lat}&max_lon={max_lon}&max_lat={max_lat}&"
            f"start_time={isoformat(start_time)}&end_time={isoformat(end_time)}&"
            f"min_fire_likelihood={app_state.filters.min_likelihood:.2f}&"
            f"include_fires=true&"
            f"include_risk={'true' if app_state.layers.show_risk else 'false'}&"
            f"include_forecast=true"
        )

        run_id = (app_state.forecast_job.last_forecast or {}).get("run", {}).get("id")
        if run_id:
            png_export_url += f"&run_id={run_id}"

        st.link_button(
            "Export map (PNG)",
            png_export_url,
            use_container_width=True,
            icon=":material/image:",
        )

    # Layers policy: fires + forecast are always on; risk follows the cluster controls.
    app_state.layers.show_fires = True
    app_state.layers.show_forecast = True
    app_state.layers.show_risk = bool(
        st.session_state.get("cluster_points", True)
        and st.session_state.get("risk_checkbox", app_state.layers.show_risk)
    )
    app_state._persist()

    # ── Map controls ──────────────────────────────────────────────────
    with st.container(border=True):
        st.caption("**Map Controls**")
        st.caption("Pan and zoom to explore. The map updates as you move.")
        if st.button("Clear selection", use_container_width=True, icon=":material/delete:"):
            app_state.selection.selected_fire = None
            app_state.selection.last_click = None
            app_state._persist()
            st.rerun()
