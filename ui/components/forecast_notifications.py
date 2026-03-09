"""Top-level in-app notifications for forecast lifecycle events."""

from __future__ import annotations

import time
from typing import Any

import pydeck as pdk
import streamlit as st

from state import app_state


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clear_notification() -> None:
    app_state.forecast_job.notification = None
    app_state._persist()


def _focus_on_notification_target(target: dict[str, Any]) -> None:
    lat = _as_float(target.get("lat"))
    lon = _as_float(target.get("lon"))
    if lat is None or lon is None:
        return

    event_snapshot = target.get("event_snapshot")
    selected = dict(event_snapshot) if isinstance(event_snapshot, dict) else {}
    selected["lat"] = lat
    selected["lon"] = lon
    if target.get("event_id") is not None and "event_id" not in selected:
        selected["event_id"] = target.get("event_id")

    app_state.selection.selected_fire = selected
    app_state.selection.last_click = {"lat": lat, "lng": lon}

    current_view = st.session_state.get("map_view_state")
    current_zoom = float(getattr(current_view, "zoom", 2.0)) if current_view is not None else 2.0
    pitch = float(getattr(current_view, "pitch", 0.0)) if current_view is not None else 0.0
    bearing = float(getattr(current_view, "bearing", 0.0)) if current_view is not None else 0.0
    target_zoom = max(current_zoom, 7.0)

    st.session_state.map_view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=target_zoom,
        pitch=pitch,
        bearing=bearing,
    )
    app_state._persist()


def render_forecast_notification() -> None:
    notice = app_state.forecast_job.notification
    if not isinstance(notice, dict):
        return

    created_at = _as_float(notice.get("created_at")) or 0.0
    ttl_seconds = _as_float(notice.get("ttl_seconds")) or 0.0
    if ttl_seconds > 0.0 and (time.time() - created_at) > ttl_seconds:
        _clear_notification()
        return

    kind = str(notice.get("kind") or "info")
    message = str(notice.get("message") or "Forecast update.")

    if kind == "ready":
        col_msg, col_action = st.columns([6, 1], gap="small")
        with col_msg:
            st.success(message)
        with col_action:
            run_id = str(notice.get("run_id") or "latest")
            if st.button("Open ->", key=f"forecast_notice_open_{run_id}", use_container_width=True):
                target = notice.get("target")
                if isinstance(target, dict):
                    _focus_on_notification_target(target)
                _clear_notification()
                st.rerun()
        return

    if kind == "error":
        st.error(message)
        return

    st.info(message)
