"""UI banner for backend data freshness and stale-data behavior."""

from __future__ import annotations

from html import escape
import os
from datetime import datetime
from typing import Any

import streamlit as st

from api_client import ApiError, ApiUnavailableError, get_data_freshness_status


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SHOW_OPS_DIAGNOSTICS = _env_bool("UI_SHOW_OPS_DIAGNOSTICS", default=False)


@st.cache_data(ttl=60, show_spinner=False)
def _load_data_freshness() -> dict[str, Any]:
    return get_data_freshness_status()


def _fmt_age(value: Any) -> str:
    if value is None:
        return "unknown age"
    try:
        return f"{float(value):.1f}m old"
    except (TypeError, ValueError):
        return str(value)


def _fmt_ts(value: Any) -> str:
    if not value:
        return "unknown time"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return str(value)


def _status_label(state: Any) -> str:
    state_norm = str(state or "").strip().lower()
    if state_norm == "fresh":
        return "Fresh"
    if state_norm == "stale":
        return "Stale"
    if state_norm == "missing":
        return "Missing"
    return "Unknown"


def _status_class(state: Any) -> str:
    state_norm = str(state or "").strip().lower()
    if state_norm in {"fresh", "stale", "missing"}:
        return f"wf-status-{state_norm}"
    return "wf-status-unknown"


def _build_badges_html(sources: dict[str, Any]) -> str:
    ordered_sources = ["firms", "weather", "terrain", "perimeters"]
    cards: list[str] = []
    for source_name in ordered_sources:
        details = sources.get(source_name) or {}
        status_text = _status_label(details.get("state"))
        status_class = _status_class(details.get("state"))
        cards.append(
            (
                '<div class="wf-source-card">'
                f'<span class="wf-source-name">{escape(source_name)}</span>'
                f'<span class="wf-source-status {status_class}">{escape(status_text)}</span>'
                f'<span class="wf-source-meta">{escape(_fmt_ts(details.get("last_seen_at")))}</span>'
                f'<span class="wf-source-meta">{escape(_fmt_age(details.get("age_minutes")))}</span>'
                "</div>"
            )
        )
    return '<div class="wf-data-badges">' + "".join(cards) + "</div>"


def render_data_freshness_banner() -> None:
    """Render user-facing data recency banner sourced from API /health/data-freshness."""
    try:
        snapshot = _load_data_freshness()
    except ApiUnavailableError:
        st.warning("Data freshness status is unavailable (API unreachable).")
        return
    except ApiError as exc:
        st.warning(f"Data freshness status error: {exc.message}")
        return

    sources = snapshot.get("sources", {})
    idempotency_dashboard = snapshot.get("idempotency_dashboard", {})
    if sources:
        st.markdown(_build_badges_html(sources), unsafe_allow_html=True)
    else:
        st.caption("Data updates unavailable")

    if SHOW_OPS_DIAGNOSTICS and idempotency_dashboard:
        with st.expander("Operational diagnostics", expanded=False):
            st.json(idempotency_dashboard)
