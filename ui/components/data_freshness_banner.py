"""UI banner for backend data freshness and stale-data behavior."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from api_client import ApiError, ApiUnavailableError, get_data_freshness_status


@st.cache_data(ttl=60, show_spinner=False)
def _load_data_freshness() -> dict[str, Any]:
    return get_data_freshness_status()


def _fmt_age(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.1f}m"
    except (TypeError, ValueError):
        return str(value)


def _fmt_ts(value: Any) -> str:
    if not value:
        return "unknown"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return str(value)


def render_data_freshness_banner() -> None:
    """Render stale-data status banner sourced from API /health/data-freshness."""
    try:
        snapshot = _load_data_freshness()
    except ApiUnavailableError:
        st.warning("Data freshness status is unavailable (API unreachable).")
        return
    except ApiError as exc:
        st.warning(f"Data freshness status error: {exc.message}")
        return

    overall = snapshot.get("overall_state", "unknown")
    sources = snapshot.get("sources", {})
    stale_sources = snapshot.get("stale_sources", [])
    stale_behavior = snapshot.get("stale_behavior", {})
    idempotency_dashboard = snapshot.get("idempotency_dashboard", {})

    if overall == "healthy":
        st.caption("Data freshness: healthy")
        if idempotency_dashboard:
            with st.expander("Ingestion idempotency dashboard", expanded=False):
                st.json(idempotency_dashboard)
        return

    lines: list[str] = []
    for source_name in stale_sources:
        details = sources.get(source_name, {})
        state = details.get("state", "unknown")
        lines.append(
            f"- `{source_name}` is **{state}** (last seen {_fmt_ts(details.get('last_seen_at'))}, age {_fmt_age(details.get('age_minutes'))})"
        )

    policy = stale_behavior.get("policy", "serve_last_known_data_with_warning")
    forecast_api_mode = stale_behavior.get("forecast_api", "allow_forecast_generation")

    body = "\n".join(lines) if lines else "- Source freshness details unavailable"
    body += (
        f"\n\nPolicy: `{policy}`\n"
        f"Forecast behavior: `{forecast_api_mode}`"
    )

    if overall == "critical":
        st.error(f"Data freshness is critical.\n\n{body}")
    else:
        st.warning(f"Data freshness is degraded.\n\n{body}")

    if idempotency_dashboard:
        with st.expander("Ingestion idempotency dashboard", expanded=False):
            st.json(idempotency_dashboard)
