"""Centralized state manager for the Streamlit UI.

Wraps st.session_state with typed dataclasses, eliminating scattered raw
key access across components.  Every piece of application state lives here
— defaults are defined once, widget-key sync is internal, and derived
values (time_window, time_range) are computed properties.

Usage
-----
    from state import app_state

    # At top of main():
    app_state.initialize()

    # Read anywhere:
    if app_state.layers.show_fires:
        ...

    # Sidebar lifecycle:
    app_state.sync_widgets_before_render()
    # ... render Streamlit widgets ...
    app_state.read_widgets_after_render()
    app_state.sync_to_url()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pydeck as pdk
import streamlit as st

from config.theme import FilterPresets, MapConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def isoformat(dt: datetime) -> str:
    """Format *dt* for API query parameters (UTC → ``…Z`` suffix)."""
    offset = dt.utcoffset() if dt.tzinfo is not None else None
    if offset is not None and offset.total_seconds() == 0:
        dt_clean = dt.replace(microsecond=0)
        return dt_clean.replace(tzinfo=None).isoformat() + "Z"
    return dt.replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FilterState:
    hours_start: int = 24
    hours_end: int = 0
    min_likelihood: float = 0.0
    apply_denoiser: bool = True


@dataclass
class LayerState:
    show_fires: bool = True
    show_forecast: bool = False
    show_risk: bool = False


@dataclass
class SelectionState:
    selected_fire: dict | None = None
    last_click: dict | None = None

    def update_click(self, coords: dict | None) -> None:
        """Update *last_click* only when the coordinates actually change."""
        if coords is None:
            return
        cur = self.last_click
        if (cur is None
                or cur.get("lat") != coords.get("lat")
                or cur.get("lng") != coords.get("lng")):
            self.last_click = coords


@dataclass
class ForecastJobState:
    job_id: str | None = None
    poll_count: int = 0
    last_forecast: dict | None = field(default=None)

    # -- convenience helpers ------------------------------------------------

    def start(self, job_id: str) -> None:
        self.job_id = job_id
        self.poll_count = 0

    def increment_poll(self) -> None:
        self.poll_count += 1

    def complete(self, run_id: str, job_id: str) -> None:
        """Record a successful forecast completion."""
        import time as _time

        self.last_forecast = {
            "run": {"id": run_id},
            "job_id": job_id,
            "completed_at": _time.time(),
        }
        self.job_id = None
        self.poll_count = 0

    def clear(self) -> None:
        """Clear all polling state (failure / timeout)."""
        self.job_id = None
        self.poll_count = 0


# ---------------------------------------------------------------------------
# Main state manager
# ---------------------------------------------------------------------------

class AppState:
    """Typed façade over ``st.session_state``."""

    def __init__(self) -> None:
        self.filters = FilterState()
        self.layers = LayerState()
        self.selection = SelectionState()
        self.forecast_job = ForecastJobState()
        self.active_preset: str | None = None
        self._preset_applied: bool = False

    # -- lifecycle -----------------------------------------------------------

    def initialize(self) -> None:
        """Restore state from ``st.session_state`` (or bootstrap on first run).

        Must be called exactly once at the top of ``main()``.
        """
        self._preset_applied = False

        if "_state_initialized" not in st.session_state:
            # First run — load from URL, then fill defaults
            self._load_from_url()
            self._persist()
            self._init_map_view_state()
            st.session_state._state_initialized = True
        else:
            self._restore()

    # -- computed properties -------------------------------------------------

    @property
    def time_window(self) -> str:
        """Human-readable label derived from the current time range."""
        hours = self.filters.hours_start - self.filters.hours_end
        if hours <= 6:
            return "Last 6 hours"
        if hours <= 12:
            return "Last 12 hours"
        if hours <= 24:
            return "Last 24 hours"
        return "Last 48 hours"

    @property
    def time_range(self) -> tuple[datetime, datetime]:
        """``(start_utc, end_utc)`` computed from the filter hours."""
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        end = now - timedelta(hours=self.filters.hours_end)
        start = end - timedelta(hours=(self.filters.hours_start - self.filters.hours_end))
        return start, end

    # -- viewport helpers ----------------------------------------------------

    @property
    def viewport_bbox(self) -> tuple[float, float, float, float]:
        """``(min_lon, min_lat, max_lon, max_lat)`` from the current map view."""
        vs = st.session_state.get("map_view_state")
        if vs is None:
            return (-180.0, -85.0, 180.0, 85.0)

        deg_per_tile = 360.0 / (2 ** vs.zoom)
        half = deg_per_tile * 0.5
        return (
            max(vs.longitude - half, -180.0),
            max(vs.latitude - half, -85.0),
            min(vs.longitude + half, 180.0),
            min(vs.latitude + half, 85.0),
        )

    # -- preset management ---------------------------------------------------

    def apply_preset(
        self, name: str, hours_start: int, hours_end: int,
        likelihood: float, denoiser: bool,
    ) -> None:
        """Atomically apply a filter preset (filters + widget keys)."""
        self.filters.hours_start = hours_start
        self.filters.hours_end = hours_end
        self.filters.min_likelihood = likelihood
        self.filters.apply_denoiser = denoiser
        self.active_preset = name
        self._preset_applied = True

        # Write widget keys so Streamlit picks up new values on rerun
        st.session_state.timeline_scrubber = (hours_end, hours_start)
        st.session_state.min_likelihood = likelihood
        st.session_state.apply_denoiser = denoiser

        self._persist()

    def get_matching_preset(self) -> str | None:
        """Return the preset name that matches current filters, or *None*."""
        f = self.filters
        for name, hs, he, lk, dn in FilterPresets.all_presets():
            if (hs == f.hours_start
                    and he == f.hours_end
                    and abs(lk - f.min_likelihood) < 0.01
                    and dn == f.apply_denoiser):
                return name
        return None

    # -- widget sync ---------------------------------------------------------

    def sync_widgets_before_render(self) -> None:
        """Push canonical state → widget keys (call *before* widgets render)."""
        f = self.filters
        if "timeline_scrubber" not in st.session_state:
            st.session_state.timeline_scrubber = (f.hours_end, f.hours_start)
        if "min_likelihood" not in st.session_state:
            st.session_state.min_likelihood = f.min_likelihood
        if "apply_denoiser" not in st.session_state:
            st.session_state.apply_denoiser = f.apply_denoiser

        lyr = self.layers
        if "fires_checkbox" not in st.session_state:
            st.session_state.fires_checkbox = lyr.show_fires
        if "forecast_checkbox" not in st.session_state:
            st.session_state.forecast_checkbox = lyr.show_forecast
        if "risk_checkbox" not in st.session_state:
            st.session_state.risk_checkbox = lyr.show_risk

    def read_widgets_after_render(self) -> None:
        """Pull filter widget values → canonical state (call *after* filter widgets render).

        Also handles change-detection for active preset.

        Note: layer checkboxes render later in the sidebar and are read
        separately in ``sidebar.py`` after they render.
        """
        f = self.filters
        prev = (f.hours_start, f.hours_end, f.min_likelihood, f.apply_denoiser)

        # Timeline scrubber → hours
        scrubber = st.session_state.get("timeline_scrubber", (f.hours_end, f.hours_start))
        end_hours, start_hours = scrubber
        if start_hours <= end_hours:
            start_hours = end_hours + 1
        f.hours_start = start_hours
        f.hours_end = end_hours

        # Likelihood & denoiser
        f.min_likelihood = st.session_state.get("min_likelihood", f.min_likelihood)
        f.apply_denoiser = st.session_state.get("apply_denoiser", f.apply_denoiser)

        # Detect manual filter changes → update active_preset
        cur = (f.hours_start, f.hours_end, f.min_likelihood, f.apply_denoiser)
        if not self._preset_applied and cur != prev:
            match = self.get_matching_preset()
            self.active_preset = match if match else "Custom"

        self._persist()

    # -- URL sync ------------------------------------------------------------

    def sync_to_url(self) -> None:
        """Write current filter state to URL query parameters."""
        f = self.filters
        st.query_params["start"] = str(f.hours_start)
        st.query_params["end"] = str(f.hours_end)
        st.query_params["likelihood"] = f"{f.min_likelihood:.2f}"
        st.query_params["denoiser"] = str(f.apply_denoiser).lower()

        if self.active_preset:
            st.query_params["preset"] = self.active_preset
        elif "preset" in st.query_params:
            del st.query_params["preset"]

    # -- persistence to / from st.session_state ------------------------------

    def _persist(self) -> None:
        """Write all canonical fields into ``st.session_state``."""
        s = st.session_state

        # Filters
        s.time_range_hours_start = self.filters.hours_start
        s.time_range_hours_end = self.filters.hours_end
        s.fires_min_likelihood = self.filters.min_likelihood
        s.fires_apply_denoiser = self.filters.apply_denoiser

        # Layers
        s.show_fires = self.layers.show_fires
        s.show_forecast = self.layers.show_forecast
        s.show_risk = self.layers.show_risk

        # Selection
        s.selected_fire = self.selection.selected_fire
        s.last_click = self.selection.last_click

        # Forecast job
        if self.forecast_job.job_id is not None:
            s.jit_job_id = self.forecast_job.job_id
        elif "jit_job_id" in s:
            del s.jit_job_id
        s.jit_poll_count = self.forecast_job.poll_count
        s.last_forecast = self.forecast_job.last_forecast

        # Preset
        s.active_preset = self.active_preset

    def _restore(self) -> None:
        """Read canonical fields from ``st.session_state``."""
        s = st.session_state

        self.filters = FilterState(
            hours_start=s.get("time_range_hours_start", 24),
            hours_end=s.get("time_range_hours_end", 0),
            min_likelihood=s.get("fires_min_likelihood", 0.0),
            apply_denoiser=s.get("fires_apply_denoiser", True),
        )
        self.layers = LayerState(
            show_fires=s.get("show_fires", True),
            show_forecast=s.get("show_forecast", False),
            show_risk=s.get("show_risk", False),
        )
        self.selection = SelectionState(
            selected_fire=s.get("selected_fire"),
            last_click=s.get("last_click"),
        )
        self.forecast_job = ForecastJobState(
            job_id=s.get("jit_job_id"),
            poll_count=s.get("jit_poll_count", 0),
            last_forecast=s.get("last_forecast"),
        )
        self.active_preset = s.get("active_preset")

    def _load_from_url(self) -> None:
        """Bootstrap state from URL query parameters (first page load)."""
        params = st.query_params
        f = self.filters

        if "start" in params:
            try:
                f.hours_start = int(params["start"])
            except ValueError:
                pass
        if "end" in params:
            try:
                f.hours_end = int(params["end"])
            except ValueError:
                pass
        if "likelihood" in params:
            try:
                f.min_likelihood = float(params["likelihood"])
            except ValueError:
                pass
        if "denoiser" in params:
            f.apply_denoiser = params["denoiser"].lower() == "true"

        # Determine active preset
        match = self.get_matching_preset()
        if match:
            self.active_preset = match
        elif any(k in params for k in ("start", "end", "likelihood", "denoiser")):
            self.active_preset = "Custom"

    @staticmethod
    def _init_map_view_state() -> None:
        """Ensure the PyDeck ViewState exists in session state."""
        if "map_view_state" not in st.session_state:
            from config.constants import DEFAULT_MAP_CENTER, DEFAULT_ZOOM_LEVEL

            st.session_state.map_view_state = pdk.ViewState(
                latitude=DEFAULT_MAP_CENTER[0],
                longitude=DEFAULT_MAP_CENTER[1],
                zoom=DEFAULT_ZOOM_LEVEL,
                pitch=0,
                bearing=0,
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

app_state = AppState()
