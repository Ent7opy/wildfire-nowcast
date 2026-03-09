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

from config.theme import FilterPresets

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
    active_only: bool = True
    cluster_points: bool = True


@dataclass
class LayerState:
    show_fires: bool = True
    show_forecast: bool = True
    show_risk: bool = False


@dataclass
class SelectionState:
    selected_fire: dict | None = None
    last_click: dict | None = None
    front_index_by_event: dict[str, dict] = field(default_factory=dict)

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
class ForecastDisplayState:
    forecast_radius_km: int = 20
    show_h24: bool = True
    show_h48: bool = False
    show_h72: bool = False
    show_t07: bool = True
    show_t05: bool = False
    show_t03: bool = False


@dataclass
class ForecastJobState:
    job_id: str | None = None
    poll_count: int = 0
    last_forecast: dict | None = field(default=None)
    active_request: dict | None = field(default=None)
    notification: dict | None = field(default=None)

    # -- convenience helpers ------------------------------------------------

    def start(self, job_id: str, request_context: dict | None = None) -> None:
        self.job_id = job_id
        self.poll_count = 0
        self.active_request = request_context if isinstance(request_context, dict) else None

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
        if isinstance(self.active_request, dict):
            self.last_forecast.update(self.active_request)
        self.job_id = None
        self.poll_count = 0
        self.active_request = None

    def clear(self) -> None:
        """Clear all polling state (failure / timeout)."""
        self.job_id = None
        self.poll_count = 0
        self.active_request = None


# ---------------------------------------------------------------------------
# Main state manager
# ---------------------------------------------------------------------------

class AppState:
    """Typed façade over ``st.session_state``."""

    def __init__(self) -> None:
        self.filters = FilterState()
        self.layers = LayerState()
        self.selection = SelectionState()
        self.forecast_display = ForecastDisplayState()
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
        if self.filters.hours_end == 0:
            if hours == 1:
                return "Last 1 hour"
            return f"Last {hours} hours"
        return f"{hours}h window ({self.filters.hours_start}h ago to {self.filters.hours_end}h ago)"

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
        likelihood: float,
    ) -> None:
        """Atomically apply a filter preset (filters + widget keys)."""
        self.filters.hours_start = hours_start
        self.filters.hours_end = hours_end
        self.filters.min_likelihood = likelihood
        self.active_preset = name
        self._preset_applied = True

        # Write widget keys so Streamlit picks up new values on rerun
        st.session_state.timeline_scrubber = (hours_end, hours_start)
        st.session_state.min_likelihood = likelihood

        self._persist()

    def get_matching_preset(self) -> str | None:
        """Return the preset name that matches current filters, or *None*."""
        f = self.filters
        for name, hs, he, lk in FilterPresets.all_presets():
            if (hs == f.hours_start
                    and he == f.hours_end
                    and abs(lk - f.min_likelihood) < 0.01):
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
        if "active_only" not in st.session_state:
            st.session_state.active_only = f.active_only
        if "cluster_points" not in st.session_state:
            st.session_state.cluster_points = f.cluster_points
        lyr = self.layers
        if "risk_checkbox" not in st.session_state:
            st.session_state.risk_checkbox = lyr.show_risk
        fd = self.forecast_display
        if "forecast_show_h24" not in st.session_state:
            st.session_state.forecast_show_h24 = bool(fd.show_h24)
        if "forecast_show_h48" not in st.session_state:
            st.session_state.forecast_show_h48 = bool(fd.show_h48)
        if "forecast_show_h72" not in st.session_state:
            st.session_state.forecast_show_h72 = bool(fd.show_h72)
        if "forecast_show_t07" not in st.session_state:
            st.session_state.forecast_show_t07 = bool(fd.show_t07)
        if "forecast_show_t05" not in st.session_state:
            st.session_state.forecast_show_t05 = bool(fd.show_t05)
        if "forecast_show_t03" not in st.session_state:
            st.session_state.forecast_show_t03 = bool(fd.show_t03)

    def read_widgets_after_render(self) -> None:
        """Pull filter widget values → canonical state (call *after* filter widgets render).

        Also handles change-detection for active preset.

        Layer controls are handled in ``sidebar.py`` after rendering.
        """
        f = self.filters
        prev = (f.hours_start, f.hours_end, f.min_likelihood, f.active_only, f.cluster_points)

        # Timeline scrubber → hours
        scrubber = st.session_state.get("timeline_scrubber", (f.hours_end, f.hours_start))
        end_hours, start_hours = scrubber
        if start_hours <= end_hours:
            start_hours = end_hours + 1
        f.hours_start = start_hours
        f.hours_end = end_hours

        # Likelihood
        f.min_likelihood = st.session_state.get("min_likelihood", f.min_likelihood)
        f.active_only = bool(st.session_state.get("active_only", f.active_only))
        f.cluster_points = bool(st.session_state.get("cluster_points", f.cluster_points))
        fd = self.forecast_display
        fd.forecast_radius_km = 20
        fd.show_h24 = bool(st.session_state.get("forecast_show_h24", fd.show_h24))
        fd.show_h48 = bool(st.session_state.get("forecast_show_h48", fd.show_h48))
        fd.show_h72 = bool(st.session_state.get("forecast_show_h72", fd.show_h72))
        # Keep at least one horizon visible.
        if not (fd.show_h24 or fd.show_h48 or fd.show_h72):
            fd.show_h24 = True

        fd.show_t07 = bool(st.session_state.get("forecast_show_t07", fd.show_t07))
        fd.show_t05 = bool(st.session_state.get("forecast_show_t05", fd.show_t05))
        fd.show_t03 = bool(st.session_state.get("forecast_show_t03", fd.show_t03))
        # Keep at least one threshold visible.
        if not (fd.show_t07 or fd.show_t05 or fd.show_t03):
            fd.show_t07 = True

        # Detect manual filter changes → update active_preset
        cur = (f.hours_start, f.hours_end, f.min_likelihood, f.active_only, f.cluster_points)
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
        st.query_params["active_only"] = "true" if f.active_only else "false"
        st.query_params["cluster"] = "true" if f.cluster_points else "false"

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
        s.fires_active_only = self.filters.active_only
        s.fires_cluster_points = self.filters.cluster_points

        # Layers
        s.show_fires = True
        s.show_forecast = True
        s.show_risk = self.layers.show_risk

        # Selection
        s.selected_fire = self.selection.selected_fire
        s.last_click = self.selection.last_click
        s.front_index_by_event = self.selection.front_index_by_event

        # Forecast display controls (persist under non-widget keys to avoid
        # Streamlit runtime errors when widget keys are already instantiated).
        s.forecast_display_radius_km = int(self.forecast_display.forecast_radius_km)
        s.forecast_display_show_h24 = bool(self.forecast_display.show_h24)
        s.forecast_display_show_h48 = bool(self.forecast_display.show_h48)
        s.forecast_display_show_h72 = bool(self.forecast_display.show_h72)
        s.forecast_display_show_t07 = bool(self.forecast_display.show_t07)
        s.forecast_display_show_t05 = bool(self.forecast_display.show_t05)
        s.forecast_display_show_t03 = bool(self.forecast_display.show_t03)

        # Forecast job
        if self.forecast_job.job_id is not None:
            s.jit_job_id = self.forecast_job.job_id
        elif "jit_job_id" in s:
            del s.jit_job_id
        s.jit_poll_count = self.forecast_job.poll_count
        s.last_forecast = self.forecast_job.last_forecast
        s.jit_active_request = self.forecast_job.active_request
        s.jit_notification = self.forecast_job.notification

        # Preset
        s.active_preset = self.active_preset

    def _restore(self) -> None:
        """Read canonical fields from ``st.session_state``."""
        s = st.session_state

        self.filters = FilterState(
            hours_start=s.get("time_range_hours_start", 24),
            hours_end=s.get("time_range_hours_end", 0),
            min_likelihood=s.get("fires_min_likelihood", 0.0),
            active_only=s.get("fires_active_only", True),
            cluster_points=s.get("fires_cluster_points", True),
        )
        self.layers = LayerState(
            show_fires=True,
            show_forecast=True,
            show_risk=s.get("show_risk", False),
        )
        self.selection = SelectionState(
            selected_fire=s.get("selected_fire"),
            last_click=s.get("last_click"),
            front_index_by_event=s.get("front_index_by_event", {}),
        )
        self.forecast_display = ForecastDisplayState(
            forecast_radius_km=20,
            show_h24=bool(
                s.get("forecast_display_show_h24", s.get("forecast_show_h24", True))
            ),
            show_h48=bool(
                s.get("forecast_display_show_h48", s.get("forecast_show_h48", False))
            ),
            show_h72=bool(
                s.get("forecast_display_show_h72", s.get("forecast_show_h72", False))
            ),
            show_t07=bool(
                s.get("forecast_display_show_t07", s.get("forecast_show_t07", True))
            ),
            show_t05=bool(
                s.get("forecast_display_show_t05", s.get("forecast_show_t05", False))
            ),
            show_t03=bool(
                s.get("forecast_display_show_t03", s.get("forecast_show_t03", False))
            ),
        )
        self.forecast_job = ForecastJobState(
            job_id=s.get("jit_job_id"),
            poll_count=s.get("jit_poll_count", 0),
            last_forecast=s.get("last_forecast"),
            active_request=s.get("jit_active_request"),
            notification=s.get("jit_notification"),
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
        if "active_only" in params:
            f.active_only = str(params["active_only"]).strip().lower() in ("1", "true", "yes", "on")
        if "cluster" in params:
            f.cluster_points = str(params["cluster"]).strip().lower() in ("1", "true", "yes", "on")
        # Determine active preset
        match = self.get_matching_preset()
        if match:
            self.active_preset = match
        elif any(k in params for k in ("start", "end", "likelihood")):
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
