"""Map view component for wildfire dashboard."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import folium
import httpx
import streamlit as st
from streamlit_folium import st_folium

from ui.api_client import ApiError, ApiUnavailableError, get_forecast, get_fires
from ui.config.constants import (
    DEFAULT_MAP_CENTER,
    DEFAULT_ZOOM_LEVEL,
    MAP_HEIGHT,
    PLACEHOLDER_RISK_POLYGON,
)


def _current_bbox() -> tuple[float, float, float, float]:
    """Return current bbox (min_lon, min_lat, max_lon, max_lat)."""
    if "map_bounds" in st.session_state and st.session_state.map_bounds:
        b = st.session_state.map_bounds
        # st-folium bounds format: {'_southWest': {'lat': 41.0, 'lng': 22.0}, ...}
        return (
            b["_southWest"]["lng"],
            b["_southWest"]["lat"],
            b["_northEast"]["lng"],
            b["_northEast"]["lat"],
        )

    # Default bbox around Bulgaria area
    return (22.0, 41.0, 29.0, 44.5)


def _current_time_range() -> tuple[datetime, datetime]:
    """Return (start_time, end_time) based on selected time window."""
    end = datetime.now(timezone.utc)
    window = getattr(st.session_state, "time_window", "Last 24 hours")
    hours = 24
    if window == "Last 6 hours":
        hours = 6
    elif window == "Last 12 hours":
        hours = 12
    elif window == "Last 48 hours":
        hours = 48
    start = end - timedelta(hours=hours)
    return start, end


def add_fire_markers(map_obj: folium.Map, detections: List[Dict[str, Any]]) -> None:
    """Add fire detection markers to the map."""
    for det in detections:
        lat = det.get("lat")
        lon = det.get("lon")
        if lat is None or lon is None:
            continue

        tooltip = "Active fire detection"
        popup_parts = [
            f"id: {det.get('id')}",
            f"time: {det.get('acq_time')}",
            f"confidence: {det.get('confidence')}",
            f"frp: {det.get('frp')}",
            f"sensor: {det.get('sensor')}",
            f"source: {det.get('source')}",
        ]
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup="<br/>".join(popup_parts),
            tooltip=tooltip,
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.7,
        ).add_to(map_obj)


def add_forecast_layers(map_obj: folium.Map) -> None:
    """Fetch and add real forecast layers to the map."""
    bbox = _current_bbox()

    try:
        forecast_data = get_forecast(bbox, horizons=[24, 48, 72])
    except ApiUnavailableError:
        st.error("API unavailable — please start the backend")
        return
    except ApiError as e:
        details = f"(status={e.status_code})" if e.status_code is not None else ""
        st.error(f"Forecast API error {details}".strip())
        if e.response_text:
            st.caption(e.response_text[:300])
        return

    # 1. Add contours
    if forecast_data.get("contours") and forecast_data["contours"].get("features"):
        folium.GeoJson(
            forecast_data["contours"],
            name="Forecast Contours",
            style_function=lambda x: {
                "color": "orange",
                "weight": 2,
                "fillOpacity": 0.1,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["horizon_hours", "threshold"], aliases=["Horizon (h)", "Probability"]
            ),
        ).add_to(map_obj)

    # 2. Add raster tiles (latest horizon)
    if forecast_data.get("rasters"):
        # Show the longest horizon by default for visual impact
        r = sorted(forecast_data["rasters"], key=lambda x: x["horizon_hours"])[-1]
        tilejson_url = r["tilejson_url"]
        try:
            # We need to resolve TileJSON to get the actual tile template for Leaflet.
            # The URL from the API is external (e.g. localhost:8080), but we are inside Docker
            # so we need to use the internal service name.
            # TODO: Make these hostnames configurable via env vars
            internal_url = tilejson_url.replace("localhost:8080", "titiler:8000")

            resp = httpx.get(internal_url, timeout=5.0)
            if resp.status_code == 200:
                tj = resp.json()
                tile_url = tj["tiles"][0]

                # The tile URL returned by TiTiler will likely be internal (titiler:8000)
                # because we accessed it via that hostname. We need to convert it back
                # to external for the browser.
                tile_url_external = tile_url.replace("titiler:8000", "localhost:8080")

                folium.TileLayer(
                    tiles=tile_url_external,
                    attr="TiTiler",
                    name=f"Spread Probability (T+{r['horizon_hours']}h)",
                    overlay=True,
                    opacity=0.5,
                    control=True,
                ).add_to(map_obj)
        except Exception as e:
            st.warning(f"Could not load raster tiles: {e}")


def add_risk_polygon(map_obj: folium.Map, polygon: List[List[float]]) -> None:
    """Add placeholder risk polygon to the map."""
    folium.Polygon(
        locations=polygon,
        popup="Fire risk index (placeholder)",
        tooltip="Risk index",
        color="purple",
        fill=True,
        fillColor="purple",
        fillOpacity=0.2,
        weight=2,
    ).add_to(map_obj)


def create_base_map() -> folium.Map:
    """Create a base Folium map with default settings."""
    return folium.Map(
        location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_ZOOM_LEVEL, tiles="OpenStreetMap"
    )


def render_map_view() -> Optional[Dict[str, float]]:
    """Render the map view and return click coordinates if any."""
    try:
        m = create_base_map()

        # Add layers based on toggles
        if st.session_state.show_fires:
            bbox = _current_bbox()
            start_time, end_time = _current_time_range()
            try:
                include_noise = not bool(getattr(st.session_state, "fires_apply_denoiser", True))
                min_confidence = float(getattr(st.session_state, "fires_min_confidence", 0.0))
                fires_data = get_fires(
                    bbox=bbox,
                    time_range=(start_time, end_time),
                    filters={
                        "limit": 10000,
                        "include_noise": include_noise,
                        "min_confidence": min_confidence,
                    },
                )
                add_fire_markers(m, fires_data.get("detections", []))
            except ApiUnavailableError:
                st.error("API unavailable — please start the backend")
            except ApiError as e:
                details = f"(status={e.status_code})" if e.status_code is not None else ""
                st.error(f"Fires API error {details}".strip())
                if e.response_text:
                    st.caption(e.response_text[:300])

        if st.session_state.show_forecast:
            add_forecast_layers(m)

        if st.session_state.show_risk:
            add_risk_polygon(m, PLACEHOLDER_RISK_POLYGON)

        # Add layer control to toggle between contours/rasters if both present
        folium.LayerControl().add_to(m)

        # Render map and capture interactions
        map_data = st_folium(
            m, width=None, height=MAP_HEIGHT, returned_objects=["last_clicked", "bounds"]
        )

        # Update bounds in session state for next rerun (for API querying)
        if map_data.get("bounds"):
            st.session_state.map_bounds = map_data["bounds"]

        # Handle map click safely
        last_clicked = map_data.get("last_clicked")
        if last_clicked is not None:
            clicked_lat = last_clicked.get("lat")
            clicked_lng = last_clicked.get("lng")
            if clicked_lat is not None and clicked_lng is not None:
                return {"lat": clicked_lat, "lng": clicked_lng}

    except Exception as e:
        st.error(f"Error rendering map: {e}")
        st.info(
            "Please refresh the page. If the problem persists, check your connection and try again."
        )
        st.stop()

    return None
