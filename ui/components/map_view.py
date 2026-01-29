"""Map view component for wildfire dashboard using PyDeck."""

import logging
from typing import Dict, Optional

import pydeck as pdk
import streamlit as st

from state import app_state, isoformat
from runtime_config import api_public_base_url
from config.theme import (
    FireColors,
    FireThresholds,
    RiskColors,
    RiskThresholds,
    ForecastColors,
    PointSizing,
    MapConfig,
    UIColors,
)

LOGGER = logging.getLogger(__name__)


def render_map_view() -> Optional[Dict[str, float]]:
    """Render the PyDeck map view and return click coordinates if any."""

    layers = []

    # 1. Fires Layer (MVT)
    if app_state.layers.show_fires:
        start_time, end_time = app_state.time_range
        include_noise = not app_state.filters.apply_denoiser
        min_likelihood = app_state.filters.min_likelihood

        # Build query params for the tile URL
        params = {
            "start_time": isoformat(start_time),
            "end_time": isoformat(end_time),
            "min_fire_likelihood": min_likelihood,
            "include_noise": str(include_noise).lower(),
        }
        query_str = "&".join([f"{k}={v}" for k, v in params.items()])

        tile_url = f"{api_public_base_url()}/tiles/fires/{{z}}/{{x}}/{{y}}.pbf?{query_str}"

        layers.append(pdk.Layer(
            "MVTLayer",
            data=tile_url,
            id="fires",
            pickable=True,
            auto_highlight=True,
            get_fill_color=f"properties.fire_likelihood >= {FireThresholds.HIGH} ? {FireColors.HIGH_FILL} : properties.fire_likelihood >= {FireThresholds.MEDIUM} ? {FireColors.MEDIUM_FILL} : {FireColors.LOW_FILL}",
            get_point_radius=f"properties.frp > {PointSizing.LARGE_FRP} ? {PointSizing.LARGE_SIZE} : properties.frp > {PointSizing.MEDIUM_FRP} ? {PointSizing.MEDIUM_SIZE} : properties.frp > {PointSizing.SMALL_FRP} ? {PointSizing.SMALL_SIZE} : {PointSizing.MIN_SIZE}",
            point_radius_min_pixels=PointSizing.MIN_PIXELS,
            point_radius_max_pixels=PointSizing.MAX_PIXELS,
            stroked=True,
            get_line_color=FireColors.OUTLINE,
            line_width_min_pixels=1,
        ))

    # 2. Forecast Contours (MVT)
    if app_state.layers.show_forecast:
        last = app_state.forecast_job.last_forecast
        run_id = (last or {}).get("run", {}).get("id")
        contour_url = f"{api_public_base_url()}/tiles/forecast_contours/{{z}}/{{x}}/{{y}}.pbf"
        if run_id:
            contour_url += f"?run_id={run_id}"

        layers.append(pdk.Layer(
            "MVTLayer",
            data=contour_url,
            id="forecast_contours",
            pickable=False,
            get_fill_color=ForecastColors.FILL,
            get_line_color=ForecastColors.STROKE,
            get_line_width=2,
            line_width_min_pixels=1,
        ))

    # 3. Risk Index Layer (GeoJSON)
    if app_state.layers.show_risk:
        view_state = st.session_state.get("map_view_state")
        if view_state:
            lat = view_state.latitude
            lon = view_state.longitude
            zoom = view_state.zoom

            degrees_per_tile = 360.0 / (2 ** zoom)
            half = degrees_per_tile * 0.5

            min_lon = max(lon - half, -180.0)
            max_lon = min(lon + half, 180.0)
            min_lat = max(lat - half, -85.0)
            max_lat = min(lat + half, 85.0)

            risk_url = (
                f"{api_public_base_url()}/risk?"
                f"min_lon={min_lon}&min_lat={min_lat}&max_lon={max_lon}&max_lat={max_lat}"
            )

            layers.append(pdk.Layer(
                "GeoJsonLayer",
                data=risk_url,
                id="risk",
                pickable=False,
                stroked=True,
                filled=True,
                get_fill_color=f"properties.risk_score < {RiskThresholds.MEDIUM} ? {RiskColors.LOW_FILL} : properties.risk_score < {RiskThresholds.HIGH} ? {RiskColors.MEDIUM_FILL} : {RiskColors.HIGH_FILL}",
                get_line_color=f"properties.risk_score < {RiskThresholds.MEDIUM} ? {RiskColors.LOW_STROKE} : properties.risk_score < {RiskThresholds.HIGH} ? {RiskColors.MEDIUM_STROKE} : {RiskColors.HIGH_STROKE}",
                line_width_min_pixels=1,
            ))

    # Create the Deck
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=st.session_state.map_view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": "<b>Time:</b> {acq_time}<br/>"
                    "<b>Sensor:</b> {sensor}<br/>"
                    "<b>Intensity (FRP):</b> {frp}<br/>"
                    "<b>Likelihood:</b> {fire_likelihood}",
            "style": {"color": UIColors.TOOLTIP_TEXT, "backgroundColor": UIColors.TOOLTIP_BG}
        },
    )

    # Render with selection support
    event = st.pydeck_chart(
        deck,
        height=MapConfig.HEIGHT,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-object",
        key="main_map",
    )

    # Handle interactions
    if event and event.selection:
        selected_fires = event.selection.objects.get("fires", [])
        if selected_fires:
            feature = selected_fires[0]

            props = feature.get("properties", feature)

            lat = props.get("lat")
            lon = props.get("lon")

            if (lat is None or lon is None) and "geometry" in feature:
                geom = feature["geometry"]
                if geom.get("type") == "Point" and "coordinates" in geom:
                    coords = geom["coordinates"]
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]

            if lat is None or lon is None:
                LOGGER.warning(
                    "Failed to extract coordinates from MVT feature. "
                    "Feature structure: %s",
                    feature,
                )

            normalized_feature = dict(props)
            if lat is not None and lon is not None:
                normalized_feature["lat"] = lat
                normalized_feature["lon"] = lon

            app_state.selection.selected_fire = normalized_feature
            app_state._persist()
            return {"lat": lat, "lng": lon}

    return None
