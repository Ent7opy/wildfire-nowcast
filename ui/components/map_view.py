"""Map view component for wildfire dashboard using PyDeck."""

import logging
import math
from typing import Any, Dict, Optional

import pydeck as pdk
import streamlit as st

from state import app_state, isoformat
from api_client import ApiError, ApiUnavailableError, get_fires
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
    DarkTheme,
)

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fire_severity(det: Dict[str, Any]) -> float:
    fire_likelihood = _safe_float(det.get("fire_likelihood"))
    if fire_likelihood is not None:
        return max(0.0, min(fire_likelihood, 1.0))

    confidence_score = _safe_float(det.get("confidence_score"))
    if confidence_score is not None:
        return max(0.0, min(confidence_score, 1.0))

    confidence = _safe_float(det.get("confidence"))
    if confidence is not None:
        return max(0.0, min(confidence / 100.0, 1.0))

    frp = _safe_float(det.get("frp"))
    if frp is None:
        return 0.0
    if frp >= 80:
        return 0.8
    if frp >= 40:
        return 0.6
    if frp >= 20:
        return 0.4
    if frp >= 5:
        return 0.2
    return 0.1


def _fire_fill_rgba(severity: float) -> list[int]:
    if severity >= FireThresholds.VERY_HIGH:
        return FireColors.VERY_HIGH_FILL
    if severity >= FireThresholds.HIGH:
        return FireColors.HIGH_FILL
    if severity >= FireThresholds.MEDIUM:
        return FireColors.MEDIUM_FILL
    if severity >= FireThresholds.LOW:
        return FireColors.LOW_FILL
    if severity >= 0:
        return FireColors.VERY_LOW_FILL
    return FireColors.UNSCORED_FILL


def _fire_line_rgba(severity: float) -> list[int]:
    if severity >= FireThresholds.HIGH:
        return FireColors.OUTLINE_HIGH
    return FireColors.OUTLINE_DEFAULT


def _fire_radius_m(frp: Optional[float]) -> float:
    if frp is None:
        return float(PointSizing.MIN_SIZE * 1000)
    if frp > PointSizing.LARGE_FRP:
        return float(PointSizing.LARGE_SIZE * 1000)
    if frp > PointSizing.MEDIUM_FRP:
        return float(PointSizing.MEDIUM_SIZE * 1000)
    if frp > PointSizing.SMALL_FRP:
        return float(PointSizing.SMALL_SIZE * 1000)
    return float(PointSizing.MIN_SIZE * 1000)


def _is_active_candidate(det: Dict[str, Any]) -> bool:
    """Heuristic gate for likely active incidents (reduces known low-signal noise)."""
    fire_likelihood = _safe_float(det.get("fire_likelihood"))
    persistence_score = _safe_float(det.get("persistence_score"))
    severity = _fire_severity(det)
    frp = _safe_float(det.get("frp")) or 0.0

    if fire_likelihood is not None and fire_likelihood >= 0.6:
        return True
    if severity >= 0.5 and frp >= 8.0:
        return True
    if persistence_score is not None and persistence_score >= 0.45 and frp >= 5.0:
        return True
    return False


def _cluster_fire_points(points: list[Dict[str, Any]], zoom: float) -> list[Dict[str, Any]]:
    """Aggregate nearby points into incident bubbles to declutter low/mid zooms."""
    if not points:
        return points

    z = max(1.0, min(float(zoom), 10.0))
    cell_deg = max(0.08, 8.0 / (2.0 ** z))
    buckets: dict[tuple[int, int], Dict[str, Any]] = {}

    for p in points:
        lat = _safe_float(p.get("lat"))
        lon = _safe_float(p.get("lon"))
        if lat is None or lon is None:
            continue
        key = (int(math.floor(lat / cell_deg)), int(math.floor(lon / cell_deg)))
        b = buckets.get(key)
        if b is None:
            buckets[key] = {
                "cluster_count": 1,
                "sum_lat": lat,
                "sum_lon": lon,
                "max_severity": float(p.get("_severity", 0.0)),
                "max_frp": _safe_float(p.get("frp")) or 0.0,
                "latest_time": p.get("acq_time"),
                "sample": p,
            }
            continue

        b["cluster_count"] += 1
        b["sum_lat"] += lat
        b["sum_lon"] += lon
        b["max_severity"] = max(float(b["max_severity"]), float(p.get("_severity", 0.0)))
        b["max_frp"] = max(float(b["max_frp"]), _safe_float(p.get("frp")) or 0.0)
        cur_t = p.get("acq_time")
        prev_t = b.get("latest_time")
        if isinstance(cur_t, str) and (not isinstance(prev_t, str) or cur_t > prev_t):
            b["latest_time"] = cur_t
            b["sample"] = p

    out: list[Dict[str, Any]] = []
    for b in buckets.values():
        count = int(b["cluster_count"])
        sample = dict(b["sample"])
        sample["lat"] = float(b["sum_lat"]) / count
        sample["lon"] = float(b["sum_lon"]) / count
        sample["cluster_count"] = count
        sample["acq_time"] = b.get("latest_time")
        sample["frp"] = float(b["max_frp"])
        sample["fire_likelihood"] = float(b["max_severity"])
        sample["sensor"] = "Cluster"
        sample["source"] = "Aggregated detections"
        sample["radius_m"] = float(max(sample.get("radius_m", 0.0), 8000.0 * math.sqrt(max(count, 1))))
        out.append(sample)

    return out


def render_map_view() -> Optional[Dict[str, float]]:
    """Render the PyDeck map view and return click coordinates if any."""

    layers = []

    # 1. Fires Layer (API-backed Scatterplot)
    start_time, end_time = app_state.time_range
    min_likelihood = app_state.filters.min_likelihood

    fire_points: list[Dict[str, Any]] = []
    bbox = app_state.viewport_bbox
    try:
        fires = get_fires(
            bbox=bbox,
            time_range=(start_time, end_time),
            filters={
                "min_fire_likelihood": min_likelihood,
                "include_noise": False,
                "include_denoiser_fields": True,
                "limit": 10000,
            },
        ).get("detections", [])
    except (ApiUnavailableError, ApiError) as exc:
        LOGGER.warning("Failed to fetch fires for map layer: %s", exc)
        fires = []

    for det in fires:
        if app_state.filters.active_only and not _is_active_candidate(det):
            continue
        lat = _safe_float(det.get("lat"))
        lon = _safe_float(det.get("lon"))
        if lat is None or lon is None:
            continue
        severity = _fire_severity(det)
        fill = _fire_fill_rgba(severity)
        line = _fire_line_rgba(severity)
        frp = _safe_float(det.get("frp"))
        point = dict(det)
        point["fill_r"] = int(fill[0])
        point["fill_g"] = int(fill[1])
        point["fill_b"] = int(fill[2])
        point["fill_a"] = int(fill[3])
        point["line_r"] = int(line[0])
        point["line_g"] = int(line[1])
        point["line_b"] = int(line[2])
        point["line_a"] = int(line[3])
        point["radius_m"] = _fire_radius_m(frp)
        point["_severity"] = severity
        point["cluster_count"] = 1
        fire_points.append(point)

    if app_state.filters.cluster_points:
        zoom = float(getattr(st.session_state.get("map_view_state"), "zoom", 2.0))
        fire_points = _cluster_fire_points(fire_points, zoom)

    # Include filter params in the layer ID so deck.gl fully recreates
    # the layer when filters change.
    fires_layer_id = f"fires-{min_likelihood}-{isoformat(start_time)}"

    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=fire_points,
        id=fires_layer_id,
        pickable=True,
        auto_highlight=True,
        get_position="[lon, lat]",
        filled=True,
        get_fill_color="[fill_r, fill_g, fill_b, fill_a]",
        get_radius="radius_m",
        radius_units="meters",
        radius_min_pixels=PointSizing.MIN_PIXELS,
        radius_max_pixels=PointSizing.MAX_PIXELS,
        stroked=True,
        get_line_color="[line_r, line_g, line_b, line_a]",
        line_width_min_pixels=1,
    ))

    # 2. Forecast Contours (MVT)
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

    # Create the Deck with dark basemap
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=st.session_state.map_view_state,
        map_style=MapConfig.BASEMAP_DARK,
        tooltip={
            "html": (
                '<div style="font-family:Inter,sans-serif;padding:2px;">'
                '<div style="font-size:13px;font-weight:600;color:#ff6b35;margin-bottom:4px;">'
                'Fire Detection</div>'
                '<div style="font-size:12px;color:#e0e0e0;">'
                '<b>Cluster size:</b> {cluster_count}<br/>'
                '<b>Time:</b> {acq_time}<br/>'
                '<b>Sensor:</b> {sensor}<br/>'
                '<b>FRP:</b> {frp} MW<br/>'
                '<b>Likelihood:</b> {fire_likelihood}<br/>'
                '<b>Confidence:</b> {confidence_score}'
                '</div></div>'
            ),
            "style": {
                "color": UIColors.TOOLTIP_TEXT,
                "backgroundColor": UIColors.TOOLTIP_BG,
                "borderRadius": "8px",
                "border": f"1px solid {DarkTheme.BORDER_SUBTLE}",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.3)",
                "fontSize": "12px",
                "padding": "8px 12px",
            },
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
        all_keys = list(event.selection.objects.keys())
        LOGGER.debug("Selection event objects keys: %s", all_keys)

        # Find selected fire objects by matching layer ID prefix, then fall back
        selected_fires = []
        for key, objects in event.selection.objects.items():
            if objects and key.startswith("fires"):
                selected_fires = objects
                break
        if not selected_fires:
            for key, objects in event.selection.objects.items():
                if objects:
                    LOGGER.debug(
                        "No objects under 'fires*'; using key '%s' (%d objects)",
                        key,
                        len(objects),
                    )
                    selected_fires = objects
                    break

        if selected_fires:
            feature = selected_fires[0]
            LOGGER.debug("Selected feature keys: %s", list(feature.keys()))

            props = feature.get("properties", feature)
            if "properties" not in feature:
                LOGGER.debug("Feature has no 'properties' key — using feature dict directly")

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
