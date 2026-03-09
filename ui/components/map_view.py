"""Map view component for wildfire dashboard using PyDeck."""

import json
import logging
import math
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional

import pydeck as pdk
import streamlit as st

from state import app_state, isoformat
from api_client import ApiError, ApiUnavailableError, get_fire_events, get_fire_fronts
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
_EVENT_CIRCLE_SEGMENTS = 40
_FORECAST_DEFAULT_HORIZONS = [24]
_FORECAST_DEFAULT_THRESHOLDS = [0.7]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_severity(event: Dict[str, Any]) -> float:
    event_score = _safe_float(event.get("event_score"))
    if event_score is None:
        return 0.0
    return max(0.0, min(event_score, 1.0))


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


def _event_radius_m(detection_count: Optional[float]) -> float:
    if detection_count is None:
        return float(PointSizing.MIN_SIZE * 1000)
    if detection_count >= 50:
        return float(PointSizing.LARGE_SIZE * 1000)
    if detection_count >= 20:
        return float(PointSizing.MEDIUM_SIZE * 1000)
    if detection_count >= 5:
        return float(PointSizing.SMALL_SIZE * 1000)
    return float(PointSizing.MIN_SIZE * 1000)


def _front_line_width(detection_count: Optional[float]) -> int:
    if detection_count is None:
        return 2
    if detection_count >= 50:
        return 5
    if detection_count >= 20:
        return 4
    if detection_count >= 5:
        return 3
    return 2


def _visible_horizons() -> list[int]:
    return list(_FORECAST_DEFAULT_HORIZONS)


def _visible_thresholds() -> list[float]:
    return list(_FORECAST_DEFAULT_THRESHOLDS)


def _horizon_visibility_expr(horizons: list[int]) -> str:
    if not horizons:
        return "false"
    parts = [f"properties.horizon_hours == {int(h)}" for h in horizons]
    return "(" + " || ".join(parts) + ")"


def _threshold_visibility_expr(thresholds: list[float]) -> str:
    if not thresholds:
        return "false"
    eps = 0.001
    parts = [
        (
            f"(properties.threshold >= {float(t) - eps:.3f} "
            f"&& properties.threshold <= {float(t) + eps:.3f})"
        )
        for t in thresholds
    ]
    return "(" + " || ".join(parts) + ")"


def _is_active_candidate(event: Dict[str, Any]) -> bool:
    """Strict event-level activity gate."""
    severity = _event_severity(event)
    decision = str(event.get("denoiser_decision") or "").strip().lower()
    review_required = bool(event.get("review_required"))
    if review_required:
        return True
    if decision in {"pass", "downweight"}:
        return True
    return severity >= 0.6


def _cluster_event_points(points: list[Dict[str, Any]], zoom: float) -> list[Dict[str, Any]]:
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
                "event_total_detections": int(_safe_float(p.get("detection_count")) or 0),
                "latest_time": p.get("end_time"),
                "sample": p,
            }
            continue

        b["cluster_count"] += 1
        b["sum_lat"] += lat
        b["sum_lon"] += lon
        b["max_severity"] = max(float(b["max_severity"]), float(p.get("_severity", 0.0)))
        b["event_total_detections"] += int(_safe_float(p.get("detection_count")) or 0)
        cur_t = p.get("end_time")
        prev_t = b.get("latest_time")
        if isinstance(cur_t, str) and (not isinstance(prev_t, str) or cur_t > prev_t):
            b["latest_time"] = cur_t
            b["sample"] = p

    out: list[Dict[str, Any]] = []
    for cluster_key, b in buckets.items():
        count = int(b["cluster_count"])
        sample = dict(b["sample"])
        sample["lat"] = float(b["sum_lat"]) / count
        sample["lon"] = float(b["sum_lon"]) / count
        sample["cluster_event_count"] = count
        sample["detection_count"] = int(b["event_total_detections"])
        sample["end_time"] = b.get("latest_time")
        sample["event_score"] = float(b["max_severity"])
        sample["event_id"] = f"cluster_{cluster_key[0]}_{cluster_key[1]}"
        sample["denoiser_decision"] = "pass"
        sample["review_required"] = False
        sample["sensor"] = "Cluster"
        sample["source"] = "Aggregated events"
        sample["radius_m"] = float(max(sample.get("radius_m", 0.0), 8000.0 * math.sqrt(max(count, 1))))
        out.append(sample)

    return out


def _cache_key(
    *,
    bbox: tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    min_likelihood: float,
    limit: int,
) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        f"{min_lon:.4f}|{min_lat:.4f}|{max_lon:.4f}|{max_lat:.4f}|"
        f"{isoformat(start_time)}|{isoformat(end_time)}|{min_likelihood:.3f}|{limit}"
    )


def _layer_id(prefix: str, identity: str) -> str:
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _cached_data_for_key(
    cache_slot: str,
    key: str,
    *,
    allow_any_fallback: bool = False,
) -> list[Dict[str, Any]]:
    payload = st.session_state.get(cache_slot)
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if payload.get("key") == key and isinstance(data, list):
        return data
    if allow_any_fallback and isinstance(data, list):
        return data
    if not isinstance(data, list):
        return []
    return []


def _store_cached_data(cache_slot: str, key: str, data: list[Dict[str, Any]]) -> None:
    st.session_state[cache_slot] = {"key": key, "data": data}


def _focus_map_on_event(lat: float, lon: float) -> None:
    current_view = st.session_state.get("map_view_state")
    current_zoom = float(getattr(current_view, "zoom", MapConfig.DEFAULT_ZOOM))
    pitch = float(getattr(current_view, "pitch", 0.0))
    bearing = float(getattr(current_view, "bearing", 0.0))
    target_zoom = max(current_zoom, 6.0)
    st.session_state.map_view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=target_zoom,
        pitch=pitch,
        bearing=bearing,
    )


def _event_ring_coords(lon: float, lat: float, radius_m: float) -> list[list[float]]:
    radius = max(float(radius_m), 300.0)
    lat_delta = radius / 111_000.0
    lon_denom = 111_000.0 * max(abs(math.cos(math.radians(lat))), 0.1)
    lon_delta = radius / lon_denom

    ring: list[list[float]] = []
    for i in range(_EVENT_CIRCLE_SEGMENTS):
        theta = 2.0 * math.pi * i / _EVENT_CIRCLE_SEGMENTS
        px = lon + lon_delta * math.cos(theta)
        py = lat + lat_delta * math.sin(theta)
        py = max(min(py, 85.0), -85.0)
        if px < -180.0:
            px += 360.0
        elif px > 180.0:
            px -= 360.0
        ring.append([px, py])
    ring.append(ring[0])
    return ring


def _event_feature(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lat = _safe_float(event.get("lat"))
    lon = _safe_float(event.get("lon"))
    if lat is None or lon is None:
        return None

    fill_a = int(_safe_float(event.get("fill_a")) or 70)
    line_a = int(_safe_float(event.get("line_a")) or 180)
    # Polygon fills need lower opacity than point markers to avoid visual overload.
    fill_alpha = min(max(fill_a, 45), 110)
    line_alpha = min(max(line_a, 120), 220)

    properties = dict(event)
    properties["lat"] = lat
    properties["lon"] = lon
    properties["fill_a"] = fill_alpha
    properties["line_a"] = line_alpha

    raw_geom = event.get("geom_geojson")
    geometry: Dict[str, Any] | None = None
    if isinstance(raw_geom, dict):
        if raw_geom.get("type") == "Feature" and isinstance(raw_geom.get("geometry"), dict):
            geometry = raw_geom.get("geometry")
        else:
            geometry = raw_geom
    elif isinstance(raw_geom, str):
        try:
            parsed = json.loads(raw_geom)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            if parsed.get("type") == "Feature" and isinstance(parsed.get("geometry"), dict):
                geometry = parsed.get("geometry")
            else:
                geometry = parsed

    if not isinstance(geometry, dict):
        radius_m = _safe_float(event.get("radius_m")) or 0.0
        radius_m = min(max(radius_m, 500.0), 20_000.0)
        ring = _event_ring_coords(lon, lat, radius_m)
        geometry = {"type": "Polygon", "coordinates": [ring]}

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": properties,
    }


def render_map_view() -> Optional[Dict[str, float]]:
    """Render the PyDeck map view and return click coordinates if any."""

    layers = []

    # 1. Events Layer (API-backed event footprints)
    start_time, end_time = app_state.time_range
    min_likelihood = app_state.filters.min_likelihood

    fire_points: list[Dict[str, Any]] = []
    bbox = app_state.viewport_bbox
    current_zoom = float(
        getattr(st.session_state.get("map_view_state"), "zoom", MapConfig.DEFAULT_ZOOM)
    )
    event_limit = 10000 if current_zoom >= 4.0 else 4000 if current_zoom >= 2.0 else 2000
    events_cache_key = _cache_key(
        bbox=bbox,
        start_time=start_time,
        end_time=end_time,
        min_likelihood=min_likelihood,
        limit=event_limit,
    )
    try:
        events = get_fire_events(
            bbox=bbox,
            time_range=(start_time, end_time),
            filters={
                "min_event_score": min_likelihood,
                "include_review_required": True,
                "limit": event_limit,
            },
        ).get("events", [])
        if isinstance(events, list):
            _store_cached_data("map_cached_events", events_cache_key, events)
        else:
            events = []
    except (ApiUnavailableError, ApiError) as exc:
        LOGGER.warning("Failed to fetch fire events for map layer: %s", exc)
        events = _cached_data_for_key(
            "map_cached_events",
            events_cache_key,
            allow_any_fallback=True,
        )

    for event in events:
        if app_state.filters.active_only and not _is_active_candidate(event):
            continue
        lat = _safe_float(event.get("lat"))
        lon = _safe_float(event.get("lon"))
        if lat is None or lon is None:
            continue
        severity = _event_severity(event)
        fill = _fire_fill_rgba(severity)
        line = _fire_line_rgba(severity)
        detection_count = _safe_float(event.get("detection_count"))
        point = dict(event)
        point["fill_r"] = int(fill[0])
        point["fill_g"] = int(fill[1])
        point["fill_b"] = int(fill[2])
        point["fill_a"] = int(fill[3])
        point["line_r"] = int(line[0])
        point["line_g"] = int(line[1])
        point["line_b"] = int(line[2])
        point["line_a"] = int(line[3])
        point["radius_m"] = _event_radius_m(detection_count)
        point["_severity"] = severity
        point["cluster_event_count"] = 1
        fire_points.append(point)

    # Keep the selected event visible during transient fetch instability.
    if not fire_points:
        selected = app_state.selection.selected_fire
        if isinstance(selected, dict):
            lat = _safe_float(selected.get("lat"))
            lon = _safe_float(selected.get("lon"))
            if lat is not None and lon is not None:
                severity = _event_severity(selected)
                fill = _fire_fill_rgba(severity)
                line = _fire_line_rgba(severity)
                fallback = dict(selected)
                fallback["lat"] = lat
                fallback["lon"] = lon
                fallback["fill_r"] = int(fill[0])
                fallback["fill_g"] = int(fill[1])
                fallback["fill_b"] = int(fill[2])
                fallback["fill_a"] = int(fill[3])
                fallback["line_r"] = int(line[0])
                fallback["line_g"] = int(line[1])
                fallback["line_b"] = int(line[2])
                fallback["line_a"] = int(line[3])
                fallback["radius_m"] = _event_radius_m(_safe_float(selected.get("detection_count")))
                fallback["_severity"] = severity
                fallback["cluster_event_count"] = 1
                fire_points.append(fallback)

    front_features: list[Dict[str, Any]] = []
    if current_zoom >= 5.0:
        front_limit = 1000 if current_zoom >= 7.0 else 600
        fronts_cache_key = _cache_key(
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
            min_likelihood=min_likelihood,
            limit=front_limit,
        )
        try:
            fronts = get_fire_fronts(
                bbox=bbox,
                time_range=(start_time, end_time),
                filters={
                    "min_event_score": min_likelihood,
                    "include_review_required": True,
                    "limit": front_limit,
                },
            ).get("fronts", [])
            if isinstance(fronts, list):
                _store_cached_data("map_cached_fronts", fronts_cache_key, fronts)
            else:
                fronts = []
        except (ApiUnavailableError, ApiError) as exc:
            LOGGER.warning("Failed to fetch fire fronts for map layer: %s", exc)
            fronts = _cached_data_for_key(
                "map_cached_fronts",
                fronts_cache_key,
                allow_any_fallback=True,
            )
    else:
        fronts = []

    visible_fronts: list[Dict[str, Any]] = []
    for front in fronts:
        if app_state.filters.active_only and not _is_active_candidate(front):
            continue
        raw_geom = front.get("geom_geojson")
        geometry: Dict[str, Any] | None = None
        if isinstance(raw_geom, dict):
            geometry = raw_geom
        elif isinstance(raw_geom, str):
            try:
                parsed = json.loads(raw_geom)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                geometry = parsed
        if not isinstance(geometry, dict):
            continue
        visible_fronts.append(front)

        severity = _event_severity(front)
        line = _fire_line_rgba(severity)
        front_features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "front_id": front.get("front_id"),
                    "event_id": front.get("event_id"),
                    "event_score": front.get("event_score"),
                    "detection_count": front.get("detection_count"),
                    "line_r": int(line[0]),
                    "line_g": int(line[1]),
                    "line_b": int(line[2]),
                    "line_a": int(max(120, line[3])),
                    "line_width": _front_line_width(_safe_float(front.get("detection_count"))),
                },
            }
        )

    # Keep the best visible front per event for front-driven forecast triggering.
    front_index_by_event: dict[str, dict[str, Any]] = {}
    for front in visible_fronts:
        event_id = front.get("event_id")
        front_id = front.get("front_id")
        if not event_id or not front_id:
            continue
        score = float(_safe_float(front.get("detection_count")) or 0.0)
        event_key = str(event_id)
        current = front_index_by_event.get(event_key)
        if current is None or score > float(current.get("detection_count") or 0.0):
            front_index_by_event[event_key] = {
                "front_id": str(front_id),
                "detection_count": score,
            }
    app_state.selection.front_index_by_event = front_index_by_event

    if front_features:
        front_identity = (
            f"{fronts_cache_key}|fronts={len(front_features)}|active={int(app_state.filters.active_only)}"
        )
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data={"type": "FeatureCollection", "features": front_features},
                id=_layer_id("fronts", front_identity),
                pickable=False,
                stroked=True,
                filled=False,
                get_line_color="[line_r, line_g, line_b, line_a]",
                get_line_width="line_width",
                line_width_min_pixels=1,
                line_width_max_pixels=6,
            )
        )

    marker_points = list(fire_points)
    if app_state.filters.cluster_points:
        marker_points = _cluster_event_points(marker_points, current_zoom)

    if app_state.filters.cluster_points:
        points_with_geom = [point for point in fire_points if point.get("geom_geojson")]
        points_without_geom = [point for point in fire_points if not point.get("geom_geojson")]
        fire_points = points_with_geom + _cluster_event_points(points_without_geom, current_zoom)

    fire_features: list[Dict[str, Any]] = []
    for point in fire_points:
        feature = _event_feature(point)
        if feature is not None:
            fire_features.append(feature)

    # Force a fresh deck layer when viewport/time/filter payload changes.
    events_layer_id = _layer_id(
        "events",
        (
            f"{events_cache_key}|events={len(fire_features)}|active={int(app_state.filters.active_only)}"
            f"|cluster={int(app_state.filters.cluster_points)}"
        ),
    )

    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data={"type": "FeatureCollection", "features": fire_features},
            id=events_layer_id,
            pickable=True,
            auto_highlight=True,
            filled=True,
            stroked=True,
            get_fill_color="[fill_r, fill_g, fill_b, fill_a]",
            get_line_color="[line_r, line_g, line_b, line_a]",
            get_line_width=3,
            line_width_min_pixels=1,
            line_width_max_pixels=4,
        )
    )

    # At global/regional zooms, authoritative event polygons can be sub-pixel.
    # Render centroid markers so events remain visible without replacing geometry.
    if marker_points and current_zoom < 4.0:
        centroid_layer_id = _layer_id(
            "events-centroids",
            f"{events_cache_key}|centroids={len(marker_points)}|zoom={current_zoom:.2f}",
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=marker_points,
                id=centroid_layer_id,
                pickable=True,
                auto_highlight=True,
                filled=True,
                stroked=True,
                get_position="[lon, lat]",
                get_fill_color="[fill_r, fill_g, fill_b, 220]",
                get_line_color="[line_r, line_g, line_b, 240]",
                get_radius=5,
                radius_units="pixels",
                radius_min_pixels=3,
                radius_max_pixels=8,
                line_width_min_pixels=1,
            )
        )

    # 2. Forecast Contours (MVT)
    last = app_state.forecast_job.last_forecast
    run_id = (last or {}).get("run", {}).get("id")
    contour_url = f"{api_public_base_url()}/tiles/forecast_contours/{{z}}/{{x}}/{{y}}.pbf"
    if run_id:
        contour_url += f"?run_id={run_id}"

    horizons = _visible_horizons()
    thresholds = _visible_thresholds()
    visible_expr = f"{_horizon_visibility_expr(horizons)} && {_threshold_visibility_expr(thresholds)}"
    fill_expr = f"{visible_expr} ? {ForecastColors.FILL} : [0, 0, 0, 0]"
    line_expr = f"{visible_expr} ? {ForecastColors.STROKE} : [0, 0, 0, 0]"

    layers.append(pdk.Layer(
        "MVTLayer",
        data=contour_url,
        id="forecast_contours",
        pickable=False,
        get_fill_color=fill_expr,
        get_line_color=line_expr,
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
                'Fire Event</div>'
                '<div style="font-size:12px;color:#e0e0e0;">'
                '<b>Event ID:</b> {event_id}<br/>'
                '<b>Cluster events:</b> {cluster_event_count}<br/>'
                '<b>Window:</b> {start_time} → {end_time}<br/>'
                '<b>Sensor:</b> {sensor}<br/>'
                '<b>Detections:</b> {detection_count}<br/>'
                '<b>Event score:</b> {event_score}<br/>'
                '<b>Decision:</b> {denoiser_decision}<br/>'
                '<b>Review required:</b> {review_required}'
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

        # Find selected fire-event objects by matching layer ID prefix.
        selected_fires = []
        for key, objects in event.selection.objects.items():
            if objects and key.startswith("events"):
                selected_fires = objects
                break

        if selected_fires:
            feature = selected_fires[0]
            if not isinstance(feature, dict):
                LOGGER.debug("Selected feature is not a dict: %r", feature)
                return None
            LOGGER.debug("Selected feature keys: %s", list(feature.keys()))

            raw_props = feature.get("properties")
            props = raw_props if isinstance(raw_props, dict) else feature
            if raw_props is None:
                LOGGER.debug("Feature has no 'properties' key — using feature dict directly")

            lat = _safe_float(props.get("lat"))
            lon = _safe_float(props.get("lon"))

            if lat is None or lon is None:
                LOGGER.warning(
                    "Selected event is missing required lat/lon fields. "
                    "Feature structure: %s",
                    feature,
                )
                return None

            normalized_feature = dict(props)
            normalized_feature["lat"] = lat
            normalized_feature["lon"] = lon

            app_state.selection.selected_fire = normalized_feature
            app_state.selection.last_click = {"lat": lat, "lng": lon}
            _focus_map_on_event(lat, lon)
            app_state._persist()
            return {"lat": lat, "lng": lon}

    return None
