from components import map_view


def test_visible_forecast_defaults_are_fixed_tactical_view() -> None:
    assert map_view._visible_horizons() == [24]
    assert map_view._visible_thresholds() == [0.7]


def test_event_ring_coords_is_closed_and_segmented() -> None:
    ring = map_view._event_ring_coords(lon=44.0, lat=33.0, radius_m=3000.0)

    assert len(ring) == 41
    assert ring[0] == ring[-1]


def test_event_feature_uses_polygon_geometry_and_keeps_centroid() -> None:
    feature = map_view._event_feature(
        {
            "event_id": "evt-1",
            "lat": 33.1,
            "lon": 44.2,
            "radius_m": 2500.0,
            "fill_r": 255,
            "fill_g": 107,
            "fill_b": 53,
            "fill_a": 220,
            "line_r": 255,
            "line_g": 255,
            "line_b": 255,
            "line_a": 180,
        }
    )

    assert feature is not None
    assert feature["geometry"]["type"] == "Polygon"
    props = feature["properties"]
    assert props["event_id"] == "evt-1"
    assert props["lat"] == 33.1
    assert props["lon"] == 44.2
    assert 45 <= props["fill_a"] <= 110


def test_event_feature_prefers_authoritative_geom_geojson() -> None:
    feature = map_view._event_feature(
        {
            "event_id": "evt-2",
            "lat": 33.0,
            "lon": 44.0,
            "geom_geojson": {
                "type": "Polygon",
                "coordinates": [[[44.0, 33.0], [44.1, 33.0], [44.1, 33.1], [44.0, 33.0]]],
            },
            "fill_r": 255,
            "fill_g": 107,
            "fill_b": 53,
            "fill_a": 120,
            "line_r": 255,
            "line_g": 255,
            "line_b": 255,
            "line_a": 180,
        }
    )

    assert feature is not None
    assert feature["geometry"]["type"] == "Polygon"
    coords = feature["geometry"]["coordinates"][0]
    assert coords[0] == [44.0, 33.0]
    assert coords[1] == [44.1, 33.0]
