from types import SimpleNamespace

from components import click_details


def test_render_click_details_no_selection_renders_aggregate_stats(monkeypatch) -> None:
    calls = {"aggregate": 0}

    monkeypatch.setattr(
        click_details,
        "app_state",
        SimpleNamespace(selection=SimpleNamespace(selected_fire=None)),
    )
    monkeypatch.setattr(click_details, "_render_aggregate_stats", lambda: calls.__setitem__("aggregate", 1))
    monkeypatch.setattr(click_details.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(click_details.st, "info", lambda *args, **kwargs: None)

    click_details.render_click_details(None)
    assert calls["aggregate"] == 1


def test_render_click_details_no_selection_with_click_shows_info(monkeypatch) -> None:
    calls = {"info": 0}

    monkeypatch.setattr(
        click_details,
        "app_state",
        SimpleNamespace(selection=SimpleNamespace(selected_fire=None)),
    )
    monkeypatch.setattr(click_details, "_render_aggregate_stats", lambda: None)
    monkeypatch.setattr(click_details.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(click_details.st, "info", lambda *args, **kwargs: calls.__setitem__("info", 1))

    click_details.render_click_details({"lat": 42.0, "lng": 21.0})
    assert calls["info"] == 1


def test_build_event_key_prefers_event_id() -> None:
    key = click_details._build_event_key({"event_id": "evt-123"}, 10.0, 20.0)
    assert key == "event_id:evt-123"


def test_build_event_key_falls_back_to_point_signature() -> None:
    key = click_details._build_event_key({"end_time": "2026-01-01T00:00:00Z"}, 12.34567, 23.45678)
    assert key.startswith("point:12.3457:23.4568:")


def test_location_label_prefers_event_country() -> None:
    label = click_details._location_label_for_event({"country": "Iraq"}, 33.0, 44.0)
    assert label == "Iraq"


def test_location_label_falls_back_to_coordinates(monkeypatch) -> None:
    monkeypatch.setattr(click_details, "_lookup_country_for_coordinates", lambda lat, lon: None)
    label = click_details._location_label_for_event({}, 33.1234, 44.5678)
    assert label == "33.12, 44.57"
