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
