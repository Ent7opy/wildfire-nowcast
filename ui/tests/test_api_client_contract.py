from datetime import datetime, timezone, timedelta

import api_client


class _StubResponse:
    def __init__(self) -> None:
        self.status_code = 200
        self.url = "http://example.test/fires"
        self.text = ""

    def json(self) -> dict:
        return {"count": 0, "detections": []}


def test_get_fires_serializes_bool_filters_as_lowercase_strings(monkeypatch) -> None:
    captured: dict = {}

    def _fake_get(url, params, timeout):  # noqa: ANN001 - test stub
        captured["url"] = url
        captured["params"] = dict(params)
        captured["timeout"] = timeout
        return _StubResponse()

    monkeypatch.setattr(api_client, "api_base_url", lambda: "http://example.test")
    monkeypatch.setattr(api_client.requests, "get", _fake_get)

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(hours=24)
    api_client.get_fires(
        bbox=(-180.0, -85.0, 180.0, 85.0),
        time_range=(start, now),
        filters={
            "include_noise": False,
            "include_denoiser_fields": True,
            "min_fire_likelihood": 0.0,
            "limit": 10000,
        },
    )

    assert captured["url"] == "http://example.test/fires"
    assert captured["params"]["include_noise"] == "false"
    assert captured["params"]["include_denoiser_fields"] == "true"
    assert captured["params"]["min_fire_likelihood"] == 0.0
    assert captured["params"]["limit"] == 10000
