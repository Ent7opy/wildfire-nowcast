from datetime import datetime, timezone, timedelta

import api_client


class _StubResponse:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self.url = "http://example.test/fires"
        self.text = ""
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _StubPostResponse:
    def __init__(self, payload: dict, status_code: int = 202) -> None:
        self.status_code = status_code
        self.url = "http://example.test/forecast/jit"
        self.text = ""
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_get_fires_serializes_bool_filters_as_lowercase_strings(monkeypatch) -> None:
    captured: dict = {}

    def _fake_get(url, params, timeout):  # noqa: ANN001 - test stub
        captured["url"] = url
        captured["params"] = dict(params)
        captured["timeout"] = timeout
        return _StubResponse({"count": 0, "detections": []})

    monkeypatch.setattr(api_client, "api_base_url", lambda: "http://example.test")
    monkeypatch.setattr(api_client, "api_base_url_candidates", lambda: ["http://example.test"])
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


def test_get_fire_events_serializes_bool_filters_as_lowercase_strings(monkeypatch) -> None:
    captured: dict = {}

    def _fake_get(url, params, timeout):  # noqa: ANN001 - test stub
        captured["url"] = url
        captured["params"] = dict(params)
        captured["timeout"] = timeout
        return _StubResponse({"count": 0, "events": []})

    monkeypatch.setattr(api_client, "api_base_url", lambda: "http://example.test")
    monkeypatch.setattr(api_client, "api_base_url_candidates", lambda: ["http://example.test"])
    monkeypatch.setattr(api_client.requests, "get", _fake_get)

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(hours=24)
    api_client.get_fire_events(
        bbox=(-180.0, -85.0, 180.0, 85.0),
        time_range=(start, now),
        filters={
            "include_review_required": True,
            "min_event_score": 0.35,
            "limit": 5000,
        },
    )

    assert captured["url"] == "http://example.test/fires/events"
    assert captured["params"]["include_review_required"] == "true"
    assert captured["params"]["min_event_score"] == 0.35
    assert captured["params"]["limit"] == 5000


def test_get_fire_events_falls_back_to_next_api_candidate(monkeypatch) -> None:
    seen_urls: list[str] = []

    def _fake_get(url, params, timeout):  # noqa: ANN001 - test stub
        seen_urls.append(url)
        if url.startswith("http://api:8000"):
            raise api_client.requests.ConnectionError("connection refused")
        return _StubResponse({"count": 0, "events": []})

    monkeypatch.setattr(
        api_client,
        "api_base_url_candidates",
        lambda: ["http://api:8000", "http://localhost:8000"],
    )
    monkeypatch.setattr(api_client.requests, "get", _fake_get)

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(hours=24)
    payload = api_client.get_fire_events(
        bbox=(-180.0, -85.0, 180.0, 85.0),
        time_range=(start, now),
        filters={"min_event_score": 0.35, "limit": 50},
    )

    assert payload["count"] == 0
    assert seen_urls[0] == "http://api:8000/fires/events"
    assert seen_urls[1] == "http://localhost:8000/fires/events"


def test_get_fire_events_retries_timeout_with_longer_read_timeout(monkeypatch) -> None:
    calls: list[tuple[str, tuple[float, float]]] = []

    def _fake_get(url, params, timeout):  # noqa: ANN001 - test stub
        calls.append((url, timeout))
        if timeout == (2.0, 8.0):
            raise api_client.requests.Timeout("read timeout")
        return _StubResponse({"count": 0, "events": []})

    monkeypatch.setattr(api_client, "api_base_url_candidates", lambda: ["http://example.test"])
    monkeypatch.setattr(api_client.requests, "get", _fake_get)

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(hours=24)
    payload = api_client.get_fire_events(
        bbox=(-180.0, -85.0, 180.0, 85.0),
        time_range=(start, now),
        filters={"min_event_score": 0.0, "limit": 50},
    )

    assert payload["count"] == 0
    assert len(calls) == 2
    assert calls[0] == ("http://example.test/fires/events", (2.0, 8.0))
    assert calls[1] == ("http://example.test/fires/events", (2.0, 15.0))


def test_get_fire_fronts_serializes_bool_filters_as_lowercase_strings(monkeypatch) -> None:
    captured: dict = {}

    def _fake_get(url, params, timeout):  # noqa: ANN001 - test stub
        captured["url"] = url
        captured["params"] = dict(params)
        captured["timeout"] = timeout
        return _StubResponse({"count": 0, "fronts": []})

    monkeypatch.setattr(api_client, "api_base_url", lambda: "http://example.test")
    monkeypatch.setattr(api_client, "api_base_url_candidates", lambda: ["http://example.test"])
    monkeypatch.setattr(api_client.requests, "get", _fake_get)

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(hours=24)
    api_client.get_fire_fronts(
        bbox=(-180.0, -85.0, 180.0, 85.0),
        time_range=(start, now),
        filters={
            "include_review_required": False,
            "min_event_score": 0.2,
            "limit": 2000,
        },
    )

    assert captured["url"] == "http://example.test/fires/fronts"
    assert captured["params"]["include_review_required"] == "false"
    assert captured["params"]["min_event_score"] == 0.2
    assert captured["params"]["limit"] == 2000


def test_get_active_spread_model_id_reads_internal_registry_payload(monkeypatch) -> None:
    captured: dict = {}

    def _fake_get(url, params, timeout):  # noqa: ANN001 - test stub
        captured["url"] = url
        captured["params"] = dict(params)
        captured["timeout"] = timeout
        return _StubResponse(
            {
                "models": {
                    "spread": {"model_id": "spread_v3_prod"},
                }
            }
        )

    monkeypatch.setattr(api_client, "api_base_url", lambda: "http://example.test")
    monkeypatch.setattr(api_client, "api_base_url_candidates", lambda: ["http://example.test"])
    monkeypatch.setattr(api_client.requests, "get", _fake_get)

    model_id = api_client.get_active_spread_model_id()
    assert model_id == "spread_v3_prod"
    assert captured["url"] == "http://example.test/internal/models/active"


def test_create_jit_forecast_includes_model_id(monkeypatch) -> None:
    captured: dict = {}

    def _fake_post(url, json, timeout):  # noqa: ANN001 - test stub
        captured["url"] = url
        captured["json"] = dict(json)
        captured["timeout"] = timeout
        return _StubPostResponse({"job_id": "abc", "status": "queued"})

    monkeypatch.setattr(api_client, "api_base_url", lambda: "http://example.test")
    monkeypatch.setattr(api_client, "api_base_url_candidates", lambda: ["http://example.test"])
    monkeypatch.setattr(api_client.requests, "post", _fake_post)

    api_client.create_jit_forecast(
        bbox=(10.0, 11.0, 12.0, 13.0),
        horizons=[24],
        model_id="spread_v3_prod",
    )
    assert captured["url"] == "http://example.test/forecast/jit"
    assert captured["json"]["model_id"] == "spread_v3_prod"


def test_create_jit_forecast_from_front_includes_model_id(monkeypatch) -> None:
    captured: dict = {}

    def _fake_post(url, json, timeout):  # noqa: ANN001 - test stub
        captured["url"] = url
        captured["json"] = dict(json)
        captured["timeout"] = timeout
        return _StubPostResponse({"job_id": "abc", "status": "queued"})

    monkeypatch.setattr(api_client, "api_base_url", lambda: "http://example.test")
    monkeypatch.setattr(api_client, "api_base_url_candidates", lambda: ["http://example.test"])
    monkeypatch.setattr(api_client.requests, "post", _fake_post)

    api_client.create_jit_forecast_from_front(
        "front-123",
        buffer_km=3.0,
        horizons=[24, 48],
        model_id="spread_v3_prod",
    )
    assert captured["url"] == "http://example.test/forecast/jit/from-front"
    assert captured["json"]["model_id"] == "spread_v3_prod"
