"""Tests for assistant proxy: circuit breaker open/close/timeout and health endpoint."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_circuit() -> None:
    """Reset the module-level circuit breaker to CLOSED state."""
    import api.routes.assistant as mod
    cb = mod._circuit
    with cb._lock:
        cb._failures = 0
        cb._opened_at = None


# ---------------------------------------------------------------------------
# /assistant/config
# ---------------------------------------------------------------------------


def test_config_returns_configured_false_when_no_key(monkeypatch) -> None:
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "")
    response = client.get("/assistant/config")
    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is False
    assert "model" in body


def test_config_returns_configured_true_when_key_set(monkeypatch) -> None:
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    response = client.get("/assistant/config")
    assert response.status_code == 200
    assert response.json()["configured"] is True


# ---------------------------------------------------------------------------
# /assistant/health — circuit breaker state reporting
# ---------------------------------------------------------------------------


def test_health_reports_closed_state(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    response = client.get("/assistant/health")
    assert response.status_code == 200
    body = response.json()
    assert body["circuit_state"] == "closed"
    assert body["failure_count"] == 0
    assert body["cooldown_remaining_seconds"] is None
    assert body["configured"] is True


def test_health_reports_open_state_after_failures(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("api.routes.assistant.settings.gemini_circuit_breaker_threshold", 3)
    monkeypatch.setattr(
        "api.routes.assistant.settings.gemini_circuit_breaker_cooldown_seconds", 120.0
    )

    import api.routes.assistant as mod
    mod._circuit.record_failure()
    mod._circuit.record_failure()
    mod._circuit.record_failure()

    response = client.get("/assistant/health")
    body = response.json()
    assert body["circuit_state"] == "open"
    assert body["failure_count"] == 3
    assert body["cooldown_remaining_seconds"] is not None
    assert body["cooldown_remaining_seconds"] > 0


def test_health_reports_half_open_after_cooldown(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("api.routes.assistant.settings.gemini_circuit_breaker_threshold", 1)
    monkeypatch.setattr(
        "api.routes.assistant.settings.gemini_circuit_breaker_cooldown_seconds", 0.01
    )

    import api.routes.assistant as mod
    mod._circuit.record_failure()
    time.sleep(0.05)  # allow cooldown to expire

    response = client.get("/assistant/health")
    body = response.json()
    assert body["circuit_state"] == "half_open"
    assert body["cooldown_remaining_seconds"] is None


# ---------------------------------------------------------------------------
# /assistant/chat — circuit breaker integration
# ---------------------------------------------------------------------------


def test_chat_returns_503_when_not_configured(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "")
    response = client.post("/assistant/chat", json={"contents": []})
    assert response.status_code == 503


def test_chat_short_circuits_when_open(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("api.routes.assistant.settings.gemini_circuit_breaker_threshold", 1)
    monkeypatch.setattr(
        "api.routes.assistant.settings.gemini_circuit_breaker_cooldown_seconds", 120.0
    )

    import api.routes.assistant as mod
    mod._circuit.record_failure()  # open the circuit

    with patch("httpx.AsyncClient") as mock_client_cls:
        response = client.post("/assistant/chat", json={"contents": []})

    # httpx should never have been called
    mock_client_cls.assert_not_called()
    assert response.status_code == 503
    assert "circuit" in response.json()["message"].lower()


def test_chat_records_failure_on_http_error(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("api.routes.assistant.settings.gemini_circuit_breaker_threshold", 10)

    import api.routes.assistant as mod
    failures_before = mod._circuit.failure_count

    mock_resp = MagicMock()
    mock_resp.is_success = False
    mock_resp.status_code = 429
    mock_resp.text = "rate limited"

    async def _mock_post(*args, **kwargs):
        return mock_resp

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)
    mock_async_client.post = _mock_post

    with patch("api.routes.assistant.httpx.AsyncClient", return_value=mock_async_client):
        response = client.post("/assistant/chat", json={"contents": []})

    assert response.status_code == 429
    assert mod._circuit.failure_count == failures_before + 1


def test_chat_records_success_and_resets_failures(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("api.routes.assistant.settings.gemini_circuit_breaker_threshold", 10)

    import api.routes.assistant as mod
    # Pre-load some failures
    mod._circuit._failures = 4

    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    async def _mock_post(*args, **kwargs):
        return mock_resp

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)
    mock_async_client.post = _mock_post

    with patch("api.routes.assistant.httpx.AsyncClient", return_value=mock_async_client):
        response = client.post("/assistant/chat", json={"contents": []})

    assert response.status_code == 200
    assert mod._circuit.failure_count == 0
    assert mod._circuit.state == "closed"


def test_chat_records_failure_on_connection_error(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("api.routes.assistant.settings.gemini_circuit_breaker_threshold", 10)

    import api.routes.assistant as mod
    failures_before = mod._circuit.failure_count

    async def _mock_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)
    mock_async_client.post = _mock_post

    with patch("api.routes.assistant.httpx.AsyncClient", return_value=mock_async_client):
        response = client.post("/assistant/chat", json={"contents": []})

    assert response.status_code == 503
    assert mod._circuit.failure_count == failures_before + 1


def test_circuit_opens_after_threshold_failures(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("api.routes.assistant.settings.gemini_circuit_breaker_threshold", 3)
    monkeypatch.setattr(
        "api.routes.assistant.settings.gemini_circuit_breaker_cooldown_seconds", 120.0
    )

    import api.routes.assistant as mod

    mock_resp = MagicMock()
    mock_resp.is_success = False
    mock_resp.status_code = 500
    mock_resp.text = "server error"

    async def _mock_post(*args, **kwargs):
        return mock_resp

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)
    mock_async_client.post = _mock_post

    with patch("api.routes.assistant.httpx.AsyncClient", return_value=mock_async_client):
        for _ in range(3):
            client.post("/assistant/chat", json={"contents": []})

    assert mod._circuit.state == "open"

    # Next request must be short-circuited without hitting httpx
    with patch("httpx.AsyncClient") as mock_cls:
        response = client.post("/assistant/chat", json={"contents": []})
    mock_cls.assert_not_called()
    assert response.status_code == 503


def test_circuit_closes_after_successful_probe(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("api.routes.assistant.settings.gemini_circuit_breaker_threshold", 1)
    monkeypatch.setattr(
        "api.routes.assistant.settings.gemini_circuit_breaker_cooldown_seconds", 0.01
    )

    import api.routes.assistant as mod
    mod._circuit.record_failure()  # open the circuit
    time.sleep(0.05)              # let cooldown expire → half_open

    assert mod._circuit.state == "half_open"

    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"candidates": []}

    async def _mock_post(*args, **kwargs):
        return mock_resp

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)
    mock_async_client.post = _mock_post

    with patch("api.routes.assistant.httpx.AsyncClient", return_value=mock_async_client):
        response = client.post("/assistant/chat", json={"contents": []})

    assert response.status_code == 200
    assert mod._circuit.state == "closed"
    assert mod._circuit.failure_count == 0


def test_timeout_is_passed_to_httpx(monkeypatch) -> None:
    _reset_circuit()
    monkeypatch.setattr("api.routes.assistant.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("api.routes.assistant.settings.gemini_timeout_seconds", 42.0)

    captured_timeout: list[float] = []

    class _CapturingClient:
        def __init__(self, timeout=None, **_kwargs):
            captured_timeout.append(timeout)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.json.return_value = {}
            return mock_resp

    with patch("api.routes.assistant.httpx.AsyncClient", _CapturingClient):
        client.post("/assistant/chat", json={"contents": []})

    assert captured_timeout == [42.0]
