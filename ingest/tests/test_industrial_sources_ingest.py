from __future__ import annotations

from types import SimpleNamespace

import pytest

import ingest.industrial_sources_ingest as mod


class _DummyResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"ok") -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": "text/plain"}
        self.request = SimpleNamespace(url="https://example.test/check")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")


class _DummyClient:
    def __init__(self, response: _DummyResponse) -> None:
        self._response = response
        self.last_method: str | None = None
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None

    def __enter__(self) -> _DummyClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def request(self, method: str, url: str, headers=None):  # noqa: ANN001
        self.last_method = method
        self.last_url = url
        self.last_headers = headers
        return self._response


def test_check_endpoint_returns_metadata_and_uses_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _DummyResponse(status_code=200, content=b"endpoint-body")
    client = _DummyClient(response)

    monkeypatch.setattr(mod.httpx, "Client", lambda **_: client)

    profile = {
        "source_profile": "test_profile",
        "endpoint_required": True,
        "endpoint_check_url": "https://example.test/check",
        "endpoint_check_method": "GET",
        "endpoint_check_headers": {"User-Agent": "UA"},
    }

    result = mod._check_endpoint(profile, timeout_seconds=5.0)

    assert result is not None
    assert result.status_code == 200
    assert result.method == "GET"
    assert result.url == "https://example.test/check"
    assert client.last_method == "GET"
    assert client.last_url == "https://example.test/check"
    assert client.last_headers == {"User-Agent": "UA"}


def test_check_endpoint_not_required_returns_none() -> None:
    profile = {"source_profile": "test_profile", "endpoint_required": False}
    assert mod._check_endpoint(profile, timeout_seconds=5.0) is None


def test_check_endpoint_invalid_method_raises() -> None:
    profile = {
        "source_profile": "test_profile",
        "endpoint_required": True,
        "endpoint_check_url": "https://example.test/check",
        "endpoint_check_method": "PATCH",
    }
    with pytest.raises(ValueError, match="Unsupported endpoint_check_method"):
        mod._check_endpoint(profile, timeout_seconds=5.0)


def test_check_endpoint_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _DummyResponse(status_code=403)
    client = _DummyClient(response)
    monkeypatch.setattr(mod.httpx, "Client", lambda **_: client)

    profile = {
        "source_profile": "test_profile",
        "endpoint_required": True,
        "endpoint_check_url": "https://example.test/check",
    }

    with pytest.raises(RuntimeError, match="Endpoint verification failed"):
        mod._check_endpoint(profile, timeout_seconds=5.0)
