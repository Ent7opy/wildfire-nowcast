import runtime_config


def test_api_public_base_url_rewrites_internal_api_host_from_public_env(monkeypatch) -> None:
    monkeypatch.setenv("API_PUBLIC_BASE_URL", "http://api:8000")
    monkeypatch.delenv("API_BASE_URL", raising=False)

    assert runtime_config.api_public_base_url() == "http://localhost:8000"


def test_api_public_base_url_rewrites_internal_api_host_from_api_base(monkeypatch) -> None:
    monkeypatch.delenv("API_PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("API_BASE_URL", "http://api:9000")

    assert runtime_config.api_public_base_url() == "http://localhost:9000"


def test_api_public_base_url_keeps_explicit_external_url(monkeypatch) -> None:
    monkeypatch.setenv("API_PUBLIC_BASE_URL", "http://localhost:8000")
    monkeypatch.setenv("API_BASE_URL", "http://api:8000")

    assert runtime_config.api_public_base_url() == "http://localhost:8000"
