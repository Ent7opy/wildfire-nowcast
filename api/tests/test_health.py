from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    """Ensure the internal /health endpoint stays wired up."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_data_freshness_endpoint_returns_snapshot(monkeypatch) -> None:
    expected = {
        "as_of": "2026-02-11T00:00:00+00:00",
        "overall_state": "healthy",
        "stale_sources": [],
        "critical_stale_sources": [],
        "forecast_inputs_ready": True,
        "stale_behavior": {
            "mode": "normal",
            "policy": "serve_last_known_data_with_warning",
            "fires_api": "returns cached/latest detections and includes freshness status endpoint",
            "forecast_api": "allow_forecast_generation",
            "ui": "show_stale_data_banner_when_state_not_healthy",
            "critical_sources": ["firms", "weather"],
        },
        "sources": {},
        "idempotency_dashboard": {},
    }
    monkeypatch.setattr(
        "api.routes.internal.build_data_status_snapshot",
        lambda: expected,
    )

    response = client.get("/health/data-freshness")
    assert response.status_code == 200
    assert response.json() == expected
