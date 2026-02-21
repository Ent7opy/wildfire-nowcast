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
        "forecast_gate": {
            "can_run": True,
            "would_block_if_fail_closed": False,
            "policy": "fail_closed",
            "reasons": [],
            "missing_or_stale_sources": [],
            "as_of": "2026-02-11T00:00:00+00:00",
            "retry_hint": None,
        },
        "stale_behavior": {
            "mode": "normal",
            "policy": "serve_last_known_data_with_warning",
            "fires_api": "returns cached/latest detections and includes freshness status endpoint",
            "forecast_api": "allow_forecast_generation",
            "ui": "show_stale_data_banner_when_state_not_healthy",
            "critical_sources": ["firms", "weather"],
        },
        "sources": {},
    }
    monkeypatch.setattr(
        "api.routes.internal.build_data_status_snapshot",
        lambda: expected,
    )

    response = client.get("/health/data-freshness")
    assert response.status_code == 200
    assert response.json() == expected


def test_internal_data_freshness_endpoint_includes_idempotency(monkeypatch) -> None:
    expected = {
        "as_of": "2026-02-11T00:00:00+00:00",
        "overall_state": "healthy",
        "stale_sources": [],
        "critical_stale_sources": [],
        "forecast_inputs_ready": True,
        "forecast_gate": {
            "can_run": True,
            "would_block_if_fail_closed": False,
            "policy": "fail_closed",
            "reasons": [],
            "missing_or_stale_sources": [],
            "as_of": "2026-02-11T00:00:00+00:00",
            "retry_hint": None,
        },
        "stale_behavior": {
            "mode": "normal",
            "policy": "serve_last_known_data_with_warning",
            "fires_api": "returns cached/latest detections and includes freshness status endpoint",
            "forecast_api": "allow_forecast_generation",
            "ui": "show_stale_data_banner_when_state_not_healthy",
            "critical_sources": ["firms", "weather"],
        },
        "sources": {},
        "idempotency_dashboard": {"firms": {"latest_batch_id": 1}},
    }
    monkeypatch.setattr(
        "api.routes.internal.build_data_status_snapshot",
        lambda include_internal=False: expected if include_internal else {**expected, "idempotency_dashboard": {}},
    )

    response = client.get("/internal/health/data-freshness")
    assert response.status_code == 200
    assert response.json() == expected


def test_active_models_endpoint_returns_registry_payload(monkeypatch) -> None:
    expected = {
        "denoiser": {
            "model_id": "denoiser-prod-1",
            "artifact_uri": "models/denoiser/run_prod",
        }
    }
    monkeypatch.setattr("api.routes.internal.list_active_models", lambda: expected)

    response = client.get("/internal/models/active")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert body["models"] == expected


def test_denoiser_latest_gate_endpoint(monkeypatch) -> None:
    expected_gate = {"run_id": "run_1", "gate_report_json": {"pass": True}}
    monkeypatch.setattr("api.routes.internal.get_latest_denoiser_gate_report", lambda: expected_gate)

    response = client.get("/internal/denoiser/gates/latest")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert body["gate"] == expected_gate


def test_denoiser_drift_endpoint(monkeypatch) -> None:
    rows = [{"metric_name": "psi_score", "metric_value": 0.03}]
    monkeypatch.setattr("api.routes.internal.list_recent_denoiser_drift", lambda limit=50: rows)

    response = client.get("/internal/denoiser/drift")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert body["rows"] == rows


def test_denoiser_review_queue_endpoints(monkeypatch) -> None:
    rows = [{"id": 1, "event_id": "evt_1", "status": "open"}]
    monkeypatch.setattr("api.routes.internal.list_denoiser_review_queue", lambda limit=200, status="open": rows)
    monkeypatch.setattr("api.routes.internal.resolve_denoiser_review_event", lambda **_: 2)

    list_resp = client.get("/internal/denoiser/review-queue")
    assert list_resp.status_code == 200
    assert list_resp.json()["rows"] == rows

    resolve_resp = client.post(
        "/internal/denoiser/review-queue/evt_1/resolve",
        json={"resolved_by": "qa", "resolved_notes": "validated"},
    )
    assert resolve_resp.status_code == 200
    payload = resolve_resp.json()
    assert payload["event_id"] == "evt_1"
    assert payload["updated"] == 2
