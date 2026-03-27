"""Tests for model registry REST endpoints.

Covers:
  GET  /internal/models/{family}/active
  POST /internal/models/{family}/promote
  POST /internal/models/{family}/rollback
"""

from fastapi.testclient import TestClient

from api.main import app
from api.model_registry import validate_gate_report, validate_model_gate

client = TestClient(app)


# ---------------------------------------------------------------------------
# validate_gate_report unit tests (shared validation logic)
# ---------------------------------------------------------------------------


def test_validate_gate_report_passes_on_true() -> None:
    ok, reason = validate_gate_report({"gate_report": {"pass": True}})
    assert ok is True
    assert reason == ""


def test_validate_gate_report_fails_on_false_with_reason() -> None:
    ok, reason = validate_gate_report({"gate_report": {"pass": False, "reason": "accuracy too low"}})
    assert ok is False
    assert "accuracy too low" in reason


def test_validate_gate_report_fails_on_false_no_reason() -> None:
    ok, reason = validate_gate_report({"gate_report": {"pass": False}})
    assert ok is False
    assert reason  # some non-empty reason


def test_validate_gate_report_fails_on_missing_gate_report() -> None:
    ok, reason = validate_gate_report({"some": "data"})
    assert ok is False
    assert "gate_report" in reason


def test_validate_gate_report_fails_on_none_metrics() -> None:
    ok, reason = validate_gate_report(None)
    assert ok is False
    assert reason


def test_validate_gate_report_fails_on_empty_metrics() -> None:
    ok, reason = validate_gate_report({})
    assert ok is False


def test_validate_gate_report_fails_on_non_dict_gate_report() -> None:
    ok, reason = validate_gate_report({"gate_report": "yes"})
    assert ok is False
    assert "JSON object" in reason


def test_validate_gate_report_uses_failure_reason_field() -> None:
    ok, reason = validate_gate_report({"gate_report": {"pass": False, "failure_reason": "f1 too low"}})
    assert ok is False
    assert "f1 too low" in reason


# ---------------------------------------------------------------------------
# GET /internal/models/{family}/active
# ---------------------------------------------------------------------------


def test_active_model_returns_promoted_model(monkeypatch) -> None:
    expected = {
        "model_id": "spread-run-123",
        "family": "spread",
        "artifact_uri": "models/spread/run_123",
        "metrics_json": {"gate_report": {"pass": True}},
        "status": "promoted",
    }
    monkeypatch.setattr("api.routes.internal.resolve_active_model", lambda family, **_: expected)

    response = client.get("/internal/models/spread/active")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert body["model"] == expected


def test_active_model_returns_none_when_no_promotion(monkeypatch) -> None:
    monkeypatch.setattr("api.routes.internal.resolve_active_model", lambda family, **_: None)

    response = client.get("/internal/models/spread/active")
    assert response.status_code == 200
    assert response.json()["model"] is None


def test_active_model_denoiser_family(monkeypatch) -> None:
    expected = {"model_id": "denoiser-run-1", "family": "denoiser", "status": "promoted"}
    monkeypatch.setattr("api.routes.internal.resolve_active_model", lambda family, **_: expected)

    response = client.get("/internal/models/denoiser/active")
    assert response.status_code == 200
    assert response.json()["model"]["family"] == "denoiser"


def test_active_model_invalid_family_returns_422() -> None:
    response = client.get("/internal/models/unknown_family/active")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# validate_model_gate unit tests (public pre-promotion gate check)
# ---------------------------------------------------------------------------


def test_validate_model_gate_raises_on_invalid_family() -> None:
    try:
        validate_model_gate("badFamily", "some-model")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported model family" in str(exc)


# ---------------------------------------------------------------------------
# POST /internal/models/{family}/promote
# ---------------------------------------------------------------------------


def test_promote_returns_200_on_valid_gate_report(monkeypatch) -> None:
    promoted = {"model_id": "spread-run-123", "family": "spread", "status": "promoted"}
    monkeypatch.setattr("api.routes.internal.validate_model_gate", lambda family, model_id, **_: None)
    monkeypatch.setattr("api.routes.internal.promote_model", lambda **_: promoted)

    response = client.post(
        "/internal/models/spread/promote",
        json={"model_id": "spread-run-123", "promoted_by": "operator", "notes": "prod release"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "promote"
    assert body["active"] == promoted
    assert "as_of" in body


def test_promote_returns_422_on_failed_gate_report(monkeypatch) -> None:
    def _gate_fail(family, model_id, **_):
        raise ValueError("Gate report validation failed: accuracy too low")

    monkeypatch.setattr("api.routes.internal.validate_model_gate", _gate_fail)

    response = client.post(
        "/internal/models/spread/promote",
        json={"model_id": "spread-run-123"},
    )
    assert response.status_code == 422
    body = response.json()
    assert "Gate report validation failed" in body["message"]
    assert "accuracy too low" in body["message"]


def test_promote_returns_422_on_missing_gate_report(monkeypatch) -> None:
    def _gate_fail(family, model_id, **_):
        raise ValueError("Gate report validation failed: No gate_report in metrics_json")

    monkeypatch.setattr("api.routes.internal.validate_model_gate", _gate_fail)

    response = client.post(
        "/internal/models/spread/promote",
        json={"model_id": "spread-run-123"},
    )
    assert response.status_code == 422
    assert "gate_report" in response.json()["message"]


def test_promote_returns_422_when_model_not_found(monkeypatch) -> None:
    def _gate_fail(family, model_id, **_):
        raise ValueError(f"Model not found for family={family}: {model_id}")

    monkeypatch.setattr("api.routes.internal.validate_model_gate", _gate_fail)

    response = client.post(
        "/internal/models/spread/promote",
        json={"model_id": "nonexistent-model"},
    )
    assert response.status_code == 422
    assert "not found" in response.json()["message"]


def test_promote_invalid_family_returns_422() -> None:
    response = client.post(
        "/internal/models/badFamily/promote",
        json={"model_id": "some-model"},
    )
    assert response.status_code == 422
    assert "Unsupported model family" in response.json()["message"]


def test_promote_works_for_denoiser_family(monkeypatch) -> None:
    promoted = {"model_id": "denoiser-run-1", "family": "denoiser", "status": "promoted"}
    monkeypatch.setattr("api.routes.internal.validate_model_gate", lambda family, model_id, **_: None)
    monkeypatch.setattr("api.routes.internal.promote_model", lambda **_: promoted)

    response = client.post(
        "/internal/models/denoiser/promote",
        json={"model_id": "denoiser-run-1"},
    )
    assert response.status_code == 200
    assert response.json()["active"]["family"] == "denoiser"


def test_promote_missing_model_id_returns_422() -> None:
    response = client.post("/internal/models/spread/promote", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /internal/models/{family}/rollback
# ---------------------------------------------------------------------------


def test_rollback_returns_200_and_previous_model(monkeypatch) -> None:
    previous = {"model_id": "spread-run-100", "family": "spread", "status": "promoted"}
    monkeypatch.setattr("api.routes.internal.rollback_model", lambda **_: previous)

    response = client.post(
        "/internal/models/spread/rollback",
        json={"promoted_by": "operator", "notes": "bad metrics"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "rollback"
    assert body["active"] == previous
    assert "as_of" in body


def test_rollback_returns_422_when_no_rollback_target(monkeypatch) -> None:
    def _no_rollback(**kwargs):
        raise ValueError("No rollback target recorded for family=spread")

    monkeypatch.setattr("api.routes.internal.rollback_model", _no_rollback)

    response = client.post("/internal/models/spread/rollback", json={})
    assert response.status_code == 422
    assert "No rollback target" in response.json()["message"]


def test_rollback_returns_422_when_no_promotion_exists(monkeypatch) -> None:
    def _no_promotion(**kwargs):
        raise ValueError("No promotion exists for family=spread")

    monkeypatch.setattr("api.routes.internal.rollback_model", _no_promotion)

    response = client.post("/internal/models/spread/rollback", json={})
    assert response.status_code == 422


def test_rollback_works_for_denoiser_family(monkeypatch) -> None:
    previous = {"model_id": "denoiser-run-0", "family": "denoiser", "status": "promoted"}
    monkeypatch.setattr("api.routes.internal.rollback_model", lambda **_: previous)

    response = client.post("/internal/models/denoiser/rollback", json={})
    assert response.status_code == 200
    assert response.json()["active"]["family"] == "denoiser"


def test_rollback_accepts_empty_body(monkeypatch) -> None:
    """Rollback body is fully optional."""
    previous = {"model_id": "spread-run-0", "family": "spread", "status": "promoted"}
    monkeypatch.setattr("api.routes.internal.rollback_model", lambda **_: previous)

    response = client.post("/internal/models/spread/rollback", json={})
    assert response.status_code == 200


def test_rollback_invalid_family_returns_422(monkeypatch) -> None:
    def _bad_family(**kwargs):
        raise ValueError("Unsupported model family: badFamily")

    monkeypatch.setattr("api.routes.internal.rollback_model", _bad_family)

    response = client.post("/internal/models/badFamily/rollback", json={})
    assert response.status_code == 422
