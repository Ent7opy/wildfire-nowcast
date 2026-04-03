"""Integration test for GET /ignition endpoint."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.mark.integration
def test_get_ignition_endpoint_returns_well_formed_response(db_available):
    response = client.get(
        "/ignition",
        params={
            "min_lon": -121.0,
            "min_lat": 37.0,
            "max_lon": -120.0,
            "max_lat": 38.0,
            "horizon": "now",
        },
    )

    assert response.status_code in (200, 503), (
        f"Expected 200 or 503 (503 if no model promoted), got {response.status_code}: {response.text}"
    )

    if response.status_code == 503:
        body = response.json()
        assert body["error"] == "ignition_model_unavailable"
        return

    body = response.json()
    assert "horizon" in body
    assert body["horizon"] == "now"
    assert "model_id" in body
    assert "cells" in body
    assert isinstance(body["cells"], list)
    assert "coverage_warnings" in body
    assert isinstance(body["coverage_warnings"], list)
    assert "valid_time" in body
    assert "low_confidence" in body
    assert body["low_confidence"] is False

    for cell in body["cells"]:
        assert "cell_id" in cell
        assert "lat" in cell
        assert "lon" in cell
        assert "probability" in cell
        assert "level" in cell
        assert "signals" in cell
        assert cell["level"] in ("low", "elevated", "high", "critical")
        assert 0.0 <= cell["probability"] <= 1.0
