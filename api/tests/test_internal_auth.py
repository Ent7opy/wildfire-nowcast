"""Tests for internal API authentication (X-Internal-API-Key)."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app
from api.config import settings

client = TestClient(app)


class TestInternalAPIKeyAuth:
    """Test suite for X-Internal-API-Key authentication on state-mutating endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear any test overrides before each test."""
        yield
        # Reset app dependency overrides after each test
        app.dependency_overrides.clear()

    def test_promote_without_key_when_key_configured(self):
        """POST /internal/models/{family}/promote returns 401 when key is configured but not provided."""
        with patch("api.config.settings.internal_api_key", "configured-key"):
            response = client.post(
                "/internal/models/denoiser/promote",
                json={"model_id": "model-123"},
            )
            assert response.status_code == 401
            assert "X-Internal-API-Key" in response.json()["message"]

    def test_promote_with_wrong_key(self):
        """POST /internal/models/{family}/promote returns 401 when key doesn't match."""
        with patch("api.config.settings.internal_api_key", "correct-key"):
            response = client.post(
                "/internal/models/denoiser/promote",
                json={"model_id": "model-123"},
                headers={"X-Internal-API-Key": "wrong-key"},
            )
            assert response.status_code == 401
            assert "Invalid X-Internal-API-Key" in response.json()["message"]

    def test_promote_with_correct_key(self):
        """POST /internal/models/{family}/promote succeeds with correct key."""
        # Mock the actual promote function to avoid DB calls
        with patch("api.routes.internal.validate_model_gate") as mock_validate, \
             patch("api.routes.internal.promote_model") as mock_promote, \
             patch("api.config.settings.internal_api_key", "correct-key"):

            mock_promote.return_value = {"model_id": "model-123", "status": "active"}

            response = client.post(
                "/internal/models/denoiser/promote",
                json={"model_id": "model-123"},
                headers={"X-Internal-API-Key": "correct-key"},
            )
            assert response.status_code == 200
            assert response.json()["action"] == "promote"
            mock_validate.assert_called_once()
            mock_promote.assert_called_once()

    def test_promote_without_key_when_key_not_configured(self, caplog):
        """POST /internal/models/{family}/promote succeeds when key is not configured (dev mode)."""
        # When key is empty string, endpoints should be accessible but log a warning
        with patch("api.config.settings.internal_api_key", ""), \
             patch("api.routes.internal.validate_model_gate") as _mock_validate, \
             patch("api.routes.internal.promote_model") as mock_promote:

            mock_promote.return_value = {"model_id": "model-123", "status": "active"}

            response = client.post(
                "/internal/models/denoiser/promote",
                json={"model_id": "model-123"},
            )
            assert response.status_code == 200
            # A warning should be logged about unprotected endpoints
            assert "not configured" in caplog.text or "unprotected" in caplog.text

    def test_rollback_without_key_when_key_configured(self):
        """POST /internal/models/{family}/rollback returns 401 when key is configured but not provided."""
        with patch("api.config.settings.internal_api_key", "configured-key"):
            response = client.post(
                "/internal/models/denoiser/rollback",
                json={},
            )
            assert response.status_code == 401
            assert "X-Internal-API-Key" in response.json()["message"]

    def test_rollback_with_correct_key(self):
        """POST /internal/models/{family}/rollback succeeds with correct key."""
        with patch("api.routes.internal.rollback_model") as mock_rollback, \
             patch("api.config.settings.internal_api_key", "correct-key"):

            mock_rollback.return_value = {"model_id": "model-456", "status": "active"}

            response = client.post(
                "/internal/models/denoiser/rollback",
                json={},
                headers={"X-Internal-API-Key": "correct-key"},
            )
            assert response.status_code == 200
            assert response.json()["action"] == "rollback"
            mock_rollback.assert_called_once()

    def test_review_queue_resolve_without_key_when_key_configured(self):
        """POST /internal/denoiser/review-queue/{event_id}/resolve returns 401 when key is configured."""
        with patch("api.config.settings.internal_api_key", "configured-key"):
            response = client.post(
                "/internal/denoiser/review-queue/event-123/resolve",
                json={"resolved_by": "operator"},
            )
            assert response.status_code == 401
            assert "X-Internal-API-Key" in response.json()["detail"]

    def test_review_queue_resolve_with_correct_key(self):
        """POST /internal/denoiser/review-queue/{event_id}/resolve succeeds with correct key."""
        with patch("api.routes.internal.resolve_denoiser_review_event") as mock_resolve, \
             patch("api.config.settings.internal_api_key", "correct-key"):

            mock_resolve.return_value = 3  # 3 rows updated

            response = client.post(
                "/internal/denoiser/review-queue/event-123/resolve",
                json={"resolved_by": "operator"},
                headers={"X-Internal-API-Key": "correct-key"},
            )
            assert response.status_code == 200
            assert response.json()["updated"] == 3
            mock_resolve.assert_called_once()

    def test_get_endpoints_no_auth_required(self):
        """GET endpoints should not require authentication."""
        # Health endpoints should work without any key
        response = client.get("/internal/health")
        assert response.status_code == 200

        response = client.get("/internal/models/active")
        assert response.status_code == 200

        response = client.get("/internal/denoiser/review-queue")
        assert response.status_code == 200

    def test_get_endpoints_work_with_key(self):
        """GET endpoints should work whether key is provided or not."""
        with patch("api.config.settings.internal_api_key", "correct-key"):
            # Should work without key
            response = client.get("/internal/health")
            assert response.status_code == 200

            # Should also work with key
            response = client.get(
                "/internal/health",
                headers={"X-Internal-API-Key": "correct-key"},
            )
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
