"""Tests for CORS policy (issue #297).

Verifies that the explicit method/header allowlists are enforced and that
non-standard methods are rejected at the CORS layer.
"""

import pytest
from starlette.testclient import TestClient

from api.main import app

ORIGIN = "http://localhost:8501"


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


# -- Allowed methods ----------------------------------------------------------


@pytest.mark.parametrize("method", ["GET", "POST", "HEAD", "OPTIONS"])
def test_preflight_allowed_methods(client, method):
    """CORS preflight returns the requested method when it is in the allowlist."""
    resp = client.options(
        "/fires",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": method,
        },
    )
    assert resp.status_code == 200
    allowed = resp.headers.get("access-control-allow-methods", "")
    assert method in allowed


# -- Disallowed methods -------------------------------------------------------


@pytest.mark.parametrize("method", ["DELETE", "PUT", "PATCH", "TRACE"])
def test_preflight_disallowed_methods(client, method):
    """CORS preflight must NOT echo back methods outside the allowlist."""
    resp = client.options(
        "/fires",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": method,
        },
    )
    allowed = resp.headers.get("access-control-allow-methods", "")
    assert method not in allowed


# -- Allowed headers ----------------------------------------------------------


@pytest.mark.parametrize("header", ["Content-Type", "Authorization", "X-Request-ID"])
def test_preflight_allowed_headers(client, header):
    """CORS preflight returns the requested header when it is in the allowlist."""
    resp = client.options(
        "/fires",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": header,
        },
    )
    assert resp.status_code == 200
    allowed = resp.headers.get("access-control-allow-headers", "")
    assert header.lower() in allowed.lower()


# -- Disallowed headers -------------------------------------------------------


def test_preflight_disallowed_header(client):
    """CORS preflight must NOT echo back headers outside the allowlist."""
    resp = client.options(
        "/fires",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Custom-Evil",
        },
    )
    allowed = resp.headers.get("access-control-allow-headers", "")
    assert "x-custom-evil" not in allowed.lower()


# -- Simple request includes CORS origin header --------------------------------


def test_simple_get_includes_cors_origin(client):
    """A simple GET with an Origin header receives Access-Control-Allow-Origin."""
    resp = client.get("/fires", headers={"Origin": ORIGIN})
    assert "access-control-allow-origin" in resp.headers
