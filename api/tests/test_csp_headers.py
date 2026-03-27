"""Tests for Content-Security-Policy frame-ancestors header (iframe embedding)."""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_csp_frame_ancestors_present_on_health():
    resp = client.get("/health")
    assert "content-security-policy" in resp.headers


def test_csp_frame_ancestors_restricts_to_earth_tools():
    resp = client.get("/health")
    csp = resp.headers["content-security-policy"]
    assert "frame-ancestors" in csp
    assert "earth-tools.org" in csp


def test_csp_frame_ancestors_present_on_fires():
    resp = client.get("/fires")
    assert "frame-ancestors" in resp.headers.get("content-security-policy", "")


def test_csp_frame_ancestors_present_on_data_status():
    resp = client.get("/data-status")
    assert "frame-ancestors" in resp.headers.get("content-security-policy", "")
