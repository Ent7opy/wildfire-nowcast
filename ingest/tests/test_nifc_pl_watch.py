"""Tests for ingest.nifc_pl_watch."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from ingest.nifc_pl_watch import (
    clear_cache,
    get_preparedness_level,
    get_preparedness_level_context,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear the in-memory PL cache before every test."""
    clear_cache()
    yield
    clear_cache()


# ── get_preparedness_level ────────────────────────────────────────────────────


def test_returns_none_when_no_source_configured(monkeypatch):
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.delenv("NIFC_PL_OVERRIDE", raising=False)
    assert get_preparedness_level() is None


def test_override_env_var_returns_pl(monkeypatch):
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.setenv("NIFC_PL_OVERRIDE", "4")
    assert get_preparedness_level() == 4


def test_override_clamped_to_valid_range(monkeypatch):
    """NIFC_PL_OVERRIDE=6 is invalid — should return None."""
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.setenv("NIFC_PL_OVERRIDE", "6")
    assert get_preparedness_level() is None


def test_override_clamped_to_valid_range_zero(monkeypatch):
    """NIFC_PL_OVERRIDE=0 is invalid — should return None."""
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.setenv("NIFC_PL_OVERRIDE", "0")
    assert get_preparedness_level() is None


def test_cache_returns_cached_value(monkeypatch):
    """Second call hits cache and does not re-read the override env var."""
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.setenv("NIFC_PL_OVERRIDE", "3")

    first = get_preparedness_level()
    assert first == 3

    # Remove the override — but cached value should still be returned.
    monkeypatch.delenv("NIFC_PL_OVERRIDE", raising=False)
    second = get_preparedness_level()
    assert second == 3


def test_clear_cache_resets(monkeypatch):
    """After clear_cache(), removing the override causes None to be returned."""
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.setenv("NIFC_PL_OVERRIDE", "2")

    assert get_preparedness_level() == 2

    clear_cache()
    monkeypatch.delenv("NIFC_PL_OVERRIDE", raising=False)

    assert get_preparedness_level() is None


# ── get_preparedness_level_context ────────────────────────────────────────────


def test_context_empty_when_pl_none(monkeypatch):
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.delenv("NIFC_PL_OVERRIDE", raising=False)
    assert get_preparedness_level_context() == {}


def test_context_pl_below_4_no_warning(monkeypatch):
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.setenv("NIFC_PL_OVERRIDE", "2")
    ctx = get_preparedness_level_context()
    assert ctx == {"nifc_preparedness_level": 2}
    assert "nifc_pl_resource_warning" not in ctx


def test_context_pl_4_includes_warning(monkeypatch):
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.setenv("NIFC_PL_OVERRIDE", "4")
    ctx = get_preparedness_level_context()
    assert ctx["nifc_preparedness_level"] == 4
    assert "nifc_pl_resource_warning" in ctx
    assert "mutual aid" in ctx["nifc_pl_resource_warning"]


def test_context_pl_5_includes_critical_warning(monkeypatch):
    monkeypatch.delenv("NIFC_PL_URL", raising=False)
    monkeypatch.setenv("NIFC_PL_OVERRIDE", "5")
    ctx = get_preparedness_level_context()
    assert ctx["nifc_preparedness_level"] == 5
    assert "all national resources" in ctx["nifc_pl_resource_warning"]


# ── URL source ────────────────────────────────────────────────────────────────


def test_url_source_fetches_and_caches(monkeypatch):
    """With NIFC_PL_URL set and a mock client returning PL=3, returns 3."""
    monkeypatch.setenv("NIFC_PL_URL", "http://fake-nifc.example.com/pl.json")
    monkeypatch.delenv("NIFC_PL_OVERRIDE", raising=False)

    mock_response = MagicMock()
    mock_response.json.return_value = {"preparedness_level": 3}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.return_value = mock_response

    result = get_preparedness_level(http_client=mock_client)
    assert result == 3

    # Confirm it was cached — second call must not call client.get again.
    result2 = get_preparedness_level(http_client=mock_client)
    assert result2 == 3
    assert mock_client.get.call_count == 1


def test_url_source_falls_back_to_override_on_failure(monkeypatch):
    """When URL fetch raises httpx.HTTPError, falls back to NIFC_PL_OVERRIDE."""
    monkeypatch.setenv("NIFC_PL_URL", "http://fake-nifc.example.com/pl.json")
    monkeypatch.setenv("NIFC_PL_OVERRIDE", "2")

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = httpx.HTTPError("connection refused")

    result = get_preparedness_level(http_client=mock_client)
    assert result == 2
