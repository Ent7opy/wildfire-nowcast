"""Tests for notify_staleness_if_degraded in api.data_status."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch


from api.data_status import notify_staleness_if_degraded

_NOW = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
_STALE_SINCE = (_NOW - timedelta(hours=4)).isoformat()


def _make_snapshot(
    overall_state: str = "healthy",
    stale_sources: list[str] | None = None,
    sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal snapshot dict suitable for notify_staleness_if_degraded."""
    if stale_sources is None:
        stale_sources = []
    if sources is None:
        sources = {
            src: {"last_seen_at": _STALE_SINCE, "state": "stale", "is_stale": True}
            for src in stale_sources
        }
    return {
        "overall_state": overall_state,
        "stale_sources": stale_sources,
        "critical_stale_sources": stale_sources if overall_state == "critical" else [],
        "sources": sources,
        "as_of": _NOW.isoformat(),
    }


# ── critical state → notify with severity "critical" ──────────────────────────


def test_critical_state_emits_critical_notification() -> None:
    """overall_state='critical' fires notify() with severity='critical'."""
    snapshot = _make_snapshot(
        overall_state="critical",
        stale_sources=["firms", "weather"],
    )
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot, aoi_id=None)

    mock_notify.assert_called_once()
    call_args = mock_notify.call_args.args
    call_kwargs = mock_notify.call_args.kwargs

    # First positional arg is event_type.
    event_type = call_args[0] if call_args else call_kwargs.get("event_type", "")
    assert "picture_stale" in event_type

    assert call_kwargs.get("severity") == "critical"


def test_critical_state_event_type_uses_global_when_no_aoi() -> None:
    """When aoi_id is None, event_type contains 'global'."""
    snapshot = _make_snapshot(overall_state="critical", stale_sources=["firms"])
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot, aoi_id=None)

    call_args = mock_notify.call_args.args
    event_type = call_args[0]
    assert event_type == "picture_stale:global"


def test_critical_state_event_type_uses_aoi_id_when_provided() -> None:
    """When aoi_id is set, event_type contains the aoi_id."""
    snapshot = _make_snapshot(overall_state="critical", stale_sources=["firms"])
    aoi_id = "aoi-42"
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot, aoi_id=aoi_id)

    call_args = mock_notify.call_args.args
    event_type = call_args[0]
    assert event_type == f"picture_stale:{aoi_id}"


# ── degraded state → notify with severity "warning" ───────────────────────────


def test_degraded_state_emits_warning_notification() -> None:
    """overall_state='degraded' fires notify() with severity='warning'."""
    snapshot = _make_snapshot(
        overall_state="degraded",
        stale_sources=["lfmc"],
    )
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot, aoi_id=None)

    mock_notify.assert_called_once()
    call_kwargs = mock_notify.call_args.kwargs
    assert call_kwargs.get("severity") == "warning"


def test_degraded_state_title_indicates_may_be_outdated() -> None:
    """Degraded title mentions 'may be outdated'."""
    snapshot = _make_snapshot(overall_state="degraded", stale_sources=["lfmc"])
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot)

    call_kwargs = mock_notify.call_args.kwargs
    title = call_kwargs.get("title", "")
    assert "outdated" in title.lower() or "updated" in title.lower() or "may" in title.lower()


# ── healthy state → notify not called ─────────────────────────────────────────


def test_healthy_state_does_not_notify() -> None:
    """overall_state='healthy' must not trigger any notification."""
    snapshot = _make_snapshot(overall_state="healthy", stale_sources=[])
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot)

    mock_notify.assert_not_called()


def test_healthy_state_with_aoi_id_does_not_notify() -> None:
    """Healthy state suppresses notification even when aoi_id is supplied."""
    snapshot = _make_snapshot(overall_state="healthy")
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot, aoi_id="aoi-99")

    mock_notify.assert_not_called()


# ── stale_sources included in context ─────────────────────────────────────────


def test_stale_sources_list_passed_in_context() -> None:
    """stale_sources is forwarded as a context field to notify()."""
    stale = ["firms", "weather", "perimeters"]
    snapshot = _make_snapshot(overall_state="critical", stale_sources=stale)
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot)

    call_kwargs = mock_notify.call_args.kwargs
    assert call_kwargs.get("stale_sources") == stale


def test_empty_stale_sources_passed_in_context() -> None:
    """Empty stale_sources list is forwarded as-is (critical with no named sources)."""
    snapshot = _make_snapshot(overall_state="critical", stale_sources=[])
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot)

    mock_notify.assert_called_once()
    call_kwargs = mock_notify.call_args.kwargs
    assert call_kwargs.get("stale_sources") == []


# ── aoi_id context field ───────────────────────────────────────────────────────


def test_aoi_id_forwarded_in_context() -> None:
    """aoi_id kwarg is passed through as a notify() context field."""
    snapshot = _make_snapshot(overall_state="critical", stale_sources=["firms"])
    aoi_id = "some-aoi-uuid"
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot, aoi_id=aoi_id)

    call_kwargs = mock_notify.call_args.kwargs
    assert call_kwargs.get("aoi_id") == aoi_id


def test_aoi_id_none_forwarded_in_context() -> None:
    """aoi_id=None is still passed as context so callers can rely on its presence."""
    snapshot = _make_snapshot(overall_state="critical", stale_sources=["firms"])
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot, aoi_id=None)

    call_kwargs = mock_notify.call_args.kwargs
    assert "aoi_id" in call_kwargs
    assert call_kwargs["aoi_id"] is None


# ── age computation ────────────────────────────────────────────────────────────


def test_age_appears_in_body() -> None:
    """The notification body mentions the data age in hours."""
    stale_since = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    snapshot = _make_snapshot(
        overall_state="critical",
        stale_sources=["firms"],
        sources={"firms": {"last_seen_at": stale_since, "state": "stale", "is_stale": True}},
    )
    mock_notify = MagicMock()

    with patch("api.data_status.notify", mock_notify):
        notify_staleness_if_degraded(snapshot)

    call_kwargs = mock_notify.call_args.kwargs
    body = call_kwargs.get("body", "")
    assert "h old" in body or "hour" in body
