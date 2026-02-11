from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from api.data_status import build_data_status_snapshot


class DummyEngine:
    @contextmanager
    def begin(self):
        yield object()


def test_build_data_status_snapshot_marks_stale_and_critical(monkeypatch):
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(
        "api.data_status._fetch_latest_firms_status",
        lambda _conn: {
            "last_seen_at": now - timedelta(minutes=5),
            "idempotency": {"records_fetched": 10, "records_skipped_duplicates": 2},
        },
    )
    monkeypatch.setattr(
        "api.data_status._fetch_latest_weather_status",
        lambda _conn: {
            "last_seen_at": now - timedelta(minutes=500),
            "idempotency": {"completed_runs_last_24h": 2},
        },
    )
    monkeypatch.setattr(
        "api.data_status._fetch_latest_terrain_status",
        lambda _conn: {
            "last_seen_at": now - timedelta(minutes=30),
            "idempotency": {"total_rows": 1},
        },
    )
    monkeypatch.setattr(
        "api.data_status._fetch_latest_perimeters_status",
        lambda _conn: {
            "last_seen_at": None,
            "idempotency": {"total_rows": 0},
        },
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine())

    assert snapshot["sources"]["firms"]["state"] == "fresh"
    assert snapshot["sources"]["weather"]["state"] == "stale"
    assert snapshot["sources"]["perimeters"]["state"] == "missing"
    assert snapshot["overall_state"] == "critical"
    assert "weather" in snapshot["critical_stale_sources"]
    assert snapshot["stale_behavior"]["mode"] == "degraded"
