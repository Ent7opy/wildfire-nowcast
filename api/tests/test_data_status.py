from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from api.data_status import _fetch_latest_firms_status, build_data_status_snapshot


class DummyEngine:
    @contextmanager
    def begin(self):
        yield object()


class _DummyMappingsResult:
    def __init__(self, row):
        self._row = row

    def mappings(self):
        return self

    def first(self):
        return self._row


class _DummyFirmsConn:
    def __init__(self, batch_row, watermark_row):
        self._batch_row = batch_row
        self._watermark_row = watermark_row

    def execute(self, stmt):
        sql = str(stmt)
        if "FROM ingest_batches" in sql:
            return _DummyMappingsResult(self._batch_row)
        if "FROM ingest_watermarks" in sql:
            return _DummyMappingsResult(self._watermark_row)
        raise AssertionError(f"Unexpected SQL in test: {sql}")


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

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine(), include_internal=True)

    assert snapshot["sources"]["firms"]["state"] == "fresh"
    assert snapshot["sources"]["weather"]["state"] == "stale"
    assert snapshot["sources"]["perimeters"]["state"] == "missing"
    assert snapshot["overall_state"] == "critical"
    assert "weather" in snapshot["critical_stale_sources"]
    assert snapshot["stale_behavior"]["mode"] == "degraded"
    assert snapshot["forecast_gate"]["can_run"] is False
    assert "weather_stale_or_missing" in snapshot["forecast_gate"]["reasons"]


def test_fetch_latest_firms_status_prefers_watermark_acq_time():
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)
    conn = _DummyFirmsConn(
        batch_row={
            "id": 11,
            "source": "VIIRS_SNPP_NRT",
            "completed_at": now,
            "records_fetched": 120,
            "records_inserted": 0,
            "records_skipped_duplicates": 0,
        },
        watermark_row={
            "latest_acq_time_utc": now - timedelta(hours=6),
            "latest_watermark_updated_at": now - timedelta(minutes=5),
        },
    )

    payload = _fetch_latest_firms_status(conn)

    assert payload["last_seen_at"] == now - timedelta(hours=6)
    assert payload["idempotency"]["latest_watermark_acq_time"] == (now - timedelta(hours=6)).isoformat()
