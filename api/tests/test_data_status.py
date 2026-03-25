from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

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


_FRESH_DEFAULTS: dict[str, dict[str, Any]] = {
    "firms": {"last_seen_at_offset": 5, "idempotency": {"records_fetched": 10, "records_skipped_duplicates": 0}},
    "weather": {"last_seen_at_offset": 30, "idempotency": {"completed_runs_last_24h": 4}},
    "terrain": {"last_seen_at_offset": 30, "idempotency": {"total_rows": 1}},
    "perimeters": {"last_seen_at_offset": 60, "idempotency": {"total_rows": 5}},
    "lfmc": {"last_seen_at_offset": 60, "idempotency": {"completed_runs_last_24h": 3, "failed_runs_last_24h": 0}},
}

_FETCH_FUNCTIONS = {
    "firms": "api.data_status._fetch_latest_firms_status",
    "weather": "api.data_status._fetch_latest_weather_status",
    "terrain": "api.data_status._fetch_latest_terrain_status",
    "perimeters": "api.data_status._fetch_latest_perimeters_status",
    "lfmc": "api.data_status._fetch_latest_lfmc_status",
}


def _stub_all_sources(
    monkeypatch,
    now: datetime,
    **overrides: dict[str, Any],
) -> None:
    """Patch all data-status fetch functions with sensible defaults, applying overrides."""
    for source, defaults in _FRESH_DEFAULTS.items():
        override = overrides.get(source, {})
        offset = override.get("last_seen_at_offset", defaults["last_seen_at_offset"])
        last_seen_at = override.get("last_seen_at", now - timedelta(minutes=offset))
        idempotency = override.get("idempotency", defaults["idempotency"])
        monkeypatch.setattr(
            _FETCH_FUNCTIONS[source],
            lambda _conn, ls=last_seen_at, idem=idempotency: {
                "last_seen_at": ls,
                "idempotency": idem,
            },
        )


def test_build_data_status_snapshot_marks_stale_and_critical(monkeypatch):
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)

    _stub_all_sources(
        monkeypatch,
        now,
        firms={"idempotency": {"records_fetched": 10, "records_skipped_duplicates": 2}},
        weather={"last_seen_at_offset": 500, "idempotency": {"completed_runs_last_24h": 2}},
        perimeters={"last_seen_at": None, "idempotency": {"total_rows": 0}},
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine(), include_internal=True)

    assert snapshot["sources"]["firms"]["state"] == "fresh"
    assert snapshot["sources"]["weather"]["state"] == "stale"
    assert snapshot["sources"]["perimeters"]["state"] == "missing"
    assert snapshot["sources"]["lfmc"]["state"] == "fresh"
    assert snapshot["overall_state"] == "critical"
    assert "weather" in snapshot["critical_stale_sources"]
    assert snapshot["stale_behavior"]["mode"] == "degraded"
    assert snapshot["forecast_gate"]["can_run"] is False
    assert "weather_stale_or_missing" in snapshot["forecast_gate"]["reasons"]


def test_lfmc_stale_when_api_unavailable(monkeypatch):
    """LFMC reports stale when last successful run exceeds threshold."""
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)

    _stub_all_sources(
        monkeypatch,
        now,
        lfmc={
            "last_seen_at_offset": 600,
            "idempotency": {"completed_runs_last_24h": 0, "failed_runs_last_24h": 4},
        },
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine(), include_internal=True)

    assert snapshot["sources"]["lfmc"]["state"] == "stale"
    assert snapshot["sources"]["lfmc"]["is_stale"] is True
    assert "lfmc" in snapshot["stale_sources"]


def test_lfmc_missing_when_no_runs(monkeypatch):
    """LFMC reports missing when no completed runs exist."""
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)

    _stub_all_sources(
        monkeypatch,
        now,
        lfmc={
            "last_seen_at": None,
            "idempotency": {"completed_runs_last_24h": 0, "failed_runs_last_24h": 0},
        },
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine(), include_internal=True)

    assert snapshot["sources"]["lfmc"]["state"] == "missing"
    assert snapshot["sources"]["lfmc"]["is_stale"] is True


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
