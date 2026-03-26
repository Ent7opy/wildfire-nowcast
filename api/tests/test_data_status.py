from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from api.data_status import _fetch_latest_firms_status, _fetch_latest_lulc_status, build_data_status_snapshot


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
    "lulc": {"last_seen_at_offset": 120, "idempotency": {"latest_version": "v200_2021", "classified_last_7d": 500, "total_last_7d": 500}},
}

_WEATHER_VARIABLES = ("wind", "temperature", "humidity", "precipitation")

_FETCH_FUNCTIONS = {
    "firms": "api.data_status._fetch_latest_firms_status",
    "weather": "api.data_status._fetch_latest_weather_status",
    "terrain": "api.data_status._fetch_latest_terrain_status",
    "perimeters": "api.data_status._fetch_latest_perimeters_status",
    "lfmc": "api.data_status._fetch_latest_lfmc_status",
    "lulc": "api.data_status._fetch_latest_lulc_status",
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

        if source == "weather":
            # Per-variable last_seen defaults: same as aggregate unless overridden.
            default_vars = {v: last_seen_at for v in _WEATHER_VARIABLES}
            variables_last_seen = override.get("variables_last_seen", default_vars)
            monkeypatch.setattr(
                _FETCH_FUNCTIONS[source],
                lambda _conn, ls=last_seen_at, idem=idempotency, vls=variables_last_seen: {
                    "last_seen_at": ls,
                    "variables_last_seen": vls,
                    "idempotency": idem,
                },
            )
        elif source == "lfmc":
            coverage_fraction = override.get("coverage_fraction", 0.92)
            monkeypatch.setattr(
                _FETCH_FUNCTIONS[source],
                lambda _conn, ls=last_seen_at, idem=idempotency, cf=coverage_fraction: {
                    "last_seen_at": ls,
                    "coverage_fraction": cf,
                    "idempotency": idem,
                },
            )
        else:
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


def test_weather_per_variable_staleness(monkeypatch):
    """Per-variable staleness is reported independently; any stale variable flags the weather source."""
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)
    threshold = 360  # DATA_STALE_WEATHER_MINUTES default

    fresh_ts = now - timedelta(minutes=30)
    stale_ts = now - timedelta(minutes=threshold + 60)

    _stub_all_sources(
        monkeypatch,
        now,
        weather={
            "last_seen_at_offset": 30,  # aggregate appears fresh
            "variables_last_seen": {
                "wind": fresh_ts,
                "temperature": fresh_ts,
                "humidity": fresh_ts,
                "precipitation": stale_ts,  # precipitation run is old
            },
            "idempotency": {"completed_runs_last_24h": 4},
        },
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine(), include_internal=True)

    weather = snapshot["sources"]["weather"]

    # Per-variable status present for all four canonical variables.
    assert set(weather["variables"].keys()) == {"wind", "temperature", "humidity", "precipitation"}

    assert weather["variables"]["wind"]["state"] == "fresh"
    assert weather["variables"]["temperature"]["state"] == "fresh"
    assert weather["variables"]["humidity"]["state"] == "fresh"
    assert weather["variables"]["precipitation"]["state"] == "stale"
    assert weather["variables"]["precipitation"]["is_stale"] is True

    # Aggregate weather is flagged stale because precipitation is stale.
    assert weather["any_variable_stale"] is True
    assert weather["is_stale"] is True
    assert weather["state"] == "stale"

    # Overall snapshot is degraded/critical.
    assert snapshot["overall_state"] in {"degraded", "critical"}


def test_weather_all_variables_fresh(monkeypatch):
    """When all variables are within threshold, weather source stays fresh."""
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)
    fresh_ts = now - timedelta(minutes=30)

    _stub_all_sources(
        monkeypatch,
        now,
        weather={
            "last_seen_at_offset": 30,
            "variables_last_seen": {v: fresh_ts for v in _WEATHER_VARIABLES},
            "idempotency": {"completed_runs_last_24h": 4},
        },
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine(), include_internal=True)

    weather = snapshot["sources"]["weather"]
    assert weather["state"] == "fresh"
    assert weather["any_variable_stale"] is False
    assert weather["is_stale"] is False
    for var in _WEATHER_VARIABLES:
        assert weather["variables"][var]["state"] == "fresh"


def test_weather_missing_variable_flags_stale(monkeypatch):
    """A variable with no data (None) is reported as missing and flags the weather source stale."""
    now = datetime(2026, 2, 11, 12, 0, tzinfo=timezone.utc)
    fresh_ts = now - timedelta(minutes=30)

    _stub_all_sources(
        monkeypatch,
        now,
        weather={
            "last_seen_at_offset": 30,
            "variables_last_seen": {
                "wind": fresh_ts,
                "temperature": fresh_ts,
                "humidity": fresh_ts,
                "precipitation": None,  # never ingested
            },
            "idempotency": {"completed_runs_last_24h": 4},
        },
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine(), include_internal=True)

    weather = snapshot["sources"]["weather"]
    assert weather["variables"]["precipitation"]["state"] == "missing"
    assert weather["variables"]["precipitation"]["is_stale"] is True
    assert weather["any_variable_stale"] is True
    assert weather["is_stale"] is True


def test_fuel_section_present_with_named_fields(monkeypatch):
    """Snapshot always includes a 'fuel' section with the three operator fields."""
    now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)

    lfmc_ts = now - timedelta(minutes=60)
    lulc_ts = now - timedelta(hours=48)

    _stub_all_sources(
        monkeypatch,
        now,
        lfmc={"last_seen_at": lfmc_ts, "coverage_fraction": 0.87, "idempotency": {"completed_runs_last_24h": 2, "failed_runs_last_24h": 0}},
        lulc={"last_seen_at": lulc_ts, "idempotency": {"latest_version": "v200_2021", "classified_last_7d": 400, "total_last_7d": 400}},
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine())

    fuel = snapshot["fuel"]
    assert fuel["lfmc_last_updated"] == lfmc_ts.isoformat()
    assert fuel["lulc_last_updated"] == lulc_ts.isoformat()
    assert fuel["lfmc_coverage_fraction"] == 0.87


def test_fuel_section_none_when_no_data(monkeypatch):
    """fuel fields are None when LFMC/LULC have never run."""
    now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)

    _stub_all_sources(
        monkeypatch,
        now,
        lfmc={"last_seen_at": None, "coverage_fraction": None, "idempotency": {"completed_runs_last_24h": 0, "failed_runs_last_24h": 0}},
        lulc={"last_seen_at": None, "idempotency": {"latest_version": None, "classified_last_7d": 0, "total_last_7d": 0}},
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine())

    fuel = snapshot["fuel"]
    assert fuel["lfmc_last_updated"] is None
    assert fuel["lulc_last_updated"] is None
    assert fuel["lfmc_coverage_fraction"] is None


def test_lulc_source_in_snapshot(monkeypatch):
    """LULC appears as a source with freshness state."""
    now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)

    _stub_all_sources(monkeypatch, now)

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine(), include_internal=True)

    assert "lulc" in snapshot["sources"]
    assert snapshot["sources"]["lulc"]["state"] == "fresh"
    assert "lulc" in snapshot["idempotency_dashboard"]


def test_lulc_stale_when_old(monkeypatch):
    """LULC reports stale when backfill exceeds the weekly threshold."""
    now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)

    _stub_all_sources(
        monkeypatch,
        now,
        lulc={
            "last_seen_at_offset": 15000,  # > 10080 min (7 days)
            "idempotency": {"latest_version": "v200_2021", "classified_last_7d": 0, "total_last_7d": 100},
        },
    )

    snapshot = build_data_status_snapshot(now=now, engine=DummyEngine(), include_internal=True)

    assert snapshot["sources"]["lulc"]["state"] == "stale"
    assert snapshot["sources"]["lulc"]["is_stale"] is True
    assert "lulc" in snapshot["stale_sources"]


class _DummyLulcConn:
    """Minimal conn stub for _fetch_latest_lulc_status unit test (single merged query)."""

    def __init__(self, row):
        self._row = row

    def execute(self, stmt):
        return _DummyMappingsResult(self._row)


def test_fetch_latest_lulc_status_returns_expected_shape():
    now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
    conn = _DummyLulcConn(
        row={
            "last_lulc_at": now - timedelta(hours=12),
            "latest_version": "v200_2021",
            "classified_last_7d": 300,
            "total_last_7d": 350,
        },
    )

    result = _fetch_latest_lulc_status(conn)

    assert result["last_seen_at"] == now - timedelta(hours=12)
    assert result["idempotency"]["latest_version"] == "v200_2021"
    assert result["idempotency"]["classified_last_7d"] == 300
    assert result["idempotency"]["total_last_7d"] == 350
    assert result["idempotency"]["coverage_ratio_last_7d"] == round(300 / 350, 4)
