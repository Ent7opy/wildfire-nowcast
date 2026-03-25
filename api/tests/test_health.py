from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    """Ensure the internal /health endpoint stays wired up."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_data_freshness_endpoint_returns_snapshot(monkeypatch) -> None:
    expected = {
        "as_of": "2026-02-11T00:00:00+00:00",
        "overall_state": "healthy",
        "stale_sources": [],
        "critical_stale_sources": [],
        "forecast_inputs_ready": True,
        "forecast_gate": {
            "can_run": True,
            "would_block_if_fail_closed": False,
            "policy": "fail_closed",
            "reasons": [],
            "missing_or_stale_sources": [],
            "as_of": "2026-02-11T00:00:00+00:00",
            "retry_hint": None,
        },
        "stale_behavior": {
            "mode": "normal",
            "policy": "serve_last_known_data_with_warning",
            "fires_api": "returns cached/latest detections and includes freshness status endpoint",
            "forecast_api": "allow_forecast_generation",
            "ui": "show_stale_data_banner_when_state_not_healthy",
            "critical_sources": ["firms", "weather"],
        },
        "sources": {},
    }
    monkeypatch.setattr(
        "api.routes.internal.build_data_status_snapshot",
        lambda: expected,
    )

    response = client.get("/health/data-freshness")
    assert response.status_code == 200
    assert response.json() == expected


def test_internal_data_freshness_endpoint_includes_idempotency(monkeypatch) -> None:
    expected = {
        "as_of": "2026-02-11T00:00:00+00:00",
        "overall_state": "healthy",
        "stale_sources": [],
        "critical_stale_sources": [],
        "forecast_inputs_ready": True,
        "forecast_gate": {
            "can_run": True,
            "would_block_if_fail_closed": False,
            "policy": "fail_closed",
            "reasons": [],
            "missing_or_stale_sources": [],
            "as_of": "2026-02-11T00:00:00+00:00",
            "retry_hint": None,
        },
        "stale_behavior": {
            "mode": "normal",
            "policy": "serve_last_known_data_with_warning",
            "fires_api": "returns cached/latest detections and includes freshness status endpoint",
            "forecast_api": "allow_forecast_generation",
            "ui": "show_stale_data_banner_when_state_not_healthy",
            "critical_sources": ["firms", "weather"],
        },
        "sources": {},
        "idempotency_dashboard": {"firms": {"latest_batch_id": 1}},
    }
    monkeypatch.setattr(
        "api.routes.internal.build_data_status_snapshot",
        lambda include_internal=False: expected if include_internal else {**expected, "idempotency_dashboard": {}},
    )

    response = client.get("/internal/health/data-freshness")
    assert response.status_code == 200
    assert response.json() == expected


def test_active_models_endpoint_returns_registry_payload(monkeypatch) -> None:
    expected = {
        "denoiser": {
            "model_id": "denoiser-prod-1",
            "artifact_uri": "models/denoiser/run_prod",
        }
    }
    monkeypatch.setattr("api.routes.internal.list_active_models", lambda: expected)

    response = client.get("/internal/models/active")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert body["models"] == expected


def test_denoiser_latest_gate_endpoint(monkeypatch) -> None:
    expected_gate = {"run_id": "run_1", "gate_report_json": {"pass": True}}
    monkeypatch.setattr("api.routes.internal.get_latest_denoiser_gate_report", lambda: expected_gate)

    response = client.get("/internal/denoiser/gates/latest")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert body["gate"] == expected_gate


def test_denoiser_drift_endpoint(monkeypatch) -> None:
    rows = [{"metric_name": "psi_score", "metric_value": 0.03}]
    monkeypatch.setattr("api.routes.internal.list_recent_denoiser_drift", lambda limit=50: rows)

    response = client.get("/internal/denoiser/drift")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert body["rows"] == rows


def test_denoiser_industrial_coverage_endpoint(monkeypatch) -> None:
    payload = {
        "latest_run": {"run_id": "industrial_run_1", "source_profile": "global_wri_gppd_silver"},
        "policy": {"policy_version": "global_authoritative_industrial_v1", "strict_no_go": True},
        "source_stats": {"gold_sources": 100, "silver_sources": 50, "active_sources": 150},
    }
    monkeypatch.setattr(
        "api.routes.internal.get_latest_denoiser_industrial_coverage_status",
        lambda source_profile=None, policy_version=None: payload,
    )

    response = client.get("/internal/denoiser/industrial-coverage/latest")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert body["coverage"] == payload


def test_terrain_coverage_inventory_endpoint(monkeypatch) -> None:
    from datetime import datetime, timezone

    preprocessed_at = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

    class _FakeRow:
        region_name = "us_west"
        bbox = (-125.0, 32.0, -114.0, 49.0)
        cell_size_deg = 0.01
        crs_epsg = 4326
        grid_n_lat = 1700
        grid_n_lon = 1100
        terrain_fallback_used = False
        coverage_fraction = 0.97
        created_at = preprocessed_at.replace(tzinfo=None)  # stored as naive UTC
        slope_path = "data/terrain/us_west/slope_us_west_epsg4326_0p01deg.tif"
        aspect_path = "data/terrain/us_west/aspect_us_west_epsg4326_0p01deg.tif"

    monkeypatch.setattr(
        "api.routes.internal.list_terrain_coverage_inventory",
        lambda: [_FakeRow()],
    )

    response = client.get("/internal/health/terrain-coverage")
    assert response.status_code == 200
    body = response.json()
    assert "as_of" in body
    assert "stale_threshold_minutes" in body
    assert body["region_count"] == 1

    region = body["regions"][0]
    assert region["region_name"] == "us_west"
    assert region["bbox"] == {"min_lon": -125.0, "min_lat": 32.0, "max_lon": -114.0, "max_lat": 49.0}
    assert region["resolution_deg"] == 0.01
    assert region["crs_epsg"] == 4326
    assert region["grid"] == {"n_lat": 1700, "n_lon": 1100}
    assert region["terrain_fallback_used"] is False
    assert region["coverage_fraction"] == 0.97
    assert region["preprocessed_at"] == "2026-03-20T12:00:00+00:00"
    assert "age_minutes" in region
    assert isinstance(region["is_stale"], bool)
    assert region["slope_path"] == "data/terrain/us_west/slope_us_west_epsg4326_0p01deg.tif"
    assert region["aspect_path"] == "data/terrain/us_west/aspect_us_west_epsg4326_0p01deg.tif"


def test_terrain_coverage_inventory_fallback_region(monkeypatch) -> None:
    """Fallback (flat-terrain stub) regions are included and flagged."""
    from datetime import datetime

    class _FallbackRow:
        region_name = "remote_stub"
        bbox = (-110.0, 35.0, -100.0, 45.0)
        cell_size_deg = 0.01
        crs_epsg = 4326
        grid_n_lat = 1000
        grid_n_lon = 1000
        terrain_fallback_used = True
        coverage_fraction = None
        created_at = datetime(2026, 3, 1, 0, 0, 0)
        slope_path = "data/terrain/remote_stub/slope_remote_stub_epsg4326_0p01deg.tif"
        aspect_path = "data/terrain/remote_stub/aspect_remote_stub_epsg4326_0p01deg.tif"

    monkeypatch.setattr(
        "api.routes.internal.list_terrain_coverage_inventory",
        lambda: [_FallbackRow()],
    )

    response = client.get("/internal/health/terrain-coverage")
    assert response.status_code == 200
    region = response.json()["regions"][0]
    assert region["terrain_fallback_used"] is True
    assert region["coverage_fraction"] is None
    assert region["is_stale"] is True  # 24+ days old, well past threshold


def test_terrain_coverage_inventory_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.routes.internal.list_terrain_coverage_inventory",
        lambda: [],
    )
    response = client.get("/internal/health/terrain-coverage")
    assert response.status_code == 200
    body = response.json()
    assert body["region_count"] == 0
    assert body["regions"] == []


def test_denoiser_review_queue_endpoints(monkeypatch) -> None:
    rows = [{"id": 1, "event_id": "evt_1", "status": "open"}]
    monkeypatch.setattr("api.routes.internal.list_denoiser_review_queue", lambda limit=200, status="open": rows)
    monkeypatch.setattr("api.routes.internal.resolve_denoiser_review_event", lambda **_: 2)

    list_resp = client.get("/internal/denoiser/review-queue")
    assert list_resp.status_code == 200
    assert list_resp.json()["rows"] == rows

    resolve_resp = client.post(
        "/internal/denoiser/review-queue/evt_1/resolve",
        json={"resolved_by": "qa", "resolved_notes": "validated"},
    )
    assert resolve_resp.status_code == 200
    payload = resolve_resp.json()
    assert payload["event_id"] == "evt_1"
    assert payload["updated"] == 2


# ---------------------------------------------------------------------------
# /internal/health/db-size
# ---------------------------------------------------------------------------


def _make_db_size_snapshot(**overrides) -> dict:
    base = {
        "as_of": "2026-03-25T12:00:00+00:00",
        "database": {"size_bytes": 524288000, "size_pretty": "500 MB"},
        "tables": {
            "fire_detections": {"row_count": 150000, "retention": {"archive_days": 3, "nrt_days": 14}},
            "ingest_batches": {"row_count": 2000, "retention": 30},
        },
        "retention_policy": {"default_retention_days": 14, "archive_retention_days": 3},
        "cleanup": {
            "last_run_at": "2026-03-25T06:00:00+00:00",
            "last_outcome": "success",
            "next_run_at": "2026-03-26T06:00:00+00:00",
            "interval_minutes": 1440.0,
            "source": "orchestrator_dashboard",
        },
    }
    base.update(overrides)
    return base


def test_db_size_health_returns_snapshot(monkeypatch) -> None:
    expected = _make_db_size_snapshot()
    monkeypatch.setattr("api.routes.internal.build_db_size_snapshot", lambda **_: expected)

    response = client.get("/internal/health/db-size")
    assert response.status_code == 200
    body = response.json()
    assert body == expected


def test_db_size_health_has_required_numeric_fields(monkeypatch) -> None:
    """Monitors need raw numeric fields — verify they are present and typed."""
    expected = _make_db_size_snapshot()
    monkeypatch.setattr("api.routes.internal.build_db_size_snapshot", lambda **_: expected)

    body = client.get("/internal/health/db-size").json()

    assert isinstance(body["database"]["size_bytes"], int)
    assert isinstance(body["database"]["size_pretty"], str)
    assert isinstance(body["cleanup"]["interval_minutes"], float)
    assert isinstance(body["retention_policy"]["default_retention_days"], int)
    assert isinstance(body["retention_policy"]["archive_retention_days"], int)
    for _table, stats in body["tables"].items():
        assert "row_count" in stats
        assert isinstance(stats["row_count"], int)


def test_db_size_health_cleanup_timing_fields(monkeypatch) -> None:
    """last_run_at and next_run_at must be present (may be null on first deploy)."""
    expected = _make_db_size_snapshot()
    monkeypatch.setattr("api.routes.internal.build_db_size_snapshot", lambda **_: expected)

    body = client.get("/internal/health/db-size").json()
    cleanup = body["cleanup"]

    assert "last_run_at" in cleanup
    assert "next_run_at" in cleanup
    assert "source" in cleanup


def test_db_size_health_null_cleanup_when_dashboard_missing(monkeypatch) -> None:
    """If orchestrator dashboard is absent, nulls with source explanation are returned."""
    expected = _make_db_size_snapshot(
        cleanup={
            "last_run_at": None,
            "last_outcome": None,
            "next_run_at": None,
            "interval_minutes": 1440.0,
            "source": "dashboard_unavailable",
            "source_detail": "/data/ingest/orchestrator_dashboard.json",
        }
    )
    monkeypatch.setattr("api.routes.internal.build_db_size_snapshot", lambda **_: expected)

    body = client.get("/internal/health/db-size").json()
    cleanup = body["cleanup"]

    assert cleanup["last_run_at"] is None
    assert cleanup["next_run_at"] is None
    assert cleanup["source"] == "dashboard_unavailable"


def test_db_size_health_fallback_on_db_error(monkeypatch) -> None:
    """Endpoint must return a structured fallback dict, not a 500, on DB errors."""

    def _raise(**_):
        raise RuntimeError("DB is down")

    monkeypatch.setattr("api.routes.internal.build_db_size_snapshot", _raise)

    response = client.get("/internal/health/db-size")
    assert response.status_code == 200
    body = response.json()
    assert "error" in body
    assert body["database"]["size_bytes"] is None
    assert body["tables"] == {}
    assert body["cleanup"]["source"] == "error"
