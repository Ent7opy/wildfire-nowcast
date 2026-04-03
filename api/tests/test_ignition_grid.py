"""Unit tests for api/ignition/grid.py."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_active_model(top_features=None, thresholds=None, artifact_uri="models/ignition/model.onnx"):
    if top_features is None:
        top_features = [
            "temperature_c", "relative_humidity", "wind_speed_kmh",
            "precip_last_7d_mm", "days_since_last_burn",
        ]
    if thresholds is None:
        thresholds = {"low_max": 0.2, "elevated_max": 0.5, "high_max": 0.8}
    return {
        "model_id": "ignition-test-model-20260101000000-abcd1234",
        "family": "ignition",
        "artifact_uri": artifact_uri,
        "metrics_json": {
            "runtime_contract": {
                "top_features": top_features,
                "thresholds": thresholds,
            }
        },
    }


def _call_compute(horizon="now", monkeypatch=None, active_model=None, env_override=None):
    from api.ignition import grid as grid_mod

    fake_engine = MagicMock()

    model = active_model if active_model is not None else _make_active_model()

    with patch.object(grid_mod, "resolve_active_model", return_value=model), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=datetime(2026, 4, 1, tzinfo=timezone.utc)), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=True), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=True), \
         patch.object(grid_mod, "_run_onnx_inference", return_value=np.full(9, 0.15)):
        if env_override:
            with patch.dict(os.environ, env_override):
                result = grid_mod.compute_ignition_grid(
                    -121.0, 37.0, -120.0, 38.0,
                    cell_size_km=40.0,
                    horizon=horizon,
                    engine=fake_engine,
                )
        else:
            result = grid_mod.compute_ignition_grid(
                -121.0, 37.0, -120.0, 38.0,
                cell_size_km=40.0,
                horizon=horizon,
                engine=fake_engine,
            )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_horizon_now_offset_zero():
    result = _call_compute(horizon="now")
    assert result["horizon"] == "now"
    assert result["low_confidence"] is False
    assert result["valid_time"] == "2026-04-03T12:00:00+00:00"


def test_horizon_24h_offset():
    from api.ignition import grid as grid_mod

    fake_engine = MagicMock()
    now = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)
    expected_valid_time = "2026-04-04T12:00:00+00:00"

    with patch.object(grid_mod, "resolve_active_model", return_value=_make_active_model()), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=now), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=now), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=True), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=True), \
         patch.object(grid_mod, "_run_onnx_inference", return_value=np.full(9, 0.15)), \
         patch("api.ignition.grid.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = grid_mod.compute_ignition_grid(
            -121.0, 37.0, -120.0, 38.0,
            cell_size_km=40.0,
            horizon="+24h",
            engine=fake_engine,
        )

    assert result["horizon"] == "+24h"
    assert result["low_confidence"] is False
    assert "2026-04-04" in result["valid_time"]


def test_horizon_48h_sets_low_confidence():
    from api.ignition import grid as grid_mod

    fake_engine = MagicMock()
    now = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)

    with patch.object(grid_mod, "resolve_active_model", return_value=_make_active_model()), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=now), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=now), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=True), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=True), \
         patch.object(grid_mod, "_run_onnx_inference", return_value=np.full(9, 0.15)), \
         patch("api.ignition.grid.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = grid_mod.compute_ignition_grid(
            -121.0, 37.0, -120.0, 38.0,
            cell_size_km=40.0,
            horizon="+48h",
            engine=fake_engine,
        )

    assert result["horizon"] == "+48h"
    assert result["low_confidence"] is True
    assert "2026-04-05" in result["valid_time"]


def test_stale_drought_index_emits_warning():
    from api.ignition import grid as grid_mod

    fake_engine = MagicMock()
    now = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)
    stale_date = datetime(2026, 3, 20, tzinfo=timezone.utc)

    with patch.object(grid_mod, "resolve_active_model", return_value=_make_active_model()), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=now), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=stale_date), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=True), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=True), \
         patch.object(grid_mod, "_run_onnx_inference", return_value=np.full(9, 0.1)), \
         patch("api.ignition.grid.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = grid_mod.compute_ignition_grid(
            -121.0, 37.0, -120.0, 38.0,
            cell_size_km=40.0,
            horizon="now",
            engine=fake_engine,
        )

    drought_warnings = [w for w in result["coverage_warnings"] if w.startswith("drought_index_stale:")]
    assert len(drought_warnings) == 1


def test_missing_thunderstorm_data_emits_warning():
    from api.ignition import grid as grid_mod

    fake_engine = MagicMock()
    now = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)

    with patch.object(grid_mod, "resolve_active_model", return_value=_make_active_model()), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=now), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=now), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=False), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=True), \
         patch.object(grid_mod, "_run_onnx_inference", return_value=np.full(9, 0.1)), \
         patch("api.ignition.grid.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = grid_mod.compute_ignition_grid(
            -121.0, 37.0, -120.0, 38.0,
            cell_size_km=40.0,
            horizon="now",
            engine=fake_engine,
        )

    assert "thunderstorm_data_missing" in result["coverage_warnings"]


def test_grid_capped_at_500_cells():
    from api.ignition import grid as grid_mod

    fake_engine = MagicMock()
    now = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)

    with patch.object(grid_mod, "resolve_active_model", return_value=_make_active_model()), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=now), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=now), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=True), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=True), \
         patch.object(grid_mod, "_run_onnx_inference") as mock_infer, \
         patch("api.ignition.grid.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        mock_infer.side_effect = lambda uri, matrix: np.full(len(matrix), 0.1)
        result = grid_mod.compute_ignition_grid(
            -130.0, 25.0, -65.0, 50.0,
            cell_size_km=1.0,
            horizon="now",
            engine=fake_engine,
        )

    assert len(result["cells"]) <= 500


def test_503_when_no_model_and_required():
    from api.ignition import grid as grid_mod
    from api.ignition.grid import IgnitionModelUnavailable

    fake_engine = MagicMock()

    with patch.object(grid_mod, "resolve_active_model", return_value=None), \
         patch.dict(os.environ, {"IGNITION_REQUIRED": "true"}):
        with pytest.raises(IgnitionModelUnavailable):
            grid_mod.compute_ignition_grid(
                -121.0, 37.0, -120.0, 38.0,
                cell_size_km=40.0,
                horizon="now",
                engine=fake_engine,
            )


def test_onnx_inference_failure_returns_503():
    """ONNX inference error must raise IgnitionInferenceFailed, not return fake data."""
    from api.ignition import grid as grid_mod
    from api.ignition.grid import IgnitionInferenceFailed

    fake_engine = MagicMock()
    now = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)

    with patch.object(grid_mod, "resolve_active_model", return_value=_make_active_model()), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=now), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=now), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=True), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=True), \
         patch.object(grid_mod, "_run_onnx_inference", side_effect=RuntimeError("ONNX session failed")), \
         patch("api.ignition.grid.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        with pytest.raises(IgnitionInferenceFailed):
            grid_mod.compute_ignition_grid(
                -121.0, 37.0, -120.0, 38.0,
                cell_size_km=40.0,
                horizon="now",
                engine=fake_engine,
            )


def test_missing_drought_index_emits_warning():
    """When drought index is unavailable, coverage_warnings must contain a drought warning."""
    from api.ignition import grid as grid_mod

    fake_engine = MagicMock()
    now = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)

    with patch.object(grid_mod, "resolve_active_model", return_value=_make_active_model()), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=now), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=None), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=True), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=True), \
         patch.object(grid_mod, "_run_onnx_inference", return_value=np.full(9, 0.15)), \
         patch("api.ignition.grid.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = grid_mod.compute_ignition_grid(
            -121.0, 37.0, -120.0, 38.0,
            cell_size_km=40.0,
            horizon="now",
            engine=fake_engine,
        )

    drought_warnings = [
        w for w in result["coverage_warnings"]
        if "drought_index_stale" in w or "drought_index_missing" in w
    ]
    assert len(drought_warnings) >= 1, (
        f"Expected a drought_index_stale/missing warning; got: {result['coverage_warnings']}"
    )


def test_ignition_required_false_no_model_returns_503():
    """With IGNITION_REQUIRED=false and no model, must raise IgnitionModelUnavailable (not return zeros)."""
    from api.ignition import grid as grid_mod
    from api.ignition.grid import IgnitionModelUnavailable

    fake_engine = MagicMock()

    with patch.object(grid_mod, "resolve_active_model", return_value=None), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=None), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=None), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=False), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=False), \
         patch.dict(os.environ, {"IGNITION_REQUIRED": "false"}):
        with pytest.raises(IgnitionModelUnavailable):
            grid_mod.compute_ignition_grid(
                -121.0, 37.0, -120.0, 38.0,
                cell_size_km=40.0,
                horizon="now",
                engine=fake_engine,
            )


def test_model_id_matches_registry():
    from api.ignition import grid as grid_mod

    fake_engine = MagicMock()
    now = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)
    expected_model_id = "ignition-my-run-20260101000000-deadbeef"
    model = _make_active_model()
    model["model_id"] = expected_model_id

    with patch.object(grid_mod, "resolve_active_model", return_value=model), \
         patch.object(grid_mod, "_query_weather_for_cells", return_value={}), \
         patch.object(grid_mod, "_query_latest_weather_run_time", return_value=now), \
         patch.object(grid_mod, "_query_drought_index_freshness", return_value=now), \
         patch.object(grid_mod, "_query_thunderstorm_present", return_value=True), \
         patch.object(grid_mod, "_check_gfs_48h_available", return_value=True), \
         patch.object(grid_mod, "_run_onnx_inference", return_value=np.full(9, 0.15)), \
         patch("api.ignition.grid.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = grid_mod.compute_ignition_grid(
            -121.0, 37.0, -120.0, 38.0,
            cell_size_km=40.0,
            horizon="now",
            engine=fake_engine,
        )

    assert result["model_id"] == expected_model_id
