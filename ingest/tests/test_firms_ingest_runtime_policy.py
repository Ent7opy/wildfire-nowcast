from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ingest.firms_ingest import _build_denoiser_argv, _resolve_denoiser_runtime_policy


def _cfg(**overrides):
    defaults = {
        "denoiser_pipeline_version": "v2",
        "denoiser_threshold_profile": "strict_v1",
        "denoiser_allow_unsafe_threshold_override": False,
        "denoiser_model_run_dir": "/models/fallback",
        "denoiser_strong_filter_threshold": 0.99,
        "denoiser_downweight_threshold": 0.98,
        "denoiser_uncertainty_band_low": 0.97,
        "denoiser_uncertainty_band_high": 0.999,
        "denoiser_event_front_radius_m": 9999.0,
        "denoiser_event_front_max_gap_minutes": 999,
        "denoiser_event_link_radius_m": 8888.0,
        "denoiser_event_link_max_gap_days": 88,
        "denoiser_event_static_persistence_threshold": 0.77,
        "denoiser_event_strict_static_split": True,
        "denoiser_shadow_mode": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _contract():
    return {
        "pipeline_version": "v2",
        "threshold_profile": "strict_v1",
        "thresholds": {
            "strong_filter_threshold": 0.5,
            "downweight_threshold": 0.7,
            "uncertainty_band_low": 0.45,
            "uncertainty_band_high": 0.55,
            "event_front_radius_m": 2500.0,
            "event_front_max_gap_minutes": 45,
            "event_link_radius_m": 10000.0,
            "event_link_max_gap_days": 11,
            "event_static_persistence_threshold": 0.85,
            "event_strict_static_split": True,
        },
    }


def test_pipeline_mismatch_fails_when_unsafe_override_disabled() -> None:
    cfg = _cfg(denoiser_pipeline_version="v1")
    active = {
        "model_id": "denoiser-123",
        "artifact_uri": "/models/promoted",
        "metrics_json": {"runtime_contract": _contract()},
    }
    with patch("ingest.firms_ingest._resolve_active_denoiser_model", return_value=active):
        with pytest.raises(RuntimeError, match="pipeline mismatch"):
            _resolve_denoiser_runtime_policy(cfg)


def test_missing_runtime_contract_fails_for_strict_profile() -> None:
    cfg = _cfg()
    active = {
        "model_id": "denoiser-123",
        "artifact_uri": "/models/promoted",
        "metrics_json": {},
    }
    with patch("ingest.firms_ingest._resolve_active_denoiser_model", return_value=active):
        with pytest.raises(RuntimeError, match="runtime_contract"):
            _resolve_denoiser_runtime_policy(cfg)


def test_strict_profile_uses_registry_thresholds_not_env_values() -> None:
    cfg = _cfg()
    active = {
        "model_id": "denoiser-123",
        "artifact_uri": "/models/promoted",
        "metrics_json": {"runtime_contract": _contract()},
    }
    with patch("ingest.firms_ingest._resolve_active_denoiser_model", return_value=active):
        policy = _resolve_denoiser_runtime_policy(cfg)

    assert policy is not None
    assert policy.threshold_source == "registry_contract"
    assert policy.thresholds["strong_filter_threshold"] == 0.5
    assert policy.thresholds["strong_filter_threshold"] != cfg.denoiser_strong_filter_threshold

    argv = _build_denoiser_argv(
        batch_id=101,
        model_run_dir=policy.model_run_dir,
        config=cfg,
        runtime_policy=policy,
    )
    assert "--strong-filter-threshold" in argv
    assert "0.5" in argv
    assert "0.99" not in argv


def test_unsafe_override_allows_missing_contract_and_uses_env_thresholds() -> None:
    cfg = _cfg(denoiser_allow_unsafe_threshold_override=True)
    active = {
        "model_id": "denoiser-123",
        "artifact_uri": "/models/promoted",
        "metrics_json": {},
    }
    with patch("ingest.firms_ingest._resolve_active_denoiser_model", return_value=active):
        policy = _resolve_denoiser_runtime_policy(cfg)

    assert policy is not None
    assert policy.threshold_source == "env_config"
    assert policy.thresholds["strong_filter_threshold"] == 0.99
