import json

import pytest

from api.forecast.model_catalog import (
    compute_catalog_signature,
    get_spread_model_catalog,
    resolve_request_model_selection,
)


def test_signed_catalog_valid_signature_resolves_model_id():
    catalog = {
        "spread_v1_prod": {
            "model_name": "LearnedSpreadModelV1",
            "model_params": {"model_run_dir": "models/spread_v1/run_123"},
        }
    }
    signing_key = "test-signing-key"
    signature = compute_catalog_signature(catalog, signing_key)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SPREAD_MODEL_CATALOG_JSON", json.dumps(catalog))
        mp.setenv("SPREAD_MODEL_CATALOG_SIGNING_KEY", signing_key)
        mp.setenv("SPREAD_MODEL_CATALOG_SIGNATURE", signature)
        mp.setenv("SPREAD_MODEL_CATALOG_REQUIRE_SIGNATURE", "true")
        get_spread_model_catalog.cache_clear()

        model_name, model_params, model_id = resolve_request_model_selection(
            model_id="spread_v1_prod",
            model_name=None,
            model_params=None,
        )
        assert model_name == "LearnedSpreadModelV1"
        assert model_params == {"model_run_dir": "models/spread_v1/run_123"}
        assert model_id == "spread_v1_prod"


def test_signed_catalog_invalid_signature_fails():
    catalog = {
        "v0_default": {
            "model_name": "HeuristicSpreadModelV0",
            "model_params": {},
        }
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SPREAD_MODEL_CATALOG_JSON", json.dumps(catalog))
        mp.setenv("SPREAD_MODEL_CATALOG_SIGNING_KEY", "test-signing-key")
        mp.setenv("SPREAD_MODEL_CATALOG_SIGNATURE", "not-valid")
        mp.setenv("SPREAD_MODEL_CATALOG_REQUIRE_SIGNATURE", "true")
        get_spread_model_catalog.cache_clear()

        with pytest.raises(ValueError, match="signature mismatch"):
            get_spread_model_catalog()


def test_signed_catalog_required_without_signature_fails():
    catalog = {
        "v0_default": {
            "model_name": "HeuristicSpreadModelV0",
            "model_params": {},
        }
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SPREAD_MODEL_CATALOG_JSON", json.dumps(catalog))
        mp.setenv("SPREAD_MODEL_CATALOG_REQUIRE_SIGNATURE", "true")
        mp.delenv("SPREAD_MODEL_CATALOG_SIGNATURE", raising=False)
        mp.delenv("SPREAD_MODEL_CATALOG_SIGNING_KEY", raising=False)
        get_spread_model_catalog.cache_clear()

        with pytest.raises(ValueError, match="SPREAD_MODEL_CATALOG_SIGNING_KEY is required"):
            get_spread_model_catalog()


def test_default_model_selection_prefers_promoted_model(monkeypatch):
    monkeypatch.setattr(
        "api.forecast.model_catalog._resolve_promoted_spread_catalog_entry",
        lambda: (
            "spread_prod_20260215",
            ("LearnedSpreadModelV1", {"model_run_dir": "models/spread_v1/run_prod"}),
        ),
    )
    get_spread_model_catalog.cache_clear()

    model_name, model_params, model_id = resolve_request_model_selection(
        model_id=None,
        model_name=None,
        model_params=None,
    )

    assert model_name == "LearnedSpreadModelV1"
    assert model_params == {"model_run_dir": "models/spread_v1/run_prod"}
    assert model_id == "spread_prod_20260215"
