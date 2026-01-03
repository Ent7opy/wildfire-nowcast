"""Tests for the spread model factory."""

import logging

import pytest

from unittest.mock import patch, MagicMock
from ml.spread.factory import get_spread_model
from ml.spread.heuristic_v0 import HeuristicSpreadModelV0
from ml.spread.learned_v1 import LearnedSpreadModelV1


def test_get_spread_model_learned_v1_success():
    """Verify that we can instantiate the learned v1 model."""
    params = {"model_run_dir": "/tmp/test_run"}
    
    # Mock the artifact loading to avoid FileNotFoundError and JSON errors
    with patch("ml.spread.learned_v1.joblib.load", return_value={}), \
         patch("ml.spread.learned_v1.os.path.exists", return_value=True), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value=["feature1", "feature2"]):
        
        model = get_spread_model("LearnedSpreadModelV1", params=params)
        assert isinstance(model, LearnedSpreadModelV1)
        assert model.model_run_dir == "/tmp/test_run"


def test_get_spread_model_success():
    """Verify that we can instantiate the known MVP model."""
    model = get_spread_model("HeuristicSpreadModelV0")
    assert isinstance(model, HeuristicSpreadModelV0)
    # Check default config
    assert model.config.base_spread_km_h == 0.05


def test_get_spread_model_with_params():
    """Verify that parameters are passed to the model config."""
    params = {"base_spread_km_h": 0.5, "wind_influence_km_h_per_ms": 1.0}
    model = get_spread_model("HeuristicSpreadModelV0", params=params)
    assert isinstance(model, HeuristicSpreadModelV0)
    assert model.config.base_spread_km_h == 0.5
    assert model.config.wind_influence_km_h_per_ms == 1.0


def test_get_spread_model_unknown_model():
    """Verify that asking for a non-existent model raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model: UnknownModel"):
        get_spread_model("UnknownModel")


def test_get_spread_model_filters_unknown_params(caplog):
    """Verify that unknown parameters are filtered and warned about."""
    params = {"base_spread_km_h": 0.1, "this_does_not_exist": 123}
    
    with caplog.at_level(logging.WARNING):
        model = get_spread_model("HeuristicSpreadModelV0", params=params)
    
    # Model should still be created with valid params
    assert isinstance(model, HeuristicSpreadModelV0)
    assert model.config.base_spread_km_h == 0.1
    
    # Warning should be logged
    assert "Ignoring unknown model_params for HeuristicSpreadModelV0" in caplog.text
    assert "this_does_not_exist" in caplog.text

