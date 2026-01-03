"""Tests for weather bias analysis metrics."""

import numpy as np
import pytest
from ml.weather_bias_analysis import compute_metrics


def test_compute_metrics_basic():
    """Test metrics with simple integer arrays."""
    forecast = np.array([2.0, 4.0, 6.0])
    truth = np.array([1.0, 2.0, 3.0])
    # bias: [1, 2, 3]
    # abs bias: [1, 2, 3]
    # sq bias: [1, 4, 9]
    
    m = compute_metrics(forecast, truth)
    
    assert m["bias_mean"] == pytest.approx(2.0)
    assert m["bias_std"] == pytest.approx(np.std([1, 2, 3]))
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 4 + 9) / 3))
    assert m["count"] == 3


def test_compute_metrics_with_nans():
    """Test metrics with NaNs in inputs."""
    forecast = np.array([2.0, np.nan, 6.0])
    truth = np.array([1.0, 2.0, 3.0])
    # valid diffs: [1.0, 3.0]
    
    m = compute_metrics(forecast, truth)
    
    assert m["bias_mean"] == pytest.approx(2.0)
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 9) / 2))
    assert m["count"] == 2


def test_compute_metrics_all_nans():
    """Test metrics with all NaNs."""
    forecast = np.array([np.nan, np.nan])
    truth = np.array([1.0, 2.0])
    
    m = compute_metrics(forecast, truth)
    
    assert np.isnan(m["bias_mean"])
    assert np.isnan(m["mae"])
    assert np.isnan(m["rmse"])
    assert m["count"] == 0


def test_compute_metrics_zero_diff():
    """Test metrics with zero difference."""
    forecast = np.array([1.0, 1.0])
    truth = np.array([1.0, 1.0])
    
    m = compute_metrics(forecast, truth)
    
    assert m["bias_mean"] == 0.0
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["count"] == 2

