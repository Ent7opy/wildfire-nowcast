"""Unit tests for the scoring_pipeline Strategy pattern.

Each test uses a plain stub that satisfies ScoringStrategy structurally —
no subclassing required, demonstrating that the Protocol is genuinely duck-typed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from api.fires.scoring_pipeline import (
    FalseSourceMaskingStrategy,
    LandcoverScoringStrategy,
    PersistenceScoringStrategy,
    WeatherScoringStrategy,
    run_scoring_stage,
)


# ── Stub strategy helpers ─────────────────────────────────────────────────────

class _SimpleStrategy:
    """Minimal duck-typed stub — no ScoringStrategy inheritance needed."""

    name = "test_score"
    select_fields = ("id", "lat", "lon")
    update_column = "test_score"
    update_param = "score"
    neutral_threshold = 0
    neutral_value = 0.5

    def __init__(self, compute_result: dict[int, Any] | None = None):
        self._result = compute_result or {1: 0.8, 2: 0.9}
        self.compute_calls: list[list[dict]] = []

    def compute(self, detections: list[dict]) -> dict[int, Any]:
        self.compute_calls.append(detections)
        return self._result

    def count_result(self, results: dict[int, Any]) -> int:
        return len(results)


def _make_mock_conn(rows: list[dict]) -> MagicMock:
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = rows
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result
    return mock_conn


# ── run_scoring_stage: core driver ────────────────────────────────────────────

def test_run_scoring_stage_empty_batch_returns_zero():
    strategy = _SimpleStrategy()
    mock_conn = _make_mock_conn([])

    with patch("api.fires.scoring_pipeline.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        result = run_scoring_stage(batch_id=1, strategy=strategy)

    assert result == 0
    assert strategy.compute_calls == []


def test_run_scoring_stage_calls_compute_and_updates():
    detections = [{"id": 1, "lat": 42.0, "lon": 23.0}, {"id": 2, "lat": 43.0, "lon": 24.0}]
    strategy = _SimpleStrategy({1: 0.8, 2: 0.9})
    mock_conn = _make_mock_conn(detections)

    with patch("api.fires.scoring_pipeline.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        result = run_scoring_stage(batch_id=7, strategy=strategy)

    assert result == 2
    assert len(strategy.compute_calls) == 1
    assert strategy.compute_calls[0][0]["id"] == 1


def test_run_scoring_stage_uses_existing_conn():
    """When conn is provided, get_engine must not be called."""
    detections = [{"id": 5, "lat": 1.0, "lon": 2.0}]
    strategy = _SimpleStrategy({5: 0.7})

    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = detections
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_result

    with patch("api.fires.scoring_pipeline.get_engine") as mock_engine:
        result = run_scoring_stage(batch_id=3, strategy=strategy, conn=mock_conn)
        mock_engine.assert_not_called()

    assert result == 1


def test_run_scoring_stage_neutral_fallback_when_threshold_exceeded():
    """Neutral fallback: batch UPDATE by batch_id instead of per-detection UPDATE."""
    class _ThresholdedStrategy(_SimpleStrategy):
        neutral_threshold = 2
        neutral_value = 0.5

    detections = [
        {"id": 1, "lat": 0.0, "lon": 0.0},
        {"id": 2, "lat": 1.0, "lon": 1.0},
    ]
    strategy = _ThresholdedStrategy()
    mock_conn = _make_mock_conn(detections)

    with patch("api.fires.scoring_pipeline.get_engine") as mock_engine, \
         patch("api.fires.scoring_pipeline._DISABLE_NEUTRAL_FALLBACK", False):
        mock_engine.return_value.begin.return_value = mock_conn
        result = run_scoring_stage(batch_id=99, strategy=strategy)

    # compute() should NOT have been called — neutral path skips it
    assert strategy.compute_calls == []
    # Returns full detection count
    assert result == 2


def test_run_scoring_stage_neutral_fallback_disabled():
    """When _DISABLE_NEUTRAL_FALLBACK is True, compute() runs even on large batches."""
    class _ThresholdedStrategy(_SimpleStrategy):
        neutral_threshold = 1  # threshold of 1 — would fire for any non-empty batch
        neutral_value = 0.5

    detections = [{"id": 10, "lat": 0.0, "lon": 0.0}]
    strategy = _ThresholdedStrategy({10: 0.8})
    mock_conn = _make_mock_conn(detections)

    with patch("api.fires.scoring_pipeline.get_engine") as mock_engine, \
         patch("api.fires.scoring_pipeline._DISABLE_NEUTRAL_FALLBACK", True):
        mock_engine.return_value.begin.return_value = mock_conn
        result = run_scoring_stage(batch_id=55, strategy=strategy)

    assert len(strategy.compute_calls) == 1
    assert result == 1


def test_run_scoring_stage_count_result_is_delegated():
    """count_result() on the strategy controls the return value."""
    class _BoolStrategy(_SimpleStrategy):
        update_column = "flag"
        update_param = "flag"

        def compute(self, detections):
            return {d["id"]: True for d in detections}

        def count_result(self, results):
            return sum(1 for v in results.values() if v)

    detections = [{"id": i, "lat": 0.0, "lon": 0.0} for i in range(3)]
    strategy = _BoolStrategy()
    mock_conn = _make_mock_conn(detections)

    with patch("api.fires.scoring_pipeline.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        result = run_scoring_stage(batch_id=1, strategy=strategy)

    assert result == 3


# ── Strategy metadata & Protocol conformance ─────────────────────────────────

@pytest.mark.parametrize("strategy_cls, expected", [
    (FalseSourceMaskingStrategy, "false_source"),
    (PersistenceScoringStrategy, "persistence"),
    (LandcoverScoringStrategy, "landcover"),
    (WeatherScoringStrategy, "weather"),
])
def test_strategy_name(strategy_cls, expected):
    assert strategy_cls().name == expected


@pytest.mark.parametrize("strategy_cls, expected_col", [
    (FalseSourceMaskingStrategy, "false_source_masked"),
    (PersistenceScoringStrategy, "persistence_score"),
    (LandcoverScoringStrategy, "landcover_score"),
    (WeatherScoringStrategy, "weather_score"),
])
def test_strategy_update_column(strategy_cls, expected_col):
    assert strategy_cls().update_column == expected_col


@pytest.mark.parametrize("strategy_cls", [
    FalseSourceMaskingStrategy,
    PersistenceScoringStrategy,
    LandcoverScoringStrategy,
    WeatherScoringStrategy,
])
def test_strategy_has_required_protocol_attributes(strategy_cls):
    s = strategy_cls()
    assert isinstance(s.name, str)
    assert isinstance(s.select_fields, tuple)
    assert "id" in s.select_fields
    assert isinstance(s.update_column, str)
    assert isinstance(s.update_param, str)
    assert isinstance(s.neutral_threshold, int)
    assert callable(s.compute)
    assert callable(s.count_result)


# ── FalseSourceMaskingStrategy specifics ─────────────────────────────────────

def test_false_source_count_result_counts_only_truthy():
    s = FalseSourceMaskingStrategy()
    assert s.count_result({1: True, 2: False, 3: True}) == 2
    assert s.count_result({1: False, 2: False}) == 0
    assert s.count_result({}) == 0


def test_false_source_has_no_neutral_threshold():
    assert FalseSourceMaskingStrategy().neutral_threshold == 0


def test_false_source_compute_delegates_to_mask_false_sources():
    s = FalseSourceMaskingStrategy()
    detections = [{"id": 1, "lat": 0.0, "lon": 0.0}]
    with patch("api.fires.scoring_pipeline.mask_false_sources") as mock_fn:
        mock_fn.return_value = {1: True}
        result = s.compute(detections)
    mock_fn.assert_called_once_with(detections)
    assert result == {1: True}


# ── PersistenceScoringStrategy specifics ─────────────────────────────────────

def test_persistence_count_result_is_len():
    s = PersistenceScoringStrategy()
    assert s.count_result({1: 0.8, 2: 0.3}) == 2


def test_persistence_select_fields_include_sensor():
    assert "sensor" in PersistenceScoringStrategy().select_fields
    assert "acq_time" in PersistenceScoringStrategy().select_fields


def test_persistence_compute_delegates_to_compute_persistence_scores():
    s = PersistenceScoringStrategy()
    detections = [{"id": 1, "lat": 0.0, "lon": 0.0, "acq_time": None, "sensor": "VIIRS"}]
    with patch("api.fires.scoring_pipeline.compute_persistence_scores") as mock_fn:
        mock_fn.return_value = {1: 0.8}
        result = s.compute(detections)
    mock_fn.assert_called_once_with(detections)
    assert result == {1: 0.8}


# ── LandcoverScoringStrategy specifics ───────────────────────────────────────

def test_landcover_has_no_neutral_threshold():
    assert LandcoverScoringStrategy().neutral_threshold == 0


def test_landcover_compute_calls_landcover_module():
    s = LandcoverScoringStrategy()
    detections = [{"id": 1, "lat": 0.0, "lon": 0.0}]
    with patch("api.fires.landcover.compute_landcover_scores") as mock_fn:
        mock_fn.return_value = {1: 0.9}
        result = s.compute(detections)
    mock_fn.assert_called_once_with(detections)
    assert result == {1: 0.9}


# ── WeatherScoringStrategy specifics ─────────────────────────────────────────

def test_weather_compute_passes_time_tolerance():
    s = WeatherScoringStrategy()
    detections = [{"id": 1, "lat": 0.0, "lon": 0.0, "acq_time": None}]
    with patch("api.fires.scoring_pipeline.compute_weather_plausibility_scores") as mock_fn:
        mock_fn.return_value = {1: 0.6}
        s.compute(detections)
    _, kwargs = mock_fn.call_args
    assert "time_tolerance_hours" in kwargs


def test_weather_count_result_is_len():
    s = WeatherScoringStrategy()
    assert s.count_result({1: 0.5, 2: 0.7, 3: 0.3}) == 3
