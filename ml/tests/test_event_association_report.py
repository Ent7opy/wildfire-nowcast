from datetime import datetime, timedelta, timezone

import pandas as pd

from ml.denoiser.eval_event_association import (
    EventizeParams,
    compute_replay_diff_rate,
    compute_updated_assignments,
    evaluate_no_regression_gates,
)


def _sample_df() -> pd.DataFrame:
    t0 = datetime(2026, 2, 1, tzinfo=timezone.utc)
    return pd.DataFrame(
        [
            {
                "id": 1,
                "source": "TEST",
                "sensor": "VIIRS",
                "acq_time": t0,
                "lat": 42.0,
                "lon": 23.0,
                "false_source_masked": False,
                "persistence_score": 0.2,
            },
            {
                "id": 2,
                "source": "TEST",
                "sensor": "VIIRS",
                "acq_time": t0 + timedelta(minutes=30),
                "lat": 42.001,
                "lon": 23.001,
                "false_source_masked": False,
                "persistence_score": 0.3,
            },
            {
                "id": 3,
                "source": "TEST",
                "sensor": "VIIRS",
                "acq_time": t0 + timedelta(minutes=30),
                "lat": 42.001,
                "lon": 23.001,
                "false_source_masked": True,
                "persistence_score": 0.95,
            },
        ]
    )


def test_compute_updated_assignments_is_deterministic() -> None:
    df = _sample_df()
    params = EventizeParams()
    a = compute_updated_assignments(df, params=params)
    b = compute_updated_assignments(df, params=params)
    assert compute_replay_diff_rate(a, b) == 0.0


def test_compute_updated_assignments_respects_strict_static_split() -> None:
    df = _sample_df()
    params = EventizeParams(strict_static_split=True)
    assignments = compute_updated_assignments(df, params=params)

    event_2 = assignments.loc[assignments["id"] == 2, "event_id"].iloc[0]
    event_3 = assignments.loc[assignments["id"] == 3, "event_id"].iloc[0]
    assert event_2 != event_3


def test_no_regression_gate_logic() -> None:
    baseline = {
        "median_event_duration_hours": 10.0,
        "singleton_event_share": 0.40,
        "mixed_static_dynamic_event_share": 0.10,
    }
    updated_good = {
        "median_event_duration_hours": 12.0,
        "singleton_event_share": 0.35,
        "mixed_static_dynamic_event_share": 0.08,
    }
    updated_bad = {
        "median_event_duration_hours": 8.0,
        "singleton_event_share": 0.50,
        "mixed_static_dynamic_event_share": 0.20,
    }

    assert evaluate_no_regression_gates(baseline, updated_good)["pass"] is True
    assert evaluate_no_regression_gates(baseline, updated_bad)["pass"] is False
