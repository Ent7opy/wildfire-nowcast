import numpy as np

from ml.eval_spread_champion_challenger import (
    compute_recommendation,
    summarize_comparison_for_horizon,
)


def test_summarize_comparison_reports_positive_primary_improvement_when_challenger_is_better():
    rng = np.random.default_rng(42)
    y_true = (rng.uniform(size=5000) < 0.25).astype(np.float32)

    # Champion is overconfident noisy copy; challenger is closer to truth.
    champion = np.clip(0.7 * y_true + 0.3 * rng.uniform(size=5000), 0.0, 1.0)
    challenger = np.clip(0.85 * y_true + 0.15 * rng.uniform(size=5000), 0.0, 1.0)

    row = summarize_comparison_for_horizon(
        horizon_hours=24,
        y_true=y_true,
        y_prob_champion=champion,
        y_prob_challenger=challenger,
    )

    assert row["brier_improvement"] > 0
    assert row["ece_improvement"] > 0


def test_compute_recommendation_conservative_gate_rejects_primary_regression():
    summary_rows = [
        {
            "horizon_hours": 24,
            "brier_improvement": 0.02,
            "ece_improvement": 0.01,
            "pr_auc_improvement": 0.001,
            "iou_03_improvement": 0.01,
            "iou_05_improvement": 0.01,
        },
        {
            "horizon_hours": 48,
            "brier_improvement": -0.01,
            "ece_improvement": 0.01,
            "pr_auc_improvement": 0.0,
            "iou_03_improvement": 0.0,
            "iou_05_improvement": 0.0,
        },
    ]

    decision = compute_recommendation(summary_rows, max_pr_auc_drop=0.01, max_iou_drop=0.02)
    assert decision["recommend_challenger"] is False
    assert decision["primary_ok"] is False


def test_compute_recommendation_conservative_gate_accepts_improvements_without_major_secondary_regression():
    summary_rows = [
        {
            "horizon_hours": 24,
            "brier_improvement": 0.02,
            "ece_improvement": 0.01,
            "pr_auc_improvement": -0.005,
            "iou_03_improvement": -0.01,
            "iou_05_improvement": -0.015,
        },
        {
            "horizon_hours": 48,
            "brier_improvement": 0.01,
            "ece_improvement": 0.02,
            "pr_auc_improvement": 0.003,
            "iou_03_improvement": 0.0,
            "iou_05_improvement": 0.001,
        },
    ]

    decision = compute_recommendation(summary_rows, max_pr_auc_drop=0.01, max_iou_drop=0.02)
    assert decision["recommend_challenger"] is True
    assert decision["primary_ok"] is True
    assert decision["secondary_ok"] is True
