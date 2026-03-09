import numpy as np

from ml.eval_spread_champion_challenger import (
    _build_stage_governance,
    compute_recommendation,
    summarize_comparison_for_horizon,
)


def test_summarize_comparison_reports_positive_primary_improvement_when_challenger_is_better():
    rng = np.random.default_rng(42)
    y_true = (rng.uniform(size=5000) < 0.25).astype(np.float32)

    champion = np.clip(0.7 * y_true + 0.3 * rng.uniform(size=5000), 0.0, 1.0)
    challenger = np.clip(0.85 * y_true + 0.15 * rng.uniform(size=5000), 0.0, 1.0)

    # Provide one synthetic case so SAL/DM metrics are populated.
    case_shape = (50, 100)
    row = summarize_comparison_for_horizon(
        horizon_hours=24,
        y_true=y_true,
        y_prob_champion=champion,
        y_prob_challenger=challenger,
        y_true_cases=[y_true.reshape(case_shape)],
        champion_cases=[champion.reshape(case_shape)],
        challenger_cases=[challenger.reshape(case_shape)],
    )

    assert row["brier_improvement"] > 0
    assert row["ece_improvement"] > 0
    assert row["bss_improvement"] > -0.1


def test_compute_recommendation_rejects_primary_regression():
    summary_rows = [
        {
            "horizon_hours": 24,
            "n": 100,
            "bss_improvement": 0.04,
            "sal_composite_improvement": 0.01,
            "dm_p_value": 0.01,
            "pr_auc_improvement": 0.001,
            "iou_03_improvement": 0.01,
            "iou_05_improvement": 0.01,
        },
        {
            "horizon_hours": 48,
            "n": 100,
            "bss_improvement": -0.02,
            "sal_composite_improvement": 0.01,
            "dm_p_value": 0.01,
            "pr_auc_improvement": 0.0,
            "iou_03_improvement": 0.0,
            "iou_05_improvement": 0.0,
        },
    ]

    decision = compute_recommendation(summary_rows)
    assert decision["recommend_challenger"] is False
    assert decision["primary_ok"] is False


def test_compute_recommendation_accepts_when_primary_and_secondary_gates_pass():
    summary_rows = [
        {
            "horizon_hours": 24,
            "n": 100,
            "bss_improvement": 0.05,
            "sal_composite_improvement": 0.02,
            "dm_p_value": 0.01,
            "pr_auc_improvement": -0.005,
            "iou_03_improvement": -0.01,
            "iou_05_improvement": -0.015,
        },
        {
            "horizon_hours": 48,
            "n": 100,
            "bss_improvement": 0.03,
            "sal_composite_improvement": 0.01,
            "dm_p_value": 0.02,
            "pr_auc_improvement": 0.003,
            "iou_03_improvement": 0.0,
            "iou_05_improvement": 0.001,
        },
        {
            "horizon_hours": 72,
            "n": 100,
            "bss_improvement": 0.03,
            "sal_composite_improvement": 0.001,
            "dm_p_value": 0.03,
            "pr_auc_improvement": 0.002,
            "iou_03_improvement": 0.0,
            "iou_05_improvement": 0.0,
        },
    ]

    decision = compute_recommendation(summary_rows)
    assert decision["recommend_challenger"] is True
    assert decision["primary_ok"] is True
    assert decision["secondary_ok"] is True


def test_stage_governance_requires_calibrator_for_v3_promotion(tmp_path):
    decision = {
        "pass": True,
        "weighted_bss_improvement": 0.02,
        "recommend_challenger": True,
        "reasons": [],
    }
    config_missing_cal = {
        "gate": {"maturity_stage": "mvp_operational"},
        "challenger": {
            "model_name": "LearnedSpreadModelV3",
            "model_params": {"model_run_dir": str(tmp_path / "run")},
        },
    }

    out_missing = _build_stage_governance(config=config_missing_cal, decision=decision, summary_rows=[])
    assert out_missing["hard_stops"]
    assert any(s["id"] == "STOP-CAL-001" for s in out_missing["hard_stops"])

    cal_dir = tmp_path / "cal"
    cal_dir.mkdir(parents=True)
    (cal_dir / "calibrator.pkl").write_bytes(b"ok")
    config_with_cal = {
        "gate": {"maturity_stage": "mvp_operational"},
        "challenger": {
            "model_name": "LearnedSpreadModelV3",
            "model_params": {
                "model_run_dir": str(tmp_path / "run"),
                "calibrator_run_dir": str(cal_dir),
            },
        },
    }
    out_with_cal = _build_stage_governance(config=config_with_cal, decision=decision, summary_rows=[])
    assert not any(s["id"] == "STOP-CAL-001" for s in out_with_cal["hard_stops"])
