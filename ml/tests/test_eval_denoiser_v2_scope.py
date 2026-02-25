import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.eval_denoiser_v2 import evaluate_denoiser_v2


def _write_bundle(path: Path) -> None:
    model = LogisticRegression(max_iter=1000)
    x = pd.DataFrame({"f1": [0.0, 0.2, 0.8, 1.0]})
    y = [0, 0, 1, 1]
    model.fit(x, y)
    bundle = {
        "model": model,
        "features": ["f1"],
        "slice_cols": ["sensor", "biome_slice"],
        "global_calibrator": {"type": "identity", "model": None},
        "slice_calibrators": {},
        "thresholds": {
            "decision": 0.5,
            "strong_filter": 0.5,
            "downweight": 0.7,
            "uncertainty_band_low": 0.45,
            "uncertainty_band_high": 0.55,
        },
        "latency_per_10k_seconds": 1.0,
    }
    joblib.dump(bundle, path / "model_bundle.pkl")


def test_eval_denoiser_v2_both_scope_outputs(tmp_path: Path) -> None:
    model_run = tmp_path / "run"
    model_run.mkdir()
    _write_bundle(model_run)

    snapshot = tmp_path / "snapshot.parquet"
    df = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.9, 0.8, 0.3, 0.7],
            "event_label": ["NEGATIVE", "NEGATIVE", "POSITIVE", "POSITIVE", "NEGATIVE", "POSITIVE"],
            "sensor": ["VIIRS", "VIIRS", "VIIRS", "VIIRS", "VIIRS", "VIIRS"],
            "biome_slice": ["mixed_fuel"] * 6,
            "is_day_ratio": [0.0, 0.1, 0.8, 0.9, 0.2, 0.7],
            "truth_covered_mask": [True, True, True, True, False, False],
            "coverage_mask_ids": [["mask_a"], ["mask_a"], ["mask_a"], ["mask_a"], ["mask_b"], ["mask_b"]],
        }
    )
    df.to_parquet(snapshot, index=False)

    out_dir = tmp_path / "report"
    evaluate_denoiser_v2(
        model_run_dir=str(model_run),
        snapshot_path=str(snapshot),
        out_dir=str(out_dir),
        gate_scope="both",
        coverage_mask_source="db_mask",
    )

    summary = json.loads((out_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    gate = json.loads((out_dir / "gate_report.json").read_text(encoding="utf-8"))

    assert summary["gate_scope"] == "both"
    assert summary["global"] is not None
    assert summary["covered"] is not None
    assert "default_metrics" in summary
    assert "gate_results" in summary
    assert "threshold_recommendations" in summary
    assert sorted(summary["coverage_mask_ids"]) == ["mask_a", "mask_b"]

    assert gate["gate_scope"] == "both"
    assert gate["global_results"] is not None
    assert gate["covered_results"] is not None

    assert (out_dir / "threshold_sweep.csv").exists()
    assert (out_dir / "threshold_sweep_global.csv").exists()
    assert (out_dir / "threshold_sweep_covered.csv").exists()


def test_eval_denoiser_v2_covered_scope_requires_mask_column(tmp_path: Path) -> None:
    model_run = tmp_path / "run"
    model_run.mkdir()
    _write_bundle(model_run)

    snapshot = tmp_path / "snapshot_no_mask.parquet"
    df = pd.DataFrame(
        {
            "f1": [0.1, 0.9],
            "event_label": ["NEGATIVE", "POSITIVE"],
            "sensor": ["VIIRS", "VIIRS"],
            "biome_slice": ["mixed_fuel", "mixed_fuel"],
            "is_day_ratio": [0.1, 0.9],
        }
    )
    df.to_parquet(snapshot, index=False)

    with pytest.raises(ValueError, match="truth_covered_mask"):
        evaluate_denoiser_v2(
            model_run_dir=str(model_run),
            snapshot_path=str(snapshot),
            out_dir=str(tmp_path / "report_no_mask"),
            gate_scope="covered",
            fail_on_missing_coverage_mask=True,
        )
