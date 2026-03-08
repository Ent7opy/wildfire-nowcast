import numpy as np
import pandas as pd

from ml.train_denoiser_v2 import (
    _apply_adasyn_high_intensity,
    _apply_pseudo_negative_caps,
    _build_pseudo_label_masks,
    _oob_mean_scores,
    _stratified_majority_sample,
)


def test_oob_mean_scores_respects_min_votes() -> None:
    score_sums = np.asarray([1.2, 0.9, 0.3], dtype=float)
    vote_counts = np.asarray([3, 2, 1], dtype=int)

    means, valid = _oob_mean_scores(score_sums, vote_counts, min_oob_votes=2)

    assert valid.tolist() == [True, True, False]
    assert np.isclose(means[0], 0.4)
    assert np.isclose(means[1], 0.45)
    assert np.isnan(means[2])


def test_oob_mean_scores_min_votes_floor_to_one() -> None:
    score_sums = np.asarray([0.0, 0.4], dtype=float)
    vote_counts = np.asarray([0, 2], dtype=int)

    means, valid = _oob_mean_scores(score_sums, vote_counts, min_oob_votes=0)

    assert valid.tolist() == [False, True]
    assert np.isnan(means[0])
    assert np.isclose(means[1], 0.2)


def test_build_pseudo_label_masks_applies_oob_margin() -> None:
    oob_mean = np.asarray([0.76, 0.73, 0.27, 0.24], dtype=float)
    valid_votes = np.asarray([True, True, True, True], dtype=bool)

    pseudo_pos, pseudo_neg, ignored, pos_cut, neg_cut = _build_pseudo_label_masks(
        oob_mean=oob_mean,
        valid_votes=valid_votes,
        pos_threshold=0.70,
        neg_threshold=0.30,
        oob_margin_min=0.05,
    )

    assert np.isclose(pos_cut, 0.75)
    assert np.isclose(neg_cut, 0.25)
    assert pseudo_pos.tolist() == [True, False, False, False]
    assert pseudo_neg.tolist() == [False, False, False, True]
    assert ignored.tolist() == [False, True, True, False]


def test_apply_pseudo_negative_caps_respects_ratio_and_absolute_cap() -> None:
    mask = np.asarray([True] * 10, dtype=bool)
    capped, stats = _apply_pseudo_negative_caps(
        pseudo_neg_mask=mask,
        pseudo_positive_rows=2,
        pseudo_negative_max_ratio=3.0,
        max_pseudo_negative_rows=5,
        rng=np.random.default_rng(42),
    )

    assert int(capped.sum()) == 5
    assert stats["pseudo_negative_rows_before_cap"] == 10
    assert stats["pseudo_negative_rows_after_cap"] == 5
    assert stats["pseudo_negative_rows_cap_target"] == 5
    assert stats["pseudo_negative_rows_dropped"] == 5


def test_apply_pseudo_negative_caps_drops_all_when_no_pseudo_positives() -> None:
    mask = np.asarray([True, True, True], dtype=bool)
    capped, stats = _apply_pseudo_negative_caps(
        pseudo_neg_mask=mask,
        pseudo_positive_rows=0,
        pseudo_negative_max_ratio=3.0,
        max_pseudo_negative_rows=0,
        rng=np.random.default_rng(11),
    )

    assert int(capped.sum()) == 0
    assert stats["pseudo_negative_rows_before_cap"] == 3
    assert stats["pseudo_negative_rows_after_cap"] == 0
    assert stats["pseudo_negative_rows_cap_target"] == 0
    assert stats["pseudo_negative_rows_dropped"] == 3


def test_stratified_majority_sample_respects_1_to_10_ratio() -> None:
    df = pd.DataFrame(
        {
            "sensor_id": ["S-NPP"] * 3 + ["NOAA-20"] * 6 + ["NOAA-21"] * 31,
            "biome_slice": ["forest"] * 40,
            "x": np.linspace(0.0, 1.0, 40),
        }
    )
    y = np.asarray([1, 1, 1] + [0] * 6 + [-1] * 31, dtype=int)

    sampled_df, sampled_y, stats = _stratified_majority_sample(
        train_df=df,
        y_train=y,
        ratio_majority_to_positive=10.0,
        slice_cols=["sensor_id", "biome_slice"],
        rng=np.random.default_rng(42),
    )

    assert len(sampled_df) == len(sampled_y)
    assert int((sampled_y == 1).sum()) == 3
    assert int((sampled_y != 1).sum()) == 30
    assert stats["sampling_applied"] is True


def test_apply_adasyn_high_intensity_generates_positive_rows() -> None:
    x = pd.DataFrame(
        {
            "frp_max": [10, 20, 40, 80, 100, 5, 6, 7, 8, 9, 11, 12],
            "f1": np.linspace(0.0, 1.0, 12),
            "f2": np.linspace(1.0, 2.0, 12),
        }
    )
    y = np.asarray([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, 0, -1], dtype=int)
    config = {
        "adasyn": {
            "enabled": True,
            "intensity_feature": "frp_max",
            "high_intensity_quantile": 0.6,
            "multiplier": 2.0,
            "k_neighbors": 3,
            "min_high_intensity_rows": 2,
            "max_synthetic_rows": 100,
        }
    }

    x_aug, y_aug, stats = _apply_adasyn_high_intensity(
        config=config,
        x_train=x,
        y_train=y,
        rng=np.random.default_rng(7),
    )

    assert len(x_aug) == len(y_aug)
    assert stats["generated_rows"] > 0
    assert int((y_aug == 1).sum()) > int((y == 1).sum())
