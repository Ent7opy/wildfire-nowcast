import numpy as np

from ml.train_denoiser_v2 import (
    _apply_pseudo_negative_caps,
    _build_pseudo_label_masks,
    _oob_mean_scores,
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
