import numpy as np

from ml.train_denoiser_v2 import _oob_mean_scores


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
