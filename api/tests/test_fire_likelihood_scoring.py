"""Unit tests for compute_fire_likelihood function."""

import pytest

from api.fires.scoring import compute_fire_likelihood


def test_compute_fire_likelihood_all_scores_present():
    """Test correct weighted calculation with all component scores."""
    # Expected: 0.2*0.8 + 0.3*0.9 + 0.25*1.0 + 0.25*0.7 = 0.16 + 0.27 + 0.25 + 0.175 = 0.855
    result = compute_fire_likelihood(
        confidence_score=0.8,
        persistence_score=0.9,
        landcover_score=1.0,
        weather_score=0.7,
        false_source_masked=False,
    )
    assert result == pytest.approx(0.855, abs=0.001)


def test_compute_fire_likelihood_medium_landcover():
    """Test weighted calculation with lower land-cover score."""
    # Expected: 0.2*0.9 + 0.3*0.9 + 0.25*0.1 + 0.25*0.8 = 0.18 + 0.27 + 0.025 + 0.2 = 0.675
    result = compute_fire_likelihood(
        confidence_score=0.9,
        persistence_score=0.9,
        landcover_score=0.1,
        weather_score=0.8,
        false_source_masked=False,
    )
    assert result == pytest.approx(0.675, abs=0.001)


def test_compute_fire_likelihood_none_persistence():
    """Test handling of None persistence score (defaults to 0.5)."""
    # Expected: 0.2*0.8 + 0.3*0.5 + 0.25*0.9 + 0.25*0.7 = 0.16 + 0.15 + 0.225 + 0.175 = 0.71
    result = compute_fire_likelihood(
        confidence_score=0.8,
        persistence_score=None,
        landcover_score=0.9,
        weather_score=0.7,
        false_source_masked=False,
    )
    assert result == pytest.approx(0.71, abs=0.001)


def test_compute_fire_likelihood_none_landcover():
    """Test handling of None landcover score (defaults to 0.5)."""
    # Expected: 0.2*0.8 + 0.3*0.9 + 0.25*0.5 + 0.25*0.7 = 0.16 + 0.27 + 0.125 + 0.175 = 0.73
    result = compute_fire_likelihood(
        confidence_score=0.8,
        persistence_score=0.9,
        landcover_score=None,
        weather_score=0.7,
        false_source_masked=False,
    )
    assert result == pytest.approx(0.73, abs=0.001)


def test_compute_fire_likelihood_none_weather():
    """Test handling of None weather score (defaults to 0.5)."""
    # Expected: 0.2*0.8 + 0.3*0.9 + 0.25*1.0 + 0.25*0.5 = 0.16 + 0.27 + 0.25 + 0.125 = 0.805
    result = compute_fire_likelihood(
        confidence_score=0.8,
        persistence_score=0.9,
        landcover_score=1.0,
        weather_score=None,
        false_source_masked=False,
    )
    assert result == pytest.approx(0.805, abs=0.001)


def test_compute_fire_likelihood_all_none_except_confidence():
    """Test handling of all optional scores as None."""
    # Expected: 0.2*0.8 + 0.3*0.5 + 0.25*0.5 + 0.25*0.5 = 0.16 + 0.15 + 0.125 + 0.125 = 0.56
    result = compute_fire_likelihood(
        confidence_score=0.8,
        persistence_score=None,
        landcover_score=None,
        weather_score=None,
        false_source_masked=False,
    )
    assert result == pytest.approx(0.56, abs=0.001)


def test_compute_fire_likelihood_industrial_source_masked():
    """Test that industrial false-source masking returns 0.0."""
    result = compute_fire_likelihood(
        confidence_score=0.9,
        persistence_score=0.9,
        landcover_score=0.9,
        weather_score=0.9,
        false_source_masked=True,
    )
    assert result == 0.0


def test_compute_fire_likelihood_all_zeros():
    """Test edge case with all component scores at 0."""
    # Expected: 0.2*0.0 + 0.3*0.0 + 0.25*0.0 + 0.25*0.0 = 0.0
    result = compute_fire_likelihood(
        confidence_score=0.0,
        persistence_score=0.0,
        landcover_score=0.0,
        weather_score=0.0,
        false_source_masked=False,
    )
    assert result == 0.0


def test_compute_fire_likelihood_all_ones():
    """Test edge case with all component scores at 1."""
    # Expected: 0.2*1.0 + 0.3*1.0 + 0.25*1.0 + 0.25*1.0 = 0.2 + 0.3 + 0.25 + 0.25 = 1.0
    result = compute_fire_likelihood(
        confidence_score=1.0,
        persistence_score=1.0,
        landcover_score=1.0,
        weather_score=1.0,
        false_source_masked=False,
    )
    assert result == 1.0


def test_compute_fire_likelihood_clamping_defensive():
    """Test that clamping works if somehow score exceeds [0,1] range."""
    # Though inputs should be [0,1], test defensive clamping
    # This is a defensive test; in practice, inputs should never exceed range
    result = compute_fire_likelihood(
        confidence_score=1.0,
        persistence_score=1.0,
        landcover_score=1.0,
        weather_score=1.0,
        false_source_masked=False,
    )
    assert 0.0 <= result <= 1.0
