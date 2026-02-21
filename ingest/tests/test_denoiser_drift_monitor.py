import numpy as np

from ingest.denoiser_drift_monitor import compute_population_stability_index


def test_population_stability_index_near_zero_for_matching_distributions() -> None:
    baseline = np.linspace(0.0, 1.0, num=1000, dtype=float)
    current = baseline.copy()
    psi = compute_population_stability_index(baseline, current, bins=10)
    assert psi < 1e-6


def test_population_stability_index_increases_for_shifted_distribution() -> None:
    baseline = np.random.RandomState(42).beta(a=2, b=5, size=2000)
    current = np.random.RandomState(7).beta(a=5, b=2, size=2000)
    psi = compute_population_stability_index(baseline, current, bins=10)
    assert psi > 0.2
