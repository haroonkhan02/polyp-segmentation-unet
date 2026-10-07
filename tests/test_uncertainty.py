import numpy as np

from polypseg.uncertainty import (
    auroc,
    binary_entropy,
    error_uncertainty_stats,
    mean_std,
    subsample_for_auroc,
    summarize_uncertainty,
)


def test_entropy_peaks_at_half_and_mean_std():
    p = np.array([0.0, 0.5, 1.0])
    h = binary_entropy(p)
    assert h[1] > h[0] and h[1] > h[2] and abs(h[1] - np.log(2)) < 1e-6
    samples = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
    mean, std = mean_std(samples)
    assert np.allclose(mean, 0.5) and np.allclose(std, 0.5)


def test_auroc_known_values():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert abs(auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) - 0.5) < 1e-9
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_error_uncertainty_stats_rewards_uncertainty_on_errors():
    target = np.zeros((10, 10), np.uint8)
    target[:, :5] = 1
    pred = target.copy()
    pred[:, 4:6] = 1 - pred[:, 4:6]  # two wrong columns
    unc = np.full((10, 10), 0.05)
    unc[:, 4:6] = 0.4
    stats = error_uncertainty_stats(unc, pred, target)
    assert stats["mean_unc_on_errors"] > stats["mean_unc_on_correct"]
    assert stats["auroc"] == 1.0 and abs(stats["error_fraction"] - 0.2) < 1e-9
    u, e = subsample_for_auroc(unc, pred, target, max_pixels=30)
    assert len(u) == 30 and len(e) == 30


def test_summarize_uncertainty_keys():
    pred = np.zeros((20, 20), np.uint8)
    pred[5:15, 5:15] = 1
    std = np.random.default_rng(0).uniform(0, 0.3, (20, 20))
    s = summarize_uncertainty(std, pred)
    assert set(s) == {"mean_std", "mean_std_in_polyp", "mean_std_on_boundary",
                      "uncertain_fraction"}
    assert 0 <= s["uncertain_fraction"] <= 1
