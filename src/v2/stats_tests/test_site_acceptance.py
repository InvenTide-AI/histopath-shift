"""Tests of the site acceptance test (src/v2/site_acceptance.py)."""
import os
import sys

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import site_acceptance as SA  # noqa: E402


def test_local_auroc_matches_mann_whitney():
    from scipy import stats
    rng = np.random.default_rng(0)
    y = rng.random((20, 150)) < 0.4
    p = rng.random((20, 150)) + 0.6 * y
    z, v, n1, n0 = SA.local_estimate(y, p)
    for i in range(20):
        u = stats.mannwhitneyu(p[i][y[i]], p[i][~y[i]]).statistic
        a = (u + 0.5) / (n1[i] * n0[i] + 1.0)
        assert abs(expit(z[i]) - a) < 1e-12
        assert v[i] > 0


def test_single_class_sample_is_nan():
    z, v, _, _ = SA.local_estimate(np.ones((1, 10), bool), np.random.default_rng(1).random((1, 10)))
    assert np.isnan(z[0]) and np.isnan(v[0])


def test_prior_is_normalized_and_centered():
    rng = np.random.default_rng(2)
    Z = 2 + rng.normal(0, 0.3, (4, 1)) + rng.normal(0, 0.1, (4, 3))
    pr = SA.fit_prior(Z)
    assert abs(pr["w"].sum() - 1) < 1e-12 and np.all(pr["V"] > 0)
    assert abs(pr["center"] - Z.mean()) < 1e-12
    med = SA.quantile(SA.posterior(pr), 0.5)[0]
    assert abs(med - Z.mean()) < 1e-6            # symmetric scale mixture: median = centre


def test_conjugate_update_single_component():
    pr = dict(center=1.0, centers=np.array([1.0]), V=np.array([0.5]), w=np.array([1.0]))
    post = SA.posterior(pr, np.array([2.0]), np.array([0.25]))
    m = (1.0 * 0.25 + 2.0 * 0.5) / 0.75
    sd = np.sqrt(0.5 * 0.25 / 0.75)
    assert abs(post["mean"][0, 0] - m) < 1e-12 and abs(post["sd"][0, 0] - sd) < 1e-12


def test_predictive_coverage_under_the_model():
    """90% predictive interval for a new site's single run covers at least nominally (K' = 4, n = 3)."""
    rng = np.random.default_rng(3)
    cov = []
    for _ in range(300):
        b = rng.normal(0, 0.4, 5)
        Z = 2.0 + b[:, None] + rng.normal(0, 0.15, (5, 3))
        post = SA.posterior(SA.fit_prior(Z[:4]))
        lo, hi = SA.quantile(post, 0.05)[0], SA.quantile(post, 0.95)[0]
        cov.append(lo <= Z[4, 0] <= hi)
    assert 0.86 <= np.mean(cov) <= 0.98, np.mean(cov)


def test_robust_prior_is_less_confident():
    rng = np.random.default_rng(4)
    Z = 2.5 + rng.normal(0, 0.2, (4, 1)) + rng.normal(0, 0.1, (4, 3))
    pr = SA.fit_prior(Z)
    rp = SA.robustify(pr, eps=0.1)
    assert abs(rp["w"].sum() - 1) < 1e-12
    t = logit(0.85)
    assert SA.prob_above(SA.posterior(rp), t)[0] < SA.prob_above(SA.posterior(pr), t)[0]
    # strong contrary local evidence pulls the robust posterior further down
    z, v = np.array([logit(0.70)]), np.array([0.05])
    assert SA.prob_above(SA.posterior(rp, z, v), t)[0] <= SA.prob_above(SA.posterior(pr, z, v), t)[0]


def test_planner_monotone():
    rng = np.random.default_rng(5)
    Z = 2.0 + rng.normal(0, 0.3, (4, 1)) + rng.normal(0, 0.1, (4, 3))
    m, curve = SA.labels_needed(SA.fit_prior(Z), 0.80, power=2.0, ms=(25, 100, 400), sims=1500)
    vals = [curve[k] for k in (25, 100, 400)]
    assert vals[0] <= vals[1] + 0.05 <= vals[2] + 0.1


if __name__ == "__main__":
    from _common import run_module
    sys.exit(run_module(globals()))
