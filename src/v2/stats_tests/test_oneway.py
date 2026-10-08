"""One-way random-effects estimators on simulated data with known parameters."""
import math

import numpy as np
from scipy import stats

from _common import S, run_module


def test_reml_equals_anova_when_positive():
    rng = np.random.default_rng(1)
    n_checked = 0
    for _ in range(300):
        y = S.simulate_oneway(6, 4, 0.5, rng=rng)
        a = S.oneway_anova(y)
        if a["s2_domain_raw"] <= 0:
            continue
        r = S.oneway_reml(y)
        assert abs(r["s2_domain"] - a["s2_domain"]) < 1e-6 * max(1, a["s2_domain"]), (r, a)
        assert abs(r["s2_run"] - a["s2_run"]) < 1e-6 * max(1, a["s2_run"])
        n_checked += 1
    assert n_checked > 200


def test_reml_boundary():
    rng = np.random.default_rng(2)
    hit = 0
    for _ in range(300):
        y = S.simulate_oneway(5, 3, 0.0, rng=rng)
        a = S.oneway_anova(y)
        if a["s2_domain_raw"] >= 0:
            continue
        r = S.oneway_reml(y)
        assert r["boundary"] and r["s2_domain"] == 0.0
        sst = ((y - y.mean()) ** 2).sum()
        assert abs(r["s2_run"] - sst / (y.size - 1)) < 1e-9  # REML at boundary = pooled variance
        assert a["truncated"] and a["icc"] == 0.0
        ci = S.icc1_exact_ci(y)
        assert ci["lo"] == 0.0 and ci["hi"] >= 0 and ci["hi"] >= ci["lo"]
        hit += 1
    assert hit > 50


def test_unbalanced_reml_against_bruteforce():
    """Unbalanced REML: compare with a direct dense-matrix REML minimised by Nelder-Mead."""
    from scipy import optimize
    rng = np.random.default_rng(3)
    for rep in range(5):
        sizes = rng.integers(1, 6, size=6)
        sizes[0] = max(sizes[0], 2)
        g = [rng.normal(rng.normal(0, 1), 0.7, n) for n in sizes]
        y = np.concatenate(g)
        Z = np.zeros((y.size, len(g)))
        o = 0
        for i, n in enumerate(sizes):
            Z[o:o + n, i] = 1
            o += n
        X = np.ones((y.size, 1))

        def nll(p):
            sb2, sw2 = np.exp(p)
            V = sb2 * Z @ Z.T + sw2 * np.eye(y.size)
            Vi = np.linalg.inv(V)
            XtViX = X.T @ Vi @ X
            b = np.linalg.solve(XtViX, X.T @ Vi @ y)
            r = y - X @ b
            return 0.5 * (np.linalg.slogdet(V)[1] + np.linalg.slogdet(XtViX)[1] + r @ Vi @ r)

        best = min((optimize.minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
                    for x0 in ([0, 0], [-3, -1], [1, -2])), key=lambda r: r.fun)
        r = S.oneway_reml(g)
        sb2, sw2 = np.exp(best.x)
        if sb2 > 1e-4:  # interior optimum
            assert abs(r["s2_domain"] - sb2) / sb2 < 1e-3, (r, sb2)
            assert abs(r["s2_run"] - sw2) / sw2 < 1e-3


def test_unbalanced_anova_n0():
    g = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]), np.array([0.0])]
    a = S.oneway_anova(g)
    N = 6
    assert abs(a["n0"] - (N - (4 + 9 + 1) / N) / 2) < 1e-12
    M = S.as_matrix(g)
    assert np.isnan(M[2, 1]) and S.oneway_anova(M)["ssb"] == a["ssb"]


def test_icc_ci_matches_closed_form():
    y = S.simulate_oneway(5, 3, 0.5, rng=np.random.default_rng(4))
    a = S.oneway_anova(y)
    c = S.icc1_exact_ci(y)
    FL = a["F"] / stats.f.ppf(0.975, 4, 10)
    assert abs(c["lo_raw"] - (FL - 1) / (FL + 2)) < 1e-12


def test_bayes_boundary_and_shrinkage():
    """At a zero-boundary ANOVA estimate the posterior is proper, its median is > 0 but its
    one-sided 95% bound is finite, and a half-normal with a small scale shrinks harder."""
    rng = np.random.default_rng(5)
    while True:
        y = S.simulate_oneway(5, 3, 0.0, rng=rng)
        if S.oneway_anova(y)["s2_domain_raw"] < 0:
            break
    b1 = S.bayes_oneway(y, "halfnormal", 1.0)
    b2 = S.bayes_oneway(y, "halfnormal", 0.1)
    bc = S.bayes_oneway(y, "halfcauchy", 1.0)
    for b in (b1, b2, bc):
        assert 0 < b["sigma_domain"]["median"] < b["sigma_domain"]["upper_one_sided"] < 10
        assert b["edge_mass"] < 1e-3
    assert b2["sigma_domain"]["upper_one_sided"] < b1["sigma_domain"]["upper_one_sided"]


def test_bayes_grid_convergence():
    y = S.simulate_oneway(5, 3, 0.5, rng=np.random.default_rng(6))
    lo = S.bayes_oneway(y, "halfcauchy", 1.0, n_domain=150, n_run=100)
    hi = S.bayes_oneway(y, "halfcauchy", 1.0, n_domain=1500, n_run=1000)
    df = S.bayes_oneway(y, "halfcauchy", 1.0)  # default 400 x 300
    for k in ("sigma_domain", "sigma_run", "icc"):
        for q in ("median", "lo", "hi"):
            assert abs(lo[k][q] - hi[k][q]) <= 0.015 * abs(hi[k][q]) + 1e-3, (k, q, lo[k][q], hi[k][q])
            assert abs(df[k][q] - hi[k][q]) <= 0.005 * abs(hi[k][q]) + 1e-3, (k, q, df[k][q], hi[k][q])


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
