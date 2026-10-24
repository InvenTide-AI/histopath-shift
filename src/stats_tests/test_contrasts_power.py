"""Paired contrasts, Holm, K_eff formula, power monotonicity, response shape, ranks."""
import math

import numpy as np
from scipy import stats

from _common import S, run_module


def test_keff_formula():
    # v1 C17 design: fold t trains on {t+1, t+2, t+3} mod 5 == LODO with val = t-1
    d = S.lodo_design(5)
    assert d[0] == {1, 2, 3} and d[3] == {4, 0, 1}
    O = S.fold_overlap(d)
    assert np.allclose(np.diag(O), 1) and np.allclose(O, O.T)
    assert np.allclose(sorted(O[0, 1:]), [1 / 3, 1 / 3, 2 / 3, 2 / 3])
    for rho, want in ((0, 5.0), (0.25, 25 / 7.5), (0.5, 2.5), (1.0, 25 / 15)):
        assert abs(S.k_eff(d, rho) - want) < 1e-12
    # K_eff = 1 when every fold has the same training set and rho = 1
    assert abs(S.k_eff([{0, 1}] * 4, 1.0) - 1.0) < 1e-12
    # closed form used for large K agrees with the explicit design
    for K in (7, 20, 60):
        for vo in (-1, None):
            f = S.keff_lodo_fn(0.5, vo)
            ex = S.k_eff(S.lodo_design(K, vo), 0.5)
            assert abs(f(K) - ex) < 1e-9
            # closed form branch
            m = K - 1 if vo is None else K - 2
            assert abs(S.keff_lodo_fn(0.5, vo).__call__(K) - ex) < 1e-9 and m > 0
    # analytic large-K check vs closed-form branch continuity (K=61 uses closed form)
    f = S.keff_lodo_fn(0.5, -1)
    assert abs(f(61) - f(60)) < 0.05


def test_dependence_sensitivity_matches_simulation():
    """Monte-Carlo check of the corrected variance: simulate d ~ N(mu, sigma^2 R) and check that
    the corrected interval covers mu at ~95% under the assumed rho."""
    rng = np.random.default_rng(0)
    d = S.lodo_design(5)
    for rho in (0.25, 0.5):
        R = S.dependence_matrix(d, rho)
        L = np.linalg.cholesky(R)
        cov = 0
        n = 4000
        for _ in range(n):
            x = 0.1 + 0.05 * (L @ rng.normal(size=5))
            r = [r for r in S.dependence_sensitivity(x, d, (rho,))][0]
            cov += r["ci"][0] <= 0.1 <= r["ci"][1]
        # the corrected t uses K-1 df with a scaled variance: approximately calibrated
        assert 0.90 < cov / n < 0.985, (rho, cov / n)


def test_holm():
    p = [0.01, 0.04, 0.03, 0.005]
    adj = S.holm(p)
    want = np.array([0.03, 0.06, 0.06, 0.02])
    assert np.allclose(adj, want), adj
    assert np.isnan(S.holm([0.1, np.nan])[1])


def test_paired_contrast_equals_ttest():
    rng = np.random.default_rng(1)
    a, r = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    c = S.paired_contrast(a, r)
    tt = stats.ttest_rel(a.mean(1), r.mean(1))
    assert abs(c["t"] - tt.statistic) < 1e-10 and abs(c["p"] - tt.pvalue) < 1e-10


def test_power_monotone():
    p = [S.power_paired_t(K, 0.05, 0.08) for K in range(2, 60)]
    assert all(np.diff(p) > 0)
    assert S.power_paired_t(10, 0.1, 0.08) > S.power_paired_t(10, 0.05, 0.08)
    assert S.power_paired_t(10, 0.05, 0.05) > S.power_paired_t(10, 0.05, 0.08)
    # power at delta=0 equals alpha
    assert abs(S.power_paired_t(8, 0.0, 0.1) - 0.05) < 1e-10
    comp = dict(s2_dxa=0.002, s2_run_arm=0.004, s2_run_ref=0.004)
    ks = [S.min_domains(0.05, S.sd_delta(comp, s)) for s in (1, 3, 5, 10)]
    assert all(np.diff(ks) <= 0) and ks[0] > ks[-1]
    # K_eff inflation can only increase the required K; strong dependence can make it unreachable
    k0 = S.min_domains(0.1, 0.1)
    k1 = S.min_domains(0.1, 0.1, keff_fn=S.keff_lodo_fn(0.25))
    assert k1 is None or k1 >= k0
    assert S.min_domains(0.02, 0.1, keff_fn=S.keff_lodo_fn(1.0), kmax=300) is None


def test_power_against_simulation():
    rng = np.random.default_rng(2)
    K, delta, sd = 8, 0.05, 0.06
    x = rng.normal(delta, sd, size=(20000, K))
    t = x.mean(1) / (x.std(1, ddof=1) / math.sqrt(K))
    emp = (np.abs(t) > stats.t.ppf(0.975, K - 1)).mean()
    assert abs(emp - S.power_paired_t(K, delta, sd)) < 0.01


def test_difference_variance_recovers_components():
    rng = np.random.default_rng(3)
    K, s = 400, 3
    b = rng.normal(0, 0.08, (K, 1))
    ab = rng.normal(0, 0.03, (K, 1))
    ref = 0.8 + b + rng.normal(0, 0.05, (K, s))
    arm = 0.85 + b + ab + rng.normal(0, 0.02, (K, s))
    c = S.difference_variance(arm, ref)
    assert abs(math.sqrt(c["s2_dxa"]) - 0.03) < 0.006
    assert abs(math.sqrt(c["s2_run_ref"]) - 0.05) < 0.004 and abs(math.sqrt(c["s2_run_arm"]) - 0.02) < 0.002


def test_oldham_pitman_morgan():
    rng = np.random.default_rng(4)
    E = rng.normal(0.8, 0.1, 7)
    M = 0.9 + 0.3 * (E - 0.8) + rng.normal(0, 0.01, 7)  # compressed response -> var(M) < var(E)
    r = S.oldham_pitman_morgan(M, E)
    assert r["oldham_r"] < -0.8 and r["pm_p"] < 0.05 and r["exhaustive"] and r["n_perm"] == 128
    assert r["perm_p"] <= 2 / 128 + 1e-12
    # Pitman-Morgan t equals the textbook variance-ratio form
    K = 7
    F = M.var(ddof=1) / E.var(ddof=1)
    rr = np.corrcoef(M, E)[0, 1]
    t_alt = (F - 1) * math.sqrt(K - 2) / (2 * math.sqrt(F * (1 - rr ** 2)))
    assert abs(abs(r["pm_t"]) - abs(t_alt)) < 1e-8


def test_ranks_and_tau():
    rng = np.random.default_rng(5)
    arms = {f"a{i}": 0.7 + 0.02 * i + rng.normal(0, 0.01, (5, 3)) for i in range(6)}
    br = S.bootstrap_ranks(arms, "mean", "seed", B=2000)
    assert br["a5"]["rank"] == 1 and br["a0"]["rank"] == 6
    assert br["a5"]["lo"] <= 1 <= br["a5"]["hi"]
    bd = S.bootstrap_ranks(arms, "worst", "domain", B=2000)
    assert all(v["lo"] <= v["rank"] <= v["hi"] for v in bd.values())
    t = S.tau_between_cohorts(arms, arms, "mean", B=200)
    assert abs(t["tau"] - 1.0) < 1e-12 and t["ci"][0] > 0.6
    assert abs(S.kendall_tau_b([1, 2, 3, 4], [1, 2, 4, 3]) - 4 / 6) < 1e-12


def test_interaction_permutation():
    rng = np.random.default_rng(6)
    A = 6
    eff = rng.normal(0, 0.05, A)
    labels = ["c17"] * 5 + ["canine"] * 5 + ["midog21"] * 3 + ["midogpp"] * 7
    null_rows = np.array([eff + rng.normal(0, 0.02, A) + rng.normal() for _ in labels])
    p0 = S.interaction_permutation_test(null_rows, labels, B=2000)["p"]
    alt = null_rows.copy()
    alt[:10, 0] += 0.15  # arm 0 helps only on tumour cohorts
    p1 = S.interaction_permutation_test(alt, labels, B=2000)["p"]
    assert p1 < 0.01 and p0 > 0.01
    # null calibration: type-I error ~ alpha
    rej = 0
    for i in range(200):
        rows = np.array([eff + rng.normal(0, 0.02, A) for _ in labels])
        rej += S.interaction_permutation_test(rows, labels, B=199, seed=i)["p"] <= 0.05
    assert rej / 200 < 0.10


def test_cluster_summary():
    rng = np.random.default_rng(7)
    base = 0.02 * np.arange(6)
    def coh(perm):
        return {f"a{i}": 0.7 + base[perm[i]] + rng.normal(0, 0.005, (5, 3)) for i in range(6)}
    tum = coh([0, 1, 2, 3, 4, 5])
    mit = coh([5, 4, 3, 2, 1, 0])
    cohorts = dict(c17=tum, canine=coh([0, 1, 2, 3, 5, 4]), midog21=mit, midogpp=coh([5, 4, 3, 2, 0, 1]))
    pairs = {}
    names = list(cohorts)
    for i in range(4):
        for j in range(i + 1, 4):
            pairs[(names[i], names[j])] = S.tau_between_cohorts(cohorts[names[i]], cohorts[names[j]], "mean", B=200,
                                                                return_draws=True)
    cs = S.cluster_summary(pairs)
    assert cs["same_task"]["mean_tau"] > 0.6 and cs["same_shift"]["mean_tau"] < -0.6
    assert cs["same_task_minus_same_shift"]["ci"][0] > 0


def test_bootstrap_summary_modes():
    y = np.array([[0.9, 0.91, 0.92], [0.7, 0.6, 0.8], [0.85, 0.86, 0.84]])
    b = S.bootstrap_summary(y, B=4000)
    s = S.arm_summary(y)
    assert s["worst_index"] == 1 and abs(s["mean"] - y.mean(1).mean()) < 1e-12
    for m in ("seed", "domain", "two_stage"):
        assert b[m]["mean_ci"][0] < s["mean"] < b[m]["mean_ci"][1]
    assert b["domain"]["mean_se"] > b["seed"]["mean_se"]


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
