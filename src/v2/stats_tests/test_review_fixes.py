"""Regression tests for the stats-track review findings (2026-10-05).

Each test reproduces a reviewer's failure scenario and checks the fixed behaviour.
"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

from _common import HERE, S, V1_RUNS, run_module

KS = dict(c17=5, canine=5, midog21=3, midogpp=7)


def _cohort_rows(rng, A, csd, sd, eff_sd=0.03):
    """Rows (one per (cohort, domain)) of gains: common arm effect + cohort x arm effect + domain noise."""
    rows, lab = [], []
    eff = rng.normal(0, eff_sd, A)
    for c, K in KS.items():
        ce = rng.normal(0, csd, A)
        for _ in range(K):
            rows.append(eff + ce + rng.normal(0, sd[c], A) + rng.normal())
            lab.append(c)
    return np.array(rows), lab


def test_task_shift_null_with_cohort_effects():
    """Cohort x arm effects without task / shift structure must not look like task or shift effects
    (reviewer: row-level shuffling rejected 0.90 / 0.83)."""
    rng = np.random.default_rng(10)
    homo = {c: 0.02 for c in KS}
    n, rt, rs = 300, 0, 0
    for _ in range(n):
        R, lab = _cohort_rows(rng, 8, 0.03, homo)
        r = S.interaction_2x2_cohort(R, lab)
        rt += r["task"]["p_F"] <= 0.05
        rs += r["shift"]["p_F"] <= 0.05
        # exact decomposition of the between-cohort interaction SS
        assert abs(r["task"]["ss"] + r["shift"]["ss"] + r["task_x_shift"]["ss"] - r["between_cohort_ss"]) < 1e-12
        assert r["task"]["p_split"] in (1 / 3, 2 / 3, 1.0)
    assert rt / n < 0.09 and rs / n < 0.09, (rt / n, rs / n)
    # power against a genuine task x arm effect
    hits = 0
    for _ in range(100):
        R, lab = _cohort_rows(rng, 8, 0.01, homo)
        te = rng.normal(0, 0.03, 8)
        R[np.array([S.COHORT_META[c]["task"] == "tumour" for c in lab])] += te
        hits += S.interaction_2x2_cohort(R, lab)["task"]["p_F"] <= 0.05
    assert hits / 100 > 0.5, hits


def test_cohort_interaction_heteroscedastic_null():
    """No interaction, MIDOG21 4x noisier: wild-bootstrap studentised test stays near alpha
    (reviewer: unstudentised permutation 0.517)."""
    rng = np.random.default_rng(11)
    sd = dict(c17=0.02, canine=0.02, midog21=0.08, midogpp=0.02)
    n, rej, rej_perm = 200, 0, 0
    for i in range(n):
        R, lab = _cohort_rows(rng, 8, 0.0, sd)
        rej += S.interaction_permutation_test(R, lab, B=199, seed=i)["p"] <= 0.05
        rej_perm += S.interaction_permutation_test(R, lab, B=199, seed=i, method="perm")["p"] <= 0.05
    assert rej / n < 0.09, rej / n
    assert rej_perm / n > 0.25  # documents why 'perm' is not the default


def test_mean_ci_t_coverage():
    rng = np.random.default_rng(12)
    for K in (3, 5):
        cov = 0
        for _ in range(1000):
            Y = S.simulate_oneway(K, 5, 0.5, 0.05, 0.8, rng)
            lo, hi = S.arm_summary(Y)["mean_ci_t"]
            cov += lo <= 0.8 <= hi
        assert 0.93 < cov / 1000 < 0.97, (K, cov / 1000)


def test_lower_better_worst():
    Y = np.array([[0.05, 0.06], [0.20, 0.22], [0.10, 0.11]])  # ECE: domain 1 is worst
    s = S.arm_summary(Y, higher_better=False)
    assert s["worst_index"] == 1 and abs(s["worst"] - 0.21) < 1e-12 and abs(s["best"] - 0.055) < 1e-12
    b = S.bootstrap_summary(Y, B=500, higher_better=False)
    assert b["seed"]["worst_ci"][0] >= 0.20 - 1e-12
    arms = {"a": Y, "b": Y + np.array([[-0.04], [0.05], [0.0]])}
    r = S.bootstrap_ranks(arms, "worst", "seed", B=300, higher_better=False)
    assert r["a"]["rank"] == 1 and r["b"]["rank"] == 2
    c = S.paired_contrast(arms["b"], arms["a"], higher_better=False)
    assert c["n_improved"] == 1 and c["n_worse"] == 1
    t = S.tau_between_cohorts(arms, arms, "worst", B=50, higher_better=False)
    assert abs(t["tau"] - 1.0) < 1e-12


def test_single_seed_not_identifiable():
    Y1 = np.array([[0.8], [0.7], [0.9]])
    a, c = S.oneway_anova(Y1), S.icc1_exact_ci(Y1)
    assert math.isnan(a["F"]) and math.isnan(a["p"]) and math.isnan(a["icc"])
    assert math.isnan(c["icc"]) and math.isnan(c["lo"]) and math.isnan(c["hi"])
    assert math.isnan(S.oneway_reml(Y1)["s2_domain"])
    assert all(math.isnan(v) for v in S.sigma_domain_mls_ci(Y1))
    # MSW = 0 with replication: genuinely infinite F
    Y0 = np.array([[0.8, 0.8], [0.7, 0.7]])
    assert math.isinf(S.oneway_anova(Y0)["F"]) and S.icc1_exact_ci(Y0)["lo"] == 1.0


def test_reml_large_variance_ratio():
    g = [np.array([m - 0.01, m, m + 0.01]) for m in (10.0, 40.0, 70.0, 90.0)]
    a, r = S.oneway_anova(g), S.oneway_reml(g)
    assert abs(r["s2_domain"] / a["s2_domain"] - 1) < 1e-6 and abs(r["s2_run"] / a["s2_run"] - 1) < 1e-4


def test_two_stage_independent_within_slot():
    g = [np.array([0.0, 1.0]), np.array([10.0, 11.0])]
    D = S._resampled_domain_means(g, 40000, np.random.default_rng(0), "two_stage")
    # P(equal) = P(same domain) * P(two independent seed-mean draws equal) = 1/2 * 3/8
    assert abs((D[:, 0] == D[:, 1]).mean() - 0.1875) < 0.01


def test_power_table_matches_dependence_band():
    """At the observed K and seeds, the SE implied by the rho>0 power rows equals the
    dependence_sensitivity SE (reviewer: 0.0468 vs 0.0541 for v1 SSL-ERM at rho=.5)."""
    rng = np.random.default_rng(13)
    ref = 0.8 + rng.normal(0, 0.05, (5, 1)) + rng.normal(0, 0.03, (5, 3))
    arm = ref + 0.05 + rng.normal(0, 0.04, (5, 1)) + rng.normal(0, 0.03, (5, 3))
    design = S.lodo_design(5)
    comp = S.difference_variance(arm, ref)
    d = S.paired_contrast(arm, ref)["d"]
    band = {r["rho"]: r for r in S.dependence_sensitivity(d, design, (0.25, 0.5))}
    rows = S.power_table(comp, deltas=(0.05,), seeds=(3,), rhos=(0.25, 0.5), train_sets=design, keff_modes=("ratio",))
    for r in rows:
        if comp["s2_dxa_raw"] > 0:
            se_pt = r["sd_delta"] / math.sqrt(5 / (len(design) / S.k_eff(design, r["rho"])))
            assert abs(se_pt - band[r["rho"]]["se"]) < 1e-12, (se_pt, band[r["rho"]]["se"])
        assert r["sd_delta"] > r["sd_delta_uncorrected"]


def test_paired_difference_variance_tta():
    """TTA pseudo-arm = same models as the reference + shift + tiny noise (reviewer scenario)."""
    rng = np.random.default_rng(14)
    K, s = 5, 5
    erm = 0.8 + rng.normal(0, 0.05, (K, 1)) + rng.normal(0, 0.04, (K, s))
    bn = erm + 0.02 + rng.normal(0, 0.003, (K, s))
    un = S.difference_variance(bn, erm)
    pa = S.difference_variance(bn, erm, paired=True)
    sd_obs = pa["sd_d_observed"]
    assert S.sd_delta(un, s) > 3 * sd_obs  # independent-noise formula is badly inflated
    assert abs(S.sd_delta(pa, s) - sd_obs) < 1e-12 or pa["s2_dxa_raw"] < 0
    assert pa["s2_run_corr"] > 0.9
    assert S.min_domains(0.02, S.sd_delta(pa, s)) <= 4
    # paired estimator stays unbiased when noises are independent
    big = 400
    r_ = rng.normal(0, 0.05, (big, 3))
    a_ = 0.03 * rng.normal(size=(big, 1)) + rng.normal(0, 0.02, (big, 3))
    c = S.difference_variance(a_, r_, paired=True)
    assert abs(math.sqrt(c["s2_run_diff"]) - math.sqrt(0.05 ** 2 + 0.02 ** 2)) < 0.004


def test_v1_power_rho_correction():
    """v1 SSL vs ERM, rho=.5 'ratio': K_min 48/14/6 after the (1 - rbar) correction (was 37/11/5)."""
    import test_v1_regression as V
    comp = S.difference_variance(V.v1_matrix("ssl"), V.v1_matrix("baseline"))
    rows = S.power_table(comp, deltas=(0.05, 0.10, 0.20), seeds=(3,), rhos=(0.5,), train_sets=S.lodo_design(5),
                         keff_modes=("ratio",))
    assert [r["K_min"] for r in rows] == [48, 14, 6], [r["K_min"] for r in rows]
    assert abs(rows[0]["sd_delta"] - 0.0855) < 5e-5 and abs(rows[0]["rbar_obs"] - 0.25) < 1e-12


def test_cli_ladder_and_v1_label():
    sys.path.insert(0, os.path.dirname(HERE))
    import analyze_v2 as A
    tmp = tempfile.mkdtemp(prefix="v2stats_ladder_")
    try:
        rng = np.random.default_rng(15)
        for f in range(5):
            d = os.path.join(tmp, "runs", "S", "c17", f"fold{f}")
            os.makedirs(d)
            for arm in ("erm", "sam", "probe_resnet50_imagenet", "probe_convnext_tiny_imagenet"):
                for s in range(3):
                    y = float(0.8 + 0.02 * f + rng.normal(0, 0.02))
                    rec = dict(schema="v2", arm=arm, tier="S", cohort="c17", fold=f, test_domain=f"D{f}",
                               val_domain=f"D{(f - 1) % 5}", seed=s, phase="final", input_res=96,
                               test_at={"id": {"auroc": y, "ece": 0.05}})
                    if arm == "erm":
                        rec["tta"] = {"bnadapt": {"auroc": y + 0.01 + rng.normal(0, 1e-3), "ece": 0.05}}
                    json.dump(rec, open(os.path.join(d, f"{arm}__default__s{s}.json"), "w"))
        out = os.path.join(tmp, "an")
        A.main(["runs", "--v2-root", tmp, "--tier", "S", "--cohorts", "c17", "--out", out, "--B", "200"])
        meta = json.load(open(os.path.join(out, "S", "c17", "meta_id.json")))
        assert "probe_convnext_tiny_imagenet" not in meta["arms"] and "probe_resnet50_imagenet" in meta["arms"]
        import csv
        pw = [r for r in csv.DictReader(open(os.path.join(out, "S", "c17", "power_id.csv"))) if r["arm"] == "bnadapt@erm"]
        assert pw and all(r["paired"] == "True" for r in pw)
        A.main(["runs", "--v2-root", tmp, "--tier", "S", "--cohorts", "c17", "--out", out, "--B", "200",
                "--ladder", "only"])
        meta = json.load(open(os.path.join(out, "S_ladder_only", "c17", "meta_id.json")))
        assert set(meta["arms"]) == {"probe_resnet50_imagenet", "probe_convnext_tiny_imagenet"}
        # variance table marks boundary rows with the one-sided bound
        tex = open(os.path.join(out, "S", "c17", "tables", "variance_id.tex")).read()
        assert "one-sided" in tex
        # v1 sub-command never labels its output 'id'
        if os.path.isdir(V1_RUNS):
            A.main(["v1", "--runs", V1_RUNS, "--cohort", "c17", "--out", os.path.join(tmp, "v1"), "--B", "200",
                    "--arms", "erm", "sam"])
            od = os.path.join(tmp, "v1", "v1", "c17")
            assert os.path.exists(os.path.join(od, "summary_v1rule.csv"))
            assert not os.path.exists(os.path.join(od, "summary_id.csv"))
            assert "selection=v1rule" in open(os.path.join(od, "tables", "variance_v1rule.tex")).read()
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
