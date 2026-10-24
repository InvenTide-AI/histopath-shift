"""Regression check against the v1 Camelyon17 records (audit 2026-10-05, /tmp/maps/analysis-code.md).

Builds the v1 ERM/SAM/SSL/SSL+SAM per-domain x seed AUROC matrices from
runs/fold*/{baseline,sam,ssl,ssl_sam}_seed{0,1,2}.json (key ood_test.auroc) and checks the nested
one-way numbers (ERM sigma_site 0.0736, sigma_within 0.0676, ICC(1) 0.543, exact-F CI ~[0.007, 0.929])
and the paired SSL-ERM contrast (+0.0957, CI [+0.004, +0.188], p 0.045), plus the K_eff band.
"""
import json
import os

import numpy as np

from _common import S, V1_RUNS, run_module

ARMS = ["baseline", "sam", "ssl", "ssl_sam"]


def v1_matrix(arm):
    return np.array([[json.load(open(os.path.join(V1_RUNS, f"fold{f}", f"{arm}_seed{s}.json")))["ood_test"]["auroc"]
                      for s in range(3)] for f in range(5)])


def test_v1_erm_oneway():
    y = v1_matrix("baseline")
    a, r, c = S.oneway_anova(y), S.oneway_reml(y), S.icc1_exact_ci(y)
    assert round(a["sigma_domain"], 4) == 0.0736 and round(a["sigma_run"], 4) == 0.0676
    assert round(a["icc"], 3) == 0.543
    assert abs(r["sigma_domain"] - a["sigma_domain"]) < 1e-6 and abs(r["sigma_run"] - a["sigma_run"]) < 1e-6
    assert round(c["lo"], 3) == 0.007 and round(c["hi"], 3) == 0.929
    assert round(a["F"], 2) == 4.56 and round(a["p"], 3) == 0.024


def test_v1_other_arms_oneway():
    want = dict(sam=(0.0, 0.1310, 0.0), ssl=(0.0210, 0.0254, 0.406), ssl_sam=(0.0276, 0.0107, 0.869))
    for arm, (sb, sw, icc) in want.items():
        a = S.oneway_anova(v1_matrix(arm))
        assert round(a["sigma_domain"], 4) == sb and round(a["sigma_run"], 4) == sw and round(a["icc"], 3) == icc, (arm, a)
    assert S.oneway_reml(v1_matrix("sam"))["boundary"]


def test_v1_ssl_vs_erm_contrast():
    c = S.paired_contrast(v1_matrix("ssl"), v1_matrix("baseline"))
    assert round(c["mean"], 4) == 0.0957
    assert round(c["ci"][0], 3) == 0.004 and round(c["ci"][1], 3) == 0.188
    assert round(c["p"], 3) == 0.045 and c["n_improved"] == 5
    band = S.dependence_sensitivity(c["d"], S.lodo_design(5))
    want = {0.0: (5.0, 0.004, 0.188), 0.25: (3.33, -0.025, 0.216), 0.5: (2.5, -0.054, 0.246), 1.0: (1.67, -0.130, 0.321)}
    for r in band:
        ke, lo, hi = want[r["rho"]]
        assert round(r["K_eff"], 2) == ke and round(r["ci"][0], 3) == lo and round(r["ci"][1], 3) == hi, r


def test_v1_power_from_difference_sd():
    # audit: SD of seed-averaged per-site differences (s=3): SSL-ERM 0.0741, SAM-ERM 0.1007, SSL+SAM-ERM 0.0950;
    # K at delta = .05/.10/.20: SSL 20/7/4, SAM 34/11/5, SSL+SAM 31/10/5
    want = dict(ssl=(0.0741, [20, 7, 4]), sam=(0.1007, [34, 11, 5]), ssl_sam=(0.0950, [31, 10, 5]))
    erm = v1_matrix("baseline")
    for arm, (sd, ks) in want.items():
        comp = S.difference_variance(v1_matrix(arm), erm)
        assert round(comp["sd_d_observed"], 4) == sd
        assert abs(S.sd_delta(comp, 3) - comp["sd_d_observed"]) < 1e-12  # decomposition is exact at observed s
        assert [S.min_domains(d, comp["sd_d_observed"]) for d in (0.05, 0.10, 0.20)] == ks


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
