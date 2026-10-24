"""Frequentist coverage of the §4(b) intervals by simulation.

2000 simulated K x S datasets (5 x 3 and 7 x 5) at true ICC in {0, 0.1, 0.3, 0.5, 0.8}
(sigma_total = 1).  Intervals checked:
  * exact-F 95% CI for ICC(1) (exact for balanced data -> expect 0.95 within MC error),
  * chi-square 95% CI for sigma_run (exact),
  * MLS 95% CI for sigma_domain (approximate, conservative-ish),
  * Bayesian 95% equal-tailed credible intervals (half-normal / half-Cauchy, scale 1) for
    sigma_domain, sigma_run and ICC (no frequentist guarantee; reported, loosely checked),
    and the one-sided [0, q95] credible bound for sigma_domain / ICC (used at the boundary).
Results are written to stats_tests/coverage_results.{json,csv}.
Run: python test_coverage.py [n_sim]
"""
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

from _common import HERE, S, run_module

N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DESIGNS = [(5, 3), (7, 5)]
ICCS = [0.0, 0.1, 0.3, 0.5, 0.8]
PRIORS = [("halfnormal", 1.0), ("halfcauchy", 1.0)]
MC_SE = math.sqrt(0.95 * 0.05 / N_SIM)


def _one(args):
    K, Sd, icc, seed = args
    rng = np.random.default_rng(seed)
    y = S.simulate_oneway(K, Sd, icc, rng=rng)
    sb, sw = math.sqrt(icc), math.sqrt(1 - icc)
    c = S.icc1_exact_ci(y)
    rw = S.sigma_run_ci(y)
    rb = S.sigma_domain_mls_ci(y)
    out = dict(icc_F=c["lo"] <= icc <= c["hi"], sw_chi2=rw[0] <= sw <= rw[1], sb_mls=rb[0] <= sb <= rb[1])
    for kind, sc in PRIORS:
        b = S.bayes_oneway(y, kind, sc, n_domain=150, n_run=100)
        out[f"{kind}_sb"] = b["sigma_domain"]["lo"] <= sb <= b["sigma_domain"]["hi"]
        out[f"{kind}_sw"] = b["sigma_run"]["lo"] <= sw <= b["sigma_run"]["hi"]
        out[f"{kind}_icc"] = b["icc"]["lo"] <= icc <= b["icc"]["hi"]
        out[f"{kind}_sb_1s"] = sb <= b["sigma_domain"]["upper_one_sided"]
        out[f"{kind}_icc_1s"] = icc <= b["icc"]["upper_one_sided"]
    return out


def compute():
    rows = []
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        for K, Sd in DESIGNS:
            for icc in ICCS:
                jobs = [(K, Sd, icc, 10_000_000 * K + 100_000 * Sd + int(icc * 1000) * 1000 + i) for i in range(N_SIM)]
                res = pool.map(_one, jobs, chunksize=50)
                row = dict(K=K, S=Sd, icc=icc, n_sim=N_SIM)
                for key in res[0]:
                    row[key] = float(np.mean([r[key] for r in res]))
                rows.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(os.path.join(HERE, "coverage_results.json"), "w") as f:
        json.dump(dict(n_sim=N_SIM, mc_se=MC_SE, priors=PRIORS, rows=rows), f, indent=1)
    keys = list(rows[0])
    with open(os.path.join(HERE, "coverage_results.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    return rows


ROWS = None


def _rows():
    global ROWS
    if ROWS is None:
        ROWS = compute()
    return ROWS


def test_exact_icc_ci_coverage():
    # exact for ICC > 0; at the boundary ICC = 0 the clipped interval [max(lo,0), max(hi,0)]
    # covers 0 whenever lo <= 0, i.e. with probability 0.975 (conservative by construction)
    tol = 3.5 * MC_SE
    for r in _rows():
        assert abs(r["icc_F"] - (0.975 if r["icc"] == 0 else 0.95)) < tol, r


def test_chi2_sigma_run_coverage():
    tol = 3.5 * MC_SE
    for r in _rows():
        assert abs(r["sw_chi2"] - 0.95) < tol, r


def test_mls_sigma_domain_coverage():
    for r in _rows():
        assert r["sb_mls"] > 0.95 - 3.5 * MC_SE, r


def test_bayes_coverage_reasonable():
    for r in _rows():
        for kind, _ in PRIORS:
            assert r[f"{kind}_sw"] > 0.85, r
            assert r[f"{kind}_sb_1s"] > 0.85, r      # one-sided bound covers the boundary
            assert r[f"{kind}_icc_1s"] > 0.85, r
            if r["icc"] >= 0.1:
                assert r[f"{kind}_icc"] > 0.80, r
            else:
                assert r[f"{kind}_sb"] == 0.0        # equal-tailed interval cannot contain 0


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
