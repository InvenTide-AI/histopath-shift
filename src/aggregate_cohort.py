"""Aggregate a scanner/site-complete grid for any cohort into the paper's tables.

Reads the per-run JSON records written by run_arm_ext.py / run_arm_probe.py and
emits the same statistics the Camelyon17 analysis reports, so cohorts can be
placed side by side:

  * per-fold and seed-averaged off-site AUROC and ECE
  * worst-site value, across-site SD, mean, mean within-fold seed SD
  * the response-shape correlation r, its analytic null r0 under independence,
    and the excess r - r0
  * a two-way (site x seed) variance decomposition with ICC
  * the number of domains K needed to detect a given mean effect

Usage:
    python src/aggregate_cohort.py --runs <dir>/midog_runs --out results/midog
"""
from __future__ import annotations
import argparse
import collections
import csv
import glob
import json
import math
import os

import numpy as np
from scipy import stats

BASELINE = "baseline"


def load(runs_dir):
    """arm -> (fold, seed) -> ood_test metric dict."""
    rec = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(runs_dir, "fold*", "*.json"))):
        fold = int(os.path.basename(os.path.dirname(f)).replace("fold", ""))
        arm, seed = os.path.basename(f)[:-5].rsplit("_seed", 1)
        rec[arm][(fold, int(seed))] = json.load(open(f))["ood_test"]
    return rec


def folds_of(rec):
    return sorted({f for arm in rec for (f, _) in rec[arm]})


def per_fold(rec, arm, folds, metric="auroc"):
    d = collections.defaultdict(list)
    for (f, _), m in rec[arm].items():
        d[f].append(m[metric])
    return np.array([np.mean(d[f]) if d[f] else np.nan for f in folds])


def seed_sd(rec, arm, folds):
    d = collections.defaultdict(list)
    for (f, _), m in rec[arm].items():
        d[f].append(m["auroc"])
    v = [np.std(d[f], ddof=1) for f in folds if len(d[f]) > 1]
    return float(np.mean(v)) if v else float("nan")


def decompose(rec, arm, folds):
    """Two-way random-effects components; needs a balanced seed grid."""
    d = collections.defaultdict(dict)
    for (f, s), m in rec[arm].items():
        d[f][s] = m["auroc"]
    seeds = sorted(set.intersection(*[set(d[f]) for f in folds])) if folds else []
    if len(seeds) < 2:
        return None
    y = np.array([[d[f][s] for s in seeds] for f in folds])
    k, ns = y.shape
    g = y.mean()
    ms_f = ns * ((y.mean(1) - g) ** 2).sum() / (k - 1)
    ms_s = k * ((y.mean(0) - g) ** 2).sum() / (ns - 1)
    ms_e = ((y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + g) ** 2
            ).sum() / ((k - 1) * (ns - 1))
    s2f = max((ms_f - ms_e) / ns, 0.0)
    s2s = max((ms_s - ms_e) / k, 0.0)
    tot = s2f + s2s + ms_e
    return dict(sigma_site=math.sqrt(s2f), sigma_seed=math.sqrt(s2s),
                sigma_resid=math.sqrt(ms_e),
                icc_site=(s2f / tot if tot > 0 else float("nan")))


def min_domains(s_site, s_run, eff, seeds=1, power=0.8):
    for K in range(2, 400):
        se = math.sqrt(s_site ** 2 / K + s_run ** 2 / (K * seeds))
        df = K - 1
        tc = stats.t.ppf(0.975, df)
        ncp = eff / se
        if 1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp) >= power:
            return K
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    label = a.label or os.path.basename(a.runs.rstrip("/"))

    rec = load(a.runs)
    folds = folds_of(rec)
    if BASELINE not in rec:
        raise SystemExit("no %s arm found in %s" % (BASELINE, a.runs))
    base = per_fold(rec, BASELINE, folds)
    sE = float(np.std(base, ddof=1))

    rows = []
    for arm in rec:
        v = per_fold(rec, arm, folds)
        ece = per_fold(rec, arm, folds, "ece")
        sM = float(np.std(v, ddof=1))
        if arm == BASELINE:
            r = r0 = exc = float("nan")
        else:
            r = float(np.corrcoef(v - base, base)[0, 1])
            r0 = -sE / math.sqrt(sM ** 2 + sE ** 2)
            exc = r - r0
        rows.append(dict(arm=arm, worst=float(np.min(v)), sd=sM,
                         mean=float(np.mean(v)), seed_sd=seed_sd(rec, arm, folds),
                         ece=float(np.mean(ece)), r=r, r_null=r0, excess=exc,
                         per_fold=[round(float(x), 4) for x in v]))
    rows.sort(key=lambda d: -d["worst"])

    print("=" * 104)
    print("%s  (K = %d domains: %s)" % (label, len(folds), folds))
    print("=" * 104)
    hdr = ("%-16s %7s %7s %7s %8s %7s %7s %7s %8s"
           % ("arm", "worst", "SD", "mean", "seedSD", "ECE", "r", "r0", "excess"))
    print(hdr)
    print("-" * 104)
    for d in rows:
        f = lambda x, p=3: ("%7.*f" % (p, x)) if not math.isnan(x) else "      -"
        print("%-16s %s %s %s %s %s %s %s %s"
              % (d["arm"], f(d["worst"]), f(d["sd"]), f(d["mean"]),
                 ("%8.3f" % d["seed_sd"]) if not math.isnan(d["seed_sd"]) else "       -",
                 f(d["ece"]), f(d["r"]), f(d["r_null"]),
                 ("%+8.3f" % d["excess"]) if not math.isnan(d["excess"]) else "       -"))

    with open(os.path.join(a.out, "spread_by_arm.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for d in rows:
            w.writerow({**d, "per_fold": json.dumps(d["per_fold"])})

    print()
    print("variance decomposition (arms with >=2 seeds at every fold)")
    vd = {}
    for arm in ["baseline", "sam", "ssl", "ssl_sam"]:
        if arm not in rec:
            continue
        c = decompose(rec, arm, folds)
        if c:
            vd[arm] = c
            print("  %-9s sigma_site=%.4f sigma_seed=%.4f sigma_resid=%.4f ICC_site=%.3f"
                  % (arm, c["sigma_site"], c["sigma_seed"], c["sigma_resid"],
                     c["icc_site"]))

    if BASELINE in vd:
        c = vd[BASELINE]
        s_site = c["sigma_site"]
        s_run = math.hypot(c["sigma_seed"], c["sigma_resid"])
        print()
        print("domains needed for 80%% power (sigma_site=%.3f, sigma_run=%.3f):"
              % (s_site, s_run))
        pw = []
        for delta in (0.05, 0.075, 0.10, 0.15, 0.20):
            k1 = min_domains(s_site, s_run, delta, 1)
            k3 = min_domains(s_site, s_run, delta, 3)
            pw.append(dict(delta=delta, K_1seed=k1, K_3seed=k3))
            print("   delta=%.3f  K(1 seed)=%-4s K(3 seeds)=%s" % (delta, k1, k3))
        json.dump(dict(label=label, K=len(folds), variance=vd, power=pw),
                  open(os.path.join(a.out, "variance_power.json"), "w"), indent=1)

    # paired contrasts against the baseline
    print()
    print("paired contrasts vs %s (n = %d domains)" % (BASELINE, len(folds)))
    for arm in ["ssl", "ssl_sam", "sam"]:
        if arm not in rec:
            continue
        d = per_fold(rec, arm, folds) - base
        t, p = stats.ttest_1samp(d, 0)
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        print("  %-9s mean=%+.3f  95%% CI [%+.3f, %+.3f]  p=%.3f"
              % (arm, d.mean(), ci[0], ci[1], p))
    print()
    print("wrote %s/spread_by_arm.csv" % a.out)


if __name__ == "__main__":
    main()
