"""Aggregate every cohort's per-run records into the tables the paper reports.

This is the script that produces `results/rules/synthesis.json` plus the three
cross-cohort analyses the manuscript leans on:

  1. Per-arm, per-cohort summaries: worst-domain AUROC, mean, across-domain SD,
     ECE, and the gain in mean AUROC over that cohort's supervised baseline.
     A domain's value is the mean over whatever seeds were run for that cell.

  2. The pooled test-time-adaptation contrast. Paired per domain across the
     three *independent* cohorts -- Camelyon17, MIDOG and canine SCC, 13
     domains in total. The canine no-scale condition is deliberately excluded:
     it is the same 44 slides as the canine cohort under different
     preprocessing, so including it would count those five domains twice.

  3. The preprocessing ablation: canine SCC with and without scale
     normalization, restricted to the arms run in both conditions.

Usage:
    python src/aggregate_rules.py --root /path/to/histopath_data --out results
"""
from __future__ import annotations
import argparse
import collections
import glob
import json
import os

import numpy as np
from scipy import stats

# cohort label -> (run directory under --root, is it an independent cohort?)
COHORTS = {
    "camelyon17": ("runs", True),
    "midog": ("midog_runs", True),
    "canine": ("canine_runs", True),
    "canine_noscale": ("canine_noscale_runs", False),
}
BASELINE = "baseline"


def load_cohort(run_dir: str) -> dict:
    """{arm: {fold: {seed: {"auroc": x, "ece": y}}}} from per-run JSON records."""
    out: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for fold_dir in sorted(glob.glob(os.path.join(run_dir, "fold*"))):
        fold = int(os.path.basename(fold_dir).replace("fold", ""))
        for path in sorted(glob.glob(os.path.join(fold_dir, "*.json"))):
            stem = os.path.basename(path)[:-5]
            arm, _, seed = stem.rpartition("_seed")
            if not arm:
                continue
            with open(path) as fh:
                rec = json.load(fh)
            if "ood_test" not in rec:
                continue
            out[arm][fold][int(seed)] = {
                "auroc": rec["ood_test"]["auroc"],
                "ece": rec["ood_test"]["ece"],
            }
    return {a: dict(f) for a, f in out.items()}


def per_domain(cell: dict, key: str) -> dict:
    """Seed-average each fold, returning {fold: value}."""
    return {f: float(np.mean([s[key] for s in seeds.values()]))
            for f, seeds in cell.items()}


def summarize(cohort: dict) -> dict:
    base_auroc = per_domain(cohort[BASELINE], "auroc")
    base_mean = float(np.mean(list(base_auroc.values())))
    arms = {}
    for arm, folds in sorted(cohort.items()):
        auroc = per_domain(folds, "auroc")
        ece = per_domain(folds, "ece")
        vals = np.array([auroc[f] for f in sorted(auroc)])
        arms[arm] = {
            "worst": round(float(vals.min()), 4),
            "mean": round(float(vals.mean()), 4),
            "sd": round(float(vals.std(ddof=1)), 4),
            "ece": round(float(np.mean(list(ece.values()))), 4),
            "gain_mean": round(float(vals.mean() - base_mean), 4),
            "n_folds": len(vals),
            "n_seeds": max(len(s) for s in folds.values()),
            "per_fold": [round(auroc[f], 4) for f in sorted(auroc)],
        }
    return {
        "n_domains": len(base_auroc),
        "erm_worst": arms[BASELINE]["worst"],
        "erm_sd": arms[BASELINE]["sd"],
        "arms": arms,
    }


def paired_contrast(data: dict, arm: str, cohorts: list) -> dict | None:
    """Pair `arm` against the baseline within each domain, pooled over cohorts."""
    d_auroc, d_ece = [], []
    for name in cohorts:
        cohort = data[name]
        if arm not in cohort:
            continue
        base_a = per_domain(cohort[BASELINE], "auroc")
        base_e = per_domain(cohort[BASELINE], "ece")
        arm_a = per_domain(cohort[arm], "auroc")
        arm_e = per_domain(cohort[arm], "ece")
        for f in sorted(set(base_a) & set(arm_a)):
            d_auroc.append(arm_a[f] - base_a[f])
            d_ece.append(arm_e[f] - base_e[f])
    if len(d_auroc) < 2:
        return None
    d = np.array(d_auroc)
    t, p = stats.ttest_1samp(d, 0.0)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {
        "arm": arm,
        "n_domains": len(d),
        "auroc_gain": round(float(d.mean()), 4),
        "ci": [round(float(d.mean() - half), 4), round(float(d.mean() + half), 4)],
        "t": round(float(t), 3),
        "p": float(p),
        "improved": int((d > 0).sum()),
        "ece_change": round(float(np.mean(d_ece)), 4),
        "better_calibrated": int((np.array(d_ece) < 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    raw = {name: load_cohort(os.path.join(a.root, d))
           for name, (d, _) in COHORTS.items()}
    independent = [n for n, (_, ind) in COHORTS.items() if ind]

    synthesis = {name: summarize(cohort) for name, cohort in raw.items()}

    # pooled test-time-adaptation contrasts over the independent cohorts only
    pooled = {}
    for arm in ["bnadapt", "tent", "bnadapt-on-ssl", "bnadapt-on-erm_henorm",
                "erm_henorm", "probe_resnet50"]:
        res = paired_contrast(raw, arm, independent)
        if res:
            pooled[arm] = res
    synthesis["pooled"] = {"cohorts": independent, "contrasts": pooled}

    # preprocessing ablation: arms present in both canine conditions
    shared = sorted(set(raw["canine"]) & set(raw["canine_noscale"]))
    synthesis["scale_ablation"] = {
        arm: {
            "scaled": synthesis["canine"]["arms"][arm]["gain_mean"],
            "noscale": synthesis["canine_noscale"]["arms"][arm]["gain_mean"],
            "worst_scaled": synthesis["canine"]["arms"][arm]["worst"],
            "worst_noscale": synthesis["canine_noscale"]["arms"][arm]["worst"],
        }
        for arm in shared
    }

    os.makedirs(os.path.join(a.out, "rules"), exist_ok=True)
    dest = os.path.join(a.out, "rules", "synthesis.json")
    with open(dest, "w") as fh:
        json.dump(synthesis, fh, indent=1)
    print("wrote", dest)

    for arm, r in pooled.items():
        print("%-24s n=%2d  gain=%+.4f  CI[%+.4f,%+.4f]  p=%.3g  %d/%d up  "
              "dECE=%+.4f  %d/%d better"
              % (arm, r["n_domains"], r["auroc_gain"], r["ci"][0], r["ci"][1],
                 r["p"], r["improved"], r["n_domains"], r["ece_change"],
                 r["better_calibrated"], r["n_domains"]))


if __name__ == "__main__":
    main()
