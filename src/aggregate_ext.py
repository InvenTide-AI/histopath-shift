"""Aggregate the extended site-complete sweep into a Table-III-ready CSV.

Reads runs/foldT/{arm}_seed{S}.json for T in 0..4, {arm} in {baseline, sam,
ssl, ssl_sam, groupdro, coral, irm, mixstyle}.  Emits both the long-form
per-fold table and the summary rows the paper reports (mean, SD across
hospitals, worst-site value, floor lift over baseline, gain-baseline
Pearson correlation).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np, pandas as pd


ARMS = ["baseline", "sam", "ssl", "ssl_sam", "groupdro", "coral", "irm", "mixstyle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="./data/runs")
    ap.add_argument("--out_dir", default="results/camelyon17_extended")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.runs_root, "fold*/*.json"))):
        fold = int(os.path.basename(os.path.dirname(f)).replace("fold", ""))
        r = json.load(open(f))
        arm = r["arm"]; seed = r["seed"]
        ot = r["ood_test"]
        rows.append(dict(fold=fold, arm=arm, seed=seed,
                         selected_epoch=r.get("selected_epoch"),
                         ood_test_auroc=ot["auroc"],
                         ood_test_ece=ot["ece"],
                         ood_test_accuracy=ot["accuracy"],
                         ood_test_f1=ot["f1"],
                         ood_test_avg_precision=ot["avg_precision"],
                         wall_s=r.get("wall_s")))
    df = pd.DataFrame(rows).sort_values(["fold", "arm"]).reset_index(drop=True)
    df.to_csv(os.path.join(args.out_dir, "results_folds_ext.csv"), index=False)

    # First seed-average per (fold, arm) so a "hospital value" is one number.
    fa = df.groupby(["fold", "arm"])["ood_test_auroc"].mean().reset_index()
    # Per-arm seed SD estimate (across seeds within a fold, then averaged across folds).
    seed_sd = (df.groupby(["fold", "arm"])["ood_test_auroc"]
                 .std(ddof=1).groupby("arm").mean().rename("mean_seed_sd_within_fold"))
    n_seeds = (df.groupby(["fold", "arm"])["seed"].nunique()
                 .groupby("arm").max().rename("max_seeds_per_fold"))

    summ = []
    baseline_per_fold = fa[fa.arm == "baseline"].set_index("fold")["ood_test_auroc"]
    for arm in ARMS:
        g = fa[fa.arm == arm].set_index("fold")["ood_test_auroc"]
        aligned = g.reindex(baseline_per_fold.index)
        gain = aligned - baseline_per_fold
        pearson = (float(np.corrcoef(baseline_per_fold.values, gain.values)[0, 1])
                   if arm != "baseline" else np.nan)
        summ.append(dict(arm=arm,
                         mean=float(aligned.mean()),
                         sd_across_hospitals=float(aligned.std(ddof=1)),
                         worst_hospital_auroc=float(aligned.min()),
                         best_hospital_auroc=float(aligned.max()),
                         range=float(aligned.max() - aligned.min()),
                         mean_gain_over_baseline=float(gain.mean()) if arm != "baseline" else 0.0,
                         worst_gain=float(gain.min()) if arm != "baseline" else 0.0,
                         floor_lift_at_worst=(float(aligned.min() - baseline_per_fold.min())
                                              if arm != "baseline" else 0.0),
                         pearson_gain_vs_baseline=pearson,
                         mean_seed_sd_within_fold=float(seed_sd.get(arm, float("nan"))),
                         max_seeds_per_fold=int(n_seeds.get(arm, 0))))
    summary = pd.DataFrame(summ)
    summary.to_csv(os.path.join(args.out_dir, "spread_by_arm_ext.csv"), index=False)

    # Per-hospital pivot: seed-averaged AUROC per (fold, arm).
    pivot = fa.pivot_table(index="fold", columns="arm", values="ood_test_auroc")[ARMS]
    pivot.loc["Mean"] = pivot.mean()
    pivot.loc["SD"] = pivot.std(ddof=1)
    pivot.round(4).to_csv(os.path.join(args.out_dir, "table_iii_pilot.csv"))
    print(pivot.round(3).to_string())
    print()
    print(summary[["arm", "mean", "sd_across_hospitals",
                    "worst_hospital_auroc", "floor_lift_at_worst",
                    "pearson_gain_vs_baseline",
                    "mean_seed_sd_within_fold", "max_seeds_per_fold"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
