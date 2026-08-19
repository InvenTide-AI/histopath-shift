"""Aggregate the NCT-CRC-HE 4-arm x 2-seed ablation.

Produces, for each of the two shift axes:
  crc_results_raw.csv        per run x arm x seed, all metrics
  crc_results_summary.csv    mean +/- sd per arm, plus ID-to-shift gap
  crc_bootstrap_ci.csv       per-arm bootstrap CIs on the shifted set
  crc_bootstrap_contrasts.csv paired bootstrap contrasts between arms
  crc_interaction.csv        the 2x2 decomposition (the primary estimand)

The interaction term is defined exactly as in the Camelyon17 analysis:
    interaction = (ssl_sam - base) - (ssl - base) - (sam - base)
i.e. how much the combination exceeds the sum of the two separate effects.
Contrasts are PAIRED: identical resampled patch indices score both arms.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

SRC = sys.argv[1] if len(sys.argv) > 1 else "runs_crc"
ARMS = ["baseline", "ssl", "sam", "ssl_sam"]
LABEL = {"baseline": "Baseline (supervised)", "ssl": "+ SSL (Pillar I)",
         "sam": "+ SAM (Pillar II)", "ssl_sam": "SSL + SAM (both)"}
AXES = ["shift_nonorm", "shift_external"]
METRICS = ["accuracy", "auroc", "avg_precision", "f1", "balanced_accuracy",
           "sensitivity", "specificity", "precision", "ece"]
B = 10000
rng = np.random.RandomState(12345)

# ---------------- per-run table ----------------
rows = []
for f in sorted(glob.glob(f"{SRC}/*.json")):
    r = json.load(open(f))
    base = dict(arm=r["arm"], seed=r["seed"], dataset=r["dataset"],
                selected_epoch=r["selected_epoch"], wall_s=r["wall_s"],
                n_train=r["n_train"])
    for split in ["id_val"] + AXES:
        for m in METRICS:
            base[f"{split}_{m}"] = r[split][m]
    rows.append(base)
if not rows:
    sys.exit(f"no run JSONs in {SRC}")
raw = pd.DataFrame(rows).sort_values(["arm", "seed"])
raw.to_csv("crc_results_raw.csv", index=False)

agg = {}
for m in METRICS:
    for split in ["id_val"] + AXES:
        c = f"{split}_{m}"
        agg[f"{c}_mean"] = raw.groupby("arm")[c].mean()
        agg[f"{c}_sd"] = raw.groupby("arm")[c].std(ddof=1)
summary = pd.DataFrame(agg).reindex(ARMS)
for ax in AXES:
    summary[f"gap_{ax}_auroc"] = (summary["id_val_auroc_mean"]
                                  - summary[f"{ax}_auroc_mean"])
summary.to_csv("crc_results_summary.csv")

# ---------------- bootstrap on per-sample predictions ----------------
preds = {}
for f in sorted(glob.glob(f"{SRC}/*_preds.npz")):
    tag = os.path.basename(f).replace("_preds.npz", "")
    arm, seed = tag.rsplit("_seed", 1)
    preds.setdefault(arm, {})[int(seed)] = np.load(f)


def metrics_at(y, p):
    return dict(auroc=roc_auc_score(y, p),
                avg_precision=average_precision_score(y, p),
                accuracy=((p > .5).astype(int) == y).mean(),
                f1=f1_score(y, (p > .5).astype(int)))


ci_rows, contrast_rows, inter_rows = [], [], []
for ax in AXES:
    # seed-averaged probabilities per arm, on the identical test set
    y0 = None
    prob = {}
    for arm in ARMS:
        if arm not in preds:
            continue
        ps = []
        for seed, z in sorted(preds[arm].items()):
            y = z[f"y_{ax}"]
            y0 = y if y0 is None else y0
            assert np.array_equal(y, y0), f"{ax}: labels differ across runs"
            ps.append(z[f"p_{ax}"])
        prob[arm] = np.mean(ps, axis=0)
    if y0 is None:
        continue
    n = len(y0)
    idx = rng.randint(0, n, size=(B, n))

    boot = {}
    for arm, p in prob.items():
        vals = {m: np.empty(B) for m in ["auroc", "avg_precision", "accuracy", "f1"]}
        for b in range(B):
            i = idx[b]
            yb, pb = y0[i], p[i]
            if yb.min() == yb.max():
                for m in vals:
                    vals[m][b] = np.nan
                continue
            mm = metrics_at(yb, pb)
            for m in vals:
                vals[m][b] = mm[m]
        boot[arm] = vals
        obs = metrics_at(y0, p)
        for m, v in vals.items():
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            ci_rows.append(dict(axis=ax, arm=arm, metric=m, observed=obs[m],
                                ci_lo=lo, ci_hi=hi, n_test=n, n_boot=B))

    for a, b_ in [("baseline", "ssl"), ("baseline", "sam"),
                  ("baseline", "ssl_sam"), ("ssl", "ssl_sam"),
                  ("sam", "ssl_sam")]:
        if a not in boot or b_ not in boot:
            continue
        for m in ["auroc", "avg_precision", "accuracy", "f1"]:
            d = boot[b_][m] - boot[a][m]          # paired: same resample indices
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            obs = metrics_at(y0, prob[b_])[m] - metrics_at(y0, prob[a])[m]
            contrast_rows.append(dict(
                axis=ax, contrast=f"{b_} - {a}", metric=m, observed=obs,
                ci_lo=lo, ci_hi=hi,
                p_two_sided=2 * min((d <= 0).mean(), (d >= 0).mean())))

    # ---- 2x2 interaction decomposition (primary estimand) ----
    if all(a in boot for a in ARMS):
        for m in ["auroc", "avg_precision"]:
            e_ssl = boot["ssl"][m] - boot["baseline"][m]
            e_sam = boot["sam"][m] - boot["baseline"][m]
            e_tot = boot["ssl_sam"][m] - boot["baseline"][m]
            e_int = e_tot - e_ssl - e_sam
            o = {a: metrics_at(y0, prob[a])[m] for a in ARMS}
            for name, d, obs in [
                    ("ssl_alone", e_ssl, o["ssl"] - o["baseline"]),
                    ("sam_alone", e_sam, o["sam"] - o["baseline"]),
                    ("interaction", e_int,
                     (o["ssl_sam"] - o["baseline"]) - (o["ssl"] - o["baseline"])
                     - (o["sam"] - o["baseline"])),
                    ("total", e_tot, o["ssl_sam"] - o["baseline"])]:
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                inter_rows.append(dict(
                    axis=ax, metric=m, term=name, observed=obs, ci_lo=lo,
                    ci_hi=hi,
                    p_two_sided=2 * min((d <= 0).mean(), (d >= 0).mean())))

pd.DataFrame(ci_rows).to_csv("crc_bootstrap_ci.csv", index=False)
pd.DataFrame(contrast_rows).to_csv("crc_bootstrap_contrasts.csv", index=False)
pd.DataFrame(inter_rows).to_csv("crc_interaction.csv", index=False)

print(summary[[f"{ax}_auroc_mean" for ax in AXES]
              + ["id_val_auroc_mean"]].round(4).to_string())
print()
if inter_rows:
    d = pd.DataFrame(inter_rows)
    print(d[d.metric == "auroc"][
        ["axis", "term", "observed", "ci_lo", "ci_hi", "p_two_sided"]
    ].round(4).to_string(index=False))
