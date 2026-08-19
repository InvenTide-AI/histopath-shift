"""Aggregate the cross-institution chest-radiograph experiment.

Identical statistics to aggregate_crc.py (paired bootstrap, B = 10,000, joint
resamples shared across arms, percentile CIs, two-sided p as twice the smaller
tail), plus the contrast that this experiment exists to test:

    H2:  Delta(pool M) > Delta(pool S)

where Delta is the SSL+SAM minus baseline OOD AUROC gap. Pool S is NIH-only
(single-site); pool M spans both institutions. A positive H2 contrast supports
"the mechanism needs a site-spanning pre-training corpus"; a null says the
Camelyon17 result did not replicate for a reason other than the pool.

Outputs: cxr_results_raw.csv, cxr_results_summary.csv, cxr_bootstrap_ci.csv,
cxr_bootstrap_contrasts.csv, cxr_interaction.csv, cxr_h2_contrast.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUNS = "runs_cxr"
B = 10000
ARMS = ["baseline", "sam", "ssl_S", "ssl_M", "ssl_S_sam", "ssl_M_sam"]
SETS = ["id_test", "ood"]


def load():
    recs, preds = [], {}
    for f in sorted(glob.glob(f"{RUNS}/*_seed*.json")):
        r = json.load(open(f))
        recs.append(dict(arm=r["arm"], seed=r["seed"],
                         selected_epoch=r["selected_epoch"],
                         id_val_auroc=r["id_val_auroc"],
                         id_test=r["id_test"]["auroc"], ood=r["ood"]["auroc"]))
        p = np.load(f.replace(".json", "_preds.npz"))
        preds[(r["arm"], r["seed"])] = {
            "id_test": (p["y_id_test"], p["p_id_test"]),
            "ood": (p["y_ood"], p["p_ood"])}
    return pd.DataFrame(recs), preds


def boot_indices(n, rng):
    return rng.integers(0, n, size=(B, n))


def auroc_boot(y, p, idx):
    """AUROC over each bootstrap resample; skips degenerate resamples."""
    out = np.empty(len(idx))
    for b, ix in enumerate(idx):
        yy = y[ix]
        out[b] = np.nan if yy.min() == yy.max() else roc_auc_score(yy, p[ix])
    return out


def main():
    df, preds = load()
    if df.empty:
        raise SystemExit("no run records in " + RUNS)
    df.to_csv("cxr_results_raw.csv", index=False)

    summ = (df.groupby("arm")[["id_test", "ood"]]
              .agg(["mean", "std", "count"]).round(4))
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv("cxr_results_summary.csv", index=False)

    rng = np.random.default_rng(0)
    seeds = sorted(df.seed.unique())
    arms = [a for a in ARMS if a in set(df.arm)]

    # one shared resample per test set, so every arm sees the same patients
    idx = {}
    for s in SETS:
        y0 = preds[(arms[0], seeds[0])][s][0]
        idx[s] = boot_indices(len(y0), rng)

    # seed-averaged bootstrap AUROC per arm
    curves = {}
    for a in arms:
        for s in SETS:
            mats = []
            for sd in seeds:
                y, p = preds[(a, sd)][s]
                mats.append(auroc_boot(y, p, idx[s]))
            curves[(a, s)] = np.nanmean(np.stack(mats), axis=0)

    rows = []
    for a in arms:
        for s in SETS:
            c = curves[(a, s)]
            rows.append(dict(arm=a, split=s, auroc=float(np.nanmean(c)),
                             ci_lo=float(np.nanpercentile(c, 2.5)),
                             ci_hi=float(np.nanpercentile(c, 97.5))))
    pd.DataFrame(rows).round(4).to_csv("cxr_bootstrap_ci.csv", index=False)

    def contrast(a1, a0, s):
        d = curves[(a1, s)] - curves[(a0, s)]
        frac = float(np.mean(d < 0)) if np.nanmean(d) > 0 else float(np.mean(d > 0))
        return dict(split=s, contrast=f"{a1} - {a0}", delta=float(np.nanmean(d)),
                    ci_lo=float(np.nanpercentile(d, 2.5)),
                    ci_hi=float(np.nanpercentile(d, 97.5)),
                    p_two_sided=max(2 * frac, 1.0 / B))

    crows = []
    for s in SETS:
        for a in arms:
            if a != "baseline":
                crows.append(contrast(a, "baseline", s))
        if {"ssl_M_sam", "ssl_S_sam"} <= set(arms):
            crows.append(contrast("ssl_M_sam", "ssl_S_sam", s))
    pd.DataFrame(crows).round(4).to_csv("cxr_bootstrap_contrasts.csv", index=False)

    # 2x2 interaction, per pool
    irows = []
    for pool in ["S", "M"]:
        need = {"baseline", "sam", f"ssl_{pool}", f"ssl_{pool}_sam"}
        if not need <= set(arms):
            continue
        for s in SETS:
            b = curves[("baseline", s)]
            ss = curves[(f"ssl_{pool}", s)]
            sa = curves[("sam", s)]
            both = curves[(f"ssl_{pool}_sam", s)]
            for nm, d in [("ssl_alone", ss - b), ("sam_alone", sa - b),
                          ("interaction", both - ss - sa + b), ("total", both - b)]:
                irows.append(dict(pool=pool, split=s, term=nm,
                                  value=float(np.nanmean(d)),
                                  ci_lo=float(np.nanpercentile(d, 2.5)),
                                  ci_hi=float(np.nanpercentile(d, 97.5))))
    pd.DataFrame(irows).round(4).to_csv("cxr_interaction.csv", index=False)

    # ---- H2: does the site-spanning pool buy more than the single-site pool?
    h2 = []
    if {"ssl_M_sam", "ssl_S_sam", "baseline"} <= set(arms):
        for s in SETS:
            dM = curves[("ssl_M_sam", s)] - curves[("baseline", s)]
            dS = curves[("ssl_S_sam", s)] - curves[("baseline", s)]
            dd = dM - dS
            frac = float(np.mean(dd < 0)) if np.nanmean(dd) > 0 else float(np.mean(dd > 0))
            h2.append(dict(split=s, quantity="delta_M_minus_delta_S",
                           delta_M=float(np.nanmean(dM)), delta_S=float(np.nanmean(dS)),
                           difference=float(np.nanmean(dd)),
                           ci_lo=float(np.nanpercentile(dd, 2.5)),
                           ci_hi=float(np.nanpercentile(dd, 97.5)),
                           p_two_sided=max(2 * frac, 1.0 / B)))
    pd.DataFrame(h2).round(4).to_csv("cxr_h2_contrast.csv", index=False)

    pd.set_option("display.width", 170)
    print("=== summary\n", summ.to_string(index=False))
    print("\n=== bootstrap CI\n", pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n=== contrasts\n", pd.DataFrame(crows).round(4).to_string(index=False))
    print("\n=== interaction\n", pd.DataFrame(irows).round(4).to_string(index=False))
    print("\n=== H2 (registered prediction)\n",
          pd.DataFrame(h2).round(4).to_string(index=False) if h2 else "n/a")


if __name__ == "__main__":
    main()
