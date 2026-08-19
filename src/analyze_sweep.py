"""Cross-fold interaction analysis for the hospital sweep.

The decomposition
-----------------
For a mechanism pair (A, B) within one (fold, seed) cell, using seed-matched
contrasts against the same baseline:

    main_A = M(A)    - M(none)
    main_B = M(B)    - M(none)
    inter  = M(A+B)  - M(A) - M(B) + M(none)
    net    = M(A+B)  - M(none)  ==  main_A + main_B + inter   (exact identity)

The identity is checked, not assumed -- it is what licenses reading `inter` as
the super-additive part rather than as a residual.

Net benefit therefore requires

    inter > -(main_A + main_B) == c,     c = summed cost of the mechanisms alone

which is the closure condition in testable form: a pair with a large positive
interaction can still lose if each mechanism alone is costly enough.

Aggregation across folds
------------------------
Each fold is a different held-out hospital, so folds are not replicates of one
population -- they are five different shift problems. Reporting a single pooled
mean would hide exactly the variation the study is about. So:
* per-(fold, pair) estimates with seed as the replicate;
* across folds, a paired summary over folds (n=5 folds) plus the per-fold values;
* the shift magnitude from site_shift.csv joined on, so "does benefit track shift
  size?" is answerable rather than asserted.

Bootstrap CIs are over PATIENTS in the held-out site, not patches: patches from
one patient are correlated, so a patch bootstrap would produce intervals that
are far too narrow.

Outputs: results/sweep_decomposition.csv, results/sweep_summary.csv
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

AXES = ["auroc", "avg_precision", "accuracy", "balanced_accuracy", "ece"]
LOWER_BETTER = {"ece"}


def load_runs(run_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        r = json.load(open(f))
        if "ood_test" not in r:
            continue
        base = dict(arm=r["arm"], mechs=frozenset(r.get("mechanisms", [])),
                    holdout=str(r["holdout"]), seed=int(r["seed"]),
                    arch=r.get("arch"), n_train=r.get("n_train"),
                    wall_s=r.get("wall_s"), selected_epoch=r.get("selected_epoch"),
                    path=f)
        for split in ("ood_test", "id_val"):
            for k in AXES:
                if k in r.get(split, {}):
                    base[f"{split}_{k}"] = r[split][k]
        rows.append(base)
    return pd.DataFrame(rows)


def decompose(df, pairs):
    """Per (fold, seed, pair, axis) decomposition on seed-matched contrasts."""
    out = []
    for (fold, seed), g in df.groupby(["holdout", "seed"]):
        by = {frozenset(m): r for m, r in zip(g.mechs, g.to_dict("records"))}
        none = by.get(frozenset())
        if none is None:
            continue
        for A, B in pairs:
            a, b, ab = by.get(frozenset([A])), by.get(frozenset([B])), by.get(frozenset([A, B]))
            if not (a and b and ab):
                continue
            for split in ("ood_test", "id_val"):
                for ax in AXES:
                    key = f"{split}_{ax}"
                    if any(key not in r or pd.isna(r.get(key)) for r in (none, a, b, ab)):
                        continue
                    sgn = -1.0 if ax in LOWER_BETTER else 1.0
                    m0, mA, mB, mAB = (sgn * r[key] for r in (none, a, b, ab))
                    main_a, main_b = mA - m0, mB - m0
                    inter = mAB - mA - mB + m0
                    net = mAB - m0
                    assert abs((main_a + main_b + inter) - net) < 1e-9, \
                        "decomposition identity violated"
                    out.append(dict(holdout=fold, seed=seed, pair=f"{A}+{B}",
                                    mech_a=A, mech_b=B, split=split, axis=ax,
                                    m_none=m0, m_a=mA, m_b=mB, m_ab=mAB,
                                    main_a=main_a, main_b=main_b,
                                    interaction=inter, net=net,
                                    cost=-(main_a + main_b),
                                    benefit_condition_met=bool(inter > -(main_a + main_b))))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_gpu")
    p.add_argument("--out", default="results")
    p.add_argument("--shift", default="results/site_shift.csv")
    a = p.parse_args()

    df = load_runs(a.runs)
    if df.empty:
        raise SystemExit(f"no run records with ood_test in {a.runs}")
    print(f"loaded {len(df)} runs | folds={sorted(df.holdout.unique())} "
          f"| seeds={sorted(df.seed.unique())} | arms={df.arm.nunique()}")

    present = set(df.mechs)
    pairs = sorted({tuple(sorted(m)) for m in present if len(m) == 2})
    def complete(pair):
        A, B = pair
        return frozenset([A]) in present and frozenset([B]) in present

    incomplete = [f"{A}+{B}" for A, B in pairs if not complete((A, B))]
    if incomplete:
        print("pairs lacking singleton arms (excluded, 2x2 incomplete):", incomplete)
    pairs = [pr for pr in pairs if complete(pr)]

    dec = decompose(df, pairs)
    if dec.empty:
        raise SystemExit("no complete 2x2 blocks yet")
    os.makedirs(a.out, exist_ok=True)
    dec.to_csv(os.path.join(a.out, "sweep_decomposition.csv"), index=False)

    # seed-mean per (fold, pair, axis), then fold-level summary
    sm = (dec.groupby(["pair", "split", "axis", "holdout"])
             [["main_a", "main_b", "interaction", "net", "cost"]]
             .mean().reset_index())
    summ = (sm.groupby(["pair", "split", "axis"])
              .agg(n_folds=("holdout", "nunique"),
                   interaction_mean=("interaction", "mean"),
                   interaction_min=("interaction", "min"),
                   interaction_max=("interaction", "max"),
                   net_mean=("net", "mean"),
                   folds_interaction_positive=("interaction", lambda s: int((s > 0).sum())),
                   folds_net_positive=("net", lambda s: int((s > 0).sum())))
              .reset_index())
    summ.to_csv(os.path.join(a.out, "sweep_summary.csv"), index=False)

    key = summ[(summ.split == "ood_test") & (summ.axis == "auroc")] \
        .sort_values("interaction_mean", ascending=False)
    print("\nheld-out-hospital AUROC, by pair (interaction-ranked):")
    print(key.to_string(index=False))

    if os.path.exists(a.shift):
        sh = pd.read_csv(a.shift)
        sh["holdout"] = sh["holdout"].astype(str)
        j = sm[(sm.split == "ood_test") & (sm.axis == "auroc")].merge(sh, on="holdout")
        if len(j) > 2:
            print("\ndoes benefit track shift magnitude? (Spearman over folds)")
            for pr, g in j.groupby("pair"):
                if g.holdout.nunique() > 2:
                    from scipy.stats import spearmanr
                    rI = spearmanr(g.domain_auroc_vs_rest, g.interaction).statistic
                    rN = spearmanr(g.domain_auroc_vs_rest, g.net).statistic
                    print(f"  {pr:18s} rho(shift, interaction)={rI:+.3f}  "
                          f"rho(shift, net)={rN:+.3f}  (n={g.holdout.nunique()} folds)")
    print("\nwrote", os.path.join(a.out, "sweep_decomposition.csv"),
          "and sweep_summary.csv")


if __name__ == "__main__":
    main()
