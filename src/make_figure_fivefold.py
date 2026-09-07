#!/usr/bin/env python3
"""Regenerate Fig. 1 of the manuscript (the site-complete factorial) from the
shipped five-hospital result table.

    python src/make_figure_fivefold.py [--out paper/fig1_site_complete_factorial.pdf]

Reads results/camelyon17_fivehospital/results_folds.csv, which is the per-run
record of the five-fold cross-hospital sweep (one seed per arm per held-out
hospital). Every annotation in the figure -- the floor-lift arrow, the Pearson
correlation, the mean interaction and its CI -- is computed from that table
rather than hard-coded, so the figure cannot drift from the data.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import apply_figure_style, panel_letter

ARMS = ["baseline", "sam", "ssl", "ssl_sam"]
LAB = {"baseline": "Baseline (ERM)", "sam": "SAM", "ssl": "SSL pre-train", "ssl_sam": "SSL + SAM"}
COL = {"baseline": "#4d4d4d", "sam": "#0072B2", "ssl": "#D55E00", "ssl_sam": "#009E73"}
MK = {"baseline": "o", "sam": "s", "ssl": "^", "ssl_sam": "D"}
MM = 1 / 25.4
PUBLISHED_I = 0.1610  # the single-centre value this sweep set out to replicate


def load_grid(path):
    res = pd.read_csv(path)
    piv = res.pivot_table(index="fold", columns="arm", values="ood_test_auroc")
    piv["ssl_only"] = piv["ssl"] - piv["baseline"]
    piv["sam_only"] = piv["sam"] - piv["baseline"]
    piv["I"] = (piv["ssl_sam"] - piv["ssl"]) - (piv["sam"] - piv["baseline"])
    piv["net"] = piv["ssl_sam"] - piv["baseline"]
    return piv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/camelyon17_fivehospital/results_folds.csv")
    ap.add_argument("--out", default="paper/fig1_site_complete_factorial.pdf")
    args = ap.parse_args()

    apply_figure_style()
    piv = load_grid(args.results)
    folds = list(piv.index)
    worst = piv["baseline"].idxmin()          # hospital with the lowest ERM AUROC
    lift = piv.loc[worst, "ssl_only"]
    r = np.corrcoef(piv["baseline"], piv["ssl_only"])[0, 1]
    m = piv["I"].mean()
    n = len(piv)
    se = piv["I"].std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=se)

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 64 * MM))

    ax = axes[0]
    for a in ARMS:
        ax.plot(folds, piv[a], marker=MK[a], color=COL[a], lw=1.0, ms=4.5,
                mfc=COL[a], mec="white", mew=0.6, label=LAB[a], zorder=3)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"C{f}" for f in folds])
    ax.set_xlim(-0.35, 4.75)
    ax.set_ylim(0.62, 1.0)
    ax.set_xlabel("Held-out hospital (test centre)")
    ax.set_ylabel("OOD test AUROC")
    ax.annotate("", xy=(worst, piv.loc[worst, "ssl"]), xytext=(worst, piv.loc[worst, "baseline"]),
                arrowprops=dict(arrowstyle="<->", color="#8c8c8c", lw=0.8, shrinkA=1.5, shrinkB=1.5))
    ax.text(worst + 0.12, (piv.loc[worst, "ssl"] + piv.loc[worst, "baseline"]) / 2,
            f"{lift:+.2f}", ha="left", va="center", fontsize=7, color="#4d4d4d")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left", handlelength=1.4,
              borderpad=0.2, labelspacing=0.22)
    ax.set_title("Pre-training raises the floor", pad=4)
    panel_letter(ax, "a", dx=-0.20)

    ax = axes[1]
    ax.axhline(0, color="#b0b0b0", lw=0.7, zorder=1)
    ax.scatter(piv["baseline"], piv["ssl_only"], s=34, color=COL["ssl"], ec="white", lw=0.6, zorder=3)
    for f in folds:
        ax.text(piv.loc[f, "baseline"] + 0.012, piv.loc[f, "ssl_only"] + 0.006, f"C{f}",
                fontsize=7, color="#4d4d4d", ha="left", va="bottom")
    ax.set_xlim(0.635, 1.005)
    ax.set_ylim(-0.06, 0.34)
    ax.set_xlabel("Baseline OOD AUROC on that hospital")
    ax.set_ylabel("Gain from SSL pre-training")
    ax.text(0.97, 0.95, f"Pearson $r$ = {r:.2f} (n = {n})".replace("-", "\u2212"),
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#4d4d4d")
    ax.set_title("Gain concentrates where ERM fails", pad=4)
    panel_letter(ax, "b", dx=-0.20)

    ax = axes[2]
    ax.axhline(0, color="#b0b0b0", lw=0.7, zorder=1)
    ax.scatter(folds, piv["I"], s=30, color="#4d4d4d", ec="white", lw=0.6, zorder=3,
               label="per-hospital $I$")
    ax.errorbar(5.6, m, yerr=[[m - lo], [hi - m]], fmt="D", ms=5, color="#0072B2", capsize=2.5,
                lw=1.1, mec="white", mew=0.6, zorder=4, label="mean, 95% CI")
    ax.axhline(PUBLISHED_I, color="#D55E00", lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.text(6.15, PUBLISHED_I, f"published\n{PUBLISHED_I:+.3f}", fontsize=7, color="#D55E00",
            va="center", ha="left")
    ax.set_xticks(folds + [5.6])
    ax.set_xticklabels([f"C{f}" for f in folds] + ["all"])
    ax.set_xlim(-0.6, 7.9)
    ax.set_ylim(-0.09, 0.215)
    ax.set_xlabel("Held-out hospital")
    ax.set_ylabel("Interaction $I$ (AUROC)")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=1.2,
              borderpad=0.2, labelspacing=0.22)
    ax.set_title("Reported synergy does not replicate", pad=4)
    panel_letter(ax, "c", dx=-0.20)

    fig.tight_layout(w_pad=2.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=400)
    print(f"wrote {args.out}")
    print(f"  floor lift at C{worst}: {lift:+.3f}   Pearson r: {r:+.2f}")
    print(f"  mean I: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
