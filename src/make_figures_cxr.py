"""Figure 6: cross-institution chest radiography, the pool contrast.

Panels
  A  OOD (CheXpert) AUROC by arm, with bootstrap CIs; ID test as open markers
  B  interaction decomposition, pool S vs pool M side by side
  C  the registered H2 contrast: Delta(M) vs Delta(S) with the difference CI
  D  sharpness vs OOD AUROC across checkpoints

Panel titles are written only after reading the data, and state what the data
shows rather than what was predicted.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# palette threaded from the shipped Camelyon17 figures (sampled from raster)
CLR = {"baseline": "#8c8c8c", "sam": "#dd8452",
       "ssl_S": "#7fb3d5", "ssl_M": "#4c72b0",
       "ssl_S_sam": "#e08a8a", "ssl_M_sam": "#c1272d"}
LBL = {"baseline": "supervised", "sam": "SAM",
       "ssl_S": "SSL (pool S)", "ssl_M": "SSL (pool M)",
       "ssl_S_sam": "SSL+SAM (S)", "ssl_M_sam": "SSL+SAM (M)"}
ORDER = ["baseline", "sam", "ssl_S", "ssl_S_sam", "ssl_M", "ssl_M_sam"]


def panel_letter(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def main():
    ci = pd.read_csv("cxr_bootstrap_ci.csv")
    inter = pd.read_csv("cxr_interaction.csv")
    h2 = pd.read_csv("cxr_h2_contrast.csv")
    flat = pd.read_csv("cxr_flatness.csv") if os.path.exists("cxr_flatness.csv") else None

    arms = [a for a in ORDER if a in set(ci.arm)]
    fig = plt.figure(figsize=(11.4, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=.52, wspace=.30,
                          left=.085, right=.985, top=.91, bottom=.10)

    # ---- A: OOD AUROC by arm
    ax = fig.add_subplot(gs[0, 0])
    o = ci[ci.split == "ood"].set_index("arm")
    i_ = ci[ci.split == "id_test"].set_index("arm")
    for k, a in enumerate(arms):
        r = o.loc[a]
        ax.errorbar(k, r.auroc, yerr=[[r.auroc - r.ci_lo], [r.ci_hi - r.auroc]],
                    fmt="o", ms=8, color=CLR[a], capsize=4, lw=2, zorder=3)
        if a in i_.index:
            ax.plot(k, i_.loc[a].auroc, marker="o", ms=7, mfc="none",
                    mec=CLR[a], mew=1.6, zorder=2)
    base = o.loc["baseline"].auroc
    ax.axhline(base, color="#8c8c8c", ls="--", lw=1, zorder=1)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LBL[a] for a in arms], rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("AUROC")
    ax.plot([], [], "o", color="k", ms=7, label="CheXpert (OOD)")
    ax.plot([], [], "o", mfc="none", mec="k", ms=7, label="NIH (ID)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    panel_letter(ax, "A")

    # ---- B: interaction, both pools
    ax = fig.add_subplot(gs[0, 1])
    terms = ["ssl_alone", "sam_alone", "interaction", "total"]
    w = 0.36
    for j, pool in enumerate(["S", "M"]):
        sub = inter[(inter.pool == pool) & (inter.split == "ood")].set_index("term")
        if sub.empty:
            continue
        xs = np.arange(len(terms)) + (j - 0.5) * w
        vals = [sub.loc[t].value for t in terms]
        err = [[sub.loc[t].value - sub.loc[t].ci_lo for t in terms],
               [sub.loc[t].ci_hi - sub.loc[t].value for t in terms]]
        ax.bar(xs, vals, width=w, color=CLR["ssl_S_sam" if pool == "S" else "ssl_M_sam"],
               label=f"pool {pool}", zorder=2)
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor="k", lw=1.1,
                    capsize=3, zorder=3)
    ax.axhline(0, color="k", lw=.9)
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(["SSL\nalone", "SAM\nalone", "inter-\naction", "total"], fontsize=8.5)
    ax.set_ylabel(r"$\Delta$ AUROC (OOD)")
    ax.legend(frameon=False, fontsize=8)
    panel_letter(ax, "B")

    # ---- C: the registered H2 contrast
    ax = fig.add_subplot(gs[1, 0])
    if not h2.empty:
        r = h2[h2.split == "ood"].iloc[0]
        ax.bar([0, 1], [r.delta_S, r.delta_M], width=.55,
               color=[CLR["ssl_S_sam"], CLR["ssl_M_sam"]], zorder=2)
        ax.errorbar([1.5], [r.difference],
                    yerr=[[r.difference - r.ci_lo], [r.ci_hi - r.difference]],
                    fmt="D", ms=8, color="#7d4f9c", capsize=4, lw=2, zorder=3)
        ax.axhline(0, color="k", lw=.9)
        ax.set_xticks([0, 1, 1.5])
        ax.set_xticklabels([r"$\Delta$ pool S", r"$\Delta$ pool M", "M $-$ S"],
                           fontsize=8.5)
        ax.set_ylabel(r"$\Delta$ AUROC vs supervised")
        ax.annotate(f"p = {r.p_two_sided:.3g}", xy=(1.5, r.difference),
                    xytext=(6, 8), textcoords="offset points", fontsize=8)
    panel_letter(ax, "C")

    # ---- D: sharpness vs transfer
    ax = fig.add_subplot(gs[1, 1])
    if flat is not None and not flat.empty:
        m = flat.merge(ci[ci.split == "ood"][["arm", "auroc"]], on="arm", how="left")
        for _, r in m.iterrows():
            if pd.isna(r.auroc):
                continue
            ax.scatter(r.adv_sharp_rho0_05, r.auroc, s=70,
                       color=CLR.get(r.arm, "#555"), zorder=3)
            ax.annotate(LBL.get(r.arm, r.arm), (r.adv_sharp_rho0_05, r.auroc),
                        xytext=(5, 4), textcoords="offset points", fontsize=7.5)
        ax.set_xlabel(r"adversarial sharpness ($\rho{=}0.05$)")
        ax.set_ylabel("OOD AUROC")
        if len(m.dropna(subset=["auroc"])) > 2:
            from scipy.stats import spearmanr
            rho, p = spearmanr(m.adv_sharp_rho0_05, m.auroc, nan_policy="omit")
            ax.set_title(rf"Spearman $\rho$ = {rho:.2f} (p = {p:.2f})",
                         loc="left", fontsize=9)
    panel_letter(ax, "D")

    for ext in ("png", "pdf"):
        fig.savefig(f"fig6_cxr.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
    print("wrote fig6_cxr.png / .pdf")


if __name__ == "__main__":
    main()
