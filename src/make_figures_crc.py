"""Build the NCT-CRC-HE replication figure (Fig. 5) and its LaTeX table.

Panels:
  a  per-arm AUROC on both shift axes, bootstrap CIs
  b  2x2 interaction decomposition on the primary (stain-shift) axis
  c  in-distribution-to-shift AUROC gap per arm
  d  adversarial sharpness vs shifted AUROC across checkpoints

Colour is threaded per ARM across every panel, matching the Camelyon17 figures.
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["baseline", "ssl", "sam", "ssl_sam"]
NICE = {"baseline": "Supervised only", "ssl": "SSL pre-training",
        "sam": "Sharpness-aware min.", "ssl_sam": "SSL + SAM"}
SHORT = {"baseline": "Sup.", "ssl": "SSL", "sam": "SAM", "ssl_sam": "SSL+SAM"}
# Threaded colours, sampled from the shipped Camelyon17 figures so the two
# datasets' panels are readable side by side (arm -> colour is invariant).
CLR = {"baseline": "#808080", "ssl": "#4c72b0", "sam": "#dd8452",
       "ssl_sam": "#c1272d"}
INTER_CLR = "#7c3aed"   # interaction term, as in fig2
AXLAB = {"shift_nonorm": "Stain shift (no colour norm.)",
         "shift_external": "External patient cohort"}

try:
    from figure_style import apply_figure_style, panel_letter
    apply_figure_style(sizes=(8, 7, 6))
except Exception:
    mpl.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150,
                         "savefig.dpi": 300, "axes.linewidth": .6})

    def panel_letter(ax, s, **k):
        ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="right")

ci = pd.read_csv("crc_bootstrap_ci.csv")
inter = pd.read_csv("crc_interaction.csv")
summ = pd.read_csv("crc_results_summary.csv", index_col=0)
raw = pd.read_csv("crc_results_raw.csv")
sharp = (pd.read_csv("crc_flatness.csv")
         if os.path.exists("crc_flatness.csv") else None)

fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 2, hspace=.52, wspace=.34,
                      left=.105, right=.985, top=.91, bottom=.10)

# ---------------- (a) AUROC per arm on both axes ----------------
ax = fig.add_subplot(gs[0, 0])
axes_order = ["shift_nonorm", "shift_external"]
w = 0.36
for j, axis in enumerate(axes_order):
    d = ci[(ci.axis == axis) & (ci.metric == "auroc")].set_index("arm")
    for i, arm in enumerate(ARMS):
        if arm not in d.index:
            continue
        r = d.loc[arm]
        x = i + (j - 0.5) * w
        ax.plot([x, x], [r.ci_lo, r.ci_hi], color=CLR[arm], lw=1.1,
                solid_capstyle="round", zorder=2)
        ax.plot(x, r.observed, marker="o" if j == 0 else "s", ms=4.6,
                mfc=CLR[arm] if j == 0 else "white", mec=CLR[arm], mew=1.2,
                zorder=3)
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([SHORT[a] for a in ARMS])
ax.set_ylabel("AUROC under shift")
# Panel (a) plots BOTH shift axes, and on the external-cohort axis SAM alone
# does edge past the supervised baseline (+0.002 ensemble, interval spanning
# zero). "No arm beats supervised training" is therefore true of the stain axis
# only; state the axis rather than overclaiming across both.
ax.set_title("Stain axis: no arm beats supervised", loc="left")
ax.margins(x=.12, y=.16)
h = [plt.Line2D([], [], marker="o", ls="none", mfc="#555", mec="#555", ms=4.6,
                label=AXLAB["shift_nonorm"]),
     plt.Line2D([], [], marker="s", ls="none", mfc="white", mec="#555", ms=4.6,
                label=AXLAB["shift_external"])]
ax.legend(handles=h, frameon=False, fontsize=6, loc="lower left",
          handletextpad=.4, borderpad=.2)
panel_letter(ax, "a")

# ---------------- (b) interaction decomposition ----------------
ax = fig.add_subplot(gs[0, 1])
d = inter[(inter.axis == "shift_nonorm") & (inter.metric == "auroc")]
terms = ["ssl_alone", "sam_alone", "interaction", "total"]
tlab = ["SSL\nalone", "SAM\nalone", "Inter-\naction", "Combined\ntotal"]
tclr = [CLR["ssl"], CLR["sam"], INTER_CLR, CLR["ssl_sam"]]
d = d.set_index("term")
for i, t in enumerate(terms):
    if t not in d.index:
        continue
    r = d.loc[t]
    ax.bar(i, r.observed, .62, color=tclr[i],
           edgecolor="none", zorder=2)
    ax.plot([i, i], [r.ci_lo, r.ci_hi], color="#222", lw=1.0, zorder=3)
ax.axhline(0, color="#444", lw=.7, zorder=1)
ax.set_xticks(range(len(terms)))
ax.set_xticklabels(tlab)
ax.set_ylabel("$\\Delta$ AUROC vs supervised")
ax.set_title("Decomposition on the stain-shift axis", loc="left")
ax.margins(y=.20)
panel_letter(ax, "b")

# ---------------- (c) ID-to-shift gap ----------------
ax = fig.add_subplot(gs[1, 0])
for j, axis in enumerate(axes_order):
    col = f"gap_{axis}_auroc"
    if col not in summ.columns:
        continue
    for i, arm in enumerate(ARMS):
        if arm not in summ.index:
            continue
        x = i + (j - 0.5) * w
        ax.plot([x, x], [0, summ.loc[arm, col]], color=CLR[arm], lw=1.0,
                zorder=2)
        ax.plot(x, summ.loc[arm, col], marker="o" if j == 0 else "s", ms=4.6,
                mfc=CLR[arm] if j == 0 else "white", mec=CLR[arm], mew=1.2,
                zorder=3)
ax.axhline(0, color="#444", lw=.7)
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([SHORT[a] for a in ARMS])
ax.set_ylabel("ID $-$ shifted AUROC")
ax.set_title("Every intervention widens the stain-shift gap", loc="left")
ax.margins(x=.12, y=.18)
panel_letter(ax, "c")

# ---------------- (d) sharpness vs shifted AUROC ----------------
ax = fig.add_subplot(gs[1, 1])
if sharp is not None and len(sharp):
    m = sharp.merge(raw[["arm", "seed", "shift_nonorm_auroc"]],
                    on=["arm", "seed"], how="left")
    for arm in ARMS:
        s = m[m.arm == arm]
        if not len(s):
            continue
        ax.plot(s["sharpness_rho0.05"], s["shift_nonorm_auroc"], marker="o",
                ls="none", ms=5, mfc=CLR[arm], mec="white", mew=.8,
                label=NICE[arm], zorder=3)
    if m["sharpness_rho0.05"].notna().sum() >= 3:
        x = m["sharpness_rho0.05"].values
        y = m["shift_nonorm_auroc"].values
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() >= 3:
            b, a0 = np.polyfit(x[ok], y[ok], 1)
            xs = np.linspace(x[ok].min(), x[ok].max(), 50)
            ax.plot(xs, a0 + b * xs, color="#666", lw=.9, ls="--", zorder=2)
            # Report the same statistic the caption and body text quote: a
            # Spearman rank correlation. n = 8 checkpoints is too small for a
            # Pearson coefficient to be robust to a single outlying arm, and
            # the earlier version of this panel annotated Pearson r while the
            # caption quoted Spearman rho -- a mismatch, not a second result.
            rho, p_rho = spearmanr(x[ok], y[ok])
            ax.text(.97, .95,
                    f"$\\rho$ = {rho:+.2f}  ($p$ = {p_rho:.2f}, n = {int(ok.sum())})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax.set_xlabel("Adversarial sharpness ($\\rho$ = 0.05)")
    ax.set_ylabel("Stain-shift AUROC")
    ax.legend(frameon=False, fontsize=6, loc="lower left", handletextpad=.3,
              borderpad=.2)
else:
    ax.text(.5, .5, "sharpness not computed", ha="center", va="center",
            transform=ax.transAxes, fontsize=7, color="#888")
    ax.set_xticks([])
    ax.set_yticks([])
ax.set_title("Flatness does not predict transfer here", loc="left")
ax.margins(.06)
panel_letter(ax, "d")

for e in ("pdf", "png"):
    fig.savefig(f"fig5_crc.{e}", bbox_inches="tight")

# ---------------- geometric overlap check (figure-style 9.1) ----------------
r_ = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
ov = [(a.get_text(), b.get_text())
      for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
      if ba.overlaps(bb)]
print("text overlaps:", ov[:6], "count", len(ov))

# ---------------- LaTeX table ----------------
rows = []
for arm in ARMS:
    if arm not in summ.index:
        continue
    s = summ.loc[arm]
    rows.append(
        f"{NICE[arm]} & "
        f"{s['id_val_auroc_mean']:.3f} $\\pm$ {s['id_val_auroc_sd']:.3f} & "
        f"{s['shift_nonorm_auroc_mean']:.3f} $\\pm$ {s['shift_nonorm_auroc_sd']:.3f} & "
        f"{s['shift_external_auroc_mean']:.3f} $\\pm$ {s['shift_external_auroc_sd']:.3f} & "
        f"{s['shift_nonorm_ece_mean']:.3f} & "
        f"{s['gap_shift_nonorm_auroc']:.3f} \\\\")
open("tab_crc.tex", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
