"""Regenerate fig_splits (Figure 1) for the multi-hospital section.

Reads results/center_population.csv, results/split_composition.csv and
results/patient_overlap.csv; writes paper/fig_splits.{png,pdf}.

Run from the repo root:  python src/make_figures_hospital.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.titlelocation": "left",
    "axes.titleweight": "regular",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

META_GREY = "#888888"

def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    base, annot, tick = sizes
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "legend.fontsize": annot,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.titlelocation": "left",
        "axes.titleweight": "regular",
        "axes.spines.top": frame != "open",
        "axes.spines.right": frame != "open",
        "axes.grid": grid,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if font:
        mpl.rcParams["font.family"] = font


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    s = letter.lower() if case == "lower" else letter.upper()
    return ax.text(
        dx, dy, s, transform=ax.transAxes,
        fontsize=fontsize or mpl.rcParams["font.size"] + 1,
        fontweight="bold", va="bottom", ha="left",
    )


apply_figure_style()

pop = pd.read_csv(RESULTS / "center_population.csv")
pop["center"] = pop["center"].astype(str)
comp = pd.read_csv(RESULTS / "split_composition.csv")
comp["holdout"] = comp["holdout"].astype(str)
ov = pd.read_csv(RESULTS / "patient_overlap.csv", index_col=0)
ov.index = ov.index.astype(str)
ov.columns = ov.columns.astype(str)
cs = pop["center"].tolist()

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.1))
ax = axes[0]
xx = np.arange(len(cs))
ax.bar(xx, pop.patches/1000, color=META_GREY, width=0.62)
for i,(p,pt) in enumerate(zip(pop.patches, pop.patients)):
    ax.text(i, p/1000 + 3.5, f"{pt}p", ha="center", va="bottom", fontsize=6)
ax.set_xticks(xx); ax.set_xticklabels([f"H{c}" for c in cs])
ax.set_xlabel("Hospital"); ax.set_ylabel("Patches available (thousands)")
ax.set_title("Hospitals differ 4.2-fold in size\n(patient count above each bar)")
ax.margins(y=0.16)

ax = axes[1]
w = 0.26
for k,(s,lab,col) in enumerate([("train","train","#2f6f9f"),("id_val","in-dist. val","#8fb8d8"),("test","held-out hospital","#c0504d")]):
    d = comp[comp.split==s].set_index("holdout").loc[cs]
    ax.bar(xx + (k-1)*w, d.n/1000, width=w, label=lab, color=col)
ax.set_xticks(xx); ax.set_xticklabels([f"H{c}" for c in cs])
ax.set_xlabel("Held-out hospital"); ax.set_ylabel("Patches used (thousands)")
ax.set_title("Every fold uses identical split sizes\nand 50/50 class balance")
ax.legend(frameon=False, fontsize=6, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=0.9,
              handlelength=1.1, handletextpad=0.4)
ax.margins(y=0.20)

ax = axes[2]
im = ax.imshow(ov.loc[cs, cs].values, cmap="Reds", vmin=0, vmax=max(1, ov.values.max()))
for i in range(len(cs)):
    for j in range(len(cs)):
        v = int(ov.loc[cs[i], cs[j]])
        ax.text(j, i, v, ha="center", va="center", fontsize=7,
                color="white" if v > ov.values.max()*0.6 else "black",
                fontweight="bold" if i==j else "normal")
ax.set_xticks(range(len(cs))); ax.set_xticklabels([f"H{c}" for c in cs])
ax.set_yticks(range(len(cs))); ax.set_yticklabels([f"H{c}" for c in cs])
ax.set_xlabel("Test patients from hospital"); ax.set_ylabel("Fold holding out hospital")
ax.set_title("Zero diagonal: no patient is in both\nthe training and test set of a fold")
for k,(a,l) in enumerate(zip(axes,"abc")): panel_letter(a, l)

fig.subplots_adjust(wspace=0.34)
fig.tight_layout(w_pad=2.6)
fig.savefig(PAPER / "fig_splits.pdf", bbox_inches="tight")
fig.savefig(PAPER / "fig_splits.png", dpi=300, bbox_inches="tight")