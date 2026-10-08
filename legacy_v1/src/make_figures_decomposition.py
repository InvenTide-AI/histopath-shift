"""Regenerate fig1_decomposition from released results tables.

The original producing script for this figure was not preserved in the
repository (only the output PDF was). It drew error bars on the net-effect
marker whose definition was not recoverable, and which could not have been
bootstrap intervals in any case: the bootstrap resamples test-set predictions,
so it is defined only for the ensemble (seed-averaged-prediction) estimator,
whereas every value in this figure is a metric average over seeds. Intervals
are therefore omitted here and reported, on the estimator that admits them, in
Table III and Fig. 8.

Run from the repository root.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

CLR = {"ssl": "#8a7a3d", "sam": "#b3452c", "inter": "#2b6f8a"}

cam = pd.read_csv("results/camelyon17/results_raw.csv")
crc = pd.read_csv("results/nct_crc_he/crc_results_raw.csv")


def dec(df, col, seeds):
    """Exact 2x2 decomposition of the seed-matched AUROC change vs baseline."""
    g = df[df.seed.isin(seeds)].groupby("arm")[col].mean()
    b, s, a, c = g["baseline"], g["ssl"], g["sam"], g["ssl_sam"]
    return dict(ssl=s - b, sam=a - b,
                inter=(c - b) - (s - b) - (a - b), net=c - b)


ROWS = [
    ("held-out\nhospital\n(Camelyon17)", dec(cam, "ood_test_auroc", [0]), "shift"),
    ("unnormalised\nstain\n(NCT-CRC-HE)", dec(crc, "shift_nonorm_auroc", [0, 1]), "shift"),
    ("external\ncohort\n(NCT-CRC-HE)", dec(crc, "shift_external_auroc", [0, 1]), "shift"),
    ("in-distribution\n(Camelyon17)", dec(cam, "id_val_auroc", [0]), "id"),
    ("in-distribution\n(NCT-CRC-HE)", dec(crc, "id_val_auroc", [0, 1]), "id"),
]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))

# ---- panel a: stacked decomposition ----
ax = axes[0]
h = 0.24
for i, (lab, d, kind) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for j, (k, name) in enumerate([("ssl", "SSL main effect"),
                                   ("sam", "SAM main effect"),
                                   ("inter", "SSL x SAM interaction")]):
        ax.barh(y + (1 - j) * h, d[k], height=h, color=CLR[k],
                label=name if i == 0 else None)
    ax.plot([d["net"]], [y - h], marker="|", ms=11, mew=1.6, color="black",
            label="net SSL+SAM effect" if i == 0 else None)

ax.axvline(0, color="black", lw=0.9)
ax.axhline(1.5, color="grey", lw=0.7, ls=":")
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[0] for r in ROWS][::-1])
ax.set_xlabel("Contribution to AUROC change vs baseline")
ax.set_title("Each mechanism alone hurts under shift;\ntheir interaction is what recovers",
             loc="left")
ax.text(0.97, 0.955, "shift", transform=ax.transAxes, ha="right",
        style="italic", color=CLR["inter"], fontsize=7.5)
ax.text(0.97, 0.30, "in-distribution", transform=ax.transAxes, ha="right",
        style="italic", color="grey", fontsize=7.5)
ax.legend(loc="lower right", frameon=False, fontsize=7)

# ---- panel b: interaction vs cost of the two mechanisms alone ----
ax = axes[1]
ANNOT = {  # (dx, dy, ha, va) offsets tuned to avoid the diagonal and each other
    "held-out\nhospital\n(Camelyon17)": (-0.004, 0.012, "right", "bottom"),
    "unnormalised\nstain\n(NCT-CRC-HE)": (-0.006, 0.008, "right", "bottom"),
    "external\ncohort\n(NCT-CRC-HE)": (0.010, 0.006, "left", "bottom"),
    "in-distribution\n(Camelyon17)": (0.008, -0.004, "left", "top"),
    "in-distribution\n(NCT-CRC-HE)": (0.010, -0.010, "left", "top"),
}
for lab, d, kind in ROWS:
    cost = -(d["ssl"] + d["sam"])
    col = CLR["inter"] if "Camelyon" in lab and kind == "shift" else (
        CLR["sam"] if kind == "shift" else "grey")
    ax.plot(cost, d["inter"], marker="o" if "Camelyon" in lab else "s",
            ms=6, color=col, ls="none")
    dx, dy, ha, va = ANNOT[lab]
    ax.annotate(lab.replace("\n", " ").replace("(", "\n("), (cost + dx, d["inter"] + dy),
                ha=ha, va=va, fontsize=7, color=col, linespacing=1.15)
lim = (-0.115, 0.205)
ax.plot(lim, lim, ls="--", color="grey", lw=0.9)
ax.fill_between(lim, lim, lim[1], color=CLR["inter"], alpha=0.07)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Cost of the two mechanisms alone\n$-$[SSL + SAM main effects]")
ax.set_ylabel("SSL x SAM interaction")
ax.set_title("Net benefit requires the interaction\nto exceed that cost", loc="left")
ax.text(0.04, 0.90, "interaction > cost\n$\\rightarrow$ net gain", transform=ax.transAxes,
        style="italic", color=CLR["inter"], fontsize=7.5)

fig.tight_layout()
fig.savefig("paper/fig1_decomposition.pdf", bbox_inches="tight")
fig.savefig("paper/fig1_decomposition.png", bbox_inches="tight")
print({lab.replace(chr(10), " "): {k: round(v, 4) for k, v in d.items()}
       for lab, d, _ in ROWS})
