"""Rebuild Fig. 4 (fig2_interaction_gap.pdf) from the released Camelyon17 results.

Replaces an AI-generated raster whose embedded labels said "center 2" while the
manuscript body is British throughout. Values are recomputed from
results_raw.csv, not copied from the old figure.

Panel a: exact 2x2 decomposition of the matched-seed change in held-out AUROC.
Panel b: in-distribution -> held-out gap per arm, with the ERM baseline marked
         non-comparable (it never reaches the ID ceiling the other arms share).
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

try:
    from figure_style import apply_figure_style, panel_letter
    apply_figure_style(sizes=(8, 7, 6))
except Exception:  # standalone: reproduce the same ladder and frame
    mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "figure.dpi": 130, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})

    def panel_letter(ax, letter, dx=-0.18, dy=1.02, **kw):
        ax.text(dx, dy, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

# Arm colours threaded to match fig5_crc / fig1_decomposition.
CLR = {"baseline": "#808080", "ssl": "#4c72b0", "sam": "#dd8452",
       "ssl_sam": "#c1272d"}
INTER_CLR = "#2b6f8a"
NICE = {"baseline": "ERM (supervised only)", "ssl": "SSL pre-training",
        "sam": "Sharpness-aware min.", "ssl_sam": "SSL + SAM"}
SHORT = {"baseline": "ERM", "ssl": "+SSL", "sam": "+SAM", "ssl_sam": "SSL+SAM"}

RES = os.environ.get("RESULTS", "results/camelyon17/results_raw.csv")
raw = pd.read_csv(RES)

# Seed 0 is the only seed completed for every arm, so the decomposition is
# matched on seed 0 throughout (see Table I note).
g = raw[raw.seed == 0].set_index("arm")
b, s, a, c = (g.loc[k, "ood_test_auroc"]
              for k in ("baseline", "ssl", "sam", "ssl_sam"))
d_ssl, d_sam = s - b, a - b
d_int = (c - b) - d_ssl - d_sam
d_net = c - b

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))

# ---- panel a: decomposition ------------------------------------------------
ax = axes[0]
terms = [("SSL\nalone", d_ssl, CLR["ssl"]),
         ("SAM\nalone", d_sam, CLR["sam"]),
         ("Interaction", d_int, INTER_CLR),
         ("Total\n(SSL+SAM)", d_net, CLR["ssl_sam"])]
for i, (lab, v, col) in enumerate(terms):
    ax.bar(i, v, width=.62, color=col, zorder=3)
    ax.annotate(f"{v:+.3f}", (i, v), textcoords="offset points",
                xytext=(0, 3 if v > 0 else -11), ha="center", fontsize=6.5)
ax.axhline(0, color="black", lw=.8, zorder=4)
ax.set_xticks(range(4))
ax.set_xticklabels([t[0] for t in terms])
ax.set_ylabel(r"$\Delta$ held-out AUROC vs ERM")
ax.set_title("The interaction carries the gain;\neach mechanism alone is negative",
             loc="left")
ax.margins(y=.22)
panel_letter(ax, "a", dx=-0.30)

# ---- panel b: ID -> held-out gap ------------------------------------------
ax = axes[1]
for arm in ("baseline", "ssl", "sam", "ssl_sam"):
    idv, ood = g.loc[arm, "id_val_auroc"], g.loc[arm, "ood_test_auroc"]
    comparable = arm != "baseline"
    ax.plot([0, 1], [idv, ood], color=CLR[arm], lw=1.6,
            ls="-" if comparable else "--",
            marker="o", ms=5, mfc=CLR[arm] if comparable else "white",
            mec=CLR[arm], mew=1.2, zorder=3, clip_on=False)
    ax.annotate(f"{SHORT[arm]}\ngap {idv - ood:.3f}", (1, ood),
                textcoords="offset points", xytext=(7, 0), va="center",
                fontsize=6.5, color=CLR[arm], linespacing=1.25)
ax.set_xlim(-.10, 1.0)
ax.set_xticks([0, 1])
ax.set_xticklabels(["In-distribution\nvalidation", "Held-out\nhospital (centre 2)"])
# The outer ticks sit on the axis edges, so centred labels drift over the
# y-tick column on the left and past the figure on the right.
ax.get_xticklabels()[0].set_ha("left")
ax.get_xticklabels()[1].set_ha("right")
ax.set_ylabel("AUROC")
ax.set_title("Only arms sharing the in-distribution\nceiling are comparable", loc="left")
ax.annotate("open marker, dashed:\nnever reached the\nin-distribution ceiling,\nso its gap is not comparable",
            (0.02, 0.03), xycoords="axes fraction", fontsize=6,
            color="#606060", va="bottom", linespacing=1.3)
ax.margins(y=.10)
panel_letter(ax, "b")

fig.subplots_adjust(right=.80, wspace=.42)
fig.savefig("fig2_interaction_gap.pdf", bbox_inches="tight")
fig.savefig("fig2_interaction_gap.png", dpi=300, bbox_inches="tight")

print({"d_ssl": round(d_ssl, 4), "d_sam": round(d_sam, 4),
       "d_int": round(d_int, 4), "d_net": round(d_net, 4),
       "gaps": {k: round(g.loc[k, "id_val_auroc"] - g.loc[k, "ood_test_auroc"], 4)
                for k in ("baseline", "ssl", "sam", "ssl_sam")},
       "id_ceiling": {k: round(g.loc[k, "id_val_auroc"], 4)
                      for k in ("ssl", "sam", "ssl_sam")}})
