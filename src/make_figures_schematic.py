"""Rebuild Fig. 1 (fig4_schematic.pdf) -- the mechanism / closure-condition schematic.

Replaces an AI-generated raster with overlapping annotations, American spelling
("centers 0, 3, 4", "Center 2") and notation that disagreed with Definition 1:
it used X_site / X_signal where the definition uses eta for site nuisance, and
labelled the conditions C1 / C2 where the body calls them condition (i) / (ii).

Nothing here is data -- it is a diagram, drawn with explicit coordinates so the
boxes and leaders cannot collide.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    from figure_style import apply_figure_style
    apply_figure_style(sizes=(8, 7, 6))
except Exception:
    mpl.rcParams.update({"font.size": 8, "figure.dpi": 130, "savefig.dpi": 300})

BLUE, ORANGE, RED, GREY = "#4c72b0", "#dd8452", "#c1272d", "#808080"
META = "#606060"

fig, ax = plt.subplots(figsize=(7.1, 3.05))
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, text, ec, fc="white", fs=7, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.35, fontweight=weight)


def arrow(x0, y0, x1, y1, color="black"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=1.0,
                                 color=color, zorder=1,
                                 shrinkA=2, shrinkB=2))


# ---- top row: the mechanism -------------------------------------------------
ax.text(0, 43.4, "Training hospitals (centres 0, 3, 4)", fontsize=7.5,
        color=META, ha="left", va="bottom")

box(0, 31.5, 21, 8.4, "Morphology\n$X_{\\mathrm{signal}}$", BLUE)
box(0, 19.5, 21, 8.4, "Scanner / stain\nnuisance $\\eta_A$", ORANGE)
box(31, 25.5, 17, 8.4, "ERM\nobjective", GREY)
box(58, 25.5, 19, 8.4, "Learned\nrepresentation", RED, fc="#fdf2f2")
box(87, 25.5, 13, 8.4, "Centre 2\n(unseen)\n$\\eta_B$", "black")

arrow(21.6, 35.7, 30.4, 31.5)
arrow(21.6, 23.7, 30.4, 28.0)
arrow(48.6, 29.7, 57.4, 29.7)
arrow(77.6, 29.7, 86.4, 29.7)

# Clear band between the grey condition panel (top edge y=12) and the
# nuisance box (bottom edge y~18.9).
ax.text(26.0, 15.4, "both predict\nthe label", fontsize=6, color=META,
        ha="center", va="center", linespacing=1.3)
ax.text(53.3, 37.4, "no pressure\nto separate them", fontsize=6, color=RED,
        ha="center", va="center", linespacing=1.3)
ax.text(93.5, 20.6, "shift $\\eta_A\\!\\rightarrow\\!\\eta_B$\n$\\Rightarrow$ accuracy falls",
        fontsize=6, color=RED, ha="center", va="center", linespacing=1.3)

# ---- bottom row: closure conditions ----------------------------------------
ax.add_patch(FancyBboxPatch((0, 0.5), 100, 11.5,
                            boxstyle="round,pad=0.4,rounding_size=1.0",
                            linewidth=0, facecolor="#f4f4f4", zorder=0))
ax.text(1.6, 10.4, "Closure conditions (Definition 1) and the mechanism assigned to each",
        fontsize=7, color="black", ha="left", va="center")

box(1.6, 1.6, 46, 6.4,
    "(i) Unlearnability: $\\eta_A$ made uninformative\n"
    "$\\rightarrow$ SSL pre-training (Pillar I)", BLUE, fc="white", fs=6.5)
box(52.4, 1.6, 46, 6.4,
    "(ii) Robustness: stable under $\\eta_A\\!\\rightarrow\\!\\eta_B$\n"
    "$\\rightarrow$ SAM fine-tuning (Pillar II)", ORANGE, fc="white", fs=6.5)

fig.savefig("fig4_schematic.pdf", bbox_inches="tight")
fig.savefig("fig4_schematic.png", dpi=300, bbox_inches="tight")
print("schematic written")
