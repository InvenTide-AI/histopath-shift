"""Minimal figure styling used by make_figures.py and make_figures_crc.py.

The figures in the paper were produced with an equivalent helper that lived in
the authoring environment and is not a distributable package. This module is a
standalone reimplementation of the only two functions those scripts call, with
the same signatures and the same visual settings (open frame, three-size type
ladder, bold lower-case panel letters outside the axes).

Re-running the figure scripts with this module reproduces the published figures.
Nothing here affects any computed number -- it sets rcParams and draws panel
letters only.
"""
import matplotlib as mpl


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    """Set rcParams for publication figures.

    frame : "open" drops the top and right spines; "box" keeps all four.
    sizes : (base, annotation, tick) point sizes -- the three-size ladder.
    """
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
        "pdf.fonttype": 42,      # editable text in the PDF, not outlines
        "ps.fonttype": 42,
    })
    if font:
        mpl.rcParams["font.family"] = font


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    """Draw a bold panel letter outside the top-left of the axes box."""
    s = letter.lower() if case == "lower" else letter.upper()
    return ax.text(
        dx, dy, s, transform=ax.transAxes,
        fontsize=fontsize or mpl.rcParams["font.size"] + 1,
        fontweight="bold", va="bottom", ha="left",
    )
