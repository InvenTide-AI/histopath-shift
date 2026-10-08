#!/usr/bin/env python
"""make_figures_v2 -- figures of the MedIA revision from analyze_v2 outputs (+ raw v2 records).

Usage
-----
module load PyTorch/2.7.0-CONDA
python src/v2/make_figures_v2.py --analysis results/v2/analysis --runs $V2 --out paper/media/figs --tier both
# one frozen manifest for every statistic and footnote (copies the records, runs analyze_v2 for all 4 rules):
python src/v2/make_figures_v2.py --runs $V2 --freeze /tmp/snap_YYYYMMDD --analysis results/v2/analysis --out ...

Inputs
  --analysis  output tree of analyze_v2.py ``runs``: {analysis}/{tier}/{cohort}/{kind}_{sel}.{csv,json}
              and {analysis}/{tier}/cross/{kendall,interaction,clustering}_{sel}.*
  --runs      v2 root ($V2): runs/{tier}/{cohort}/fold*/*.json(+_preds.npz), runs_tune/..., cache64/*/manifest.json
  --domainbed results/v2/external/domainbed_icc.csv (analyze_v2.py domainbed)

Every figure is drawn from whatever exists (missing tiers / cohorts / arms are left blank or
skipped) and carries a muted footnote with the runs / folds / complete arms the analysis actually
used (from per_run_*.csv and meta_*.json) while the record set is incomplete.  Drawing stops if the
analysis files of different selection rules come from different record sets (see --freeze).
Each figure is written as vector PDF + 300-dpi PNG at a fixed 6.5-in width (height cropped to the
content); a layout check reports text-on-text / text-under-legend overlaps, text below 7 pt and
content outside the canvas (exit code 3).

Figures
  F1 fig_design         2x2 task x shift design, LODO roles, 10-checkpoint selection rules (schematic)
  F2 fig_perdomain_{T}  heatmap arm x domain of seed-mean test AUROC, worst domain outlined
  F3 fig_ranks_{T}      rank by worst-domain AUROC with bootstrap rank intervals + bump chart
  F4 fig_variance_{T}   ICC(1) forest (exact-F CI, Bayesian CrI) + sigma_domain vs sigma_run
  F5 fig_power_{T}      domains needed for 80% power vs delta, s in {1,3,5}, K_eff (rho=.5) dashed
  F6 fig_scale          compact vs standard tier vs foundation models (worst / mean AUROC)
  F7 fig_selection_{T}  gain over ERM under ID-val / OOD-val / last / oracle selection
  F8 fig_kendall_{T}    Kendall tau-b between cohort rankings + 2x2 decomposition of between-cohort SS
  S1 fig_calibration_{T} ECE per arm + reliability curves at each cohort's hardest domain
  S2 fig_domainbed      ICC with CIs for DomainBed algorithms x datasets
  S3 fig_tuning_{T}     tuning sensitivity: worst-domain AUROC vs hyperparameter (runs_tune)
  S4 fig_seedspread     every seed per domain for ERM, SimCLR, H&E jitter, best standard-tier arm
  S5 fig_sensitivity_{T} MIDOG 2021 scale-normalized (primary) vs un-normalized sensitivity cohort (midog21)
  S6 fig_ladder         representation ladder: frozen linear probes by backbone and pre-training corpus (tier S)
  GA graphical_abstract 2x2 design + rank instability + variance-decomposition icon

Style: dataviz method (fixed-order family hues validated with validate_palette.js, light surface:
adjacent CVD dE >= 9.1, normal-vision >= 19.6; aqua / yellow / magenta are < 3:1 so every arm is
labelled directly, never identified by colour alone); sequential = one blue ramp; diverging =
blue <-> red with a gray midpoint; hairline recessive axes; text always in ink, never series colour.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
V2 = os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "v2")

try:
    import stats_v2 as S  # noqa: E402
except Exception:  # noqa: BLE001
    S = None
try:
    import make_plan as _MP  # noqa: E402
    TIER_ARMS = {"C": list(_MP.TIER_C_ARMS), "S": list(_MP.TIER_S_ARMS)}
    TTA = {t: dict(v) for t, v in _MP.TTA.items()}
    SEEDS = {t: list(v) for t, v in _MP.SEEDS.items()}
    GRID = {k: list(v) for k, v in _MP.GRID.items()}
except Exception:  # noqa: BLE001  (fallback copy of make_plan.py as of 2026-10-05)
    TIER_ARMS = {"C": ["erm", "sam", "simclr", "simclr_sam", "augonly", "compute_matched", "groupdro", "irm", "coral",
                       "mixstyle", "fish", "lisa", "macenko", "he_jitter", "tia_style", "macenko_tile"],
                 "S": ["erm", "sam", "groupdro", "irm", "coral", "mixstyle", "fish", "lisa", "macenko", "he_jitter",
                       "tia_style", "simclr", "erm_lunit_bt", "he_jitter_lunit_bt", "erm_lunit_dino",
                       "he_jitter_lunit_dino", "probe_resnet50_imagenet", "probe_resnet50_lunit_bt",
                       "probe_vit_s16_lunit_dino", "macenko_tile"]}
    TTA = {"C": {"erm": ["bnadapt", "tent"], "simclr": ["bnadapt"], "he_jitter": ["bnadapt"]}, "S": {"erm": ["bnadapt"]}}
    SEEDS = {"C": [0, 1, 2, 3, 4], "S": [0, 1, 2]}
    GRID = {}

# ----------------------------------------------------------------------------------------
# cohorts, domains, arms
# ----------------------------------------------------------------------------------------

COHORTS = ["c17", "canine", "midog21sn", "midogpp"]          # 2x2 order: tumour (inst, acq), mitosis (acq, inst)
K_EXPECTED = {"c17": 5, "canine": 5, "midog21sn": 3, "midogpp": 7, "midog21": 3}
COHORT_LABEL = {"c17": "Camelyon17", "canine": "Canine cSCC", "midog21sn": "MIDOG 2021", "midogpp": "MIDOG++",
                "midog21": "MIDOG 2021 un-normalized"}
COHORT_META = {"c17": dict(task="tumor", shift="institutional"), "canine": dict(task="tumor", shift="acquisition"),
               "midog21sn": dict(task="mitosis", shift="acquisition"), "midog21": dict(task="mitosis", shift="acquisition"),
               "midogpp": dict(task="mitosis", shift="institutional and biological")}
DOMAINS_FALLBACK = {"c17": [f"center{i}" for i in range(5)], "canine": ["cs2", "p1000", "nz20", "nz210", "gt450"],
                    "midog21sn": ["HamamatsuXR", "HamamatsuS360", "AperioCS2"],
                    "midog21": ["HamamatsuXR", "HamamatsuS360", "AperioCS2"],
                    "midogpp": ["canine cutaneous mast cell tumor", "canine lung cancer", "canine lymphosarcoma",
                                "canine soft tissue sarcoma", "human breast cancer", "human melanoma",
                                "human neuroendocrine tumor"]}
DOMAIN_SHORT = {"HamamatsuXR": "XR", "HamamatsuS360": "S360", "AperioCS2": "CS2",
                "canine cutaneous mast cell tumor": "MCT (c)", "canine lung cancer": "lung (c)",
                "canine lymphosarcoma": "lymph. (c)", "canine soft tissue sarcoma": "STS (c)",
                "human breast cancer": "breast (h)", "human melanoma": "melan. (h)",
                "human neuroendocrine tumor": "NET (h)"}


def dshort(d, multiline=False):
    d = str(d)
    if d in DOMAIN_SHORT:
        v = DOMAIN_SHORT[d]
        return v.replace(" (", "\n(") if multiline else v
    m = re.match(r"center(\d+)$", d)
    return f"H{m.group(1)}" if m else d


COHORT_SHORT = {"c17": "C17", "canine": "Canine", "midog21sn": "M21", "midogpp": "M++", "midog21": "M21raw"}
SELS = ["id", "ood", "last", "oracle"]
SEL_LABEL = {"id": "ID-val", "ood": "OOD-val", "last": "Last", "oracle": "Oracle"}

# families: 0 = reference (neutral gray), 1..6 = fixed-order categorical slots (validated set)
FAMILY_NAME = {0: "Reference", 1: "DG objectives", 2: "Stain & color", 3: "Self-supervision", 4: "Sharpness",
               5: "Test-time adaptation", 6: "Pretrained representations"}
FAMILY_COLOR = {0: "#6b6a66", 1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4", 6: "#008300"}
ARM_FAMILY = {"erm": 0, "compute_matched": 0,
              "groupdro": 1, "irm": 1, "coral": 1, "mixstyle": 1, "fish": 1, "lisa": 1,
              "macenko": 2, "macenko_tile": 2, "he_jitter": 2, "tia_style": 2, "augonly": 2,
              "simclr": 3, "simclr_sam": 3, "sam": 4}
ARM_LABEL = {"erm": "ERM", "compute_matched": "Compute-matched ERM", "groupdro": "GroupDRO", "irm": "IRM",
             "coral": "CORAL", "mixstyle": "MixStyle", "fish": "Fish", "lisa": "LISA", "macenko": "Macenko",
             "macenko_tile": "Macenko (per tile)",
             "he_jitter": "H&E jitter", "tia_style": "TIA-style", "augonly": "Aug-only control", "simclr": "SimCLR",
             "simclr_sam": "SimCLR+SAM", "sam": "SAM", "erm_lunit_bt": "Lunit BT FT",
             "he_jitter_lunit_bt": "Lunit BT FT + H&E jitter", "erm_lunit_dino": "Lunit DINO FT",
             "he_jitter_lunit_dino": "Lunit DINO FT + H&E jitter", "probe_resnet50_imagenet": "RN50-IN probe",
             "probe_resnet50_lunit_bt": "Lunit BT probe", "probe_vit_s16_lunit_dino": "Lunit DINO probe",
             "probe_resnet18_imagenet": "RN18-IN probe", "probe_convnext_tiny_imagenet": "ConvNeXt-T-IN probe",
             "probe_vit_b16_imagenet": "ViT-B/16-IN probe", "probe_vit_b16_swag": "ViT-B/16 SWAG probe",
             "probe_resnet50_lunit_swav": "Lunit SwAV probe", "probe_resnet50_lunit_mocov2": "Lunit MoCo probe"}
TTA_LABEL = {"bnadapt": "BN-adapt", "tent": "TENT"}
# transductive arms (use the unlabeled images of the test domain at prediction time): Macenko, TIA-style and every
# TTA pseudo-arm (bnadapt@*, tent@*).  Marked with a dagger wherever an arm is named; save() adds the note.
TRANSDUCTIVE = {"macenko", "tia_style"}
DAGGER = "\u2020"
TRANSDUCTIVE_NOTE = DAGGER + " Transductive: uses the unlabeled images of the test domain at prediction time."
ARM_ORDER = (["erm", "compute_matched", "groupdro", "irm", "coral", "mixstyle", "fish", "lisa", "macenko",
              "macenko_tile", "he_jitter",
              "tia_style", "augonly", "simclr", "simclr_sam", "sam"]
             + [f"{t}@{a}" for a in ("erm", "simclr", "he_jitter") for t in ("bnadapt", "tent")]
             + ["erm_lunit_bt", "he_jitter_lunit_bt", "erm_lunit_dino", "he_jitter_lunit_dino",
                "probe_resnet50_imagenet", "probe_resnet50_lunit_bt", "probe_vit_s16_lunit_dino"])
DG_ARMS = ["groupdro", "irm", "coral", "mixstyle", "fish", "lisa"]


def base_arm(label):
    return str(label).split("@", 1)[-1].split("[", 1)[0]


def family(label, scale_view=False):
    lab = str(label)
    if "@" in lab:
        return 5
    b = base_arm(lab)
    if b.startswith("probe_") or "lunit" in b:
        return 6
    if scale_view and b == "erm_s":
        return 6
    return ARM_FAMILY.get(b, 0)


def arm_label(label):
    lab = str(label)
    hp = ""
    m = re.match(r"^(.*?)\[(.+)\]$", lab)
    if m:
        lab, hp = m.group(1), f" [{m.group(2)}]"
    if "@" in lab:
        t, a = lab.split("@", 1)
        return f"{TTA_LABEL.get(t, t)} ({arm_label(a)})" + DAGGER + hp
    return ARM_LABEL.get(lab, lab) + (DAGGER if lab in TRANSDUCTIVE else "") + hp


def order_arms(arms):
    arms = list(dict.fromkeys(arms))

    def key(a):
        b = a.split("[", 1)[0]
        idx = ARM_ORDER.index(b) if b in ARM_ORDER else len(ARM_ORDER)
        fam = family(a)
        return (0 if base_arm(a) == "erm" and "@" not in a else 1, fam, idx, a)
    return sorted(arms, key=key)


# ----------------------------------------------------------------------------------------
# style (dataviz method; print)
# ----------------------------------------------------------------------------------------

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDC, AXISC, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"
CONTEXT = "#cfcec8"            # de-emphasis gray for context series
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]          # steps 100..700
SEQ = mcolors.LinearSegmentedColormap.from_list("seqblue", BLUE)
DIV = mcolors.LinearSegmentedColormap.from_list("divbr", ["#a3302e", "#e34948", "#f0efec", "#3987e5", "#184f95"])
FULL_W, HALF_W = 6.5, 3.3
LW, LW_THIN, LW_HAIR = 1.1, 0.7, 0.45
MS = 4.5


def setup_style():
    fams = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    sans = [f for f in ("Nimbus Sans", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans") if f in fams] or ["sans-serif"]
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans, "font.size": 7.5, "axes.titlesize": 8,
        "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 4,
        "axes.edgecolor": AXISC, "axes.linewidth": LW_HAIR, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": LW_HAIR, "ytick.major.width": LW_HAIR,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.labelcolor": INK2, "ytick.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRIDC, "grid.linewidth": LW_HAIR, "grid.linestyle": "-",
        "lines.linewidth": LW, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": MS, "legend.frameon": False, "legend.handlelength": 1.4,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
    })


def xgrid(ax):
    ax.grid(axis="x", color=GRIDC, lw=LW_HAIR)
    ax.set_axisbelow(True)


def ygrid(ax):
    ax.grid(axis="y", color=GRIDC, lw=LW_HAIR)
    ax.set_axisbelow(True)


def panel_letter(ax, s, dx=0.0, dy=1.0, fig=None):
    ax.annotate(s, xy=(dx, dy), xycoords="axes fraction", xytext=(-2, 6), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="right", va="bottom")


def text_on(color):
    r, g, b = mcolors.to_rgb(color)
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return "white" if lum < 0.33 else INK


def placeholder(ax, msg, blank=False):
    if blank:
        ax.set_xticks([])
        ax.grid(False)
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=7, wrap=True)


def arm_rows(ax, arms, ys, labels=True, chip=True, scale_view=False, fontsize=7):
    """Arm names as y tick labels with a family colour chip beside the text (identity never colour-alone)."""
    ax.set_yticks(ys)
    if labels:
        ax.set_yticklabels([arm_label(a) for a in arms], fontsize=fontsize)
        ax.tick_params(axis="y", length=0, pad=10 if chip else 3)
        if chip:
            tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData) + \
                matplotlib.transforms.ScaledTranslation(-4.5 / 72, 0, ax.figure.dpi_scale_trans)
            for a, y in zip(arms, ys):   # chip 4.5 pt left of the axes, clear of the first cell / mark
                ax.plot([0.0], [y], marker="s", ms=4, color=FAMILY_COLOR[family(a, scale_view)], transform=tr,
                        clip_on=False, mew=0)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def family_legend(fig, fams, y=0.0, x=0.5, ncol=None, anchor="upper center", ax=None):
    hs = [Line2D([], [], marker="s", ls="", color=FAMILY_COLOR[f], ms=5, mew=0, label=FAMILY_NAME[f]) for f in fams]
    if not hs:
        return None
    kw = dict(handles=hs, ncol=ncol or len(hs), loc=anchor, handletextpad=0.3, columnspacing=1.0, fontsize=7)
    if ax is not None:
        return ax.legend(bbox_to_anchor=(x, y), **kw)
    return fig.legend(bbox_to_anchor=(x, y), **kw)


def label_left(arms, extra=0.26, fs=7.0):
    """Left margin (figure fraction of FULL_W) that fits the longest arm label + colour chip."""
    n = max([len(arm_label(a)) for a in arms] + [8])
    return min(0.42, (n * 0.0072 * fs + extra) / FULL_W)


def row_positions(arms, gap=0.45):
    """y positions (top to bottom) with a small gap between families."""
    ys, y, prev = [], 0.0, None
    for a in arms:
        f = (0 if base_arm(a) == "erm" and "@" not in a else 1, family(a))
        if prev is not None and f != prev:
            y += gap
        ys.append(y)
        y += 1.0
        prev = f
    return np.array(ys)


def _nolead(v, pos=None):
    if abs(v) < 1e-12:
        return "0"
    t = f"{v:.3g}"
    return t.replace("0.", ".", 1) if t.startswith(("0.", "-0.")) else t


NOLEAD = matplotlib.ticker.FuncFormatter(_nolead)


def fy(fig, inches):
    """Figure-fraction height of a distance given in inches."""
    return inches / fig.get_figheight()


MIN_FS = 7.0                    # print floor (pt)
LAYOUT_ISSUES = []              # (figure, message) collected by save(); main() exits non-zero if any


def _drawn_texts(fig, r):
    """(text artist, window extent, owner) for every visible, non-empty text that is actually drawn."""
    out = []

    def add(t, owner):
        if t is None or not t.get_visible() or not str(t.get_text()).strip():
            return
        try:   # text box only (an Annotation's own extent also spans its leader line)
            if isinstance(t, matplotlib.text.Annotation):
                t.update_positions(r)
            e = matplotlib.text.Text.get_window_extent(t, r)
        except Exception:  # noqa: BLE001
            return
        if e.width > 0 and e.height > 0:
            out.append((t, e, owner))
    for t in fig.texts:
        add(t, "fig")
    for leg in fig.legends:
        for t in leg.get_texts():
            add(t, id(leg))
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in ax.texts:
            add(t, id(ax))
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
            add(t, id(ax))
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                for tk in axis._update_ticks():
                    add(tk.label1, id(axis))
                    add(tk.label2, id(axis))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                add(t, id(leg))
    return out


def _text_poly(t, e, r):
    """Corner polygon of a text in display coords (the rotated box for rotation_mode='anchor' labels)."""
    ang = t.get_rotation() % 180
    if ang in (0.0, 90.0) or t.get_rotation_mode() != "anchor":
        return np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]])
    rot = t.get_rotation()
    t.set_rotation(0)
    e0 = t.get_window_extent(r)
    t.set_rotation(rot)
    ax_, ay_ = t.get_transform().transform(t.get_unitless_position())
    c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    pts = np.array([[e0.x0, e0.y0], [e0.x1, e0.y0], [e0.x1, e0.y1], [e0.x0, e0.y1]]) - [ax_, ay_]
    return pts @ np.array([[c, s_], [-s_, c]]) + [ax_, ay_]


def _poly_overlap(P, Q, tol):
    """Separating-axis test for two convex quads; True if they overlap by more than tol px on every axis."""
    for poly in (P, Q):
        for k in range(4):
            ex = poly[(k + 1) % 4] - poly[k]
            n = np.array([-ex[1], ex[0]])
            nn = np.hypot(*n)
            if nn == 0:
                continue
            n = n / nn
            a, b = P @ n, Q @ n
            if min(a.max(), b.max()) - max(a.min(), b.min()) <= tol:
                return False
    return True


def layout_check(fig, name, tol=0.0):
    """Text-on-text overlap, text under a legend, text below the 7-pt floor, content outside the canvas width."""
    r = fig.canvas.get_renderer()
    items = _drawn_texts(fig, r)
    issues = []
    for t, e, _ in items:
        if t.get_fontsize() < MIN_FS - 1e-6:
            issues.append(f"font {t.get_fontsize():.1f} pt < {MIN_FS:g}: {t.get_text()[:40]!r}")
    polys = [_text_poly(t, e, r) for t, e, _ in items]
    for i in range(len(items)):
        ti, ei, _ = items[i]
        for j in range(i + 1, len(items)):
            tj, ej, _ = items[j]
            w = min(ei.x1, ej.x1) - max(ei.x0, ej.x0)
            h = min(ei.y1, ej.y1) - max(ei.y0, ej.y0)
            if w > tol and h > tol and _poly_overlap(polys[i], polys[j], tol):
                issues.append(f"overlap: {ti.get_text()[:40]!r} <-> {tj.get_text()[:40]!r}")
    legs = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    for leg in legs:
        le = leg.get_window_extent(r)
        own = set(map(id, leg.get_texts()))
        for t, e, _ in items:
            if id(t) in own:
                continue
            w = min(le.x1, e.x1) - max(le.x0, e.x0)
            h = min(le.y1, e.y1) - max(le.y0, e.y0)
            if w > tol and h > tol:
                issues.append(f"legend over text {t.get_text()[:40]!r}")
    # in-plot text of one axes running under a later-drawn (opaque) axes is hidden there
    axes_ = [ax for ax in fig.axes if ax.get_visible()]
    for k, ax in enumerate(axes_):
        for t in ax.texts:
            hit = [(t2, e2) for t2, e2, _ in items if t2 is t]
            if not hit:
                continue
            e = hit[0][1]
            for ax2 in axes_[k + 1:]:
                if not (ax2.axison and ax2.patch.get_visible() and ax2.patch.get_alpha() in (None, 1)):
                    continue
                b = ax2.get_window_extent(r)
                w = min(b.x1, e.x1) - max(b.x0, e.x0)
                h = min(b.y1, e.y1) - max(b.y0, e.y0)
                if w > 2 and h > 2:
                    issues.append(f"text hidden under a later panel: {t.get_text()[:40]!r}")
    W = fig.get_figwidth() * fig.dpi
    for t, e, _ in items:
        if e.x0 < -0.5 or e.x1 > W + 0.5:
            issues.append(f"outside canvas width: {t.get_text()[:40]!r} (x {e.x0 / fig.dpi:.2f}..{e.x1 / fig.dpi:.2f} in)")
    bb = fig.get_tightbbox(r)
    if bb.x0 < -0.01 or bb.x1 > fig.get_figwidth() + 0.01:
        issues.append(f"content spans x {bb.x0:.2f}..{bb.x1:.2f} in (canvas {fig.get_figwidth():.2f} in)")
    for s in dict.fromkeys(issues):
        print(f"  [layout] {name}: {s}")
        LAYOUT_ISSUES.append((name, s))
    return issues


def lowest_y(fig):
    """Lowest point of the drawn content, in figure fraction."""
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return bb.y0 / fig.get_figheight()


def save(fig, out, name, footnote=None, written=None, fn_y=None):
    """Fixed print width (canvas width, 6.5 in), height cropped to the content; the footnote goes below
    the lowest drawn element (never at a fixed offset) and is checked like everything else."""
    import textwrap
    fig.canvas.draw()
    if any(DAGGER in str(t.get_text()) for t, _, _ in _drawn_texts(fig, fig.canvas.get_renderer())) and \
            not (footnote and DAGGER in footnote):
        footnote = TRANSDUCTIVE_NOTE + ("\n" + footnote if footnote else "")
    if footnote:
        wmax = int(fig.get_figwidth() * 72 / (MIN_FS * 0.50))
        footnote = "\n".join(textwrap.fill(par, wmax) for par in footnote.split("\n"))
        y = lowest_y(fig) - fy(fig, 0.07)
        fig.text(0.0, y, footnote, ha="left", va="top", fontsize=MIN_FS, color=MUTED, wrap=False, linespacing=1.25)
        fig.canvas.draw()
    layout_check(fig, name)
    os.makedirs(out, exist_ok=True)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    W = fig.get_figwidth()
    box = matplotlib.transforms.Bbox([[0.0, bb.y0 - 0.03], [W, bb.y1 + 0.03]])
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches=box)
        paths.append(p)
    plt.close(fig)
    if written is not None:
        written.extend(paths)
    print(f"  wrote {name}.pdf/.png ({W:.2f} x {bb.y1 - bb.y0 + 0.06:.2f} in)")
    return paths


def top_y(fig, axes):
    """Top of the drawn extent (titles, tick labels, annotations) of ``axes``, in figure fraction."""
    r = fig.canvas.get_renderer()
    fig.canvas.draw()
    return max(ax.get_tightbbox(r).y1 for ax in axes) / (fig.get_figheight() * fig.dpi)


def bottom_y(fig, axes):
    r = fig.canvas.get_renderer()
    fig.canvas.draw()
    return min(ax.get_tightbbox(r).y0 for ax in axes) / (fig.get_figheight() * fig.dpi)


def legend_above(fig, axes, handles, x, gap=0.06, **kw):
    """Figure legend placed above the drawn top of ``axes`` (titles included), never on top of them."""
    y = top_y(fig, axes) + fy(fig, gap)
    kw.setdefault("fontsize", MIN_FS)
    return fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x, y), borderaxespad=0.0, **kw)


def caption_below(fig, axes, x, text, gap=0.08, **kw):
    """Muted explanatory text placed below the drawn bottom of ``axes`` (tick labels, x labels included)."""
    y = bottom_y(fig, axes) - fy(fig, gap)
    kw.setdefault("fontsize", MIN_FS)
    kw.setdefault("color", INK2)
    return fig.text(x, y, text, va="top", **kw)


# ----------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------


def _parse(v):
    if v is None:
        return None
    v = v.strip()
    if v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    if v[0] in "[{":
        try:
            return json.loads(v)
        except Exception:  # noqa: BLE001
            return v
    try:
        f = float(v)
        return f
    except ValueError:
        return v


def read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        return [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def fin(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


class Ctx:
    def __init__(self, args):
        self.args = args
        self.an = args.analysis
        self.runs = args.runs
        self.out = args.out
        self.cohorts = args.cohorts
        self.tiers = ["C", "S"] if args.tier == "both" else [args.tier]
        self._scan = {}
        self._domains = {}
        self.missing_fields = set()

    # ---- analysis tree
    def p(self, tier, cohort, fname):
        return os.path.join(self.an, tier, cohort, fname)

    def csv(self, tier, cohort, kind, sel="id"):
        return read_csv(self.p(tier, cohort, f"{kind}_{sel}.csv"))

    def json(self, tier, cohort, kind, sel="id"):
        return read_json(self.p(tier, cohort, f"{kind}_{sel}.json"))

    def meta(self, tier, cohort, sel="id"):
        return self.json(tier, cohort, "meta", sel)

    def avail(self, tier, sel="id", min_k=1):
        out = []
        for c in self.cohorts:
            m = self.meta(tier, c, sel)
            if m and m.get("K", 0) >= min_k:
                out.append(c)
        return out

    def K(self, tier, cohort, sel="id"):
        m = self.meta(tier, cohort, sel)
        return int(m["K"]) if m else 0

    def domains(self, cohort):
        """Domain names in fold order: manifests > fallback."""
        if cohort in self._domains:
            return self._domains[cohort]
        names = []
        for f in range(K_EXPECTED.get(cohort, 0)):
            m = read_json(os.path.join(self.runs, "cache64", cohort, f"fold{f}", "manifest.json"))
            if not m:
                names = None
                break
            names.append(str(m["test_domain"]))
        names = names or DOMAINS_FALLBACK.get(cohort, [])
        self._domains[cohort] = names
        return names

    def summary(self, tier, cohort, sel="id"):
        return {r["arm"]: r for r in self.csv(tier, cohort, "summary", sel)}

    def per_run(self, tier, cohort, sel="id"):
        """arm -> domain -> {seed: (metric, ece)} (all arms, incl. those missing folds)."""
        out = defaultdict(lambda: defaultdict(dict))
        for r in self.csv(tier, cohort, "per_run", sel):
            out[str(r["arm"])][str(r["domain"])][int(r["seed"])] = (r["metric"], r.get("ece"))
        return out

    # ---- snapshot provenance (what the analysis actually used)
    def snapshot(self, tier):
        return read_json(os.path.join(self.an, tier, "SNAPSHOT.json"))

    def used(self, tier, cohort, sel="id"):
        """Training runs the analysis used (from per_run_{sel}.csv): (n_runs, folds, files)."""
        files, folds = set(), set()
        exp_arms = TIER_ARMS.get(tier, [])
        for r in self.csv(tier, cohort, "per_run", sel):
            a = str(r["arm"])
            if "@" in a or (exp_arms and base_arm(a) not in exp_arms):
                continue
            files.add(str(r["file"]))
            folds.add(int(r["fold"]))
        return len(files), folds, files

    def check_snapshot(self, tier, cohorts=None):
        """Every selection rule of every cohort must have been analysed from one record set."""
        problems = []
        sn = self.snapshot(tier)
        rsn = read_json(os.path.join(self.runs, "SNAPSHOT.json"))
        if sn is None:
            problems.append(f"tier {tier}: no SNAPSHOT.json in {self.an}/{tier} (analysis not run from a frozen "
                            f"manifest; use --freeze)")
        elif rsn is not None and rsn.get("id") != sn.get("id"):
            problems.append(f"tier {tier}: analysis snapshot {sn.get('id')} != records snapshot {rsn.get('id')}")
        for c in cohorts or (self.cohorts + ["midog21"]):
            sets = {s_: self.used(tier, c, s_)[2] for s_ in SELS if self.meta(tier, c, s_)}
            if not sets:
                continue
            ref = sets.get("id") or next(iter(sets.values()))
            for s_, fs in sets.items():
                if fs != ref:
                    problems.append(f"tier {tier} {c}: selection '{s_}' analysed {len(fs)} runs, 'id' {len(ref)} "
                                    f"({len(fs ^ ref)} differ)")
            for s_ in sets:   # per-file provenance (analyze_v2 records the snapshot of --v2-root in meta_*)
                mid = ((self.meta(tier, c, s_) or {}).get("snapshot") or {}).get("id")
                if mid and sn is not None and mid != sn.get("id"):
                    problems.append(f"tier {tier} {c}: meta_{s_}.json from snapshot {mid}, tier stamp {sn.get('id')}")
            mt = [os.path.getmtime(self.p(tier, c, f"meta_{s_}.json")) for s_ in sets]
            if sn is None and max(mt) - min(mt) > 120:
                problems.append(f"tier {tier} {c}: meta_* written {max(mt) - min(mt):.0f} s apart")
        return problems

    def completeness(self, tier, cohorts=None, sel="id"):
        """(is_partial, text) for the footnote, from the runs and folds the analysis actually used."""
        cohorts = cohorts or self.cohorts
        parts, partial = [], False
        exp_arms = TIER_ARMS.get(tier, [])
        ns = len(SEEDS.get(tier, []))
        for c in cohorts:
            K = K_EXPECTED.get(c, 0)
            n, folds, _ = self.used(tier, c, sel)
            exp = len(exp_arms) * K * ns
            m = self.meta(tier, c, sel) or {}
            na = len([a for a in m.get("arms", []) if "@" not in a])
            if n < exp or len(folds) < K:
                partial = True
            parts.append(f"{COHORT_LABEL.get(c, c)} {n}/{exp} runs, {len(folds)}/{K} folds, "
                         f"{na}/{len(exp_arms)} arms complete")
        sn = self.snapshot(tier) or {}
        src = f"snapshot {sn['id']} of {sn['created']}" if sn.get("id") else "unfrozen records"
        txt = (f"Tier {tier} ({src}): " + "; ".join(parts) +
               ". Cross-domain statistics use only arms with every fold present.")
        return partial, txt

    def footnote(self, tiers=None, cohorts=None, extra=None):
        lines = []
        for t in tiers or self.tiers:
            p, txt = self.completeness(t, cohorts)
            if p:
                lines.append("PARTIAL DATA. " + txt)
        if extra and lines:
            lines.append(extra)
        return "\n".join(lines) if lines else None


def cohort_title(ctx, tier, c, sel="id", K=None):
    K = ctx.K(tier, c, sel) if K is None else K
    Ke = K_EXPECTED.get(c, K)
    lab = COHORT_LABEL.get(c, c)
    if K == 0:
        return f"{lab} (no runs)"
    return f"{lab} (K = {K})" if K >= Ke else f"{lab} (K = {K} of {Ke})"


def facet_widths(ctx, cohorts):
    return [K_EXPECTED.get(c, 4) for c in cohorts]


# ----------------------------------------------------------------------------------------
# F1  design schematic (no data)
# ----------------------------------------------------------------------------------------

# roles are not arms: ink and grays only (family hues mean arm identity); OOD-val = open cell with an ink outline
ROLE_STYLE = {"train": dict(fc="#e6e5e0", ec="none"), "idval": dict(fc="#a8a7a0", ec="none"),
              "ood": dict(fc="white", ec=INK, lw=0.9), "test": dict(fc=INK, ec="none")}


def fig_design(ctx, written):
    fig = plt.figure(figsize=(FULL_W, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], height_ratios=[1.12, 1.0], wspace=0.06, hspace=0.30,
                          left=0.0, right=1.0, top=0.97, bottom=0.02)
    # (a) 2 x 2 design ------------------------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    ax.set_xlim(-0.50, 2.02)
    ax.set_ylim(-0.02, 2.30)
    ax.axis("off")
    ax.text(-0.50, 2.30, "a", fontsize=9, fontweight="bold", va="top")
    ax.text(-0.38, 2.30, "Task × shift-type design (one cohort per cell)", fontsize=8, fontweight="bold", va="top")
    ax.text(0.5, 1.99, "Acquisition-only shift", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.text(1.5, 1.99, "Institutional shift", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.text(-0.05, 1.5, "Tumor vs\nnon-tumor", ha="right", va="center", fontsize=7.5, color=INK2, linespacing=1.2)
    ax.text(-0.05, 0.5, "Mitotic\nfigure vs\nlook-alike", ha="right", va="center", fontsize=7.5, color=INK2,
            linespacing=1.2)
    cells = {
        (0, 1): ("Canine cSCC", "domain = scanner", "same slides on five\nscanners; slide-disjoint", "K = 5, OOD-val"),
        (1, 1): ("Camelyon17", "domain = hospital", "scanner, stain and\npatients differ", "K = 5, OOD-val"),
        (0, 0): ("MIDOG 2021", "domain = scanner", "one lab; magnification\nnormalized", "K = 3, no OOD-val"),
        (1, 0): ("MIDOG++", "domain = tumor type", "institutional and\nbiological shift", "K = 7, OOD-val"),
    }
    for (cx, cy), (name, dom, det, k) in cells.items():
        x0, y0 = cx + 0.025, cy + 0.025
        ax.add_patch(FancyBboxPatch((x0, y0), 0.95, 0.95, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="#f4f3f0", ec="none"))
        ax.text(x0 + 0.06, y0 + 0.86, name, fontsize=8, fontweight="bold", va="top")
        ax.text(x0 + 0.06, y0 + 0.64, dom, fontsize=7, va="top", color=INK)
        ax.text(x0 + 0.06, y0 + 0.46, det, fontsize=7, va="top", color=INK2, linespacing=1.15)
        ax.text(x0 + 0.06, y0 + 0.09, k, fontsize=7, va="bottom", color=INK)
    # (b) LODO roles ---------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    K = 5
    ax.set_xlim(-1.25, K + 0.05)
    ax.set_ylim(K + 2.35, -1.75)
    ax.axis("off")
    ax.text(-1.25, -1.75, "b", fontsize=9, fontweight="bold", va="top")
    ax.text(-1.0, -1.75, "Leave-one-domain-out folds (K = 5 shown)", fontsize=8, fontweight="bold", va="top")
    for j in range(K):
        ax.text(j + 0.5, -0.15, f"domain {j}", ha="center", va="bottom", fontsize=7, color=INK2)
    g = 0.05
    for t in range(K):
        ax.text(-0.1, t + 0.5, f"fold {t}", ha="right", va="center", fontsize=7, color=INK2)
        for j in range(K):
            role = "test" if j == t else ("ood" if j == (t - 1) % K else "train")
            gg = g + (0.035 if role == "ood" else 0)
            ax.add_patch(Rectangle((j + gg, t + gg), 1 - 2 * gg, 1 - 2 * gg, **ROLE_STYLE[role]))
            if role == "train":   # ID-val = held-out groups of each training domain
                ax.add_patch(Rectangle((j + 1 - g - 0.24, t + g), 0.24, 1 - 2 * g, **ROLE_STYLE["idval"]))
    leg = [("train", "train"), ("idval", "ID-val"), ("ood", "OOD-val, (t − 1) mod K"), ("test", "test")]
    xs = [-1.2, 0.05, 1.25, 3.85]
    for (k, lab), x in zip(leg, xs):
        ax.add_patch(Rectangle((x, K + 0.30), 0.22, 0.42, **ROLE_STYLE[k]))
        ax.text(x + 0.30, K + 0.52, lab, fontsize=7, va="center", color=INK)
    ax.text(-1.2, K + 1.02, "ID-val: held-out groups of each training domain.\nMIDOG 2021 (K = 3) has no OOD-val domain.",
            fontsize=7, color=INK2, va="top")
    # (c) checkpoints and selection rules -------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    X0 = 260                                  # x offset of the step axis (room for the rule names)
    ax.set_xlim(-520, 2480)
    ax.set_ylim(-4.2, 1.45)
    ax.axis("off")
    ax.text(-520, 1.45, "c", fontsize=9, fontweight="bold", va="top")
    ax.text(-400, 1.45, "Ten checkpoints per run, four selection rules", fontsize=8, fontweight="bold", va="top")
    steps = np.arange(120, 1201, 120)
    rules = [("ID-val (primary)", "max ID-val AUROC", 6), ("OOD-val", "max OOD-val AUROC", 4),
             ("Last", "final checkpoint", 10), ("Oracle (bound)", "max test AUROC", 7)]
    for i, (name, how, k) in enumerate(rules):
        y = -i
        ax.text(-400, y, name, va="center", fontsize=7, color=INK, fontweight="bold" if i == 0 else "normal")
        ax.plot([X0 + 120, X0 + 1200], [y, y], color=GRIDC, lw=LW_HAIR, zorder=1)
        ax.plot(X0 + steps, np.full(10, y), "o", ms=3.0, color=CONTEXT, mew=0, zorder=2)
        ax.plot([X0 + steps[k - 1]], [y], "o", ms=6.0, color=INK if i == 0 else INK2, mec="white", mew=1.0, zorder=3)
        ax.text(X0 + 1270, y, how, va="center", fontsize=7, color=INK2)
    yb = -3.55
    ax.plot([X0 + 120, X0 + 1200], [yb, yb], color=AXISC, lw=LW_HAIR)
    for s_ in (120, 600, 1200):
        ax.plot([X0 + s_, X0 + s_], [yb, yb - 0.12], color=AXISC, lw=LW_HAIR)
        ax.text(X0 + s_, yb - 0.22, f"{s_:,}", ha="center", va="top", fontsize=7, color=INK2)
    ax.text(X0 + 1450, yb - 0.22, "SGD step", va="top", fontsize=7, color=INK2)
    ax.text(-400, 0.60, "1,200 steps, batch 128, cosine LR; picks illustrative; same runs, all rules",
            fontsize=7, color=INK2, va="center")
    save(fig, ctx.out, "fig_design", written=written)


# ----------------------------------------------------------------------------------------
# F2  per-domain AUROC
# ----------------------------------------------------------------------------------------
# Form choice: heatmap (arms x domains, sequential blue, values printed) rather than a dot strip.
# The reader's jobs are (i) read the exact per-domain value, (ii) see which domain is hard for
# everyone (a dark/light column) and (iii) find each arm's worst domain.  A dot strip puts all K
# domains of an arm on one line, so domain identity would need K extra hues (> the 3-colour cap
# of overlapping forms) or labels on every dot; the heatmap keeps identity in the column position,
# prints the value (colour is only a secondary cue) and outlines the worst cell per row.


def fig_perdomain(ctx, tier, written, sel="id"):
    cohorts = [c for c in ctx.cohorts]
    data = {c: ctx.per_run(tier, c, sel) for c in cohorts}
    if not any(data.values()):
        print(f"  [F2 {tier}] no per_run files -> skipped")
        return
    arms = order_arms([a for c in cohorts for a in data[c]])
    ys = row_positions(arms)
    vals = []
    cells = {}
    for c in cohorts:
        doms = ctx.domains(c)
        for a in arms:
            for j, d in enumerate(doms):
                rec = data[c].get(a, {}).get(d)
                if rec:
                    v = [x[0] for x in rec.values() if fin(x[0])]
                    if v:
                        cells[(c, a, j)] = (float(np.mean(v)), len(v))
                        vals.append(np.mean(v))
    lo = math.floor(min(vals) * 20) / 20 if vals else 0.5
    norm = mcolors.Normalize(vmin=min(lo, 0.9), vmax=1.0)
    widths = facet_widths(ctx, cohorts)
    H = 0.62 + 0.165 * (ys[-1] + 1)
    fig, axs = plt.subplots(1, len(cohorts), figsize=(FULL_W, H), gridspec_kw=dict(width_ratios=widths, wspace=0.05),
                            squeeze=False)
    fig.subplots_adjust(left=label_left(arms), right=0.965)
    axs = axs[0]
    nseed = len(SEEDS.get(tier, [])) or 1
    few = False
    for i, (c, ax) in enumerate(zip(cohorts, axs)):
        doms = ctx.domains(c)
        K = len(doms)
        ax.set_xlim(0, K)
        ax.set_ylim(ys[-1] + 1, ys[0])
        for a, y in zip(arms, ys):
            row = [cells.get((c, a, j)) for j in range(K)]
            have = [r for r in row if r]
            worst_j = None
            if len(have) == K:
                worst_j = int(np.argmin([r[0] for r in row]))
            for j, r in enumerate(row):
                g = 0.04
                if r is None:
                    ax.add_patch(Rectangle((j + g, y + g), 1 - 2 * g, 1 - 2 * g, fc="#f6f6f4", ec="none"))
                    continue
                col = SEQ(norm(r[0]))
                ax.add_patch(Rectangle((j + g, y + g), 1 - 2 * g, 1 - 2 * g, fc=col, ec="none"))
                s = f"{r[0]:.3f}".lstrip("0")
                if r[1] < nseed:
                    s += "'"
                    few = True
                ax.text(j + 0.5, y + 0.53, s, ha="center", va="center", fontsize=7, color=text_on(col))
                if j == worst_j:
                    ax.add_patch(Rectangle((j + 0.07, y + 0.07), 0.86, 0.86, fc="none", ec=INK, lw=0.9))
        ax.set_xticks(np.arange(K) + 0.5)
        ax.set_xticklabels([dshort(d) for d in doms], fontsize=7, rotation=45, ha="left", va="bottom",
                           rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.xaxis.tick_top()
        for sp in ax.spines.values():
            sp.set_visible(False)
        Ka = sum(1 for j in range(K) if any((c, a, j) in cells for a in arms))
        ax.set_title(COHORT_LABEL.get(c, c), fontsize=7.5, pad=42, loc="left")
        if Ka < K:
            ax.annotate(f"{Ka} of {K} domains", xy=(0, 1), xycoords="axes fraction", xytext=(0, 33),
                        textcoords="offset points", fontsize=7, color=MUTED, va="bottom")
        arm_rows(ax, arms, ys + 0.5, labels=(i == 0))
    # colour scale + mark legend below the panels (positions in inches below the axes)
    fig.canvas.draw()
    bot = min(ax.get_position().y0 for ax in axs)
    cax = fig.add_axes([0.64, bot - fy(fig, 0.42), 0.25, fy(fig, 0.07)])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm, SEQ), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, width=LW_HAIR)
    cb.set_label(f"test AUROC (seed mean, {SEL_LABEL[sel]} selection)", fontsize=7, labelpad=2)
    y = bot - fy(fig, 0.10)
    fig.text(0.0, y, "Outlined: worst domain of the arm (rows with every domain present)." +
             (f"  ' : fewer than {nseed} seeds in the cell." if few else ""), fontsize=7, color=INK2, va="top")
    fams = sorted({family(a) for a in arms})
    family_legend(fig, fams, y=y - fy(fig, 0.16), x=0.0, anchor="upper left", ncol=3)
    fn = ctx.footnote([tier], cohorts)
    save(fig, ctx.out, f"fig_perdomain_{tier}", footnote=fn, written=written, fn_y=y - fy(fig, 0.78))


# ----------------------------------------------------------------------------------------
# F3  rank stability
# ----------------------------------------------------------------------------------------


def _common_ranks(ctx, tier, cohorts, sel="id", key="worst"):
    """{cohort: {arm: rank among arms common to all cohorts}} by the 'worst' (or 'mean') score."""
    summ = {c: ctx.summary(tier, c, sel) for c in cohorts}
    summ = {c: v for c, v in summ.items() if v}
    if not summ:
        return [], {}
    common = None
    for c, v in summ.items():
        st = {a for a, r in v.items() if fin(r.get(key))}
        common = st if common is None else common & st
    common = order_arms(common or [])
    out = {}
    for c, v in summ.items():
        sc = np.array([v[a][key] for a in common], float)
        order = (-sc).argsort(kind="stable")
        rk = np.empty(len(common))
        # average ranks for ties
        from scipy.stats import rankdata
        rk = rankdata(-sc, method="average")
        out[c] = dict(zip(common, rk.tolist()))
    return common, out


def pick_highlights(common, ranks):
    """3 coloured arms for the bump chart: SimCLR, H&E jitter and the best DG objective (mean rank)."""
    hi = []
    for a in ("simclr", "he_jitter"):
        if a in common:
            hi.append(a)
    dg = [a for a in common if base_arm(a) in DG_ARMS]
    if dg:
        hi.append(min(dg, key=lambda a: np.mean([ranks[c][a] for c in ranks])))
    for a in common:  # fill up to 3 with arms of families not yet shown (colour = family, never repeated)
        if len(hi) >= 3:
            break
        if a not in hi and family(a) != 0 and family(a) not in {family(h) for h in hi}:
            hi.append(a)
    return hi[:3]


HI_COLORS = ["#1baf7a", "#eb6834", "#2a78d6"]
MIN_COMMON_ARMS = 6     # rank comparisons / headlines need at least this many arms common to all cohorts   # validated all-pairs trio (aqua, orange, blue)


def bump_chart(ax, ctx, cohorts, common, ranks, fs=7, lw_hi=1.6, ms_hi=5, label_all=True, title=True, room=None,
               rank_axis=True, context_lines=True):
    cs = [c for c in cohorts if c in ranks]
    x = np.arange(len(cs))
    hi = pick_highlights(common, ranks)
    n = len(common)

    def col(a):
        if a in hi:
            # family colour of the highlighted arm (SimCLR aqua, H&E jitter orange, DG blue)
            return FAMILY_COLOR[family(a)]
        if base_arm(a) == "erm" and "@" not in a:
            return FAMILY_COLOR[0]
        return CONTEXT
    order = [a for a in common if a not in hi and not (base_arm(a) == "erm" and "@" not in a)] + \
            [a for a in common if base_arm(a) == "erm" and "@" not in a] + hi
    for a in order:
        y = [ranks[c][a] for c in cs]
        emph = a in hi or (base_arm(a) == "erm" and "@" not in a)
        if not emph and not (context_lines or label_all):
            continue   # never draw a line that carries no label
        ax.plot(x, y, color=col(a), lw=lw_hi if emph else 0.8, zorder=3 if emph else 2,
                marker="o", ms=ms_hi if emph else 3, mec="white", mew=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COHORT_LABEL[c].replace(" ", "\n", 1) for c in cs], fontsize=fs, linespacing=1.05)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylim(n + 0.6, 0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if rank_axis:   # rank 1 / rank n ticks on a hairline axis at the left edge
        ax.set_yticks([1, n])
        ax.set_yticklabels([f"rank 1", f"rank {n}"], fontsize=MIN_FS, color=INK2)
        ax.tick_params(axis="y", length=2.5, width=LW_HAIR, pad=2)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_bounds(1, n)
    else:
        ax.set_yticks([])
    if room is None:   # x-units needed so the longest direct label fits beside the end columns
        labs = [a for a in common if label_all or a in hi or (base_arm(a) == "erm" and "@" not in a)]
        w = 0.0072 * fs * max([len(arm_label(a)) for a in labs] + [4]) + 0.15
        Win = ax.get_position().width * ax.figure.get_figwidth()
        room = min(1.6, max(0.5, w * (len(cs) - 0.7) / max(Win - 2 * w, 0.5)))
    ax.set_xlim(-0.15 - room, len(cs) - 1 + 0.15 + room)
    # direct labels at both ends; min spacing from the font size (labels nudged apart get a leader line)
    fig = ax.figure
    unit = ax.get_position().height * fig.get_figheight() * 72 / (n + 0.2)
    minsep = max(0.95, fs * 1.18 / unit)
    for side, ci, ha, dx in (("L", 0, "right", -0.12), ("R", len(cs) - 1, "left", 0.12)):
        if not cs:
            break
        items = [a for a in sorted(common, key=lambda a: ranks[cs[ci]][a])
                 if label_all or a in hi or (base_arm(a) == "erm" and "@" not in a)]
        ypos, last = [], -99
        for a in items:
            y = max(ranks[cs[ci]][a], last + minsep)
            ypos.append([a, y])
            last = y
        over = ypos[-1][1] - n if ypos else 0     # pull back up if pushed past the bottom
        if over > 0:
            for k in range(len(ypos) - 1, -1, -1):
                ypos[k][1] -= over
                if k and ypos[k][1] - ypos[k - 1][1] >= minsep:
                    break
                over = max(0, ypos[k - 1][1] + minsep - ypos[k][1]) if k else 0
        for a, y in ypos:
            emph = a in hi or (base_arm(a) == "erm" and "@" not in a)
            ax.text(ci + dx, y, f"{arm_label(a)}", ha=ha, va="center", fontsize=max(MIN_FS, fs - 0.5) if not emph else fs,
                    color=INK if emph else MUTED, fontweight="bold" if emph else "normal")
            if abs(y - ranks[cs[ci]][a]) > 0.05:
                ax.plot([ci, ci + dx * 0.8], [ranks[cs[ci]][a], y], color=AXISC, lw=LW_HAIR)
    return hi


def fig_ranks(ctx, tier, written, sel="id"):
    cohorts = [c for c in ctx.cohorts if ctx.csv(tier, c, "ranks", sel)]
    if not cohorts:
        print(f"  [F3 {tier}] no ranks files -> skipped")
        return
    allc = ctx.cohorts
    rk = {c: {r["arm"]: r for r in ctx.csv(tier, c, "ranks", sel)} for c in allc}
    arms = order_arms([a for c in allc for a in rk[c]])
    ys = row_positions(arms)
    L = label_left(arms)
    common, ranks = _common_ranks(ctx, tier, cohorts, sel)
    H_a = 0.45 + 0.15 * (ys[-1] + 1)
    H_b = 0.35 + 0.16 * max(len(common), 4)
    fig = plt.figure(figsize=(FULL_W, H_a + H_b + 0.55))
    gs = fig.add_gridspec(2, len(allc), height_ratios=[H_a, H_b], hspace=0.30, wspace=0.08,
                          left=L, right=0.99, top=1 - fy(fig, 0.35), bottom=fy(fig, 0.12))
    top_axes = []
    for i, c in enumerate(allc):
        ax = fig.add_subplot(gs[0, i])
        top_axes.append(ax)
        n = len(rk[c])
        ax.set_ylim(ys[-1] + 0.7, ys[0] - 0.7)
        ax.set_xlim(0.3, max(n, 2) + 0.7)
        xgrid(ax)
        if i == 0:
            panel_letter(ax, "a", dx=-0.42, dy=1.0)
        if not n:
            placeholder(ax, "no complete arms yet", blank=True)
        for a, y in zip(arms, ys):
            r = rk[c].get(a)
            if not r:
                continue
            col = FAMILY_COLOR[family(a)]
            lo, hi_ = r.get("rank_worst_domain_ci") or [None, None]
            if fin(lo):
                ax.plot([lo, hi_], [y, y], color=col, lw=0.8, alpha=0.55, solid_capstyle="butt")
            lo, hi_ = r.get("rank_worst_seed_ci") or [None, None]
            if fin(lo):
                ax.plot([lo, hi_], [y, y], color=col, lw=2.6, solid_capstyle="butt")
            ax.plot([r["rank_worst"]], [y], "o", ms=MS, color=col, mec="white", mew=0.8, zorder=3)
        ax.set_title(cohort_title(ctx, tier, c, sel), fontsize=7.5)
        ax.set_xlabel("rank (1 = best worst-domain AUROC)" if i == 0 else "", fontsize=7, loc="left")
        if n:
            step = 5 if n > 10 else (2 if n > 5 else 1)
            ax.set_xticks([1] + list(range(step, n + 1, step)) if step > 1 else range(1, n + 1))
        arm_rows(ax, arms, ys, labels=(i == 0))
    # legend for interval marks
    lh = [Line2D([], [], color=INK2, lw=2.6, solid_capstyle="butt", label="seed bootstrap 95% interval"),
          Line2D([], [], color=INK2, lw=0.8, alpha=0.55, label="domain bootstrap 95% interval"),
          Line2D([], [], color=INK2, marker="o", ls="", ms=MS, label="observed rank")]
    legend_above(fig, top_axes, lh, L, ncol=3, handlelength=1.6)
    # (b) bump chart (only when the comparison is not degenerate: enough common arms, every cohort at full K)
    ax = fig.add_subplot(gs[1, :])
    panel_letter(ax, "b", dx=-0.2, dy=1.0)
    full_k = [c for c in cohorts if ctx.K(tier, c, sel) >= K_EXPECTED.get(c, 0)]
    ok_bump = len(ranks) >= 2 and len(common) >= MIN_COMMON_ARMS and len(full_k) == len(ctx.cohorts)
    if ok_bump:
        hi = bump_chart(ax, ctx, ctx.cohorts, common, ranks)
        ax.text(-0.02, -0.02, f"Rank among the {len(common)} arms complete in every shown cohort\n(worst-domain AUROC, "
                f"{SEL_LABEL[sel]} selection)." + (" Colored: " + ", ".join(arm_label(a) for a in hi) + ";"
                                                   if hi else "") + " ERM in dark gray.",
                transform=ax.transAxes, fontsize=7, color=INK2, va="top")
    else:
        ax.axis("off")
        miss = [COHORT_LABEL[c] for c in ctx.cohorts if c not in full_k]
        placeholder(ax, f"Rank comparison across cohorts not drawn: {len(common)} arm(s) complete in every cohort "
                        f"(needs ≥ {MIN_COMMON_ARMS})" + (f"; incomplete K: {', '.join(miss)}" if miss else "") + ".")
    fn = ctx.footnote([tier])
    save(fig, ctx.out, f"fig_ranks_{tier}", footnote=fn, written=written, fn_y=-fy(fig, 0.15))


# ----------------------------------------------------------------------------------------
# F4  variance decomposition
# ----------------------------------------------------------------------------------------


def _bayes_icc(r):
    """Half-Cauchy posterior median and 95% CrI; one-sided [0, q95] at the zero boundary."""
    med = r.get("halfc_icc_med")
    if r.get("reml_boundary") or r.get("anova_truncated"):
        return med, [0.0, r.get("halfc_icc_u95")]
    return med, r.get("halfc_icc_ci")


def place_labels(ax, pts, labels, fs=MIN_FS, radii=(8, 13, 19, 26, 34), point_r=3.0, avoid_axes_text=True):
    """Greedy direct labels with hairline leaders.  pts: [(x, y)] of every drawn point (data units);
    labels: [(x, y, text)].  Every candidate offset (8 directions x radii) is scored by its overlap with
    the points, the labels placed so far, every text already drawn in the figure (ray labels, ticks,
    titles) and the area outside the axes; the first collision-free candidate wins, else the least bad."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axb = ax.get_window_extent(r)
    px = fig.dpi / 72.0
    pbox = []
    for x, y in pts:
        X, Y = ax.transData.transform((x, y))
        pbox.append(matplotlib.transforms.Bbox([[X - point_r * px, Y - point_r * px], [X + point_r * px, Y + point_r * px]]))
    taken = [e for _, e, _ in _drawn_texts(fig, r)] if avoid_axes_text else []
    P = np.array([ax.transData.transform((x, y)) for x, y in pts]) if pts else np.zeros((0, 2))
    L_ = np.array([ax.transData.transform((x, y)) for x, y, _ in labels]) if labels else np.zeros((0, 2))

    def dist(b, Q):   # px from each point of Q to box b (0 inside)
        dx = np.maximum.reduce([b.x0 - Q[:, 0], np.zeros(len(Q)), Q[:, 0] - b.x1])
        dy = np.maximum.reduce([b.y0 - Q[:, 1], np.zeros(len(Q)), Q[:, 1] - b.y1])
        return np.hypot(dx, dy)

    def ambiguity(b, own):   # labelled points nearer than its own point weigh 10x an unlabelled one
        d0 = dist(b, own[None, :])[0]
        n_lab = int(np.sum((dist(b, L_) < d0 - 1.0) & (np.hypot(*(L_ - own).T) > 1.0))) if len(L_) else 0
        n_pt = int(np.sum((dist(b, P) < d0 - 1.0) & (np.hypot(*(P - own).T) > 1.0))) if len(P) else 0
        return 10 * n_lab + max(0, n_pt - n_lab)

    def ov(b, others, m=2.0):   # 2-px clearance
        return sum(max(0, min(b.x1 + m, o.x1) - max(b.x0 - m, o.x0)) * max(0, min(b.y1 + m, o.y1) - max(b.y0 - m, o.y0))
                   for o in others)
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
    for x, y, txt in labels:
        own = np.array(ax.transData.transform((x, y)))
        best, best_sc = None, None
        for rad in radii:
            for dx, dy in dirs:
                n = math.hypot(dx, dy)
                ox, oy = rad * dx / n, rad * dy / n
                ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
                va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
                b = ax.annotate(txt, (x, y), xytext=(ox, oy), textcoords="offset points", ha=ha, va=va,
                                fontsize=fs).get_window_extent(r)
                ax.texts[-1].remove()
                out = b.width * b.height - max(0, min(b.x1, axb.x1) - max(b.x0, axb.x0)) * \
                    max(0, min(b.y1, axb.y1) - max(b.y0, axb.y0))
                sc = 50 * ov(b, taken) + 30 * ov(b, pbox) + 20 * out + 300 * ambiguity(b, own) + 0.01 * rad
                if best_sc is None or sc < best_sc:
                    best, best_sc = (ox, oy, ha, va, b), sc
            if best_sc is not None and best_sc < 0.01 * radii[-1] + 1e-9:
                break
        ox, oy, ha, va, b = best
        ax.annotate(txt, (x, y), xytext=(ox, oy), textcoords="offset points", ha=ha, va=va, fontsize=fs, color=INK,
                    zorder=6, arrowprops=dict(arrowstyle="-", color=INK2, lw=LW_HAIR, shrinkA=0, shrinkB=point_r))
        taken.append(b)


def fig_variance(ctx, tier, written, sel="id"):
    allc = ctx.cohorts
    var = {c: {r["arm"]: r for r in ctx.csv(tier, c, "variance", sel)} if ctx.K(tier, c, sel) >= 2 else {}
           for c in allc}
    if not any(var.values()):
        print(f"  [F4 {tier}] no variance files with K >= 2 -> skipped")
        return
    arms = order_arms([a for c in allc for a in var[c]])
    ys = row_positions(arms)
    L = label_left(arms)
    H_a = 0.55 + 0.15 * (ys[-1] + 1)
    H_b = 1.55
    fig = plt.figure(figsize=(FULL_W, H_a + H_b + 0.6))
    gs = fig.add_gridspec(2, len(allc), height_ratios=[H_a, H_b], hspace=0.34, wspace=0.10, left=L, right=0.975,
                          top=0.97, bottom=fy(fig, 0.12))
    top_axes, any_approx = [], False
    for i, c in enumerate(allc):
        ax = fig.add_subplot(gs[0, i])
        top_axes.append(ax)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(ys[-1] + 0.75, ys[0] - 0.6)
        xgrid(ax)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", ".5", "1"])
        if i == 0:
            panel_letter(ax, "a", dx=-0.42, dy=1.0)
            ax.set_xlabel("ICC(1) = σ²dom / (σ²dom + σ²run)", fontsize=7, loc="left")
        if not var[c]:
            placeholder(ax, "K < 2 domains" if ctx.K(tier, c, sel) == 1 else "no runs yet", blank=True)
        for a, y in zip(arms, ys):
            r = var[c].get(a)
            if not r:
                continue
            col = FAMILY_COLOR[family(a)]
            if fin(r.get("icc_ci_lo")):
                exact = r.get("icc_ci_exact") is not False
                any_approx |= not exact
                ax.plot([r["icc_ci_lo"], r["icc_ci_hi"]], [y - 0.13, y - 0.13], color=col, lw=1.6 if exact else 1.2,
                        ls="-" if exact else (0, (1.6, 1.0)), solid_capstyle="butt", dash_capstyle="butt")
            if fin(r.get("icc_reml")):
                ax.plot([r["icc_reml"]], [y - 0.13], "o", ms=MS, color=col, mec="white", mew=0.7, zorder=3)
            med, ci = _bayes_icc(r)
            if ci and fin(ci[0]) and fin(ci[1]):
                ax.plot(ci, [y + 0.22, y + 0.22], color=col, lw=0.7, alpha=0.8, solid_capstyle="butt")
            if fin(med):
                ax.plot([med], [y + 0.22], "D", ms=3.6, mfc="white", mec=col, mew=0.9, zorder=3)
        ax.set_title(cohort_title(ctx, tier, c, sel), fontsize=7.5)
        arm_rows(ax, arms, ys, labels=(i == 0))
    lh = [Line2D([], [], color=INK2, lw=1.6, marker="o", ms=MS, mec="white", solid_capstyle="butt",
                 label="REML estimate, exact-F 95% CI (balanced)"),
          Line2D([], [], color=INK2, lw=0.7, marker="D", ms=3.6, mfc="white", mec=INK2,
                 label="Bayesian (half-Cauchy) median, 95% CrI\n([0, q95] at the zero boundary)")]
    if any_approx:
        lh.insert(1, Line2D([], [], color=INK2, lw=1.2, ls=(0, (1.6, 1.0)),
                            label="approximate F-based CI ($n_0$; unbalanced)"))
    legend_above(fig, top_axes, lh, L, ncol=2, handlelength=2.0, columnspacing=1.2)
    # (b) sigma_domain vs sigma_run (all-pairs form: neutral open dots, ERM filled; iso-ICC rays)
    xm = ym = 0
    for c in allc:
        for r in var[c].values():
            if fin(r.get("sigma_run_reml")):
                xm = max(xm, r["sigma_run_reml"])
            if fin(r.get("sigma_domain_reml")):
                ym = max(ym, r["sigma_domain_reml"])
    xm = xm * 1.6 if xm > 0 else 0.05   # room on the right for the ray labels, clear of the points
    ym = ym * 1.15 if ym > 0 else 0.1
    bot_axes = []
    for i, c in enumerate(allc):
        ax = fig.add_subplot(gs[1, i])
        bot_axes.append(ax)
        ax.set_xlim(0, xm)
        ax.set_ylim(0, ym)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        if i == 0:
            panel_letter(ax, "b", dx=-0.25, dy=1.0)
            ax.set_ylabel("σ domain (REML)", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("σ run (seeds)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(NOLEAD)
        ax.yaxis.set_major_formatter(NOLEAD)
        if i:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(COHORT_LABEL[c], fontsize=7.5, fontweight="normal")
        ray_txt = []
        for icc, lab in ((0.5, ".5"), (0.9, ".9"), (0.99, ".99")):
            k = math.sqrt(icc / (1 - icc))
            if k * xm <= ym:
                xe, ye, ha, va = xm, k * xm, "right", "bottom"
            else:
                xe, ye, ha, va = ym / k, ym, "left", "top"
            ax.plot([0, xe], [0, ye], color=GRIDC, lw=0.6, zorder=1)
            # 'ICC' spelled out once (shallowest ray); steeper rays that end close together on the top edge
            # get the bare value, and drop down along their ray if they still collide
            t_ = ax.text(xe + (0.01 * xm if ha == "left" else -0.01 * xm), ye - 0.01 * ym,
                         f"ICC {lab}" if icc == 0.5 else lab, fontsize=7, color=MUTED, ha=ha, va=va)
            ray_txt.append((t_, k))
        fig.canvas.draw()
        rr_ = fig.canvas.get_renderer()
        placed = []
        for t_, k in ray_txt:
            for frac in (1.0, 0.86, 0.72, 0.58):
                if frac < 1.0:
                    yy = frac * ym
                    t_.set_position((yy / k + 0.012 * xm, yy))
                    t_.set_ha("left")
                    t_.set_va("center")
                b = t_.get_window_extent(rr_)
                if not any(b.overlaps(o) for o in placed):
                    break
            placed.append(t_.get_window_extent(rr_))
        if not var[c]:
            placeholder(ax, "")
            continue
        pts = [(a, r["sigma_run_reml"], r["sigma_domain_reml"]) for a, r in var[c].items()
               if fin(r.get("sigma_run_reml")) and fin(r.get("sigma_domain_reml"))]
        for a, x, y in pts:
            erm = base_arm(a) == "erm" and "@" not in a
            if erm:
                ax.plot([x], [y], "o", ms=MS + 0.6, color=FAMILY_COLOR[0], mec="white", mew=0.7, zorder=4)
            else:
                ax.plot([x], [y], "o", ms=MS, mfc="white", mec=MUTED, mew=0.9, zorder=3)
        # direct labels with leaders: ERM, largest run noise, largest domain spread
        lab = []
        for a, x, y in pts:
            if base_arm(a) == "erm" and "@" not in a:
                lab.append((a, x, y))
        if pts:
            for p in (max(pts, key=lambda p: p[1]), max(pts, key=lambda p: p[2])):
                if p[0] not in [q[0] for q in lab]:
                    lab.append(p)
        place_labels(ax, [(x, y) for _, x, y in pts],
                     [(x, y, arm_label(a).replace("Compute-matched ", "Compute-\nmatched ")
                       .replace(" + ", "\n+ ")) for a, x, y in lab])
    caption_below(fig, bot_axes, L, "b: one dot per arm, REML estimates (ERM filled gray, other arms open); rays mark "
                  "constant ICC.\nLabelled: ERM, the arm with the largest run noise and the arm with the largest "
                  "domain spread.")
    fn = ctx.footnote([tier])
    save(fig, ctx.out, f"fig_variance_{tier}", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# F5  power: domains needed
# ----------------------------------------------------------------------------------------


def _typical_arm(rows):
    """Training arm (no TTA pseudo-arm) whose sigma_Delta at s = 3, rho = 0 is the median over arms."""
    cand = {}
    for r in rows:
        if "@" in str(r["arm"]) or r.get("seeds") != 3 or r.get("rho") != 0 or r.get("keff_mode") != "none":
            continue
        if fin(r.get("sd_delta")):
            cand[r["arm"]] = r
    if not cand:
        return None, None
    arms = sorted(cand, key=lambda a: cand[a]["sd_delta"])
    a = arms[(len(arms) - 1) // 2]
    return a, cand[a]


def label_hline(ax, yv, txt, color, fs=7):
    """Label a horizontal reference line at the left or right end, above or below it, wherever the fewest
    drawn curve points (sampled densely in display space) fall under the text."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if len(xy) < 2 or ln.get_linestyle() in ("None", "") or ln.get_transform() is not ax.transData:
            continue   # skip markers and axhline/axvline references (blended transforms)
        d = ax.transData.transform(xy[np.all(np.isfinite(xy), axis=1)])
        for (x0, y0), (x1, y1) in zip(d[:-1], d[1:]):
            n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 2))
            pts.append(np.c_[np.linspace(x0, x1, n), np.linspace(y0, y1, n)])
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    x0, x1 = ax.get_xlim()
    xl = x0 * (x1 / x0) ** 0.02 if ax.get_xscale() == "log" else x0 + 0.02 * (x1 - x0)
    xr = x1 / (x1 / x0) ** 0.02 if ax.get_xscale() == "log" else x1 - 0.02 * (x1 - x0)
    best = None
    for k, (xx, ha) in enumerate(((xl, "left"), (xr, "right"))):
        for yy, va in ((yv / 1.09, "top"), (yv * 1.09, "bottom")):
            t_ = ax.text(xx, yy, txt, fontsize=fs, color=color, ha=ha, va=va, zorder=6)
            b = t_.get_window_extent(r).padded(1.5)
            t_.remove()
            ab = ax.get_window_extent(r)
            inside = b.y0 >= ab.y0 - 4 and b.y1 <= ab.y1 + 4
            hit = int(np.sum((pts[:, 0] >= b.x0) & (pts[:, 0] <= b.x1) & (pts[:, 1] >= b.y0) & (pts[:, 1] <= b.y1)))
            sc = hit + (0 if inside else 10 ** 6) + 0.5 * k   # ties: prefer the left end, below the line
            if best is None or sc < best[0]:
                best = (sc, xx, yy, ha, va)
    _, xx, yy, ha, va = best
    ax.text(xx, yy, txt, fontsize=fs, color=color, ha=ha, va=va, zorder=6)


def fig_power(ctx, tier, written, sel="id", kmax=1000):
    if S is None:
        print("  [F5] stats_v2 not importable -> skipped")
        return
    allc = ctx.cohorts
    deltas = np.geomspace(0.01, 0.2, 40)
    seeds = (1, 3, 5)
    scol = {1: BLUE[3], 3: BLUE[7], 5: BLUE[11]}
    fig, axs = plt.subplots(1, len(allc), figsize=(FULL_W, 2.45), sharey=True, gridspec_kw=dict(wspace=0.22),
                            squeeze=False)
    axs = axs[0]
    if not any(ctx.csv(tier, c, "power", sel) for c in allc if ctx.K(tier, c, sel) >= 2):
        plt.close(fig)
        print(f"  [F5 {tier}] no power rows -> skipped")
        return
    any_ = False
    notes, subs, titles = [], [], []
    for i, (c, ax) in enumerate(zip(allc, axs)):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01, 0.2)
        ax.set_ylim(2, kmax)
        ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2])
        ax.xaxis.set_major_formatter(NOLEAD)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([2, 5, 10, 20, 50, 100, 200, 500, 1000])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(which="major", color=GRIDC, lw=LW_HAIR)
        ax.set_axisbelow(True)
        ax.set_xlabel("effect δ (AUROC)", fontsize=7)
        if i == 0:
            ax.set_ylabel("domains K for 80% power", fontsize=7)
        Ka = ctx.K(tier, c, sel)
        Ke = K_EXPECTED.get(c, Ka)
        # reference K drawn above the curves (dotted ink, distinct from the rho = .5 dashes) so it is never hidden
        # (a light band behind the curves keeps it visible where K_min sits on it, e.g. the K = 3 floor);
        # labels go just below the line at the left edge, where the curves are far above
        klabels = []
        if Ka >= 2:
            ax.axhspan(Ka / 1.07, Ka * 1.07, color="#e6e5e0", lw=0, zorder=1)
            ax.axhline(Ka, color=INK, lw=0.7, ls=(0, (1, 1.4)), zorder=5)
            klabels.append((Ka, f"K = {Ka}", INK))      # 'K of this study' in the legend
        if Ka != Ke:
            ax.axhline(Ke, color=MUTED, lw=0.7, ls=(0, (1, 1.4)), zorder=5)
            klabels.append((Ke, f"K = {Ke} planned", INK2))
        rows = ctx.csv(tier, c, "power", sel) if Ka >= 2 else []
        meta = ctx.meta(tier, c, sel)
        arm, r = _typical_arm(rows)
        titles.append((ax, COHORT_LABEL[c]))
        if r is None:
            for kk, txt, colr in klabels:
                ax.text(0.0107, kk / 1.09, txt, fontsize=7, color=colr, va="top", zorder=6)
            placeholder(ax, "no power rows yet (K < 2)" if not rows else "no non-TTA arm complete\n(TTA pseudo-arms "
                        "are excluded)")
            continue
        comp = dict(K=int(Ka), s2_dxa=r["sigma_dxa"] ** 2, s2_run_diff=r["sigma_run_diff"] ** 2,
                    s2_run_arm=r["sigma_run_arm"] ** 2, s2_run_ref=r["sigma_run_ref"] ** 2)
        design = [set(d) for d in meta.get("design", [])] if meta else None
        has_val = any(len(d) < Ka - 1 for d in design) if design else True
        tab = S.power_table(comp, deltas=tuple(deltas), seeds=seeds, rhos=(0.0, 0.5), kmax=kmax,
                            val_offset=-1 if has_val else None, train_sets=design, keff_modes=("ratio",))
        for s_ in seeds:
            for rho, ls in ((0.0, "-"), (0.5, (0, (3, 1.6)))):
                rr = [t for t in tab if t["seeds"] == s_ and t["rho"] == rho]
                rr.sort(key=lambda t: t["delta"])
                x = np.array([t["delta"] for t in rr])
                y = np.array([t["K_min"] if t["K_min"] is not None else np.nan for t in rr], float)
                ax.plot(x, y, color=scol[s_], lw=LW if rho == 0 else 0.9, ls=ls, zorder=3)
        any_ = True
        for kk, txt, colr in klabels:   # reference-K label on the side of the line with the fewest curve points
            label_hline(ax, kk, txt, colr)
        subs.append((ax, arm_label(arm), f"$\\sigma_\\Delta$(3) = " + f"{r['sd_delta']:.3f}".lstrip("0")))
        notes.append(c)
    # subtitles: '<arm> − ERM' wrapped at ' + ' when wider than the panel (+ gutter); one title pad for all
    fig.canvas.draw()
    rr_ = fig.canvas.get_renderer()
    nl = 2
    for ax, lab, sd in subs:
        t_ = ax.annotate(f"{lab} − ERM\n{sd}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 3),
                         textcoords="offset points", fontsize=7, color=INK2, va="bottom")
        lim = ax.get_window_extent(rr_).x1 + 0.08 * fig.dpi
        if t_.get_window_extent(rr_).x1 > lim and " + " in lab:
            t_.set_text(lab.replace(" + ", "\n+ ", 1) + f" − ERM\n{sd}")
            nl = max(nl, 3)
    for ax, ttl in titles:
        ax.set_title(ttl, fontsize=7.5, pad=2 + 11 * nl)
    hs = [Line2D([], [], color=scol[s_], lw=LW, label=f"s = {s_} seed{'s' if s_ > 1 else ''}") for s_ in seeds]
    hs.append(Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 1.6)), label="fold dependence ρ = 0.5 ($K_{eff}$)"))
    hs.append(Line2D([], [], color=INK, lw=0.7, ls=(0, (1, 1.4)), label="K of this study"))
    legend_above(fig, list(axs), hs, 0.08, ncol=5, handlelength=2.2, columnspacing=1.0)
    caption_below(fig, list(axs), 0.08,
                  "Two-sided paired t over domains, α = .05; $\\sigma_\\Delta^2 = \\sigma^2_{dom \\times arm} + "
                  "\\sigma^2_{run,diff}\\,/\\,s$, from the median-$\\sigma_\\Delta$ training arm of each "
                  "cohort.\nLines end where K > 1,000.")
    fn = ctx.footnote([tier])
    save(fig, ctx.out, f"fig_power_{tier}", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# F6  model scale and pretrained representations
# ----------------------------------------------------------------------------------------


def best_dg(summ):
    """DG objective with the highest worst-domain AUROC in one cohort summary (None if none)."""
    c = [a for a in summ if base_arm(a) in DG_ARMS and "@" not in a and fin(summ[a].get("worst"))]
    return max(c, key=lambda a: summ[a]["worst"]) if c else None


def fig_scale(ctx, written, sel="id"):
    allc = ctx.cohorts
    summ = {t: {c: ctx.summary(t, c, sel) for c in allc} for t in ("C", "S")}
    tiers = [t for t in ("C", "S") if any(summ[t][c] for c in allc)]
    if not tiers:
        print("  [F6] no summaries -> skipped")
        return
    rows = []   # (tier, key, label, family)
    groups = {"C": "Compact CNN, 64 px (tier C)", "S": "ImageNet ResNet-50, 96 px (tier S)",
              "F": "Pathology FMs and frozen probes (tier S)"}
    for t in tiers:
        rows.append(("hdr", groups[t]))
        rows += [(t, "erm", "ERM" if t == "C" else "ERM (RN50-IN FT)", 0),
                 (t, "@bestdg", "best DG objective", 1), (t, "he_jitter", "H&E jitter", 2), (t, "simclr", "SimCLR", 3)]
    if "S" in tiers:
        rows.append(("hdr", groups["F"]))
        rows += [("S", a, ARM_LABEL[a], 6) for a in ("erm_lunit_bt", "erm_lunit_dino", "probe_resnet50_imagenet",
                                                      "probe_resnet50_lunit_bt", "probe_vit_s16_lunit_dino")]
    ys, y = [], 0.0
    for r in rows:
        if r[0] == "hdr" and ys:
            y += 0.5
        ys.append(y)
        y += 1.0
    H = 0.75 + 0.16 * y
    fig, axs = plt.subplots(1, len(allc), figsize=(FULL_W, H), gridspec_kw=dict(wspace=0.10), squeeze=False)
    fig.subplots_adjust(left=0.215, right=0.99, top=1 - fy(fig, 0.42), bottom=fy(fig, 0.35))
    axs = axs[0]
    for i, (c, ax) in enumerate(zip(allc, axs)):
        vals, bd_lab = [], []
        bd = {t: best_dg(summ[t][c]) for t in tiers}
        for (r, yy) in zip(rows, ys):
            if r[0] == "hdr":
                continue
            t, key = r[0], r[1]
            arm = bd[t] if key == "@bestdg" else key
            s = summ[t][c].get(arm) if arm else None
            if not s:
                continue
            col = FAMILY_COLOR[r[3]]
            ci = s.get("worst_ci_seed")
            if ci and fin(ci[0]):
                ax.plot(ci, [yy, yy], color=col, lw=2.2, alpha=0.45, solid_capstyle="butt", zorder=2)
            ax.plot([s["worst"], s["mean"]], [yy, yy], color=col, lw=0.8, zorder=2)
            ax.plot([s["worst"]], [yy], "o", ms=MS, color=col, mec="white", mew=0.7, zorder=4)
            ax.plot([s["mean"]], [yy], "o", ms=MS, mfc="white", mec=col, mew=1.0, zorder=4)
            vals += [s["worst"], s["mean"]] + ([ci[0]] if ci and fin(ci[0]) else [])
            if key == "@bestdg":
                bd_lab.append((arm_label(arm), yy, min([s["worst"]] + ([ci[0]] if ci and fin(ci[0]) else [])),
                               max([s["mean"]] + ([ci[1]] if ci and fin(ci[1]) else []))))
        if vals:
            lo, hi = min(vals), max(vals)
            rng_ = max(hi - lo, 0.02)
            ax.set_xlim(max(0.4, lo - 0.06 * rng_), min(1.0, hi + 0.55 * rng_))   # AUROC <= 1
        # name of the best DG objective: right of its marks if it fits inside the axes, else left of them
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        axb = ax.get_window_extent(rr)
        for txt, yy, xl, xr in bd_lab:
            t = ax.annotate(txt, (xr, yy), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK2,
                            va="center", ha="left")
            if t.get_window_extent(rr).x1 > axb.x1:
                t.remove()
                ax.annotate(txt, (xl, yy), xytext=(-4, 0), textcoords="offset points", fontsize=7, color=INK2,
                            va="center", ha="right")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(NOLEAD)
        xgrid(ax)
        if not vals:
            placeholder(ax, "no runs yet", blank=True)
        ax.set_ylim(ys[-1] + 0.7, -0.7)
        ax.set_title(cohort_title(ctx, tiers[-1], c, sel) if len(tiers) == 1 else COHORT_LABEL[c], fontsize=7.5)
        ax.set_yticks([yy for r, yy in zip(rows, ys) if r[0] != "hdr"])
        ax.tick_params(axis="y", length=0, pad=9)
        ax.spines["left"].set_visible(False)
        if i == 0:
            ax.set_yticklabels([r[2] for r in rows if r[0] != "hdr"], fontsize=7)
            tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
            for r, yy in zip(rows, ys):
                if r[0] == "hdr":
                    ax.text(-0.75, yy, r[1], transform=tr, fontsize=7, fontweight="bold", va="center", ha="left")
                else:
                    ax.plot([-0.012], [yy], "s", ms=4, color=FAMILY_COLOR[r[3]], transform=tr, clip_on=False, mew=0)
            ax.set_xlabel("test AUROC", fontsize=7, loc="left")
        else:
            ax.set_yticklabels([])
    lh = [Line2D([], [], color=INK2, marker="o", ls="", ms=MS, label="worst domain"),
          Line2D([], [], color=INK2, lw=2.2, alpha=0.45, label="its seed-bootstrap 95% CI"),
          Line2D([], [], color=INK2, marker="o", ls="", ms=MS, mfc="white", label="mean over domains")]
    legend_above(fig, list(axs), lh, 0.215, ncol=3)
    cap = "x-axes are scaled per cohort (compare positions within a panel, not across panels)."
    if "S" in tiers:   # input size of the pathology block, from the run records (it is not uniform)
        res = defaultdict(list)
        for a in ("erm_lunit_bt", "erm_lunit_dino", "probe_resnet50_imagenet", "probe_resnet50_lunit_bt",
                  "probe_vit_s16_lunit_dino"):
            fp = sorted(glob.glob(os.path.join(ctx.runs, "runs", "S", allc[0], "fold*", f"{a}__*.json")))
            px = (read_json(fp[0]) or {}).get("input_res") if fp else None
            if px:
                res[int(px)].append("probes" if a.startswith("probe_") else ARM_LABEL[a])
        if len(res) > 1:
            cap += "\nPathology block input: " + "; ".join(
                f"{', '.join(dict.fromkeys(v))} at {k} px" for k, v in sorted(res.items())) + "."
    caption_below(fig, list(axs), 0.215, cap)
    extra = None if "S" in tiers else "Tier S (ResNet-50, Lunit FMs, probes) not available yet: only tier C is shown."
    fn = ctx.footnote([t for t in ("C", "S") if t in tiers], extra=extra) or extra
    save(fig, ctx.out, "fig_scale", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# F7  model-selection rule
# ----------------------------------------------------------------------------------------

# selection rules are not arms: shape + fill in ink / grays (family hues stay reserved for arm identity)
SEL_STYLE = {"id": dict(marker="o", color=INK, mfc=INK, mec="white", ms=MS + 0.3, mew=0.6),
             "ood": dict(marker="^", color=INK, mfc="white", mec=INK, ms=MS + 0.4, mew=0.9),
             "last": dict(marker="s", color=INK2, mfc=MUTED, mec="white", ms=MS - 0.2, mew=0.5),
             "oracle": dict(marker="D", color=MUTED, mfc="white", mec=MUTED, ms=MS - 0.6, mew=0.9)}


def signed_formatter(ax, axis="x"):
    """'+.05' style labels with as many decimals as the tick step needs (no two ticks print alike)."""
    locs = (ax.get_xticks() if axis == "x" else ax.get_yticks())
    lim = ax.get_xlim() if axis == "x" else ax.get_ylim()
    locs = [v for v in locs if lim[0] - 1e-12 <= v <= lim[1] + 1e-12]
    step = min(np.diff(sorted(locs))) if len(locs) > 1 else 0.1
    d = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    if abs(round(step * 10 ** d) - step * 10 ** d) > 1e-6:
        d += 1

    def f(v, p=None):
        if abs(v) < step * 1e-6:
            return "0"
        return f"{v:+.{d}f}".replace("0.", ".", 1).replace("-", "−")
    return matplotlib.ticker.FuncFormatter(f)


def fig_selection(ctx, tier, written):
    from scipy.stats import kendalltau
    allc = ctx.cohorts
    con = {c: {s_: {r["arm"]: r for r in ctx.csv(tier, c, "contrasts", s_)} if ctx.K(tier, c, s_) >= 2 else {}
               for s_ in SELS} for c in allc}
    # per cohort: the rules that exist (MIDOG 2021 has no OOD-val) and the arms present under every one of them
    rules = {c: [s_ for s_ in SELS if con[c][s_]] for c in allc}
    common = {c: ({a for a in con[c][rules[c][0]] if "@" not in a and fin(con[c][rules[c][0]][a].get("mean_diff"))}
                  .intersection(*[{a for a in con[c][s_] if fin(con[c][s_][a].get("mean_diff"))}
                                  for s_ in rules[c][1:]]) if rules[c] else set()) for c in allc}
    arms = order_arms([a for c in allc for a in common[c]])
    if not arms:
        print(f"  [F7 {tier}] no contrasts -> skipped")
        return
    ys = row_positions(arms)
    L = label_left(arms)
    H = 1.0 + 0.15 * (ys[-1] + 1)
    fig, axs = plt.subplots(1, len(allc), figsize=(FULL_W, H), gridspec_kw=dict(wspace=0.10), squeeze=False)
    fig.subplots_adjust(left=L, right=0.99, top=0.97, bottom=fy(fig, 0.45))
    axs = axs[0]
    taus = {}
    for i, (c, ax) in enumerate(zip(allc, axs)):
        vals = []
        for a, y in zip(arms, ys):
            if a not in common[c]:
                continue
            pts = {s_: con[c][s_][a]["mean_diff"] for s_ in rules[c]}
            v = list(pts.values())
            ax.plot([min(v), max(v)], [y, y], color="#e6e5e0", lw=3.2, solid_capstyle="round", zorder=1)
            r = con[c]["id"].get(a)
            if r and r.get("ci") and fin(r["ci"][0]):
                ax.plot(r["ci"], [y, y], color=INK, lw=0.7, zorder=2)
                vals += r["ci"]
            for s_ in ("oracle", "last", "ood", "id"):
                if s_ in pts:
                    st = SEL_STYLE[s_]
                    ax.plot([pts[s_]], [y], ls="", marker=st["marker"], ms=st["ms"], color=st["color"], mfc=st["mfc"],
                            mec=st["mec"], mew=st["mew"], zorder=4 if s_ == "id" else 3)
            vals += v
        ax.axvline(0, color=AXISC, lw=0.8, zorder=0)
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.10 * (hi - lo + 1e-3)
            ax.set_xlim(min(lo - pad, -0.01), max(hi + pad, 0.01))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(signed_formatter(ax))
        xgrid(ax)
        if not vals:
            placeholder(ax, "no arm complete\nunder every rule", blank=True)
        ax.set_ylim(ys[-1] + 0.7, ys[0] - 0.7)
        ax.set_title(cohort_title(ctx, tier, c), fontsize=7.5)
        arm_rows(ax, arms, ys, labels=(i == 0))
        if i == 0:
            ax.set_xlabel("Δ mean AUROC vs ERM", fontsize=7, loc="left")
        # rank agreement of the gains with the primary (ID-val) rule, over the same arm set for every rule
        com = [a for a in arms if a in common[c]]
        txt = []
        for s_ in ("ood", "last", "oracle"):
            if s_ not in rules[c]:
                txt.append(f"{SEL_LABEL[s_]} n/a")
            elif len(com) >= 3 and "id" in rules[c]:
                t = kendalltau([con[c]["id"][a]["mean_diff"] for a in com], [con[c][s_][a]["mean_diff"] for a in com])[0]
                tt = f"{t:.2f}"
                tt = tt.replace("0.", ".", 1) if tt.lstrip("-").startswith("0.") else tt   # no leading zero
                txt.append(f"{SEL_LABEL[s_]} " + tt.replace("-", "−"))
        taus[c] = (txt, len(com))
    # tau_b block under each panel, below the lowest tick label / x label of the row
    y0 = bottom_y(fig, list(axs)) - fy(fig, 0.06)
    for c, ax in zip(allc, axs):
        txt, n = taus.get(c, ([], 0))
        if txt:
            fig.text(ax.get_position().x0, y0, f"τb vs ID-val ({n} arms):\n" + "\n".join(txt), fontsize=7, color=INK2,
                     va="top", linespacing=1.2)
    lh = [Line2D([], [], ls="", marker=SEL_STYLE[s_]["marker"], ms=SEL_STYLE[s_]["ms"], color=SEL_STYLE[s_]["color"],
                 mfc=SEL_STYLE[s_]["mfc"], mec=SEL_STYLE[s_]["mec"] if s_ != "id" else INK, mew=SEL_STYLE[s_]["mew"],
                 label=SEL_LABEL[s_] + (" (primary)" if s_ == "id" else " (bound)" if s_ == "oracle" else ""))
          for s_ in SELS]
    lh += [Line2D([], [], color=INK, lw=0.7, label="ID-val 95% CI"),
           Line2D([], [], color="#e6e5e0", lw=3.2, solid_capstyle="butt", label="range over rules")]
    legend_above(fig, list(axs), lh, L, ncol=6, handletextpad=0.3, columnspacing=0.9)
    note = ("Each panel shows only the arms analysed under every rule available in that cohort, so all markers\n"
            "and τb use one arm set. x-axes are scaled per cohort (ranges differ up to several-fold).")
    fig.text(L, lowest_y(fig) - fy(fig, 0.06), note, fontsize=7, color=INK2, va="top")
    fn = ctx.footnote([tier])
    save(fig, ctx.out, f"fig_selection_{tier}", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# F8  Kendall tau between cohort rankings + 2x2 decomposition
# ----------------------------------------------------------------------------------------


def _sg(v):
    """signed, no leading zero: +.27 / -.05 (with a true minus sign)."""
    t = f"{v:+.2f}".replace("0.", ".", 1)
    if abs(v) >= 0.995:
        t = "+1" if v > 0 else "-1"
    return t.replace("-", "\u2212")


def fmt_p(p):
    """'p < .001' / 'p = .012' / 'p = .43' (APA: no leading zero, never 'p = .00')."""
    if not fin(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    t = f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    return "p = " + (t[1:] if t.startswith("0.") else t)


def kendall_matrix(ax, ctx, rows, cohorts, fs=7, show_ci=True, label_fs=None, wrap=True):
    """Lower triangle (rows = cohorts[1:], columns = cohorts[:-1]) of tau_b on a blue<->red diverging scale."""
    n = len(cohorts)
    label_fs = label_fs or fs
    norm = mcolors.Normalize(-1, 1)
    idx = {c: i for i, c in enumerate(cohorts)}
    look = {}
    for r in rows:
        if r.get("key") != "worst":
            continue
        a, b = r["cohort_a"], r["cohort_b"]
        if a in idx and b in idx:
            look[(a, b)] = look[(b, a)] = r
    ax.set_xlim(-1.05, n - 1)
    ax.set_ylim(n - 1, -0.42)
    ax.axis("off")
    g = 0.03
    lab = (lambda c: COHORT_LABEL[c].replace(" ", "\n", 1)) if wrap else (lambda c: COHORT_LABEL[c])
    for j in range(n - 1):
        ax.text(j + 0.5, -0.06, lab(cohorts[j]), ha="center", va="bottom", fontsize=label_fs, linespacing=1.1)
    for i in range(1, n):
        ax.text(-0.06, i - 0.5, lab(cohorts[i]), ha="right", va="center", fontsize=label_fs, linespacing=1.1)
        for j in range(i):
            r = look.get((cohorts[i], cohorts[j]))
            x0, y0 = j + g, i - 1 + g
            if not r or not fin(r.get("tau_b")):
                ax.add_patch(Rectangle((x0, y0), 1 - 2 * g, 1 - 2 * g, fc="#f6f6f4", ec="none"))
                ax.text(j + 0.5, i - 0.5, "–", ha="center", va="center", color=MUTED, fontsize=fs)
                continue
            col = DIV(norm(r["tau_b"]))
            ax.add_patch(Rectangle((x0, y0), 1 - 2 * g, 1 - 2 * g, fc=col, ec="none"))
            t = _sg(r["tau_b"])
            if show_ci and r.get("ci") and fin(r["ci"][0]):
                t += f"\n[{_sg(r['ci'][0]).lstrip('+')}, {_sg(r['ci'][1]).lstrip('+')}]"
            ax.text(j + 0.5, i - 0.5, t, ha="center", va="center", fontsize=fs, color=text_on(col), linespacing=1.25)
    return norm


def fig_kendall(ctx, tier, written, sel="id"):
    cross = os.path.join(ctx.an, tier, "cross")
    rows = read_csv(os.path.join(cross, f"kendall_{sel}.csv"))
    inter = read_json(os.path.join(cross, f"interaction_{sel}.json")) or {}
    clus = read_json(os.path.join(cross, f"clustering_{sel}.json")) or {}
    if not rows and not ctx.avail(tier, sel):
        print(f"  [F8 {tier}] no cross-cohort output -> skipped")
        return
    fig = plt.figure(figsize=(FULL_W, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.85, 0.95, 1.2], wspace=0.6, left=0.0, right=0.99, top=0.84,
                          bottom=0.22)
    axA = fig.add_subplot(gs[0])
    norm = kendall_matrix(axA, ctx, rows, ctx.cohorts)
    n_ = len(ctx.cohorts)
    fx = lambda x: (x + 1.05) / (n_ - 1 + 1.05)  # noqa: E731
    fyy = lambda y: (n_ - 1 - y) / (n_ - 1 + 0.42)  # noqa: E731
    cax = axA.inset_axes([fx(1.25), fyy(0.42), fx(2.75) - fx(1.25), 0.035])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm, DIV), cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, width=LW_HAIR)
    cb.ax.set_xticklabels(["−1", "0", "1"])
    cb.set_label("τb  [95% CI]", fontsize=7, labelpad=1)
    na = sorted({int(r["n_arms"]) for r in rows if r.get("key") == "worst" and fin(r.get("n_arms"))})
    axA.text(-1.05, len(ctx.cohorts) - 0.95, "Arms ranked by worst-domain AUROC;\n" +
             (f"{'–'.join(map(str, [na[0], na[-1]])) if len(na) > 1 else na[0]} common complete arms per pair."
              if na else "no pair with ≥ 3 common complete arms yet."), fontsize=7, color=INK2, va="top")
    # (b) 2x2 decomposition of the between-cohort interaction SS (a quantity, not an arm: neutral ink)
    axB = fig.add_subplot(gs[1])
    t2 = inter.get("task_shift_2x2") or {}
    keys = [("task", "task"), ("shift", "shift type"), ("task_x_shift", "task × shift")]
    if t2 and "task" in t2:
        n_arm = int(t2.get("n_arms", 0))
        fr = [t2[k]["frac"] for k, _ in keys]
        axB.barh(range(3), fr, height=0.5, color=INK2, zorder=2)
        fig.canvas.draw()
        rr_ = fig.canvas.get_renderer()
        xmax_px = axB.get_window_extent(rr_).x1 + 0.12 * fig.dpi      # gutter before panel c
        for i, (k, _) in enumerate(keys):
            lab = f"{fr[i] * 100:.0f}%"
            if k in ("task", "shift") and fin(t2[k].get("p_F")) and n_arm >= MIN_COMMON_ARMS:
                lab += "  (F " + fmt_p(t2[k]["p_F"]) + ")"
            t_ = axB.text(fr[i] + 0.02, i, lab, va="center", fontsize=7, color=INK)
            if t_.get_window_extent(rr_).x1 > xmax_px:      # does not fit right of the bar: break the line
                t_.set_text(lab.replace("  (", "\n("))
                t_.set_linespacing(1.0)
                if t_.get_window_extent(rr_).x1 > xmax_px:  # still too wide: inside the bar, white on ink
                    t_.set_position((fr[i] - 0.02, i))
                    t_.set_ha("right")
                    t_.set_color("white")
        note = (f"Per-domain \u0394 AUROC vs ERM of the\n{n_arm} other arms common to all 4 cohorts.\n"
                f"Cohort-level split descriptive\n(3 balanced 2-vs-2 splits)"
                + (f";\nF p over arms needs ≥ {MIN_COMMON_ARMS} arms." if n_arm < MIN_COMMON_ARMS else "."))
        axB.annotate(note, xy=(0, 0), xycoords="axes fraction", xytext=(-46, -22), textcoords="offset points",
                     fontsize=7, color=INK2, va="top")
    else:
        placeholder(axB, "needs all 4 cohorts\nof the 2 × 2 design")
    axB.set_yticks(range(3))
    axB.set_yticklabels([k[1] for k in keys])
    axB.set_ylim(2.6, -0.6)
    axB.set_xlim(0, 1.0)
    axB.set_xticks([0, 0.5, 1])
    axB.set_xticklabels(["0", "50%", "100%"])
    axB.tick_params(axis="y", length=0)
    axB.spines["left"].set_visible(False)
    xgrid(axB)
    # (c) tau by pair class: one row per pair (pair names as tick labels), class name as a group header
    axC = fig.add_subplot(gs[2])
    classes = [("same_task", "same task"), ("same_shift", "same shift"), ("neither", "neither")]
    pairs = {r_["cohort_a"] + "-" + r_["cohort_b"]: r_ for r_ in rows if r_.get("key") == "worst"}
    yt, yl, y = [], [], 0.0
    if clus:
        for k, lab in classes:
            if k not in clus:
                continue
            axC.text(-1.08, y, lab, ha="left", va="center", fontsize=7, fontweight="bold", color=INK)
            y0 = y + 0.55
            for j, pn in enumerate(clus[k]["pairs"]):
                yy = y + 1.0 + j
                r_ = pairs.get(pn)
                yt.append(yy)
                yl.append("–".join(COHORT_SHORT.get(x, x) for x in pn.split("-")))
                if not r_:
                    continue
                if r_.get("ci") and fin(r_["ci"][0]):
                    axC.plot(r_["ci"], [yy, yy], color=INK2, lw=0.7, zorder=3)
                axC.plot([r_["tau_b"]], [yy], "o", ms=MS - 0.5, color=INK2, mec="white", mew=0.6, zorder=4)
            y1 = y + 1.0 + len(clus[k]["pairs"]) - 0.55
            axC.plot([clus[k]["mean_tau"]] * 2, [y0, y1], color=INK, lw=1.6, zorder=2, solid_capstyle="butt")
            y += len(clus[k]["pairs"]) + 1.35
        d = clus.get("same_task_minus_same_shift")
        if d:
            axC.annotate(f"same task − same shift:\n{_sg(d['point'])} [{_sg(d['ci'][0])}, {_sg(d['ci'][1])}]",
                         xy=(0, 0), xycoords="axes fraction", xytext=(-52, -24), textcoords="offset points",
                         fontsize=7, color=INK2, va="top")
    else:
        placeholder(axC, "needs all 4 cohorts")
    axC.axvline(0, color=AXISC, lw=0.8, zorder=0)
    axC.set_xlim(-1.08, 1.08)
    axC.set_xticks([-1, 0, 1])
    axC.set_xticklabels(["−1", "0", "1"])
    axC.set_yticks(yt)
    axC.set_yticklabels(yl, fontsize=7)
    axC.set_ylim(max(y - 0.85, 1), -0.6)
    axC.tick_params(axis="y", length=0)
    axC.spines["left"].set_visible(False)
    xgrid(axC)
    hs = [Line2D([], [], color=INK2, marker="o", ms=MS - 0.5, lw=0.7, label="pair τb, 95% CI"),
          Line2D([], [], color=INK, lw=1.6, label="class mean")]
    # panel titles a / b / c on one baseline above the tallest panel
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    Hpx = fig.get_figheight() * fig.dpi
    ytop = max(axA.get_tightbbox(rr).y1, axB.get_window_extent(rr).y1, axC.get_window_extent(rr).y1) / Hpx
    yttl = ytop + fy(fig, 0.10)
    for ax, letter, ttl in ((axA, "a", "Kendall τb between cohort rankings"), (axB, "b", "Split of between-cohort SS"),
                            (axC, "c", "τb by cohort-pair type")):
        x0 = ax.get_tightbbox(rr).x0 / (fig.get_figwidth() * fig.dpi) if ax is not axA else 0.0
        fig.text(x0, yttl, letter, fontsize=9, fontweight="bold", va="baseline")
        fig.text(x0 + 0.025, yttl, ttl, fontsize=8, fontweight="bold", va="baseline")
    leg_y = bottom_y(fig, [axC]) - fy(fig, 0.04)
    x0c = axC.get_tightbbox(rr).x0 / (fig.get_figwidth() * fig.dpi)
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(x0c, leg_y), ncol=2, fontsize=7, handlelength=1.3,
               columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0)
    fn = ctx.footnote([tier])
    save(fig, ctx.out, f"fig_kendall_{tier}", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# S1  calibration
# ----------------------------------------------------------------------------------------


def _preds(ctx, tier, cohort, fold, arm, key="p_ood_test_id"):
    ys, ps = [], []
    for fp in sorted(glob.glob(os.path.join(ctx.runs, "runs", tier, cohort, f"fold{fold}", f"{arm}__*__s*_preds.npz"))):
        try:
            d = np.load(fp)
            if key in d and "y_ood_test" in d:
                ys.append(d["y_ood_test"].astype(int))
                ps.append(d[key].astype(float))
        except Exception:  # noqa: BLE001
            continue
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(ps), len(ys)


def reliability(y, p, bins=10, min_n=30):
    e = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    xs, fs, ns = [], [], []
    for k in range(bins):
        m = b == k
        if m.sum() >= min_n:
            xs.append(p[m].mean())
            fs.append(y[m].mean())
            ns.append(int(m.sum()))
    ece = float(sum(n * abs(f - x) for x, f, n in zip(xs, fs, ns)) / max(1, len(y)))
    return np.array(xs), np.array(fs), ece


def fig_calibration(ctx, tier, written, sel="id"):
    allc = ctx.cohorts
    summ = {c: ctx.summary(tier, c, sel) for c in allc}
    if not any(summ.values()):
        print(f"  [S1 {tier}] no summaries -> skipped")
        return
    arms = order_arms([a for c in allc for a in summ[c] if fin(summ[c][a].get("ece_mean"))])
    ys = row_positions(arms)
    L = label_left(arms)
    H_a = 0.5 + 0.15 * (ys[-1] + 1)
    H_b = 1.55
    fig = plt.figure(figsize=(FULL_W, H_a + H_b + 0.6))
    gs = fig.add_gridspec(2, len(allc), height_ratios=[H_a, H_b], hspace=0.30, wspace=0.12, left=L, right=0.99,
                          top=1 - fy(fig, 0.33), bottom=fy(fig, 0.35))
    emax = max([summ[c][a].get("ece_worst") or 0 for c in allc for a in summ[c]] + [0.05])
    top_axes = []
    for i, c in enumerate(allc):
        ax = fig.add_subplot(gs[0, i])
        top_axes.append(ax)
        ax.set_xlim(0, emax * 1.08)
        ax.set_ylim(ys[-1] + 0.7, ys[0] - 0.7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.xaxis.set_major_formatter(NOLEAD)
        xgrid(ax)
        if i == 0:
            panel_letter(ax, "a", dx=-0.42, dy=1.0)
            ax.set_xlabel("ECE (test, 15-bin)", fontsize=7, loc="left")
        if not summ[c]:
            placeholder(ax, "no runs yet")
        for a, y in zip(arms, ys):
            s = summ[c].get(a)
            if not s or not fin(s.get("ece_mean")):
                continue
            col = FAMILY_COLOR[family(a)]
            if fin(s.get("ece_worst")):
                ax.plot([s["ece_mean"], s["ece_worst"]], [y, y], color=col, lw=0.8)
                ax.plot([s["ece_worst"]], [y], "o", ms=MS, color=col, mec="white", mew=0.7, zorder=4)
            ax.plot([s["ece_mean"]], [y], "o", ms=MS, mfc="white", mec=col, mew=1.0, zorder=3)
        ax.set_title(cohort_title(ctx, tier, c, sel), fontsize=7.5)
        arm_rows(ax, arms, ys, labels=(i == 0))
    lh = [Line2D([], [], color=INK2, marker="o", ls="", ms=MS, label="worst domain"),
          Line2D([], [], color=INK2, marker="o", ls="", ms=MS, mfc="white", label="mean over domains")]
    legend_above(fig, top_axes, lh, L, ncol=2)
    # (b) reliability at ERM's hardest domain
    hs_all, bot_axes = {}, []
    for i, c in enumerate(allc):
        ax = fig.add_subplot(gs[1, i])
        bot_axes.append(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.plot([0, 1], [0, 1], color=AXISC, lw=0.7, zorder=1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(NOLEAD)
        ax.yaxis.set_major_formatter(NOLEAD)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("predicted P(positive)", fontsize=7)
        if i == 0:
            panel_letter(ax, "b", dx=-0.25, dy=1.0)
            ax.set_ylabel("observed fraction", fontsize=7)
        else:
            ax.set_yticklabels([])
        s = summ[c]
        meta = ctx.meta(tier, c, sel)
        if "erm" not in s or not meta:
            ax.set_title(COHORT_LABEL[c], fontsize=7, fontweight="normal")
            placeholder(ax, "ERM not complete")
            continue
        dom = s["erm"]["worst_domain"]
        fold = meta["domains"].index(dom) if dom in meta["domains"] else None
        ax.set_title(f"{COHORT_LABEL[c]}: {dshort(dom)}", fontsize=7, fontweight="normal")
        sel_arms = [("erm", FAMILY_COLOR[0]), ("simclr", FAMILY_COLOR[3]), ("he_jitter", FAMILY_COLOR[2])]
        bd = best_dg(s)
        if bd:
            sel_arms.append((bd, FAMILY_COLOR[1]))
        for a, col in sel_arms:
            if fold is None or a not in s:
                continue
            pr = _preds(ctx, tier, c, fold, base_arm(a))
            if pr is None:
                continue
            x, f, e = reliability(pr[0], pr[1])
            ax.plot(x, f, color=col, lw=1.0 if a != "erm" else 1.3, marker="o", ms=2.8, mec="white", mew=0.4,
                    zorder=4 if a == "erm" else 3)
            hs_all[a] = col
        # the best DG objective is named in the panel (4 crossing curves leave no room for clean direct labels)
        if bd and bd in hs_all:
            ax.text(0.97, 0.03, f"best DG:\n{arm_label(bd)}", transform=ax.transAxes, fontsize=7, color=INK,
                    va="bottom", ha="right")   # under-confident corner: empty in these panels
    order = [a for a in ("erm", "simclr", "he_jitter") if a in hs_all] + [a for a in hs_all if base_arm(a) in DG_ARMS]
    lh = [Line2D([], [], color=hs_all[a], lw=1.1, marker="o", ms=2.8,
                 label=arm_label(a) if base_arm(a) not in DG_ARMS else "best DG objective (per cohort)") for a in order]
    lh = list({h.get_label(): h for h in lh}.values())
    if lh:
        fig.legend(handles=lh, loc="upper left", bbox_to_anchor=(L, bottom_y(fig, bot_axes) - fy(fig, 0.04)), ncol=4,
                   fontsize=7, borderaxespad=0.0)
    fig.text(L, lowest_y(fig) - fy(fig, 0.06),
             "b: test predictions (ID-val-selected checkpoint) of the fold whose test domain is ERM's worst,\n"
             "pooled over seeds; 10 equal-width bins with ≥ 30 patches; the best DG objective (highest worst-domain\n"
             "AUROC) of each cohort is named in its panel.", fontsize=7,
             color=INK2, va="top")
    fn = ctx.footnote([tier])
    save(fig, ctx.out, f"fig_calibration_{tier}", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# S2  DomainBed external example
# ----------------------------------------------------------------------------------------


# arXiv 2007.01434 appendix, Colored MNIST table: "ADA" / "CondADA"; every other table (and the main table, whose
# CMNIST averages 51.5 / 51.9 equal the means of the ADA / CondADA rows) calls them DANN / C-DANN.
DOMAINBED_ALIAS = {"ADA": "DANN", "CondADA": "C-DANN"}


def fig_domainbed(ctx, written, selection="training_domain"):
    rows = [r for r in read_csv(ctx.args.domainbed) if r.get("selection") == selection]
    for r in rows:   # the Colored MNIST per-environment table of the e-print uses the old names ADA / CondADA
        r["algorithm"] = DOMAINBED_ALIAS.get(r["algorithm"], r["algorithm"])
    if not rows:
        print("  [S2] no DomainBed rows -> skipped")
        return
    dsets = list(dict.fromkeys(r["dataset"] for r in rows))
    algs = list(dict.fromkeys(r["algorithm"] for r in rows))
    ncol = 4
    nrow = math.ceil((len(dsets) + 1) / ncol)
    H = nrow * (0.45 + 0.15 * len(algs)) + 0.4
    fig, axs = plt.subplots(nrow, ncol, figsize=(FULL_W, H), squeeze=False,
                            gridspec_kw=dict(wspace=0.10, hspace=0.42))
    fig.subplots_adjust(left=0.12, right=0.99, top=1 - fy(fig, 0.30), bottom=fy(fig, 0.35))
    look = {(r["dataset"], r["algorithm"]): r for r in rows}
    y = np.arange(len(algs))
    for k, ax in enumerate(axs.flat):
        if k >= len(dsets):
            ax.axis("off")
            continue
        ds = dsets[k]
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(len(algs) - 0.4, -0.6)
        xgrid(ax)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", ".5", "1"])
        Kd = None
        for j, a in enumerate(algs):
            r = look.get((ds, a))
            if not r:
                continue
            Kd = int(r["K"])
            col = FAMILY_COLOR[0] if a == "ERM" else FAMILY_COLOR[1]
            if fin(r.get("code_icc_lo")):
                ax.plot([r["code_icc_lo"], r["code_icc_hi"]], [j - 0.13] * 2, color=col, lw=1.6, solid_capstyle="butt")
            ax.plot([r["code_icc"]], [j - 0.13], "o", ms=MS - 0.3, color=col, mec="white", mew=0.6, zorder=3)
            ci = r.get("halfcauchy10_icc_ci")
            if ci and fin(ci[0]):
                ax.plot(ci, [j + 0.22] * 2, color=col, lw=0.7, solid_capstyle="butt")
            if fin(r.get("halfcauchy10_icc_med")):
                ax.plot([r["halfcauchy10_icc_med"]], [j + 0.22], "D", ms=3.3, mfc="white", mec=col, mew=0.8, zorder=3)
        ax.set_title(f"{ds} (K = {Kd})", fontsize=7.5)
        ax.set_yticks(y)
        if k % ncol == 0:
            ax.set_yticklabels(algs, fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        if k // ncol == nrow - 1 or k + ncol >= len(dsets):
            ax.set_xlabel("ICC(1) of test accuracy", fontsize=7)
    lh = [Line2D([], [], color=INK2, lw=1.6, marker="o", ms=MS - 0.3, mec="white", solid_capstyle="butt",
                 label="ANOVA ICC,\nexact-F 95% CI"),
          Line2D([], [], color=INK2, lw=0.7, marker="D", ms=3.3, mfc="white", mec=INK2,
                 label="Bayesian median,\n95% CrI (half-\nCauchy, 10 pp)"),
          Line2D([], [], color=FAMILY_COLOR[0], marker="s", ls="", ms=4.5, label="ERM"),
          Line2D([], [], color=FAMILY_COLOR[1], marker="s", ls="", ms=4.5, label="DG algorithms")]
    last = axs.flat[len(dsets)] if len(dsets) < nrow * ncol else None
    if last is not None:
        last.legend(handles=lh, loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=7, ncol=1, handlelength=1.4,
                    borderaxespad=0.0)
    caption_below(fig, list(axs.flat), 0.12, "DomainBed (Gulrajani & Lopez-Paz 2021), per-environment tables of the "
                  "e-print appendix, training-domain validation, 3 trials;\nSD from SE as in DomainBed's "
                  "collect_results. Colored MNIST rows 'ADA' / 'CondADA' of the source table are DANN / C-DANN.")
    save(fig, ctx.out, "fig_domainbed", written=written)


# ----------------------------------------------------------------------------------------
# S3  tuning sensitivity (runs_tune)
# ----------------------------------------------------------------------------------------

HP_SYM = {"eta": "η", "irm_lambda": "λ", "coral_lambda": "λ", "ms_p": "p", "ms_alpha": "α", "meta_lr": "meta-lr",
          "lisa_alpha": "α", "p_sel": "p_sel", "rho": "ρ", "jitter": "strength", "lr": "lr"}


def load_tune(ctx, tier, cohort):
    """arm -> config (tuple of (key, value)) -> fold -> dict(test, idval, ok)."""
    out = defaultdict(lambda: defaultdict(dict))
    hps = defaultdict(list)
    files = glob.glob(os.path.join(ctx.runs, "runs_tune", tier, cohort, "fold*", "*.json"))
    recs = []
    for fp in files:
        try:
            r = json.load(open(fp))
        except Exception:  # noqa: BLE001
            continue
        if r.get("schema") != "v2":
            continue
        recs.append(r)
        hps[r["arm"]].append(r.get("hparams") or {})
    keys, fold_keys = {}, {}
    per_fold = defaultdict(lambda: defaultdict(list))
    for r in recs:
        per_fold[r["arm"]][int(r["fold"])].append(r.get("hparams") or {})
    for a, L in hps.items():
        ks = sorted({k for h in L for k in h if k not in ("total_steps",)})
        var = [k for k in ks if len({json.dumps(h.get(k)) for h in L}) > 1]
        # fold-level keys: vary across folds but are constant within each fold (tier-S stage-2 grids run
        # at the fold's own selected base LR) -> not a grid axis; each fold contributes its own slice
        fk = [k for k in var if all(len({json.dumps(h.get(k)) for h in Lf}) == 1 for Lf in per_fold[a].values())]
        if fk and len(fk) < len(var):
            keys[a] = [k for k in var if k not in fk]
            fold_keys[a] = fk
        else:
            keys[a] = var
    for r in recs:
        a = r["arm"]
        if not keys.get(a):
            continue
        h = r.get("hparams") or {}
        cfg = tuple((k, h.get(k)) for k in keys[a])
        sid = (r.get("selected") or {}).get("id")
        iv = None
        for ck in r.get("checkpoints") or []:
            if ck.get("step") == sid:
                iv = (ck.get("id_val") or {}).get("auroc")
        te = ((r.get("test_at") or {}).get("id") or {}).get("auroc")
        out[a][cfg][int(r["fold"])] = dict(test=te, idval=iv, ok=r.get("status", "ok") == "ok",
                                            fold_hp={k: h.get(k) for k in fold_keys.get(a, [])})
    return out, keys, fold_keys


def _num(v):
    """Hyperparameter value for a tick label: 10⁻⁵ / .0003 instead of 1e-05 / 0.0003 (exponents only below
    1e-4: '3×10⁻⁴' is wider than the 5-rate BT LR panel allows)."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return str(v)
    if v != 0 and (abs(v) < 1e-4 or abs(v) >= 1e4):
        e = int(math.floor(math.log10(abs(v))))
        m = v / 10 ** e
        ms = f"{m:.2g}"
        return (f"$10^{{{e}}}$" if ms == "1" else f"${ms}\\times10^{{{e}}}$")
    t = f"{v:g}"
    return t[1:] if t.startswith("0.") else t   # .003, no leading zero (as on every axis)


def _cfg_label(cfg, e=0):
    return "\n".join(_num(v / 10 ** e if e else v) for _, v in cfg)


def _row_exp(cfgs):
    """Power of ten shared by the tick labels of a one-hyperparameter row whose smallest value is < 1e-3
    (.0003 / 3x10^-5 tick labels crowd a 5-value panel); 0 = plain values."""
    vals = [v for cf in cfgs for _, v in cf]
    if not vals or any(len(cf) != 1 for cf in cfgs) or \
            not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in vals):
        return 0
    m = min(vals)
    return int(math.floor(math.log10(m) + 1e-9)) if m < 1e-3 else 0


def fig_tuning(ctx, tier, written):
    allc = ctx.cohorts
    tune = {c: load_tune(ctx, tier, c) for c in allc}
    arms = order_arms({a for c in allc for a in tune[c][0]})
    if not arms:
        print(f"  [S3 {tier}] no tuning records -> skipped")
        return
    nr, nc = len(arms), len(allc)
    rh = min(1.0, 8.6 / nr)        # tier S has 10 tuned arms: keep the page height below ~10 in
    fig, axs = plt.subplots(nr, nc, figsize=(FULL_W, 0.35 + rh * nr), squeeze=False,
                            gridspec_kw=dict(wspace=0.2, hspace=0.85))
    fig.subplots_adjust(left=0.165, right=0.99, top=0.97, bottom=fy(fig, 0.3))
    # ERM reference (final runs, seed 0, ID-val selection): worst over folds
    erm_ref = {}
    for c in allc:
        v = {}
        for fp in glob.glob(os.path.join(ctx.runs, "runs", tier, c, "fold*", "erm__*__s0.json")):
            r = read_json(fp) or {}
            te = ((r.get("test_at") or {}).get("id") or {}).get("auroc")
            if fin(te):
                v[int(r["fold"])] = te
        if len(v) == K_EXPECTED.get(c, 0):
            erm_ref[c] = min(v.values())
    partial, any_note = False, False
    erm_drawn, erm_missing = set(), set()
    row_e = {a: _row_exp([cf for c in allc for cf in tune[c][0].get(a, {})]) for a in arms}
    for j, c in enumerate(allc):
        data, keys, fkeys = tune[c]
        Ke = K_EXPECTED.get(c, 0)
        col_vals = []
        for i, a in enumerate(arms):
            ax = axs[i, j]
            cfgs = sorted(data.get(a, {}), key=lambda cf: tuple(float(v) if isinstance(v, (int, float)) else 0
                                                                for _, v in cf))
            if not cfgs:
                ax.axis("off")
                continue
            folds = sorted({f for cf in cfgs for f in data[a][cf]})
            if len(folds) < Ke:
                partial = True
            nsel = defaultdict(int)
            for f in folds:
                cand = [(data[a][cf][f]["idval"], k) for k, cf in enumerate(cfgs) if f in data[a][cf]
                        and data[a][cf][f]["ok"] and fin(data[a][cf][f]["idval"])]
                if cand:
                    nsel[max(cand)[1]] += 1
            col = FAMILY_COLOR[family(a)]
            for k, cf in enumerate(cfgs):
                te = [data[a][cf][f]["test"] for f in folds if f in data[a][cf] and data[a][cf][f]["ok"]
                      and fin(data[a][cf][f]["test"])]
                bad = sum(1 for f in folds if f in data[a][cf] and not data[a][cf][f]["ok"])
                if not te:
                    if bad:   # every fold of this configuration diverged: say so where the marks would be
                        tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
                        ax.text(k, 0.5, f"all\n{bad} div.", transform=tr, ha="center", va="center", fontsize=7,
                                color=INK2, linespacing=1.0, zorder=5, bbox=dict(fc="white", ec="none", pad=0.6))
                    continue
                w, m = min(te), float(np.mean(te))
                ax.plot([k, k], [w, m], color=col, lw=0.8, zorder=2)
                ax.plot([k], [w], "o", ms=MS, color=col, mec="white", mew=0.6, zorder=3)
                ax.plot([k], [m], "o", ms=MS, mfc="white", mec=col, mew=0.9, zorder=3)
                col_vals += [w, m]
                note = []
                if bad:
                    note.append(f"{bad} div.")
                if len(te) + bad < len(folds):   # folds missing for another reason than divergence
                    note.append(f"n = {len(te)}")
                if note:
                    ax.annotate("\n".join(note), (k, max(w, m)), xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=7, color=INK2, va="bottom", linespacing=1.0, zorder=5,
                                bbox=dict(fc="white", ec="none", pad=0.6))
                    any_note = True
            ax.set_xticks(range(len(cfgs)))
            ax.set_xticklabels([_cfg_label(cf, row_e[a]) + f"\n[{nsel.get(k, 0)}]" for k, cf in enumerate(cfgs)],
                               fontsize=7,
                               linespacing=1.05)
            ax.tick_params(axis="x", length=0, pad=2)
            ax.set_xlim(-0.6, len(cfgs) - 0.4)
            if c in erm_ref:
                ax.axhline(erm_ref[c], color=FAMILY_COLOR[0], lw=0.7, zorder=1)
                erm_drawn.add(c)
            else:
                erm_missing.add(c)
            ygrid(ax)
            if j == 0:
                ks = [HP_SYM.get(k, k) for k in keys.get(a, [])]
                ax.set_ylabel(arm_label(a) + "\n(" + ", ".join(ks) + (f" / $10^{{{row_e[a]}}}$" if row_e[a] else "")
                              + ")", fontsize=7, rotation=0, ha="right",
                              va="center", labelpad=6)
            if i == 0:
                ax.set_title(COHORT_LABEL[c] + ("" if len(folds) >= Ke else f" ({len(folds)}/{Ke} folds)"),
                             fontsize=7.5, pad=6)
        if col_vals:
            lo, hi = min(col_vals + ([erm_ref[c]] if c in erm_ref else [])), max(col_vals)
            pad = 0.08 * (hi - lo + 1e-3)
            for i in range(nr):
                if axs[i, j].axison:
                    y0, y1 = lo - pad, hi + pad * (4.0 if any_note else 1.0)
                    axs[i, j].set_ylim(y0, y1)
                    tv = [t for t in matplotlib.ticker.MaxNLocator(3).tick_values(y0, min(y1, 1.0))
                          if y0 <= t <= min(y1, 1.0) + 1e-9]          # no AUROC tick above 1
                    axs[i, j].yaxis.set_major_locator(matplotlib.ticker.FixedLocator(tv))
                    axs[i, j].yaxis.set_major_formatter(NOLEAD)
                    axs[i, j].tick_params(axis="y", labelsize=7)
    lh = [Line2D([], [], color=INK2, marker="o", ls="", ms=MS, label="worst fold (test domain)"),
          Line2D([], [], color=INK2, marker="o", ls="", ms=MS, mfc="white", label="mean over surviving folds")]
    if erm_drawn:
        lh.append(Line2D([], [], color=FAMILY_COLOR[0], lw=0.7, label="ERM final seed 0, worst domain"))
    vis = [ax for ax in axs.flat if ax.axison]
    legend_above(fig, vis, lh, 0.165, ncol=3)
    cap = ("Tuning runs (seed 0), test AUROC at the ID-val-selected checkpoint; diverged runs excluded ('k div.': k folds\n"
           "diverged; 'all k div.': no fold survived). Tick labels: hyperparameter value(s) and [number of folds whose\n"
           "ID-val AUROC chose it].")
    fk_arms = sorted({a for c in allc for a in tune[c][2]})
    if fk_arms:
        fk = sorted({k for c in allc for a in tune[c][2] for k in tune[c][2][a]})
        cap += (f"\nMethod grids ran at each fold's own selected {', '.join(HP_SYM.get(k, k) for k in fk)} "
                f"(top row); every fold contributes its own slice.")
    if erm_drawn and erm_missing:
        cap += ("\nNo ERM line in " + ", ".join(COHORT_LABEL[c] for c in allc if c in erm_missing) +
                f": tier-{tier} ERM final runs (seed 0) are not complete there.")
    elif not erm_drawn:
        cap += f"\nNo ERM reference line: tier-{tier} ERM final runs (seed 0) are not complete in any cohort."
    fig.text(0.165, lowest_y(fig) - fy(fig, 0.06), cap, fontsize=7, color=INK2, va="top")
    fn = None
    if partial:
        fn = "PARTIAL DATA: some cohorts have fewer tuning folds than expected."
    save(fig, ctx.out, f"fig_tuning_{tier}", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# S4  every seed per domain
# ----------------------------------------------------------------------------------------


def fig_seedspread(ctx, written, sel="id"):
    allc = ctx.cohorts
    pr = {t: {c: ctx.per_run(t, c, sel) for c in allc} for t in ("C", "S")}
    if not any(pr["C"].values()) and not any(pr["S"].values()):
        print("  [S4] no per_run files -> skipped")
        return
    # best standard-tier arm: highest mean rank-free score = mean over cohorts of worst-domain AUROC (complete arms)
    best_s = None
    sc = defaultdict(list)
    for c in allc:
        for a, r in ctx.summary("S", c, sel).items():
            if "@" not in a and fin(r.get("worst")):
                sc[a].append(r["worst"])
    full = [a for a, v in sc.items() if len(v) == max(len(x) for x in sc.values())] if sc else []
    if full:
        best_s = max(full, key=lambda a: np.mean(sc[a]))
    series = [("C", "erm", FAMILY_COLOR[0], "ERM (tier C)"), ("C", "simclr", FAMILY_COLOR[3], "SimCLR (tier C)"),
              ("C", "he_jitter", FAMILY_COLOR[2], "H&E jitter (tier C)")]
    if best_s:   # its own family hue (identity) + diamond as the 2nd channel; <= 3 hues besides the gray reference
        series.append(("S", best_s, FAMILY_COLOR[family(best_s)], f"best tier-S arm: {arm_label(best_s)}"))
    widths = facet_widths(ctx, allc)
    fig, axs = plt.subplots(1, len(allc), figsize=(FULL_W, 2.5), gridspec_kw=dict(width_ratios=widths, wspace=0.18),
                            squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=fy(fig, 0.45))
    axs = axs[0]
    off = np.linspace(-0.3, 0.3, len(series))
    for i, (c, ax) in enumerate(zip(allc, axs)):
        doms = ctx.domains(c)
        vals = []
        for (t, a, col, _), dx in zip(series, off):
            d = pr[t][c].get(a, {})
            for j, dom in enumerate(doms):
                rec = d.get(dom)
                if not rec:
                    continue
                v = np.array([x[0] for x in rec.values() if fin(x[0])])
                if not v.size:
                    continue
                ax.plot(np.full(v.size, j + dx), v, "D" if t == "S" else "o", ms=3.3 if t == "S" else 3.6, color=col,
                        mec="white", mew=0.4, alpha=0.9, zorder=3)
                ax.plot([j + dx - 0.07, j + dx + 0.07], [v.mean()] * 2, color=INK, lw=0.9, zorder=4)
                vals += list(v)
        ax.set_xticks(range(len(doms)))
        ax.set_xticklabels([dshort(x) for x in doms], fontsize=7, rotation=45 if len(doms) > 5 or c == "canine" else 0,
                           ha="right" if len(doms) > 5 or c == "canine" else "center", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(doms) - 0.4)
        for j in range(1, len(doms)):
            ax.axvline(j - 0.5, color=GRIDC, lw=LW_HAIR, zorder=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(NOLEAD)
        ygrid(ax)
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.06 * (hi - lo + 1e-3)
            ax.set_ylim(lo - pad, min(1.0, hi + pad) if hi < 1 else 1.0 + pad)
        else:
            ax.set_yticks([])
            ax.grid(False)
            placeholder(ax, "no runs yet")
        ax.set_title(COHORT_LABEL[c], fontsize=7.5)
        if i == 0:
            ax.set_ylabel("test AUROC (each seed)", fontsize=7)
    lh = [Line2D([], [], color=col, marker="D" if t == "S" else "o", ls="", ms=4, label=lab) for t, _, col, lab in series]
    lh.append(Line2D([], [], color=INK, lw=0.9, label="seed mean"))
    legend_above(fig, list(axs), lh, 0.08, ncol=len(lh), handletextpad=0.2, columnspacing=0.8)
    extra = None if best_s else "Tier S not available yet: the best standard-tier arm is not shown."
    fn = ctx.footnote(["C", "S"] if best_s else ["C"], extra=extra) or extra
    save(fig, ctx.out, "fig_seedspread", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# S5  MIDOG 2021 sensitivity: scale-normalized (primary) vs un-normalized
# ----------------------------------------------------------------------------------------


def fig_sensitivity(ctx, tier, written, sel="id", prim="midog21sn", sens="midog21"):
    from scipy.stats import kendalltau
    sp, ss = ctx.summary(tier, prim, sel), ctx.summary(tier, sens, sel)
    if not sp or not ss:
        print(f"  [S5 {tier}] needs summaries of {prim} and {sens} -> skipped")
        return
    arms = order_arms([a for a in sp if a in ss and fin(sp[a].get("worst")) and fin(ss[a].get("worst"))])
    if not arms:
        print(f"  [S5 {tier}] no arm complete in both -> skipped")
        return
    ys = row_positions(arms)
    L = label_left(arms)
    H = 0.6 + 0.15 * (ys[-1] + 1)
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, H), gridspec_kw=dict(wspace=0.08), squeeze=False)
    fig.subplots_adjust(left=L, right=0.97, top=0.95, bottom=fy(fig, 0.4))
    axs = axs[0]
    st_p = dict(marker="o", ls="", ms=MS, color=INK, mec="white", mew=0.6, zorder=4)
    st_s = dict(marker="o", ls="", ms=MS, mfc="white", mec=INK2, mew=0.9, zorder=3)
    for k, (ax, key, ttl) in enumerate(zip(axs, ("worst", "mean"), ("worst-domain AUROC", "mean AUROC over domains"))):
        vals = []
        for a, y in zip(arms, ys):
            p, q = sp[a][key], ss[a][key]
            ax.plot([p, q], [y, y], color="#e6e5e0", lw=3.0, zorder=1, solid_capstyle="round")
            ax.plot([q], [y], **st_s)
            ax.plot([p], [y], **st_p)
            vals += [p, q]
        lo, hi = min(vals), max(vals)
        pad = 0.06 * (hi - lo + 1e-3)
        ax.set_xlim(lo - pad, min(1.0, hi + pad))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(NOLEAD)
        xgrid(ax)
        ax.set_ylim(ys[-1] + 0.7, ys[0] - 0.7)
        ax.set_xlabel(f"test {ttl} ({SEL_LABEL[sel]} selection)", fontsize=7, loc="left")
        tau = kendalltau([sp[a][key] for a in arms], [ss[a][key] for a in arms])[0] if len(arms) >= 3 else float("nan")
        ax.set_title(f"{'ab'[k]}   {ttl} (τb = " + f"{tau:.2f})".replace("-", "−"), fontsize=7.5)
        arm_rows(ax, arms, ys, labels=(k == 0))
    lh = [Line2D([], [], **{k: v for k, v in st_p.items() if k != "zorder"}, label=f"{COHORT_LABEL[prim]} (scale-normalized, primary)"),
          Line2D([], [], **{k: v for k, v in st_s.items() if k != "zorder"}, label=f"{COHORT_LABEL[sens]} (sensitivity)")]
    legend_above(fig, list(axs), lh, L, ncol=2)
    caption_below(fig, list(axs), L, f"{len(arms)} arms complete in both variants; same seeds and folds "
                  f"(K = {ctx.K(tier, prim, sel)}).\nτb: Kendall rank agreement of the two arm orderings "
                  "(descriptive).")
    fn = ctx.footnote([tier], cohorts=[prim, sens])
    save(fig, ctx.out, f"fig_sensitivity_{tier}", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# GA  graphical abstract
# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
# S6  representation ladder: frozen linear probes by backbone / pre-training corpus
# ----------------------------------------------------------------------------------------

# (arm, short label, corpus group); groups ordered by pre-training corpus and domain
LADDER_ROWS = [("probe_resnet18_imagenet", "ResNet-18", "in"), ("probe_resnet50_imagenet", "ResNet-50", "in"),
               ("probe_convnext_tiny_imagenet", "ConvNeXt-T", "in"), ("probe_vit_b16_imagenet", "ViT-B/16", "in"),
               ("probe_vit_b16_swag", "ViT-B/16", "swag"),
               ("probe_resnet50_lunit_bt", "ResNet-50 Barlow Twins", "path"),
               ("probe_resnet50_lunit_swav", "ResNet-50 SwAV", "path"),
               ("probe_resnet50_lunit_mocov2", "ResNet-50 MoCo v2", "path"),
               ("probe_vit_s16_lunit_dino", "ViT-S/16 DINO", "path")]
LADDER_GROUP = {"in": "ImageNet-1k (1.3 M natural images)", "swag": "IG-3.6B SWAG (3.6 B natural images)",
                "path": "Lunit pathology (19 M patches)"}
# natural-image pre-training: neutral gray circles; pathology pre-training: family-6 green squares
LADDER_STYLE = {"in": (FAMILY_COLOR[0], "o"), "swag": (FAMILY_COLOR[0], "o"), "path": (FAMILY_COLOR[6], "s")}


def fig_ladder(ctx, written, sel="id"):
    allc = ctx.cohorts
    tier_l = "S_ladder_only"
    summ = {}
    for c in allc:   # ladder-only analysis; the 3 rungs that are also final arms fall back to the main tier S
        d = ctx.summary(tier_l, c, sel)
        main = ctx.summary("S", c, sel)
        summ[c] = {a: d.get(a) or main.get(a) for a, _, _ in LADDER_ROWS if d.get(a) or main.get(a)}
    if not any(summ.values()):
        print("  [S6] no ladder summaries -> skipped")
        return
    rows = [r for r in LADDER_ROWS if any(r[0] in summ[c] for c in allc)]
    # y positions: one group-header slot above every corpus group
    ys, heads, y, prev = [], [], 0.0, None
    for a, lab, g in rows:
        if g != prev:
            if prev is not None:
                y += 0.35
            heads.append((y, LADDER_GROUP[g]))
            y += 0.95
            prev = g
        ys.append(y)
        y += 1.0
    ys = np.array(ys)
    L = 0.205
    H = 0.42 + 0.19 * (ys[-1] + 1)
    fig, axs = plt.subplots(1, len(allc), figsize=(FULL_W, H + 0.55), sharey=True, squeeze=False,
                            gridspec_kw=dict(wspace=0.12))
    axs = axs[0]
    fig.subplots_adjust(left=L, right=0.985, top=1 - fy(fig, 0.30), bottom=fy(fig, 0.42))
    for i, (c, ax) in enumerate(zip(allc, axs)):
        S_ = summ[c]
        vals = []
        for (a, lab, g), yy in zip(rows, ys):
            r = S_.get(a)
            if not r:
                continue
            col, mk = LADDER_STYLE[g]
            for key, dy, filled in (("worst", -0.14, True), ("mean", 0.20, False)):
                v, ci = r.get(key), r.get(f"{key}_ci_seed")
                if not fin(v):
                    continue
                if ci and fin(ci[0]) and fin(ci[1]):
                    ax.plot(ci, [yy + dy] * 2, color=col, lw=1.4 if filled else 0.8, solid_capstyle="butt", zorder=2)
                    vals += list(ci)
                ax.plot([v], [yy + dy], mk, ms=MS if mk == "o" else MS - 0.4, color=col,
                        mfc=col if filled else "white", mec="white" if filled else col, mew=0.6 if filled else 0.9,
                        zorder=3)
                vals.append(v)
        erm = (ctx.summary("S", c, sel) or {}).get("erm")
        if erm and fin(erm.get("worst")):
            ax.axvline(erm["worst"], color=INK2, lw=0.7, zorder=1)
            vals.append(erm["worst"])
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.06 * (hi - lo + 1e-3)
            ax.set_xlim(lo - pad, min(1.0, hi + pad) if hi < 1 else hi + pad)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(NOLEAD)
        xgrid(ax)
        ax.set_ylim(ys[-1] + 0.7, -0.6)
        ax.set_yticks(ys)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        if i == 0:
            ax.set_yticklabels([lab for _, lab, _ in rows], fontsize=7)
            tr = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
            for yh, txt in heads:
                ax.text(0.004, yh, txt, transform=tr, fontsize=7, fontweight="bold", va="center", ha="left")
            ax.set_xlabel("test AUROC", fontsize=7, loc="left")
        ax.set_title(cohort_title(ctx, tier_l, c, sel), fontsize=7.5)
    hs = [Line2D([], [], color=INK2, marker="o", ls="", ms=MS, mec="white", label="worst domain"),
          Line2D([], [], color=INK2, marker="o", ls="", ms=MS, mfc="white", label="mean over domains"),
          Line2D([], [], color=FAMILY_COLOR[0], marker="o", ls="", ms=MS, mec="white", label="natural-image pre-training"),
          Line2D([], [], color=FAMILY_COLOR[6], marker="s", ls="", ms=MS - 0.4, mec="white",
                 label="pathology pre-training"),
          Line2D([], [], color=INK2, lw=0.7, label="ERM (RN50-IN fine-tuned), worst domain")]
    legend_above(fig, list(axs), hs, L, ncol=3, handletextpad=0.3, columnspacing=1.0)
    caption_below(fig, list(axs), L, "Frozen backbone + linear probe (tier S, 224-px input, 3 seeds, ID-val "
                  "selection); bars: 95% seed-bootstrap CIs.\nSWAG: weakly supervised on IG-3.6B, then fine-tuned "
                  "on ImageNet-1k.")
    # completeness footnote from the runs the ladder analysis used
    parts, partial = [], False
    for c in allc:
        n, folds, _ = ctx.used(tier_l, c, sel)
        exp = len(LADDER_ROWS) * K_EXPECTED.get(c, 0) * len(SEEDS.get("S", []))
        if n < exp or len(folds) < K_EXPECTED.get(c, 0):
            partial = True
        parts.append(f"{COHORT_LABEL.get(c, c)} {n}/{exp} runs")
    fn = ("PARTIAL DATA. Ladder: " + "; ".join(parts) + ".") if partial else None
    save(fig, ctx.out, "fig_ladder", footnote=fn, written=written)


def graphical_abstract(ctx, written, tier="C", sel="id"):
    W, H = FULL_W, 2.5          # inches; at 300 dpi = 1950 x 750 px (>= 1328 x 531)
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.8, 0.95], wspace=0.16, left=0.01, right=0.99, top=0.80,
                          bottom=0.08)
    # 1: design
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-0.26, 2.0)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    ax.text(-0.26, 2.62, "2 × 2 design,\nleave-one-domain-out", fontsize=8.5, fontweight="bold", va="top",
            linespacing=1.05)
    ax.text(0.5, 2.03, "acquisition", ha="center", fontsize=7.5, color=INK2)
    ax.text(1.5, 2.03, "institutional", ha="center", fontsize=7.5, color=INK2)
    ax.text(-0.02, 1.5, "tumor", ha="right", va="center", fontsize=7.5, color=INK2, rotation=90)
    ax.text(-0.02, 0.5, "mitosis", ha="right", va="center", fontsize=7.5, color=INK2, rotation=90)
    cards = []
    for (cx, cy), (nm, k) in {(0, 1): ("Canine", 5), (1, 1): ("Camelyon17", 5), (0, 0): ("MIDOG 21", 3),
                              (1, 0): ("MIDOG++", 7)}.items():
        card = FancyBboxPatch((cx + 0.04, cy + 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.06",
                              fc="#f4f3f0", ec="none")
        ax.add_patch(card)
        t = ax.text(cx + 0.5, cy + 0.6, nm, ha="center", va="center", fontsize=8, fontweight="bold")
        cards.append((card, t))
        ax.text(cx + 0.5, cy + 0.3, f"K = {k}", ha="center", va="center", fontsize=8, color=INK2)
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    for card, t in cards:   # shrink a name until it sits inside its card with a margin (never below 7 pt)
        cb_ = card.get_window_extent(rr)
        while t.get_fontsize() > MIN_FS and t.get_window_extent(rr).width > cb_.width - 6 * fig.dpi / 72:
            t.set_fontsize(t.get_fontsize() - 0.25)
        if t.get_window_extent(rr).width > cb_.width:
            LAYOUT_ISSUES.append(("graphical_abstract", f"card text overflows: {t.get_text()!r}"))
    # 2: rank instability (headline only when the comparison is not degenerate)
    ax = fig.add_subplot(gs[1])
    cohorts = [c for c in ctx.cohorts if ctx.summary(tier, c, sel)]
    common, ranks = _common_ranks(ctx, tier, cohorts, sel)
    full_k = all(ctx.K(tier, c, sel) >= K_EXPECTED.get(c, 0) for c in ctx.cohorts)
    ok = len(ranks) == len(ctx.cohorts) and len(common) >= MIN_COMMON_ARMS and full_k
    krows = [r for r in read_csv(os.path.join(ctx.an, tier, "cross", f"kendall_{sel}.csv")) if r.get("key") == "worst"
             and fin(r.get("tau_b"))]
    mt = float(np.mean([r["tau_b"] for r in krows])) if krows else None
    if ok and mt is not None and all(r["ci"][1] < 0.8 for r in krows if r.get("ci") and fin(r["ci"][1])):
        ttl = "Method ranks change across cohorts"
    elif ok and mt is not None and all(r["ci"][0] > 0.5 for r in krows if r.get("ci") and fin(r["ci"][0])):
        ttl = "Method ranks agree across cohorts"
    else:
        ttl = "Method rank per cohort"
    ax.set_title(ttl, fontsize=8.5, loc="left", pad=16, x=0.0)
    if ok:
        bump_chart(ax, ctx, ctx.cohorts, common, ranks, fs=7.5, lw_hi=1.8, ms_hi=5, label_all=False, room=1.75,
                   rank_axis=False, context_lines=False)
        ax.set_xticklabels([COHORT_SHORT[c] if c != "canine" else "Canine" for c in ctx.cohorts if c in ranks],
                           fontsize=7.5)
        ax.text(0.5, -0.04, f"rank among {len(common)} tier-{tier} arms, by worst-domain AUROC (1 = top)",
                transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=INK2)
    else:
        ax.axis("off")
        placeholder(ax, f"rank panel needs ≥ {MIN_COMMON_ARMS} arms complete\nin all 4 cohorts "
                        f"(now {len(common)})")
    # 3: variance decomposition icon (ERM): share of variance from domain vs seed (a quantity: ink / gray)
    ax = fig.add_subplot(gs[2])
    rows = []
    for c in ctx.cohorts:
        v = {r["arm"]: r for r in ctx.csv(tier, c, "variance", sel)} if ctx.K(tier, c, sel) >= 2 else {}
        r = v.get("erm")
        if r and fin(r.get("icc_reml")):
            rows.append((c, r["icc_reml"], r.get("icc_ci_lo")))
    los = [lo for _, _, lo in rows]
    strong = len(rows) == len(ctx.cohorts) and full_k and all(fin(lo) and lo > 0.5 for lo in los)
    ttl = "Domain, not seed,\ndrives ERM variance" if strong else "ERM variance:\ndomain vs seed"
    ax.set_title(ttl, fontsize=8.5, loc="left", pad=6, linespacing=1.05)
    ax.set_xlim(0, 1)
    st = 1.3                        # row pitch: name, bar, bound marker
    ax.set_ylim(st * (len(ctx.cohorts) - 1) + 0.75, -0.6)
    ax.axis("off")
    look = {c: (v, lo) for c, v, lo in rows}
    for i_, c in enumerate(ctx.cohorts):
        i = st * i_
        ax.text(0, i - 0.36, COHORT_LABEL[c], fontsize=7.5, va="bottom", color=INK)
        if c not in look:
            ax.add_patch(Rectangle((0, i - 0.12), 1, 0.36, fc="#f6f6f4", ec="none"))
            ax.text(0.5, i + 0.06, "pending", fontsize=7, color=MUTED, ha="center", va="center")
            continue
        r, lo = look[c]
        gap = 0.008
        ax.add_patch(Rectangle((0, i - 0.12), max(r - gap, 0), 0.36, fc=INK2, ec="none"))
        ax.add_patch(Rectangle((r + gap, i - 0.12), max(1 - r - gap, 0), 0.36, fc=CONTEXT, ec="none"))
        if r > 0.42:
            ax.text(0.03, i + 0.06, f"domain {r * 100:.0f}%", fontsize=7.5, color="white", va="center",
                    fontweight="bold")
        if fin(lo):   # lower 95% bound of the ICC as a pointer under the bar, so a wide interval is visible
            ax.plot([lo], [i + 0.36], marker="^", ms=4.5, color=INK, mew=0, clip_on=False)
    ax.text(1.0, st * (len(ctx.cohorts) - 1) + 0.55, "gray = seed; ▲ = lower 95% bound", fontsize=7, color=INK2,
            ha="right", va="top")
    fn = ctx.footnote([tier])
    save(fig, ctx.out, "graphical_abstract", footnote=fn, written=written)


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------

def _snapshot_id(entries):
    import hashlib
    h = hashlib.sha1()
    for e in entries:
        h.update(f"{e['file']}|{e['size']}|{e['mtime']:.3f}\n".encode())
    return h.hexdigest()[:12]


def freeze(src, snap, analysis, tiers=("C", "S"), cohorts=None, analyze=True, extra_args=()):
    """Freeze one record manifest and analyse every selection rule from it.

    Copies every readable v2 run record (runs/{tier}/{cohort}/fold*/*.json) and every tuning record
    (runs_tune/**.json) from ``src`` into ``snap`` (prediction .npz files and the cache manifests are
    symlinked), writes ``snap/SNAPSHOT.json`` (file list, sizes, mtimes, id), then runs analyze_v2.py
    ``runs`` for all four selection rules with ``--v2-root snap --out analysis`` and stamps the
    snapshot id into ``analysis/{tier}/SNAPSHOT.json``.  The source records are never modified."""
    import shutil
    import subprocess
    import time
    cohorts = cohorts or (COHORTS + ["midog21"])
    if os.path.exists(os.path.join(snap, "SNAPSHOT.json")):
        raise SystemExit(f"{snap} already holds a snapshot; pass a new directory")
    t0 = time.time()
    entries, counts, skipped = [], defaultdict(dict), []

    def _copy(fp):
        rel = os.path.relpath(fp, src)
        st = os.stat(fp)
        try:
            with open(fp) as f:
                txt = f.read()
            json.loads(txt)
        except Exception:  # noqa: BLE001  (record being written right now: leave it out of the snapshot)
            skipped.append(rel)
            return None
        dst = os.path.join(snap, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            f.write(txt)
        os.utime(dst, (st.st_atime, st.st_mtime))
        entries.append(dict(file=rel, size=st.st_size, mtime=st.st_mtime))
        return dst
    for t in tiers:
        for c in cohorts:
            n = 0
            for fp in sorted(glob.glob(os.path.join(src, "runs", t, c, "fold*", "*.json"))):
                dst = _copy(fp)
                if dst is None:
                    continue
                n += 1
                npz = fp[:-5] + "_preds.npz"
                if os.path.exists(npz):
                    os.symlink(npz, dst[:-5] + "_preds.npz")
            if n:
                counts[t][c] = n
    n_tune = 0
    for fp in sorted(glob.glob(os.path.join(src, "runs_tune", "*", "*", "fold*", "*.json"))):
        n_tune += _copy(fp) is not None
    for cdir in glob.glob(os.path.join(src, "cache*")):
        if os.path.isdir(cdir) and not os.path.exists(os.path.join(snap, os.path.basename(cdir))):
            os.symlink(cdir, os.path.join(snap, os.path.basename(cdir)))
    run_entries = [e for e in entries if e["file"].startswith("runs" + os.sep)]
    sid = _snapshot_id(run_entries)
    man = dict(id=sid, created=time.strftime("%Y-%m-%d %H:%M", time.localtime(t0)), source=src,
               counts=counts, n_tune=n_tune, skipped_unreadable=skipped, files=entries)
    with open(os.path.join(snap, "SNAPSHOT.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"[freeze] snapshot {sid}: {len(run_entries)} run records, {n_tune} tuning records -> {snap}"
          + (f"; {len(skipped)} unreadable skipped" if skipped else ""))
    if not analyze:
        return sid
    an_py = os.path.join(HERE, "analyze_v2.py")
    for t in tiers:
        main_c = [c for c in COHORTS if c in counts[t]]
        sens_c = [c for c in counts[t] if c not in COHORTS]
        if not main_c and not sens_c:
            continue
        for sel in SELS:
            for group in (main_c, sens_c):   # sensitivity cohorts separately: they must not enter the 2x2 cross
                if not group:
                    continue
                cmd = [sys.executable, an_py, "runs", "--v2-root", snap, "--tier", t, "--select", sel,
                       "--out", analysis, "--cohorts", *group, *extra_args]
                print("[freeze] " + " ".join(cmd[2:]))
                subprocess.run(cmd, check=True)
        os.makedirs(os.path.join(analysis, t), exist_ok=True)
        with open(os.path.join(analysis, t, "SNAPSHOT.json"), "w") as f:
            json.dump(dict(id=sid, created=man["created"], source=src, snapshot=snap, counts=counts[t]), f, indent=1)
    return sid


# ----------------------------------------------------------------------------------------
# F9: practitioner roadmap (schematic). Every number on it is computed from the analysis tree.
# ----------------------------------------------------------------------------------------
def roadmap_numbers(ctx):
    S = "S"
    summ = {c: ctx.csv(S, c, "summary") for c in COHORTS}
    con = {c: ctx.csv(S, c, "contrasts") for c in COHORTS}
    powr = {c: ctx.csv(S, c, "power") for c in COHORTS}

    def row(df, arm):
        return next(r for r in df if r["arm"] == arm)

    erm_can = row(summ["canine"], "erm")
    kmin = []
    for c in ("c17", "canine"):
        ks = [float(r["K_min"]) for r in powr[c] if r["ref"] == "erm" and int(float(r["seeds"])) == 3
              and float(r["rho"]) == 0 and abs(float(r["delta"]) - 0.05) < 1e-9]
        kmin.append(int(math.ceil(float(np.median(ks)))))
    jit = sum(int(float(row(con[c], "he_jitter")["n_improved"])) for c in COHORTS)
    ktot = sum(int(float(row(con[c], "he_jitter")["K"])) for c in COHORTS)
    dj = [float(row(summ[c], "he_jitter_lunit_dino")["worst"]) for c in COHORTS]
    erm_w = [float(row(summ[c], "erm")["worst"]) for c in COHORTS]
    bn_can, bn_mpp = row(con["canine"], "bnadapt@erm"), row(con["midogpp"], "bnadapt@erm")
    taus = []
    for c in COHORTS:
        base = {r["arm"]: float(r["worst"]) for r in summ[c]}
        for sel in ("ood", "last", "oracle"):
            other = ctx.csv(S, c, "summary", sel)
            if not other:
                continue
            o = {r["arm"]: float(r["worst"]) for r in other}
            common = sorted(set(base) & set(o))
            taus.append(kendall_tau_b([base[a] for a in common], [o[a] for a in common]))
    sat = json.load(open(os.path.join(REPO, "results", "v2", "site_acceptance", "sat_numbers.json")))["S"]
    sat_far = max(sat[f"eb/{m}"][T]["far"] for m in (25, 50, 100, 200, 400) for T in ("0.8", "0.85", "0.9"))
    return dict(sat_far=sat_far, sat_dev=sat["dev/0"]["0.8"]["far"],
                sat_pw=(sat["eb/100"]["0.8"]["power"], sat["local/100"]["0.8"]["power"]),
                erm_can_mean=float(erm_can["mean"]), erm_can_worst=float(erm_can["worst"]), kmin=kmin,
                jit=jit, ktot=ktot, dj=(min(dj), max(dj)), erm_w=(min(erm_w), max(erm_w)),
                bn_can=(float(bn_can["mean_diff"]), int(float(bn_can["n_improved"])), int(float(bn_can["K"]))),
                bn_mpp=float(bn_mpp["mean_diff"]), tau_min=min(taus))


def kendall_tau_b(x, y):
    from scipy import stats as _st
    return float(_st.kendalltau(x, y).statistic)


def fig_roadmap(ctx, written):
    n = roadmap_numbers(ctx)
    f2 = lambda v: f"{v:.2f}".replace("0.", ".", 1)   # noqa: E731
    steps = [
        ("1", "Define the target", "Name the task and the sites; fix a\nworst-site acceptance level first.",
         f"Canine: mean AUROC {f2(n['erm_can_mean'])} hid a\nscanner at {f2(n['erm_can_worst'])} (ERM).", "strong"),
        ("2", "Collect sites, not slides", "Hold out whole sites; size the\nnumber of sites from a pilot.",
         f"A .05 gain needs {n['kmin'][0]}–{n['kmin'][1]} sites\n(tumor cohorts).", "strong"),
        ("3", "Train once, strongly", "Fine-tune a pathology FM with\ncolor jitter; tune its own LR.",
         f"Jitter beat ERM in {n['jit']}/{n['ktot']} domains;\nFM + jitter worst site {f2(n['dj'][0])}–{f2(n['dj'][1])}.",
         "moderate"),
        ("4", "Adapt without labels", "Optional: re-estimate BN statistics\nor normalize stain per slide.",
         f"BN-adapt {n['bn_can'][0]:+.2f} on canine ({n['bn_can'][1]}/{n['bn_can'][2]});\n"
         f"{n['bn_mpp']:+.3f} on MIDOG++ (no gain).", "mixed"),
        ("5", "Select and validate", "Select on training-site validation;\nreport every site, seeds, ECE.",
         f"Selection rules agree at standard\nscale (τb ≥ {f2(n['tau_min'])}).", "strong"),
        ("6", "Accept each new site", "Label a small sample; run the\nsite acceptance test; monitor.",
         f"False acceptance ≤ {100 * n['sat_far']:.1f}% (prior alone {100 * n['sat_dev']:.1f}%);\n"
         f"100 labels: {100 * n['sat_pw'][0]:.0f}% vs {100 * n['sat_pw'][1]:.0f}% power.", "moderate"),
    ]
    fig = plt.figure(figsize=(FULL_W, 2.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 2.0)
    ax.axis("off")
    W, H, gx = 0.94, 0.80, 0.06
    for i, (num, title, action, evid, grade) in enumerate(steps):
        r, cidx = divmod(i, 3)
        x0 = cidx * (W + gx) + 0.0
        y0 = 2.0 - (r + 1) * (H + 0.11) + 0.05
        ax.add_patch(FancyBboxPatch((x0, y0), W, H, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="#f4f3f0", ec="none"))
        ax.text(x0 + 0.04, y0 + H - 0.05, num, fontsize=9, fontweight="bold", va="top")
        ax.text(x0 + 0.12, y0 + H - 0.055, title, fontsize=8, fontweight="bold", va="top")
        ax.text(x0 + 0.04, y0 + H - 0.22, action, fontsize=7, va="top", color=INK, linespacing=1.2)
        ax.text(x0 + 0.04, y0 + 0.27, evid, fontsize=7, va="top", color=INK2, linespacing=1.2)
        ax.text(x0 + W - 0.04, y0 + H - 0.055, grade, fontsize=7, va="top", ha="right", color=INK2,
                style="italic", linespacing=1.1)
        if cidx < 2:
            ax.annotate("", xy=(x0 + W + gx - 0.005, y0 + H / 2), xytext=(x0 + W + 0.005, y0 + H / 2),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0))
    save(fig, ctx.out, "fig_roadmap",
         footnote="Evidence grades: strong = holds in all four cohorts at the standard tier; moderate = consistent "
                  "direction but not significant after correction, or holding with exceptions; mixed = large gains "
                  "in some cohorts, none in others. Numbers: standard tier, ID-val selection.",
         written=written)


# ----------------------------------------------------------------------------------------
# F10: site acceptance test (results/v2/site_acceptance/sat_rows_{tier}.csv, src/v2/site_acceptance.py)
# Jobs: read how often each rule accepts a good site (power) and a bad site (false acceptance) as the
# number of local labels grows. One line per rule; small multiples by target; identity by colour, marker,
# line style and a direct label at the right end (the reference rule, local labels only, is neutral gray).
# ----------------------------------------------------------------------------------------
SAT_STYLE = {"local": dict(color=FAMILY_COLOR[0], marker="s", ls="-", label="local labels only"),
             "eb": dict(color=FAMILY_COLOR[1], marker="o", ls="-", label="site acceptance test"),
             "eb_robust": dict(color=FAMILY_COLOR[2], marker="^", ls="--", label="robust variant")}
SAT_TARGETS = (0.80, 0.85, 0.90)


def fig_sat(ctx, tier, written):
    import pandas as pd
    f = os.path.join(REPO, "results", "v2", "site_acceptance", f"sat_rows_{tier}.csv")
    if not os.path.exists(f):
        print(f"  [F10 {tier}] {f} missing -> skipped")
        return
    df = pd.read_csv(f)
    ms = sorted(m for m in df.m.unique() if m > 0)
    X0 = ms[0] / 2                                   # x position of the "no local labels" (development prior) point
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.9), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.95, top=0.84, bottom=0.12, wspace=0.18, hspace=0.22)
    for j, T in enumerate(SAT_TARGETS):
        bad, good = df.truth < T, df.truth >= T
        for i, (mask, ylab) in enumerate(((good, "accepts good site"), (bad, "accepts bad site"))):
            ax = axes[i, j]
            ygrid(ax)
            for meth, st in SAT_STYLE.items():
                g = df[mask & (df.method == meth)].groupby("m")[f"acc_{T}"].mean()
                g = g[g.index > 0]
                ax.plot(g.index, g.values, color=st["color"], ls=st["ls"], lw=LW, marker=st["marker"], ms=MS,
                        mec=SURF, mew=0.6, zorder=3)
            d = df[mask & (df.method == "dev")][f"acc_{T}"].mean()
            ax.plot([X0], [d], marker="D", ms=MS, mfc="white", mec=INK, mew=0.9, ls="none", zorder=4)
            if i == 1:
                ax.axhline(0.05, color=MUTED, lw=LW_THIN, ls=":", zorder=2)
                ax.set_ylim(0, max(0.12, float(np.nanmax([d, 0.1])) * 1.15))
            else:
                ax.set_ylim(0, 1.0)
            ax.set_xscale("log")
            ax.set_xticks([X0] + ms)
            ax.set_xticklabels(["0"] + [str(m) for m in ms])
            ax.minorticks_off()
            ax.set_xlim(X0 / 1.3, ms[-1] * 1.3)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(f"target AUROC {T:.2f}".replace("0.", ".", 1), fontsize=8)
            if i == 1:
                ax.set_xlabel("local labeled patches")
            if i == 1 and j == 2:
                ax.annotate("5%", (ms[-1] * 1.3, 0.05), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=MIN_FS, color=MUTED, annotation_clip=False)
    handles = [Line2D([], [], color=st["color"], ls=st["ls"], lw=LW, marker=st["marker"], ms=MS, mec=SURF, mew=0.6,
                      label=st["label"]) for st in SAT_STYLE.values()]
    handles.append(Line2D([], [], ls="none", marker="D", ms=MS, mfc="white", mec=INK, mew=0.9,
                          label="development prior only (0 labels)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=4, frameon=False,
               fontsize=MIN_FS, handlelength=2.2, columnspacing=1.4)
    tier_name = "standard tier" if tier == "S" else "compact tier"
    n_runs = df[df.method == "dev"].shape[0]
    save(fig, ctx.out, f"fig_sat_{tier}",
         footnote=f"{n_runs} runs of the {tier_name}; each held-out site plays the new site, with the prior "
                  f"built from the other folds only. Labeled samples drawn at random from the site's test patches "
                  f"(200 draws per run and size). Accept if P(AUROC ≥ target) ≥ 0.95 (local only: one-sided 95% "
                  f"bound ≥ target).",
         written=written)


FIGS = {"F1": ("fig_design", False), "F2": ("fig_perdomain", True), "F3": ("fig_ranks", True),
        "F4": ("fig_variance", True), "F5": ("fig_power", True), "F6": ("fig_scale", False),
        "F7": ("fig_selection", True), "F8": ("fig_kendall", True), "S1": ("fig_calibration", True),
        "S2": ("fig_domainbed", False), "S3": ("fig_tuning", True), "S4": ("fig_seedspread", False), "S5": ("fig_sensitivity", True),
        "S6": ("fig_ladder", False), "F9": ("fig_roadmap", False),
        "F10": ("fig_sat", True),
        "GA": ("graphical_abstract", False)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default=os.path.join(REPO, "results", "v2", "analysis"))
    ap.add_argument("--runs", default=V2, help="v2 root with runs/, runs_tune/, cache64/")
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "media", "figs"))
    ap.add_argument("--tier", default="both", choices=["C", "S", "both"])
    ap.add_argument("--cohorts", nargs="+", default=COHORTS, help="facet order (2x2: c17 canine midog21sn midogpp)")
    ap.add_argument("--domainbed", default=os.path.join(REPO, "results", "v2", "external", "domainbed_icc.csv"))
    ap.add_argument("--only", nargs="+", default=None, help=f"subset of {sorted(FIGS)}")
    ap.add_argument("--ga-tier", default="C", choices=["C", "S"])
    ap.add_argument("--freeze", metavar="SNAPDIR", default=None,
                    help="first copy the records of --runs into SNAPDIR (one frozen manifest), run analyze_v2.py for "
                         "all four selection rules from it into --analysis, then draw from SNAPDIR")
    ap.add_argument("--freeze-only", action="store_true", help="with --freeze: stop after the analysis")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="draw even if the analysis files do not all come from one record snapshot")
    args = ap.parse_args(argv)
    if args.freeze:
        freeze(args.runs, args.freeze, args.analysis)
        args.runs = args.freeze
        if args.freeze_only:
            return 0
    setup_style()
    ctx = Ctx(args)
    problems = []
    for t in ("C", "S", "S_ladder_only"):
        if os.path.isdir(os.path.join(args.analysis, t)):
            problems += ctx.check_snapshot(t)
    for p in problems:
        print(f"[snapshot] {p}", file=sys.stderr)
    if problems and not args.allow_mixed:
        print("[snapshot] analysis files come from different record sets: re-run with --freeze NEWDIR "
              "(or --allow-mixed to draw anyway)", file=sys.stderr)
        return 2
    written, failed = [], []
    for key in sorted(FIGS, key=lambda k: (k[0] != "F", k[0] == "G", k)):
        if args.only and key not in args.only:
            continue
        name, per_tier = FIGS[key]
        fn = globals()[name]
        print(f"[{key}] {name}")
        try:
            if key == "GA":
                fn(ctx, written, tier=args.ga_tier)
            elif per_tier:
                for t in ctx.tiers:
                    fn(ctx, t, written)
            else:
                fn(ctx, written)
        except Exception:  # noqa: BLE001  (one broken figure must not stop the others)
            failed.append(key)
            traceback.print_exc()
            plt.close("all")
    print(f"wrote {len(written)} files to {args.out}" + (f"; FAILED: {failed}" if failed else ""))
    if LAYOUT_ISSUES:
        print(f"[layout] {len(LAYOUT_ISSUES)} layout issue(s) in "
              f"{sorted({n for n, _ in LAYOUT_ISSUES})}", file=sys.stderr)
    return 1 if failed else (3 if LAYOUT_ISSUES else 0)


if __name__ == "__main__":
    sys.exit(main())
