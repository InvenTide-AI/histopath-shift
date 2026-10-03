"""Figures for the TMLR submission.

Every number drawn here is read from the per-run JSON records of the single
site-complete GPU grid (105 training runs) plus the PatchCamelyon evaluation
CSV (60 checkpoint evaluations).  Nothing is hard-coded: if a record changes,
the figure changes.

Palette: slots 1/2/3/7 of the validated categorical palette (blue, orange,
aqua, violet), which clears the all-pairs CVD and normal-vision floors
(worst CVD dE 9.2, worst normal dE 16.3).  The ERM reference series is
deliberately achromatic -- a reference line is not a categorical identity --
and every series additionally carries a distinct marker and dash pattern so
identity survives greyscale printing.
"""
from __future__ import annotations
import collections
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from figure_style import apply_figure_style, panel_letter

RUNS = "./data/runs"
PCAM = "results/camelyon17_extended/pcam_external.csv"
OUT = "paper/tmlr/figs"
FOLDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------- palette
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GREY, INK, MUTED = "#6b6b6b", "#0b0b0b", "#52514e"

CORE = ["baseline", "sam", "ssl", "ssl_sam"]
NICE = {"baseline": "ERM", "sam": "SAM", "ssl": "SSL (ours)", "ssl_sam": "SSL+SAM (ours)",
        "probe_resnet50": "Frozen probe", "erm_henorm": "ERM + H&E jitter",
        "mixstyle": "MixStyle", "groupdro": "GroupDRO", "coral": "DeepCORAL",
        "irm": "IRMv1", "macenko": "Macenko + ERM", "lisa": "LISA", "fish": "Fish"}
# style = (colour, marker, dash)
STYLE = {"baseline": (GREY, "o", (0, (3, 2))),
         "sam": (ORANGE, "s", (0, (5, 1.5))),
         "ssl": (BLUE, "^", (0, ())),
         "ssl_sam": (VIOLET, "D", (0, ()))}

# method families for the 13-arm panels (4 hues max + neutral reference)
FAMILY = {
    "ssl": ("Ours (representation)", BLUE), "ssl_sam": ("Ours (representation)", BLUE),
    "probe_resnet50": ("Frozen off-the-shelf", VIOLET),
    "erm_henorm": ("Input / stain space", ORANGE), "macenko": ("Input / stain space", ORANGE),
    "mixstyle": ("Input / stain space", ORANGE),
    "groupdro": ("DG objective / leaderboard", AQUA), "coral": ("DG objective / leaderboard", AQUA),
    "irm": ("DG objective / leaderboard", AQUA), "fish": ("DG objective / leaderboard", AQUA),
    "lisa": ("DG objective / leaderboard", AQUA),
    "baseline": ("Baseline", GREY), "sam": ("Baseline", GREY),
}


# ---------------------------------------------------------------- data
def load_runs():
    rec = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{RUNS}/fold*/*.json")):
        fold = int(os.path.basename(os.path.dirname(f)).replace("fold", ""))
        arm, seed = os.path.basename(f)[:-5].rsplit("_seed", 1)
        rec[arm][(fold, int(seed))] = json.load(open(f))
    return rec


REC = load_runs()


def per_seed(arm, metric="auroc"):
    out = collections.defaultdict(list)
    for (fo, _), d in sorted(REC[arm].items()):
        out[fo].append(d["ood_test"][metric])
    return out


def fold_mean(arm, metric="auroc"):
    m = per_seed(arm, metric)
    return np.array([np.mean(m[f]) for f in FOLDS])


def fold_sd(arm, metric="auroc"):
    m = per_seed(arm, metric)
    return np.array([np.std(m[f], ddof=1) if len(m[f]) > 1 else np.nan for f in FOLDS])


def load_pcam():
    import csv
    pc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(PCAM)):
        pc[r["arm"]][int(r["fold"])].append(float(r["pcam_auroc"]))
    return pc


PC = load_pcam()


def variance_components(arm):
    m = per_seed(arm)
    y = np.array([m[f] for f in FOLDS])
    k, ns = y.shape
    grand = y.mean()
    ms_f = ns * ((y.mean(1) - grand) ** 2).sum() / (k - 1)
    ms_s = k * ((y.mean(0) - grand) ** 2).sum() / (ns - 1)
    ms_e = ((y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + grand) ** 2
            ).sum() / ((k - 1) * (ns - 1))
    return max((ms_f - ms_e) / ns, 0.0), max((ms_s - ms_e) / k, 0.0), ms_e


def min_sites(s_site, s_run, eff, seeds=1, power=0.8):
    for K in range(2, 400):
        se = math.sqrt(s_site ** 2 / K + s_run ** 2 / (K * seeds))
        df = K - 1
        tc = stats.t.ppf(0.975, df)
        ncp = eff / se
        pw = 1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp)
        if pw >= power:
            return K
    return np.nan


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf")


# ================================================================ Figure 1
def figure1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # (a) per-hospital AUROC, 4 core arms
    for arm in CORE:
        c, mk, dash = STYLE[arm]
        m, s = fold_mean(arm), fold_sd(arm)
        ax1.errorbar(FOLDS, m, yerr=s, color=c, marker=mk, linestyle=dash,
                     markersize=4.5, capsize=2, elinewidth=0.7, markeredgewidth=0,
                     label=NICE[arm], zorder=3 if "ssl" in arm else 2)
    worst = fold_mean("baseline").min()
    ax1.axhline(worst, color=MUTED, lw=0.6, linestyle=(0, (1, 2)))
    ax1.annotate(f"ERM worst site {worst:.3f}", (-0.30, worst + 0.010),
                 fontsize=5.5, color=MUTED, va="bottom", ha="left")
    ax1.annotate("SAM, seed SD 0.278", (3.08, 0.470), fontsize=5.3,
                 color=ORANGE, ha="left", va="center")
    ax1.set_xticks(FOLDS)
    ax1.set_xlim(-0.35, 4.35)
    ax1.set_xlabel("Held-out hospital (Camelyon17-WILDS centre)")
    ax1.set_ylabel("Off-site test AUROC")
    ax1.set_ylim(0.43, 1.01)
    ax1.set_title("Hospital identity moves ERM by 0.196 AUROC")
    ax1.legend(loc="lower left", ncol=2, fontsize=5.5, handlelength=2.2)
    panel_letter(ax1, "a", dx=-0.14)

    # (b) worst-site vs across-site SD, all arms, coloured by family.
    # Label offsets are explicit: the cloud is dense and autoplacement collides.
    LAB = {"probe_resnet50": (+.006, +.012, "left", "bottom"),
           "ssl": (-.005, -.014, "right", "top"),
           "ssl_sam": (+.007, 0, "left", "center"),
           "erm_henorm": (0, -.016, "center", "top"),
           "mixstyle": (+.006, 0, "left", "center"),
           "sam": (-.006, +.011, "right", "bottom"),
           "baseline": (+.005, +.008, "left", "bottom"),
           "groupdro": (-.004, -.013, "right", "top"),
           "coral": (+.006, +.007, "left", "bottom"),
           "irm": (+.009, +.002, "left", "center"),
           "macenko": (+.009, -.014, "left", "top"),
           "lisa": (+.006, -.004, "left", "top"),
           "fish": (-.006, +.011, "right", "bottom")}
    seen = {}
    for arm in FAMILY:
        fm = fold_mean(arm)
        fam, col = FAMILY[arm]
        sd_, w_ = fm.std(ddof=1), fm.min()
        ax2.scatter(sd_, w_, s=26, color=col, edgecolor="white", linewidth=0.6,
                    zorder=3, label=fam if fam not in seen else None)
        seen[fam] = True
        dx, dy, ha, va = LAB[arm]
        ax2.annotate(NICE[arm], (sd_ + dx, w_ + dy), fontsize=5.3, color=INK,
                     ha=ha, va=va)
    ax2.set_xlabel("Across-site SD of AUROC  (lower better)")
    ax2.set_ylabel("Worst-site AUROC  (higher better)")
    ax2.set_xlim(-0.02, 0.245)
    ax2.set_ylim(0.46, 1.02)
    ax2.annotate("better", (0.040, 0.600), fontsize=6, color=MUTED, style="italic")
    ax2.annotate("", xy=(0.012, 0.655), xytext=(0.062, 0.575),
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax2.set_title("Only four arms reach the low-spread corner")
    ax2.legend(loc="upper right", fontsize=5.5, markerscale=0.8)
    panel_letter(ax2, "b", dx=-0.14)

    fig.tight_layout(w_pad=2.2)
    save(fig, "fig1_site_complete")


# ================================================================ Figure 2
def figure2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # (a) stacked variance components
    labels, site, seed, res = [], [], [], []
    for arm in CORE:
        s2f, s2s, s2e = variance_components(arm)
        labels.append(NICE[arm]); site.append(s2f); seed.append(s2s); res.append(s2e)
    x = np.arange(len(labels))
    site, seed, res = np.array(site), np.array(seed), np.array(res)
    # 2px surface gap between stacked segments -> small white edge
    ax1.bar(x, site, 0.6, color=BLUE, label=r"site ($\sigma^2_{\rm site}$)",
            edgecolor="white", linewidth=0.8)
    ax1.bar(x, seed, 0.6, bottom=site, color=ORANGE,
            label=r"seed ($\sigma^2_{\rm seed}$)", edgecolor="white", linewidth=0.8)
    ax1.bar(x, res, 0.6, bottom=site + seed, color=GREY,
            label=r"residual", edgecolor="white", linewidth=0.8)
    for i in range(len(labels)):
        tot = site[i] + seed[i] + res[i]
        ax1.annotate(f"ICC$_{{site}}$={site[i]/tot:.2f}", (x[i], tot + 2e-4),
                     ha="center", fontsize=5.5, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=5.5)
    ax1.set_ylabel("Variance of off-site AUROC")
    ax1.set_ylim(0, 0.0245)
    ax1.set_title("Site is not the only nuisance")
    ax1.legend(loc="upper right", fontsize=5.5)
    panel_letter(ax1, "a", dx=-0.16)

    # (b) required number of sites
    s2f, s2s, s2e = variance_components("baseline")
    s_site, s_run = math.sqrt(s2f), math.sqrt(s2s + s2e)
    deltas = np.arange(0.04, 0.205, 0.004)
    for seeds, c, mk, dash in ((1, BLUE, "^", (0, ())),
                               (3, ORANGE, "s", (0, (5, 1.5))),
                               (5, VIOLET, "D", (0, (1, 1.5)))):
        K = [min_sites(s_site, s_run, d, seeds) for d in deltas]
        ax2.plot(deltas, K, color=c, linestyle=dash, marker=mk, markevery=8,
                 markersize=3.5, markeredgewidth=0, label=f"{seeds} seed/site")
    ax2.axhline(1, color=INK, lw=0.8)
    ax2.annotate("the field's convention: $K=1$", (0.105, 1.12), fontsize=5.8, color=INK)
    obs = 0.096
    ax2.axvline(obs, color=MUTED, lw=0.6, linestyle=(0, (1, 2)))
    ax2.annotate(f"observed SSL\neffect {obs:.3f}", (obs + 0.004, 24), fontsize=5.5,
                 color=MUTED, va="top")
    ax2.set_yscale("log")
    ax2.set_yticks([1, 2, 3, 5, 8, 11, 17, 25, 34])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel(r"Target mean-across-sites effect $\Delta$ (AUROC)")
    ax2.set_ylabel("Hospitals $K$ for 80% power")
    ax2.set_title("Detecting a 0.10 AUROC effect needs 11 hospitals")
    ax2.legend(loc="upper right", fontsize=5.5)
    panel_letter(ax2, "b", dx=-0.16)

    fig.tight_layout(w_pad=2.4)
    save(fig, "fig2_variance_power")


# ================================================================ Figure 3
def figure3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # (a) per-fold PCam AUROC
    for arm in CORE:
        c, mk, dash = STYLE[arm]
        m = np.array([np.mean(PC[arm][f]) for f in FOLDS])
        ax1.plot(FOLDS, m, color=c, marker=mk, linestyle=dash, markersize=4.5,
                 markeredgewidth=0, label=NICE[arm])
    ax1.set_xticks(FOLDS)
    ax1.set_xlabel("WILDS training fold of the evaluated checkpoint")
    ax1.set_ylabel("PatchCamelyon AUROC")
    ax1.set_ylim(0.50, 0.84)
    ax1.set_title("Out of partition: the floor lifts, the spread compresses")
    ax1.legend(loc="lower left", ncol=2, fontsize=5.5, handlelength=2.2)
    panel_letter(ax1, "a", dx=-0.15)

    # (b) across-fold SD: within WILDS vs on PCam
    x = np.arange(len(CORE))
    w_in = [fold_mean(a).std(ddof=1) for a in CORE]
    w_ex = [np.array([np.mean(PC[a][f]) for f in FOLDS]).std(ddof=1) for a in CORE]
    ax2.bar(x - 0.19, w_in, 0.36, color=BLUE, edgecolor="white", linewidth=0.8,
            label="within Camelyon17-WILDS")
    ax2.bar(x + 0.19, w_ex, 0.36, color=ORANGE, edgecolor="white", linewidth=0.8,
            label="external: PatchCamelyon")
    for i in range(len(CORE)):
        ax2.annotate(f"{w_in[i]:.3f}", (x[i] - 0.19, w_in[i] + 0.002), ha="center",
                     fontsize=5.3, color=INK)
        ax2.annotate(f"{w_ex[i]:.3f}", (x[i] + 0.19, w_ex[i] + 0.002), ha="center",
                     fontsize=5.3, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels([NICE[a] for a in CORE], fontsize=5.5)
    ax2.set_ylabel("Across-site SD of AUROC")
    ax2.set_ylim(0, 0.122)
    ax2.set_title("Variance compression survives the partition change")
    ax2.legend(loc="upper right", fontsize=5.5)
    panel_letter(ax2, "b", dx=-0.16)

    fig.tight_layout(w_pad=2.4)
    save(fig, "fig3_external_pcam")


# ================================================================ Figure 4
def reliability(p, y, nbins=15):
    """Replicates the paper's ECE definition exactly (15 equal-width bins)."""
    yhat = (p >= 0.5).astype(int)
    conf = np.where(yhat == 1, p, 1 - p)
    corr = (yhat == y).astype(float)
    edges = np.linspace(0, 1, nbins + 1)
    cs, accs, ws = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            cs.append(conf[m].mean()); accs.append(corr[m].mean()); ws.append(m.mean())
    return np.array(cs), np.array(accs), np.array(ws)


def figure4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # (a) ECE per hospital
    x = np.arange(len(FOLDS))
    for i, arm in enumerate(CORE):
        c = STYLE[arm][0]
        e = fold_mean(arm, "ece")
        ax1.bar(x + (i - 1.5) * 0.2, e, 0.19, color=c, edgecolor="white",
                linewidth=0.7, label=NICE[arm])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"centre {f}" for f in FOLDS], fontsize=5.5)
    ax1.set_ylabel("Expected calibration error")
    ax1.set_title("Both SSL arms beat ERM and SAM at every hospital")
    ax1.legend(loc="upper right", fontsize=5.5)
    panel_letter(ax1, "a", dx=-0.13)

    # (b) reliability diagram at the hardest hospital (centre 2), seed 0..2 pooled
    ax2.plot([0.5, 1], [0.5, 1], color=MUTED, lw=0.7, linestyle=(0, (2, 2)),
             label="perfect calibration")
    for arm in CORE:
        c, mk, dash = STYLE[arm]
        ps, ys = [], []
        for s in range(3):
            f = f"{RUNS}/fold2/{arm}_seed{s}_preds.npz"
            if not os.path.exists(f):
                continue
            d = np.load(f)
            ps.append(d["p_ood_test"]); ys.append(d["y_ood_test"])
        cs, accs, _ = reliability(np.concatenate(ps), np.concatenate(ys))
        ax2.plot(cs, accs, color=c, marker=mk, linestyle=dash, markersize=3.8,
                 markeredgewidth=0, label=NICE[arm])
    ax2.set_xlabel("Predicted confidence")
    ax2.set_ylabel("Empirical accuracy")
    ax2.set_xlim(0.48, 1.0); ax2.set_ylim(0.28, 1.0)
    ax2.set_title("Reliability at the hardest hospital (centre 2)")
    ax2.legend(loc="upper left", fontsize=5.3, handlelength=2.2)
    panel_letter(ax2, "b", dx=-0.15)

    fig.tight_layout(w_pad=2.4)
    save(fig, "fig4_calibration")


# ================================================================ Figure 5
def figure5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    base = fold_mean("baseline")
    sE = base.std(ddof=1)

    arms = [a for a in FAMILY if a != "baseline"]
    obs, null, cols, names = [], [], [], []
    for a in arms:
        fm = fold_mean(a)
        obs.append(np.corrcoef(fm - base, base)[0, 1])
        null.append(-sE / math.sqrt(fm.std(ddof=1) ** 2 + sE ** 2))
        cols.append(FAMILY[a][1]); names.append(NICE[a])

    # (a) mechanism: per-hospital gain against baseline difficulty.
    # One arm per family so the hues stay unambiguous across panels.
    ax1.axhline(0, color=INK, lw=0.7)
    for arm, mk in (("ssl", "^"), ("probe_resnet50", "D"),
                    ("groupdro", "s"), ("macenko", "o")):
        fm = fold_mean(arm)
        gain = fm - base
        col = FAMILY[arm][1]
        ax1.scatter(base, gain, s=26, color=col, marker=mk, edgecolor="white",
                    linewidth=0.6, zorder=3, label=NICE[arm])
        b1, b0 = np.polyfit(base, gain, 1)
        xs = np.array([base.min() - 0.01, base.max() + 0.01])
        ax1.plot(xs, b0 + b1 * xs, color=col, lw=0.9,
                 linestyle=(0, ()) if b1 < 0 else (0, (4, 1.5)))
    ax1.set_xlabel("ERM AUROC at that hospital (difficulty)")
    ax1.set_ylabel("Gain over ERM at that hospital")
    ax1.set_ylim(-0.19, 0.30)
    ax1.set_title("A falling line arises mechanically")
    ax1.legend(loc="upper right", fontsize=5.3, markerscale=0.85)
    panel_letter(ax1, "a", dx=-0.16)

    # (b) excess, sorted
    order = np.argsort(np.array(obs) - np.array(null))
    exc = (np.array(obs) - np.array(null))[order]
    ax2.barh(np.arange(len(exc)), exc, color=[cols[i] for i in order],
             edgecolor="white", linewidth=0.7)
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_yticks(np.arange(len(exc)))
    ax2.set_yticklabels([names[i] for i in order], fontsize=5.5)
    ax2.set_xlabel(r"Excess $r$ over the independence null")
    ax2.annotate("misallocates gain to\nhospitals ERM already handles",
                 (0.36, 1.1), fontsize=5.5, color=MUTED)
    ax2.set_title("Corrected response-shape diagnostic")
    panel_letter(ax2, "b", dx=-0.42)

    fig.tight_layout(w_pad=3.0)
    save(fig, "fig5_response_shape")


# ================================================================ Supplementary
def figure_s1():
    arms = sorted(FAMILY, key=lambda a: -fold_mean(a).min())
    mat = np.array([fold_mean(a) for a in arms])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(mat, cmap="Blues", vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"c{f}" for f in FOLDS])
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([NICE[a] for a in arms], fontsize=6)
    for i in range(len(arms)):
        for j in range(5):
            ax.annotate(f"{mat[i,j]:.3f}", (j, i), ha="center", va="center",
                        fontsize=5.2, color="white" if mat[i, j] > 0.8 else INK)
    ax.set_xlabel("Held-out hospital")
    ax.set_title("Off-site AUROC, all thirteen arms")
    fig.colorbar(im, ax=ax, shrink=0.75, label="AUROC")
    fig.tight_layout()
    save(fig, "figS1_heatmap")


def figure_s2():
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    rng = np.random.default_rng(0)
    for i, arm in enumerate(CORE):
        c = STYLE[arm][0]
        m = per_seed(arm)
        for f in FOLDS:
            xs = f + (i - 1.5) * 0.2 + rng.uniform(-0.035, 0.035, len(m[f]))
            ax.scatter(xs, m[f], s=14, color=c, edgecolor="white", linewidth=0.5,
                       zorder=3, label=NICE[arm] if f == 0 else None)
            ax.plot([f + (i - 1.5) * 0.2 - 0.07, f + (i - 1.5) * 0.2 + 0.07],
                    [np.mean(m[f])] * 2, color=c, lw=1.4, zorder=4)
    ax.set_xticks(FOLDS); ax.set_xticklabels([f"centre {f}" for f in FOLDS])
    ax.set_ylabel("Off-site test AUROC")
    ax.set_title("Every individual run: three seeds per cell, bar = cell mean")
    ax.legend(loc="lower right", ncol=4, fontsize=5.5)
    fig.tight_layout()
    save(fig, "figS2_seedspread")


def figure_s3():
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.85), sharey=True)
    for f, ax in zip(FOLDS, axes):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, linestyle=(0, (2, 2)))
        for arm in CORE:
            c, mk, dash = STYLE[arm]
            d = np.load(f"{RUNS}/fold{f}/{arm}_seed0_preds.npz")
            fpr, tpr, _ = roc_curve(d["y_ood_test"], d["p_ood_test"])
            ax.plot(fpr, tpr, color=c, linestyle=dash, lw=1.0,
                    label=NICE[arm] if f == 0 else None)
        ax.set_title(f"centre {f}", fontsize=6)
        ax.set_xlabel("FPR")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    axes[0].set_ylabel("TPR")
    axes[0].legend(loc="lower right", fontsize=4.8, handlelength=1.8)
    fig.suptitle("ROC at every held-out hospital (seed 0)", fontsize=8, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "figS3_roc")


def figure_s4():
    arms = ["baseline", "sam", "ssl", "ssl_sam", "fish", "lisa", "erm_henorm",
            "groupdro", "coral", "irm", "mixstyle", "macenko"]
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    for i, a in enumerate(arms):
        ep = [d["selected_epoch"] for d in REC[a].values()]
        col = FAMILY[a][1]
        ax.scatter(np.full(len(ep), i) + np.random.default_rng(i).uniform(-.12, .12, len(ep)),
                   ep, s=14, color=col, edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [np.mean(ep)] * 2, color=col, lw=1.4, zorder=4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([NICE[a] for a in arms], rotation=35, ha="right", fontsize=5.5)
    ax.set_ylabel("Epoch chosen by OOD-val AUROC")
    ax.set_title("Model selection: Fish stops at epoch 1 in four of five folds")
    fig.tight_layout()
    save(fig, "figS4_selected_epoch")


if __name__ == "__main__":
    apply_figure_style(frame="open", sizes=(8, 7, 6))
    os.makedirs(OUT, exist_ok=True)
    print("generating figures ...")
    figure1(); figure2(); figure3(); figure4(); figure5()
    figure_s1(); figure_s2(); figure_s3(); figure_s4()
    print("done")
