"""Cross-cohort figures: how the domain effect and the method ranking change
when the thing that varies between domains changes.

Three cohorts, all run through the identical pipeline and the identical
aggregator:

  Camelyon17-WILDS   hospital shift  (scanner + stain + case mix), K = 5
  MIDOG 2021         scanner only, one lab, same slides,           K = 3
  Canine SCC         scanner only, scale-normalized, slide-disjoint, K = 5

Palette and redundant-encoding rules follow make_figures_tmlr.py.
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
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from figure_style import apply_figure_style, panel_letter

ROOT = "./data"
OUT = "paper/tmlr/figs"

COHORTS = [
    ("Camelyon17", "%s/runs" % ROOT, "hospital\n(scanner+stain+case mix)", 5),
    ("MIDOG 2021", "%s/midog_runs" % ROOT, "scanner only\n(one lab)", 3),
    ("Canine SCC", "%s/canine_runs" % ROOT, "scanner only\n(slide-disjoint)", 5),
]

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GREY, INK, MUTED = "#6b6b6b", "#0b0b0b", "#52514e"

NICE = {"baseline": "ERM", "sam": "SAM", "ssl": "SSL", "ssl_sam": "SSL+SAM",
        "probe_resnet50": "Frozen probe", "erm_henorm": "H&E jitter",
        "mixstyle": "MixStyle", "groupdro": "GroupDRO", "coral": "DeepCORAL",
        "irm": "IRMv1", "macenko": "Macenko", "lisa": "LISA", "fish": "Fish",
        "rotinv": "RotInv", "tia_style": "TIA-style"}
FAMILY = {
    "ssl": BLUE, "ssl_sam": BLUE,
    "probe_resnet50": VIOLET,
    "erm_henorm": ORANGE, "macenko": ORANGE, "mixstyle": ORANGE,
    "tia_style": ORANGE, "rotinv": ORANGE,
    "groupdro": AQUA, "coral": AQUA, "irm": AQUA, "fish": AQUA, "lisa": AQUA,
    "baseline": GREY, "sam": GREY,
}
COMMON = ["probe_resnet50", "ssl", "ssl_sam", "erm_henorm", "mixstyle", "sam",
          "baseline", "groupdro", "coral", "irm", "macenko", "fish", "lisa"]


def load(runs):
    rec = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(runs, "fold*", "*.json"))):
        fold = int(os.path.basename(os.path.dirname(f)).replace("fold", ""))
        arm, seed = os.path.basename(f)[:-5].rsplit("_seed", 1)
        rec[arm][(fold, int(seed))] = json.load(open(f))["ood_test"]
    return rec


def per_fold(rec, arm, metric="auroc"):
    d = collections.defaultdict(list)
    for (f, _), m in rec[arm].items():
        d[f].append(m[metric])
    return np.array([np.mean(d[f]) for f in sorted(d)])


def components(rec, arm):
    d = collections.defaultdict(dict)
    for (f, s), m in rec[arm].items():
        d[f][s] = m["auroc"]
    folds = sorted(d)
    seeds = sorted(set.intersection(*[set(d[f]) for f in folds]))
    y = np.array([[d[f][s] for s in seeds] for f in folds])
    k, ns = y.shape
    g = y.mean()
    ms_f = ns * ((y.mean(1) - g) ** 2).sum() / (k - 1)
    ms_s = k * ((y.mean(0) - g) ** 2).sum() / (ns - 1)
    ms_e = ((y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + g) ** 2
            ).sum() / ((k - 1) * (ns - 1))
    s2f = max((ms_f - ms_e) / ns, 0.0)
    s2s = max((ms_s - ms_e) / k, 0.0)
    return s2f, s2s, ms_e


def min_domains(s_site, s_run, eff, seeds=1, power=0.8):
    for K in range(2, 400):
        se = math.sqrt(s_site ** 2 / K + s_run ** 2 / (K * seeds))
        df = K - 1
        tc = stats.t.ppf(0.975, df)
        if 1 - stats.nct.cdf(tc, df, eff / se) + stats.nct.cdf(-tc, df, eff / se) >= power:
            return K
    return np.nan


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig("%s/%s.%s" % (OUT, name, ext))
    plt.close(fig)
    print("  wrote %s/%s.pdf" % (OUT, name))


REC = {n: load(p) for n, p, _, _ in COHORTS}


def figure6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    names = [c[0] for c in COHORTS]
    x = np.arange(len(names))

    site, run = [], []
    for n in names:
        s2f, s2s, s2e = components(REC[n], "baseline")
        site.append(math.sqrt(s2f))
        run.append(math.hypot(math.sqrt(s2s), math.sqrt(s2e)))

    ax1.bar(x - 0.19, site, 0.36, color=BLUE, edgecolor="white", linewidth=0.8,
            label=r"$\sigma_{\rm site}$ (between domains)")
    ax1.bar(x + 0.19, run, 0.36, color=ORANGE, edgecolor="white", linewidth=0.8,
            label=r"$\sigma_{\rm run}$ (seed + residual)")
    for i in range(len(names)):
        ax1.annotate("%.3f" % site[i], (x[i] - 0.19, site[i] + 0.002),
                     ha="center", fontsize=5.4, color=INK)
        ax1.annotate("%.3f" % run[i], (x[i] + 0.19, run[i] + 0.002),
                     ha="center", fontsize=5.4, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["%s\n%s" % (c[0], c[2]) for c in COHORTS], fontsize=5.3)
    ax1.set_ylabel("SD of off-site AUROC (ERM)")
    ax1.set_ylim(0, 0.095)
    ax1.set_title("Hospital shift is larger than scanner shift")
    ax1.legend(loc="upper right", fontsize=5.5)
    panel_letter(ax1, "a", dx=-0.14)

    for i, n in enumerate(names):
        s2f, s2s, s2e = components(REC[n], "baseline")
        sf, sr = math.sqrt(s2f), math.hypot(math.sqrt(s2s), math.sqrt(s2e))
        deltas = np.arange(0.04, 0.205, 0.004)
        K = [min_domains(sf, sr, d, 3) for d in deltas]
        col, mk = [BLUE, ORANGE, VIOLET][i], ["^", "s", "D"][i]
        ax2.plot(deltas, K, color=col, marker=mk, markevery=9, markersize=3.6,
                 markeredgewidth=0, label="%s (K=%d available)" % (n, COHORTS[i][3]))
    ax2.axhline(1, color=INK, lw=0.8)
    ax2.annotate("the single-domain convention", (0.10, 1.12), fontsize=5.6, color=INK)
    ax2.set_yscale("log")
    ax2.set_yticks([1, 2, 3, 5, 8, 12, 25])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel(r"Target mean effect $\Delta$ (AUROC)")
    ax2.set_ylabel("Domains needed, 3 seeds each")
    ax2.set_title("Every cohort needs more than one domain")
    ax2.legend(loc="upper right", fontsize=5.4)
    panel_letter(ax2, "b", dx=-0.16)

    fig.tight_layout(w_pad=2.4)
    save(fig, "fig6_cohort_variance")


def figure7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    names = [c[0] for c in COHORTS]

    # (a) rank of worst-site AUROC across cohorts
    worst = {n: {a: per_fold(REC[n], a).min() for a in COMMON if a in REC[n]}
             for n in names}
    ranks = {}
    for n in names:
        order = sorted(worst[n], key=lambda a: -worst[n][a])
        ranks[n] = {a: i + 1 for i, a in enumerate(order)}
    xs = np.arange(len(names))
    for a in COMMON:
        ys = [ranks[n].get(a, np.nan) for n in names]
        moved = max(ys) - min(ys)
        lw, alpha = (1.9, 1.0) if moved >= 8 else (0.8, 0.45)
        ax1.plot(xs, ys, color=FAMILY[a], lw=lw, alpha=alpha,
                 marker="o", markersize=3.6, markeredgewidth=0, zorder=3 if lw > 1 else 2)
        ax1.annotate(NICE[a], (-0.06, ys[0]), fontsize=5.2, ha="right",
                     va="center", color=INK if lw > 1 else MUTED)
        ax1.annotate(NICE[a], (len(names) - 1 + 0.06, ys[-1]), fontsize=5.2,
                     ha="left", va="center", color=INK if lw > 1 else MUTED)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, fontsize=6)
    ax1.set_xlim(-0.80, len(names) - 1 + 0.80)
    ax1.invert_yaxis()
    ax1.set_yticks([1, 4, 7, 10, 13])
    ax1.set_ylabel("Rank by worst-domain AUROC")
    ax1.set_title("Two arms swap ends between shift types")
    panel_letter(ax1, "a", dx=-0.20)

    # (b) SSL mean effect with CI per cohort
    ypos = np.arange(len(names))[::-1]
    for i, n in enumerate(names):
        base = per_fold(REC[n], "baseline")
        d = per_fold(REC[n], "ssl") - base
        m = d.mean()
        ci = stats.t.interval(0.95, len(d) - 1, loc=m, scale=stats.sem(d))
        _, p = stats.ttest_1samp(d, 0)
        ax2.plot(ci, [ypos[i]] * 2, color=BLUE, lw=1.4,
                 solid_capstyle="butt", zorder=2)
        ax2.plot([m], [ypos[i]], marker="D", color=BLUE, markersize=5,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax2.annotate("%+.3f  (p=%.2f)" % (m, p), (ci[1] + 0.012, ypos[i]),
                     fontsize=5.6, va="center", color=INK)
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(["%s\n%s" % (c[0], c[2].replace("\n", " ")) for c in COHORTS],
                        fontsize=5.4)
    ax2.set_xlim(-0.07, 0.30)
    ax2.set_xlabel("SSL mean effect over ERM (AUROC)")
    ax2.set_title("The benefit of pre-training narrows with the shift")
    panel_letter(ax2, "b", dx=-0.30)

    fig.tight_layout(w_pad=3.2)
    save(fig, "fig7_cohort_ranking")


if __name__ == "__main__":
    apply_figure_style(frame="open", sizes=(8, 7, 6))
    os.makedirs(OUT, exist_ok=True)
    figure6()
    figure7()
    # cross-cohort summary table, for the paper text
    print()
    rows = []
    for n, _, shift, K in COHORTS:
        s2f, s2s, s2e = components(REC[n], "baseline")
        sf, sr = math.sqrt(s2f), math.hypot(math.sqrt(s2s), math.sqrt(s2e))
        base = per_fold(REC[n], "baseline")
        best = max((a for a in REC[n]), key=lambda a: per_fold(REC[n], a).min())
        d = per_fold(REC[n], "ssl") - base
        _, p = stats.ttest_1samp(d, 0)
        rows.append(dict(cohort=n, K=K, sigma_site=round(sf, 4),
                         sigma_run=round(sr, 4),
                         icc=round(s2f / (s2f + s2s + s2e), 3),
                         erm_worst=round(base.min(), 4),
                         best_arm=best,
                         best_worst=round(per_fold(REC[n], best).min(), 4),
                         ssl_effect=round(d.mean(), 4), ssl_p=round(p, 4),
                         K_for_010=min_domains(sf, sr, 0.10, 3)))
    os.makedirs("results/cohorts", exist_ok=True)
    json.dump(rows, open("results/cohorts/summary.json", "w"), indent=1)
    hdr = "%-12s %2s %11s %10s %6s %10s %-16s %10s %11s %8s"
    print(hdr % ("cohort", "K", "sigma_site", "sigma_run", "ICC",
                 "ERM worst", "best arm", "best worst", "SSL effect", "K@0.10"))
    for r in rows:
        print("%-12s %2d %11.4f %10.4f %6.3f %10.4f %-16s %10.4f %+11.4f %8s"
              % (r["cohort"], r["K"], r["sigma_site"], r["sigma_run"], r["icc"],
                 r["erm_worst"], NICE.get(r["best_arm"], r["best_arm"]),
                 r["best_worst"], r["ssl_effect"], r["K_for_010"]))
