"""Figures for the Camelyon17 four-arm ablation."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

try:
    from figure_style import apply_figure_style; apply_figure_style()
except Exception:
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200,
                         "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})

ARMS = ["baseline", "ssl", "sam", "ssl_sam"]
LBL = {"baseline": "Baseline", "ssl": "+SSL", "sam": "+SAM", "ssl_sam": "SSL+SAM"}
CLR = {"baseline": "#6b7280", "ssl": "#2563eb", "sam": "#d97706", "ssl_sam": "#059669"}
SRC = "runs_final" if glob.glob("runs_final/*.json") else "runs"
raw = pd.read_csv("results_raw.csv")
present = [a for a in ARMS if a in set(raw.arm)]

# ---------- FIG 1: metric bars with per-seed points ----------
METRICS = [("accuracy", "Accuracy"), ("auroc", "AUROC"),
           ("avg_precision", "Average Precision"), ("f1", "F1 Score"),
           ("balanced_accuracy", "Balanced Accuracy"), ("ece", "Calibration Error (ECE, lower=better)")]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (m, title) in zip(axes.ravel(), METRICS):
    col = f"ood_test_{m}"
    mu = [raw[raw.arm == a][col].mean() for a in present]
    sd = [raw[raw.arm == a][col].std(ddof=1) for a in present]
    xs = np.arange(len(present))
    ax.bar(xs, mu, yerr=sd, capsize=4, color=[CLR[a] for a in present],
           alpha=.85, edgecolor="black", linewidth=.6)
    for i, a in enumerate(present):
        v = raw[raw.arm == a][col].values
        ax.scatter(np.full(len(v), i) + np.linspace(-.09, .09, len(v)), v,
                   color="black", s=14, zorder=3, alpha=.8)
    ax.set_xticks(xs); ax.set_xticklabels([LBL[a] for a in present], fontsize=8)
    ax.set_title(title, fontsize=10)
    lo, hi = min(mu) - max(sd) * 2.2, max(mu) + max(sd) * 2.2
    pad = (hi - lo) * .15 + 1e-3
    ax.set_ylim(max(0, lo - pad), min(1, hi + pad) if m != "ece" else hi + pad)
    for i, (v, s) in enumerate(zip(mu, sd)):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold")
nseed = {a: raw[raw.arm == a].seed.nunique() for a in present}
fig.suptitle("Camelyon17-WILDS \u2014 held-out hospital (center 2, never seen in training)\n"
             "bars = mean over seeds (" + ", ".join(f"{LBL[a]}: n={nseed[a]}" for a in present)
             + "); error bars = s.d.; dots = individual seeds", fontsize=10.5)
fig.tight_layout(); fig.savefig("fig1_metric_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------- FIG 2: ROC + PR on the OOD hospital ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for a in present:
    f = f"{SRC}/{a}_seed0_preds.npz"
    if not os.path.exists(f): continue
    z = np.load(f); y, p = z["y_ood_test"], z["p_ood_test"]
    fpr, tpr, _ = roc_curve(y, p)
    au = raw[raw.arm == a]["ood_test_auroc"].mean()
    axes[0].plot(fpr, tpr, color=CLR[a], lw=1.8, label=f"{LBL[a]} (AUROC {au:.3f})")
    pr, rc, _ = precision_recall_curve(y, p)
    ap = raw[raw.arm == a]["ood_test_avg_precision"].mean()
    axes[1].plot(rc, pr, color=CLR[a], lw=1.8, label=f"{LBL[a]} (AP {ap:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=.8, alpha=.5)
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC — held-out hospital 2"); axes[0].legend(fontsize=8, loc="lower right")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision–Recall — held-out hospital 2"); axes[1].legend(fontsize=8, loc="lower left")

# generalizability gap
w = .35; xs = np.arange(len(present))
idv = [raw[raw.arm == a]["id_val_auroc"].mean() for a in present]
ood = [raw[raw.arm == a]["ood_test_auroc"].mean() for a in present]
axes[2].bar(xs - w/2, idv, w, label="In-distribution (val)", color="#93c5fd", edgecolor="k", lw=.6)
axes[2].bar(xs + w/2, ood, w, label="Held-out hospital", color="#1d4ed8", edgecolor="k", lw=.6)
for i, (a, b) in enumerate(zip(idv, ood)):
    axes[2].annotate("", xy=(i + w/2, b), xytext=(i - w/2, a),
                     arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
    axes[2].text(i, max(a, b) + .012, f"gap {a-b:.3f}", ha="center", fontsize=7.5, color="crimson")
axes[2].set_xticks(xs); axes[2].set_xticklabels([LBL[a] for a in present], fontsize=8)
axes[2].set_ylabel("AUROC"); axes[2].set_ylim(.5, 1.02)
axes[2].set_title("Generalizability gap"); axes[2].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig("fig2_roc_pr_gap.png", bbox_inches="tight")
plt.close(fig)

# ---------- FIG 3: training curves ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for a in present:
    for s in sorted(raw[raw.arm == a].seed.unique()):
        f = f"{SRC}/{a}_seed{s}.json"
        if not os.path.exists(f): continue
        log = pd.DataFrame(json.load(open(f))["log"])
        al = .95 if s == 0 else .35
        axes[0].plot(log.epoch, log.train_loss, color=CLR[a], alpha=al, lw=1.5,
                     label=LBL[a] if s == 0 else None)
        axes[1].plot(log.epoch, log.id_val_auroc, color=CLR[a], alpha=al, lw=1.5,
                     label=LBL[a] if s == 0 else None)
        axes[2].plot(log.epoch, log.ood_val_auroc, color=CLR[a], alpha=al, lw=1.5,
                     label=LBL[a] if s == 0 else None)
for ax, t, yl in zip(axes, ["Training loss", "In-distribution val AUROC",
                            "Out-of-distribution val AUROC (center 1)"],
                     ["cross-entropy", "AUROC", "AUROC"]):
    ax.set_xlabel("epoch"); ax.set_ylabel(yl); ax.set_title(t, fontsize=10)
    ax.legend(fontsize=8)
fig.suptitle("Training dynamics (bold = seed 0, faint = seeds 1–2)", fontsize=11)
fig.tight_layout(); fig.savefig("fig3_training_curves.png", bbox_inches="tight")
plt.close(fig)

# ---------- FIG 4: flatness ----------
if os.path.exists("flatness_curves.json"):
    cur = json.load(open("flatness_curves.json"))
    fl = pd.read_csv("flatness_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for a in present:
        ks = [k for k in cur if k.startswith(a + "_seed")]
        if not ks: continue
        rh = cur[ks[0]]["rhos"]
        A = np.array([cur[k]["adv"] for k in ks]); R = np.array([cur[k]["rand_mean"] for k in ks])
        axes[0].errorbar(rh, A.mean(0), yerr=A.std(0), color=CLR[a], marker="o", ms=4,
                         lw=1.6, capsize=3, label=LBL[a])
        axes[1].errorbar(rh, R.mean(0), yerr=R.std(0), color=CLR[a], marker="s", ms=4,
                         lw=1.6, capsize=3, label=LBL[a])
    axes[0].set_title("Adversarial sharpness (worst-case)", fontsize=10)
    axes[1].set_title("Random-direction sharpness (filter-normalized)", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel(r"perturbation radius $\rho$"); ax.set_ylabel(r"$L(w+\epsilon)-L(w)$")
        ax.set_xscale("log"); ax.legend(fontsize=8)
    # sharpness vs OOD performance — the mechanism claim
    mrg = fl.merge(raw, on=["arm", "seed"])
    for a in present:
        d = mrg[mrg.arm == a]
        axes[2].scatter(d["adv_sharp_rho0.2"], d["ood_test_auroc"], color=CLR[a],
                        s=52, edgecolor="k", lw=.6, label=LBL[a], zorder=3)
    if len(mrg) > 2:
        x, y = mrg["adv_sharp_rho0.2"].values, mrg["ood_test_auroc"].values
        r = np.corrcoef(x, y)[0, 1]
        b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        axes[2].plot(xx, np.polyval(b, xx), "k--", lw=1, alpha=.6)
        axes[2].set_title(f"Sharper minima → worse OOD transfer (r = {r:.3f})", fontsize=10)
    axes[2].set_xlabel(r"adversarial sharpness at $\rho=0.2$")
    axes[2].set_ylabel("held-out hospital AUROC"); axes[2].legend(fontsize=8)
    fig.suptitle("Loss-landscape sharpness — the paper's proposed evaluation metric (Proposal ii)", fontsize=11)
    fig.tight_layout(); fig.savefig("fig4_flatness.png", bbox_inches="tight")
    plt.close(fig)

# ---------- FIG 5: contribution waterfall ----------
if os.path.exists("nonredundancy.csv"):
    nr = pd.read_csv("nonredundancy.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    row = nr[nr.metric == "ood_test_auroc"].iloc[0]
    base = raw[raw.arm == "baseline"]["ood_test_auroc"].mean()
    vals = [base, row["gain_ssl"], row["gain_sam"], row["gain_both"] - row["gain_ssl"] - row["gain_sam"]]
    labels = ["Baseline", "+SSL\n(Pillar I)", "+SAM\n(Pillar II)", "Interaction"]
    run = base; ax = axes[0]
    ax.bar(0, base, color=CLR["baseline"], edgecolor="k", lw=.6)
    ax.text(0, base, f"{base:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for i, (v, l, c) in enumerate(zip(vals[1:], labels[1:], [CLR["ssl"], CLR["sam"], "#7c3aed"]), start=1):
        ax.bar(i, v, bottom=run, color=c, edgecolor="k", lw=.6)
        ax.text(i, run + v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
        run += v
    ax.bar(4, run, color=CLR["ssl_sam"], edgecolor="k", lw=.6)
    ax.text(4, run, f"{run:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels + ["SSL+SAM"], fontsize=8)
    ax.set_ylabel("held-out hospital AUROC")
    lo = min([base, run] + list(np.cumsum([base]+vals[1:])))
    hi = max([base, run] + list(np.cumsum([base]+vals[1:])))
    ax.set_ylim(lo - .05, hi + .03)
    ax.set_title("Decomposition of the OOD AUROC gain", fontsize=10)

    ax = axes[1]; w = .2
    ms = ["auroc", "accuracy", "avg_precision", "f1"]
    xs = np.arange(len(ms))
    for j, a in enumerate(present):
        g = [raw[raw.arm == a][f"ood_test_{m}"].mean() - raw[raw.arm == "baseline"][f"ood_test_{m}"].mean()
             for m in ms]
        ax.bar(xs + (j - 1.5) * w, g, w, color=CLR[a], edgecolor="k", lw=.6, label=LBL[a])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(xs); ax.set_xticklabels(["AUROC", "Accuracy", "Avg. Prec.", "F1"], fontsize=8)
    ax.set_ylabel("improvement over baseline"); ax.legend(fontsize=8)
    ax.set_title("Gain over baseline, all metrics", fontsize=10)
    fig.tight_layout(); fig.savefig("fig5_contributions.png", bbox_inches="tight")
    plt.close(fig)

print("figures written:", sorted(glob.glob("fig*.png")))
