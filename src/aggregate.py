"""Aggregate the 4-arm x 3-seed ablation; paired stats; markdown table."""
import glob, json, os
import numpy as np, pandas as pd
from scipy import stats

ARMS = ["baseline", "ssl", "sam", "ssl_sam"]
LABEL = {"baseline": "Baseline (supervised)", "ssl": "+ SSL (Pillar I)",
         "sam": "+ SAM (Pillar II)", "ssl_sam": "SSL + SAM (both)"}
METRICS = ["accuracy", "auroc", "avg_precision", "f1", "balanced_accuracy",
           "sensitivity", "specificity", "precision", "ece"]

SRC = "runs_final" if glob.glob("runs_final/*.json") else "runs"
EXPECT = (5, 30000)          # (epochs, n_train) of the reported configuration

rows = []
for f in sorted(glob.glob(f"{SRC}/*.json")):
    r = json.load(open(f))
    if (r["epochs"], r["n_train"]) != EXPECT:
        print(f"SKIP {os.path.basename(f)}: config {(r['epochs'], r['n_train'])} "
              f"!= {EXPECT}"); continue
    base = dict(arm=r["arm"], seed=r["seed"], selected_epoch=r["selected_epoch"],
                wall_s=r["wall_s"], n_train=r["n_train"])
    for split in ("ood_test", "id_val", "ood_val"):
        for m in METRICS:
            base[f"{split}_{m}"] = r[split][m]
    rows.append(base)
raw = pd.DataFrame(rows).sort_values(["arm", "seed"])
raw.to_csv("results_raw.csv", index=False)

# ---- summary: mean +/- std per arm ----
agg = {}
for m in METRICS:
    for split in ("ood_test", "id_val"):
        c = f"{split}_{m}"
        agg[f"{c}_mean"] = raw.groupby("arm")[c].mean()
        agg[f"{c}_std"] = raw.groupby("arm")[c].std(ddof=1)
summary = pd.DataFrame(agg).reindex(ARMS)
summary["gen_gap_auroc"] = summary["id_val_auroc_mean"] - summary["ood_test_auroc_mean"]
summary["gen_gap_acc"] = summary["id_val_accuracy_mean"] - summary["ood_test_accuracy_mean"]
summary.to_csv("results_summary.csv")

# ---- paired tests on the contrasts the argument needs ----
def paired(a, b, col):
    x = raw[raw.arm == a].sort_values("seed")[col].values
    y = raw[raw.arm == b].sort_values("seed")[col].values
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    dif = y - x
    if n < 2 or np.allclose(dif, 0):
        return dict(contrast=f"{a}->{b}", metric=col, delta=float(dif.mean()),
                    t=np.nan, p=np.nan, cohen_d=np.nan, n=n)
    t, p = stats.ttest_rel(y, x)
    return dict(contrast=f"{a}->{b}", metric=col, delta=float(dif.mean()),
                t=float(t), p=float(p),
                cohen_d=float(dif.mean() / (dif.std(ddof=1) + 1e-12)), n=n)

tests = []
for col in ["ood_test_auroc", "ood_test_accuracy", "ood_test_avg_precision",
            "ood_test_f1"]:
    for a, b in [("baseline", "ssl"), ("baseline", "sam"), ("baseline", "ssl_sam"),
                 ("ssl", "ssl_sam"), ("sam", "ssl_sam")]:
        tests.append(paired(a, b, col))
tests = pd.DataFrame(tests)
tests.to_csv("statistical_tests.csv", index=False)

# ---- non-redundancy: is the joint gain >= sum of individual gains? ----
nr = []
for col in ["ood_test_auroc", "ood_test_accuracy", "ood_test_avg_precision", "ood_test_f1"]:
    mb = raw[raw.arm == "baseline"][col].mean()
    g_ssl = raw[raw.arm == "ssl"][col].mean() - mb
    g_sam = raw[raw.arm == "sam"][col].mean() - mb
    g_both = raw[raw.arm == "ssl_sam"][col].mean() - mb
    nr.append(dict(metric=col, gain_ssl=g_ssl, gain_sam=g_sam, gain_both=g_both,
                   sum_individual=g_ssl + g_sam,
                   interaction=g_both - (g_ssl + g_sam),
                   both_beats_best_single=g_both > max(g_ssl, g_sam)))
nr = pd.DataFrame(nr)
nr.to_csv("nonredundancy.csv", index=False)


def star(p):
    if not np.isfinite(p): return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


# ---- markdown table ----
L = []
L.append("# Camelyon17-WILDS cross-hospital ablation\n")
L.append(f"Held-out test hospital: **center 2** (never seen in training, SSL "
         f"pre-training, or model selection). Model selection on center 1 "
         f"(ood_val), per the official WILDS protocol. "
         f"{raw.groupby('arm').size().min()} seeds per arm.\n")
L.append("## Out-of-distribution test hospital (center 2)\n")
hdr = "| Arm | Accuracy | AUROC | Avg. Precision | F1 | Balanced Acc. | Sensitivity | Specificity | ECE |"
L.append(hdr); L.append("|" + "---|" * 9)
for a in ARMS:
    if a not in summary.index or not np.isfinite(summary.loc[a, "ood_test_auroc_mean"]):
        continue
    r = summary.loc[a]
    cells = [LABEL[a]]
    for m in ["accuracy", "auroc", "avg_precision", "f1", "balanced_accuracy",
              "sensitivity", "specificity", "ece"]:
        cells.append(f"{r[f'ood_test_{m}_mean']:.4f} ± {r[f'ood_test_{m}_std']:.4f}")
    L.append("| " + " | ".join(cells) + " |")

L.append("\n## In-distribution vs out-of-distribution (the generalizability gap)\n")
L.append("| Arm | ID val AUROC | OOD test AUROC | Gap | ID val Acc | OOD test Acc | Gap |")
L.append("|" + "---|" * 7)
for a in ARMS:
    if a not in summary.index or not np.isfinite(summary.loc[a, "ood_test_auroc_mean"]):
        continue
    r = summary.loc[a]
    L.append(f"| {LABEL[a]} | {r['id_val_auroc_mean']:.4f} | {r['ood_test_auroc_mean']:.4f} "
             f"| {r['gen_gap_auroc']:.4f} | {r['id_val_accuracy_mean']:.4f} "
             f"| {r['ood_test_accuracy_mean']:.4f} | {r['gen_gap_acc']:.4f} |")

L.append("\n## Paired significance tests (OOD test hospital, paired by seed)\n")
L.append("| Contrast | Metric | Δ | t | p | Cohen's d | |")
L.append("|" + "---|" * 7)
for _, t in tests.iterrows():
    L.append(f"| {t['contrast']} | {t['metric'].replace('ood_test_','')} | "
             f"{t['delta']:+.4f} | {t['t']:.3f} | {t['p']:.4f} | {t['cohen_d']:+.2f} | {star(t['p'])} |")

L.append("\n## Non-redundancy of the two pillars\n")
L.append("| Metric | Gain SSL | Gain SAM | Gain both | Sum of singles | Interaction | Both > best single |")
L.append("|" + "---|" * 7)
for _, r in nr.iterrows():
    L.append(f"| {r['metric'].replace('ood_test_','')} | {r['gain_ssl']:+.4f} | "
             f"{r['gain_sam']:+.4f} | {r['gain_both']:+.4f} | {r['sum_individual']:+.4f} | "
             f"{r['interaction']:+.4f} | {'yes' if r['both_beats_best_single'] else 'no'} |")

if os.path.exists("flatness_metrics.csv"):
    fl = pd.read_csv("flatness_metrics.csv")
    fm = fl.groupby("arm").mean(numeric_only=True).reindex(ARMS)
    L.append("\n## Loss-landscape sharpness (paper's Proposal ii)\n")
    L.append("| Arm | Train loss | Adversarial sharpness ρ=0.05 | ρ=0.2 | Random-dir ρ=0.2 |")
    L.append("|" + "---|" * 5)
    for a in ARMS:
        if a not in fm.index or not np.isfinite(fm.loc[a, "base_loss"]):
            continue
        r = fm.loc[a]
        L.append(f"| {LABEL[a]} | {r['base_loss']:.4f} | {r['adv_sharp_rho0.05']:.4f} "
                 f"| {r['adv_sharp_rho0.2']:.4f} | {r['rand_sharp_rho0.2']:.4f} |")

open("performance_comparison.md", "w").write("\n".join(L) + "\n")
print("\n".join(L[:40]))
print("\nwrote results_raw.csv, results_summary.csv, statistical_tests.csv, "
      "nonredundancy.csv, performance_comparison.md")
