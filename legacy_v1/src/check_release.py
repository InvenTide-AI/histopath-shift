#!/usr/bin/env python3
"""Check that the shipped result tables reproduce the numbers printed in the
submitted manuscript (tracked outside this repository).

    python src/check_release.py

Each block below re-derives one published table or abstract claim from the CSVs
in results/ and compares it against the value in the paper, hard-coded here as
the reference. Nothing is recomputed from raw data -- this checks the release
against the paper, so that a table edited or replaced by mistake cannot pass
silently. Exits non-zero on any mismatch.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

TOL = 1.1e-3          # published values are rounded to 3 decimals
fails, checks = [], 0


def eq(name, got, want, tol=TOL):
    global checks
    checks += 1
    if got is None or not np.isfinite(got) or abs(got - want) > tol:
        fails.append(f"{name}: shipped {got!r}, paper {want!r}")


def ok(name, cond):
    global checks
    checks += 1
    if not cond:
        fails.append(f"{name}: failed")


# ---------------------------------------------------------------- Table II ---
# Site-complete cross-hospital factorial (five hospitals, one seed per arm).
res = pd.read_csv("results/camelyon17_fivehospital/results_folds.csv")
piv = res.pivot_table(index="fold", columns="arm", values="ood_test_auroc")
piv["I"] = (piv["ssl_sam"] - piv["ssl"]) - (piv["sam"] - piv["baseline"])
piv["net"] = piv["ssl_sam"] - piv["baseline"]
piv["dssl"] = piv["ssl"] - piv["baseline"]
piv["dsam"] = piv["sam"] - piv["baseline"]

TABLE_II = {   # held-out center: ERM, SAM, SSL, SSL+SAM, dSSL, dSAM, I, Net
    0: (0.954, 0.954, 0.961, 0.960, +0.007, +0.000, -0.001, +0.006),
    1: (0.769, 0.801, 0.934, 0.933, +0.165, +0.032, -0.032, +0.164),
    2: (0.920, 0.901, 0.910, 0.941, -0.011, -0.019, +0.050, +0.021),
    3: (0.944, 0.880, 0.967, 0.963, +0.023, -0.065, +0.061, +0.019),
    4: (0.667, 0.704, 0.941, 0.936, +0.274, +0.037, -0.043, +0.268),
}
cols = ["baseline", "sam", "ssl", "ssl_sam", "dssl", "dsam", "I", "net"]
ok("Table II: five held-out hospitals present", sorted(piv.index) == [0, 1, 2, 3, 4])
for fold, row in TABLE_II.items():
    for c, want in zip(cols, row):
        eq(f"Table II C{fold} {c}", float(piv.loc[fold, c]), want)
eq("Table II mean ERM", float(piv["baseline"].mean()), 0.851)
eq("Table II mean SSL", float(piv["ssl"].mean()), 0.943)
eq("Table II mean I", float(piv["I"].mean()), 0.007)
eq("Table II SD across hospitals, ERM", float(piv["baseline"].std(ddof=1)), 0.127)
eq("Table II SD across hospitals, SAM", float(piv["sam"].std(ddof=1)), 0.098)
eq("Table II SD across hospitals, SSL", float(piv["ssl"].std(ddof=1)), 0.023)
eq("Table II SD across hospitals, SSL+SAM", float(piv["ssl_sam"].std(ddof=1)), 0.014)
eq("Table II SD across hospitals, I", float(piv["I"].std(ddof=1)), 0.047)

# --------------------------------------------------------- abstract claims ---
eq("abstract: best ERM hospital", float(piv["baseline"].max()), 0.954)
eq("abstract: worst ERM hospital", float(piv["baseline"].min()), 0.667)
eq("abstract: lift at worst hospital", float(piv.loc[piv["baseline"].idxmin(), "dssl"]), 0.274)
ok("abstract: spread cut more than five-fold",
   piv["baseline"].std(ddof=1) / piv["ssl"].std(ddof=1) > 5.0)
eq("Fig. 1b: Pearson r(baseline, SSL gain)",
   float(np.corrcoef(piv["baseline"], piv["dssl"])[0, 1]), -0.98, tol=5e-3)
ok("abstract: mean interaction CI covers zero",
   piv["I"].mean() - 2.776 * piv["I"].std(ddof=1) / np.sqrt(5) < 0 <
   piv["I"].mean() + 2.776 * piv["I"].std(ddof=1) / np.sqrt(5))

# derived spread table must agree with the fold table it summarises
spread = pd.read_csv("results/camelyon17_fivehospital/spread_by_arm.csv").set_index("arm")
for arm in ["baseline", "sam", "ssl", "ssl_sam"]:
    eq(f"spread_by_arm {arm} sd", float(spread.loc[arm, "sd_across_hospitals"]),
       round(float(piv[arm].std(ddof=1)), 4))
    eq(f"spread_by_arm {arm} range", float(spread.loc[arm, "range"]),
       round(float(piv[arm].max() - piv[arm].min()), 4))

# --------------------------------------------------------------- Table III ---
# Held-out hospital = Camelyon17 center 2, seed-matched (the seed run in all arms).
raw = pd.read_csv("results/camelyon17_center2/results_raw.csv")
s0 = raw[raw["seed"] == 0].set_index("arm")
TABLE_III = {          # AUROC, AP, Acc, F1, ECE
    "baseline": (0.931, 0.922, 0.819, 0.794, 0.074),
    "ssl":      (0.895, 0.872, 0.805, 0.799, 0.063),
    "sam":      (0.826, 0.807, 0.733, 0.719, 0.141),
    "ssl_sam":  (0.951, 0.953, 0.883, 0.886, 0.015),
}
mcols = ["ood_test_auroc", "ood_test_avg_precision", "ood_test_accuracy",
         "ood_test_f1", "ood_test_ece"]
for arm, row in TABLE_III.items():
    for c, want in zip(mcols, row):
        eq(f"Table III {arm} {c}", float(s0.loc[arm, c]), want)
eq("center 2: ECE ratio baseline / SSL+SAM",
   float(s0.loc["baseline", "ood_test_ece"] / s0.loc["ssl_sam", "ood_test_ece"]), 5.06, tol=0.02)

# ---------------------------------------------------------------- Table IV ---
# NCT-CRC-HE, stain-appearance shift, metric average over two seeds.
crc = pd.read_csv("results/nct_crc_he/crc_results_summary.csv").set_index("arm")
TABLE_IV = {"baseline": (0.791, 0.156), "ssl": (0.702, 0.235),
            "sam": (0.726, 0.206), "ssl_sam": (0.724, 0.232)}
for arm, (auroc, ece) in TABLE_IV.items():
    eq(f"Table IV {arm} AUROC", float(crc.loc[arm, "shift_nonorm_auroc_mean"]), auroc)
    eq(f"Table IV {arm} ECE", float(crc.loc[arm, "shift_nonorm_ece_mean"]), ece)


def decompose(col):
    b, s, m, sm = (float(crc.loc[a, col]) for a in ["baseline", "ssl", "sam", "ssl_sam"])
    return s - b, m - b, (sm - s) - (m - b), sm - b


d_ssl, d_sam, inter, net = decompose("shift_nonorm_auroc_mean")
eq("Table IV SSL main effect", d_ssl, -0.089)
eq("Table IV SAM main effect", d_sam, -0.065)
eq("Table IV interaction", inter, +0.087)
eq("Table IV net", net, -0.067)
_, _, inter_x, net_x = decompose("shift_external_auroc_mean")
eq("Sec. III-E external cohort interaction", inter_x, +0.004)
eq("Sec. III-E external cohort net", net_x, -0.0045)

# ------------------------------------------------------ structural checks ----
runs = [f for f in os.listdir("runs/camelyon17_center2") if f.endswith(".json")]
ok("center-2 per-run records present (6)", len(runs) == 6)
for f in runs:
    r = json.load(open(os.path.join("runs/camelyon17_center2", f)))
    ok(f"run record {f} has an arm and a test metric",
       "arm" in r and any("auroc" in json.dumps(v) for v in r.values() if isinstance(v, dict)))
flat = pd.read_csv("results/camelyon17_center2/flatness_metrics.csv")
ok("sharpness records cover all four arms", set(flat["arm"]) == {"baseline", "sam", "ssl", "ssl_sam"})
ok("sharpness measured at four radii",
   sum(c.startswith("adv_sharp_rho") for c in flat.columns) == 4)
site = pd.read_csv("results/site_characterization/site_shift.csv")
ok("site-shift characterisation covers five hospitals", len(site) == 5)
ok("every hospital is separable from the rest by a domain classifier",
   (site["domain_auroc_vs_rest"] > 0.9).all())
for p in ["protocol/protocol_matched_folds.json", "protocol/hospital_splits_manifest.json",
          "results/nct_crc_he/crc_data_manifest.json"]:
    ok(f"{p} present", os.path.exists(p))

# --------------------------------------------------------------------------- #
print(f"{checks} checks run, {len(fails)} failed")
for f in fails:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
