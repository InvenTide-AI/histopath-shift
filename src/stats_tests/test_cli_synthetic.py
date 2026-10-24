"""End-to-end check of analyze_v2.py on a synthetic v2 run tree (CONTRACTS.md B layout).

Four cohorts with the PLAN.md K's (c17 5, canine 5, midog21 3 without val domain, midogpp 7),
4 arms x 3 seeds, a TTA entry, and a manifest for one cohort.  Checks that every output is
written, that the variance table equals stats_v2 on the same matrix, and that the K_eff design
is read from manifest / test+val domains.
"""
import csv
import json
import os
import shutil
import sys
import tempfile

import numpy as np

from _common import HERE, S, run_module

sys.path.insert(0, os.path.dirname(HERE))
import analyze_v2 as A  # noqa: E402

KS = dict(c17=5, canine=5, midog21=3, midogpp=7)
ARMS = dict(erm=0.0, sam=0.01, simclr=0.05, groupdro=-0.01)


def make_tree(root):
    rng = np.random.default_rng(0)
    truth = {}
    for coh, K in KS.items():
        b = rng.normal(0, 0.05, K)
        for f in range(K):
            val = None if coh == "midog21" else f"D{(f - 1) % K}"
            d = os.path.join(root, "runs", "C", coh, f"fold{f}")
            os.makedirs(d, exist_ok=True)
            if coh == "c17":
                md = os.path.join(root, "cache64", coh, f"fold{f}")
                os.makedirs(md, exist_ok=True)
                tr = [f"D{t}" for t in range(K) if t not in (f, (f - 1) % K)]
                json.dump(dict(train_domains=tr), open(os.path.join(md, "manifest.json"), "w"))
            for arm, eff in ARMS.items():
                for s in range(3):
                    y = float(np.clip(0.8 + b[f] + eff + rng.normal(0, 0.03), 0, 1))
                    truth[(coh, arm, f, s)] = y
                    rec = dict(schema="v2", arm=arm, tier="C", cohort=coh, fold=f, test_domain=f"D{f}", val_domain=val,
                               seed=s, phase="final", input_res=64,
                               test_at={"id": {"auroc": y, "ece": 0.05}, "last": {"auroc": y - 0.01, "ece": 0.06}})
                    if val is not None:
                        rec["test_at"]["ood"] = {"auroc": y + 0.005, "ece": 0.05}
                    if arm == "erm":
                        rec["tta"] = {"bnadapt": {"auroc": y + 0.02, "ece": 0.04}}
                    json.dump(rec, open(os.path.join(d, f"{arm}__default__s{s}.json"), "w"))
            # a tune-phase record that must be ignored
            json.dump(dict(schema="v2", arm="erm", fold=f, seed=9, phase="tune", test_domain=f"D{f}",
                           test_at={"id": {"auroc": 0.0}}), open(os.path.join(d, "erm__lr0.1__s9.json"), "w"))
    return truth


def test_cli_end_to_end():
    tmp = tempfile.mkdtemp(prefix="v2stats_")
    try:
        truth = make_tree(tmp)
        out = os.path.join(tmp, "analysis")
        # the synthetic tree uses the cohorts of KS (incl. the un-normalized midog21), not the CLI default
        A.main(["runs", "--v2-root", tmp, "--tier", "C", "--cohorts", *KS, "--out", out,
                "--B", "500", "--B-tau", "200", "--B-perm", "500"])
        for coh, K in KS.items():
            od = os.path.join(out, "C", coh)
            for fn in ("per_run_id.csv", "summary_id.csv", "variance_id.csv", "contrasts_id.csv", "power_id.csv",
                       "ranks_id.csv", "response_id.csv", "tables/summary_id.tex", "tables/variance_id.tex",
                       "tables/contrasts_id.tex", "tables/power_id.tex"):
                assert os.path.getsize(os.path.join(od, fn)) > 0, (coh, fn)
            meta = json.load(open(os.path.join(od, "meta_id.json")))
            assert meta["K"] == K and "bnadapt@erm" in meta["arms"] and len(meta["arms"]) == 5
            y = np.array([[truth[(coh, "sam", f, s)] for s in range(3)] for f in range(K)])
            row = [r for r in csv.DictReader(open(os.path.join(od, "variance_id.csv"))) if r["arm"] == "sam"][0]
            assert abs(float(row["sigma_domain_reml"]) - S.oneway_reml(y)["sigma_domain"]) < 1e-9
            assert abs(float(row["icc_ci_lo"]) - S.icc1_exact_ci(y)["lo"]) < 1e-9
            con = {r["arm"]: r for r in csv.DictReader(open(os.path.join(od, "contrasts_id.csv")))}
            if coh in ("c17", "canine"):
                assert abs(float(con["sam"]["rho1_Keff"]) - 25 / 15) < 1e-9, con["sam"]["rho1_Keff"]
            if coh == "midog21":  # no val domain: trains on 2 of 3, every pair shares 1 domain
                assert abs(float(con["sam"]["rho1_Keff"]) - 9 / (3 + 6 * 0.5)) < 1e-9
        for fn in ("kendall_id.csv", "interaction_id.json", "clustering_id.json", "kendall_id.tex"):
            assert os.path.getsize(os.path.join(out, "C", "cross", fn)) > 0
        cl = json.load(open(os.path.join(out, "C", "cross", "clustering_id.json")))
        assert set(cl) >= {"same_task", "same_shift", "neither"}
        it = json.load(open(os.path.join(out, "C", "cross", "interaction_id.json")))
        assert it["cohort"]["method"] == "wild" and "task" not in it and "shift" not in it
        assert set(it["task_shift_2x2"]) >= {"task", "shift", "task_x_shift"} and it["task_shift_2x2"]["unit"] == "cohort"
        # OOD rule: midog21 has no test_at.ood -> no records, other cohorts fine
        A.main(["runs", "--v2-root", tmp, "--tier", "C", "--out", out, "--select", "ood", "--B", "200", "--B-tau", "100",
                "--B-perm", "200", "--cohorts", "c17", "midog21"])
        assert os.path.exists(os.path.join(out, "C", "c17", "summary_ood.csv"))
        assert not os.path.exists(os.path.join(out, "C", "midog21", "summary_ood.csv"))
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    raise SystemExit(run_module(globals()))
