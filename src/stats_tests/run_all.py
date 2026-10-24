"""Run all stats_v2 tests.  Usage: python run_all.py [--with-coverage [n_sim]]
(the coverage simulation takes ~3-7 min on 8 CPUs for n_sim=2000)."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import run_module  # noqa: E402

mods = ["test_oneway", "test_contrasts_power", "test_v1_regression", "test_cli_synthetic", "test_review_fixes",
        "test_site_acceptance"]
if "--with-coverage" in sys.argv:
    i = sys.argv.index("--with-coverage")
    n = sys.argv[i + 1] if len(sys.argv) > i + 1 else "2000"
    sys.argv = [sys.argv[0], n]
    mods.append("test_coverage")
fails = 0
for m in mods:
    print(f"== {m}")
    fails += run_module(vars(importlib.import_module(m)))
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
raise SystemExit(fails)
