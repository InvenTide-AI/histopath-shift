"""Shared helpers for the stats_v2 tests (no pytest needed: each test module has a runner)."""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # src/v2

import stats_v2 as S  # noqa: E402,F401

V1_RUNS = os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "runs")


def run_module(mod_globals):
    """Run every test_* function in a module; print PASS/FAIL; return number of failures."""
    fails = 0
    for name, fn in sorted(mod_globals.items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)")
            except Exception:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    return fails
