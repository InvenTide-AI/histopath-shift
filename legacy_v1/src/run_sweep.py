"""Drive the leave-one-hospital-out x mechanism sweep, resumably.

Design constraints this encodes
-------------------------------
* The interaction decomposition needs a COMPLETE 2x2 per (fold, seed, pair):
  {none, A, B, A+B}. A partially-run cell yields no decomposition at all, so
  arms are ordered to close 2x2 blocks rather than to finish one mechanism
  across all folds. An interrupted sweep therefore still analyses.
* Every arm is skipped if its record already exists, so re-invocation is free
  and interruption costs only the arm in flight.
* Shared singletons: `baseline` and each single mechanism appear in several
  pairs' 2x2 blocks. They are run ONCE per (fold, seed) and reused, which is
  what makes the full design affordable -- 8 mechanisms pairwise would be 16
  cells x 4 arms = 64 runs per fold-seed if run naively, versus 15 distinct arms.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import mechanisms as M


def arms_for_design(pairs):
    """Distinct arms needed to complete every 2x2 block, in an order that closes
    blocks early: baseline, then each pair's singletons, then the combination."""
    need, seen = [], set()

    def add(a):
        if a not in seen:
            seen.add(a); need.append(a)

    add("baseline")
    for a, b in pairs:
        add(a); add(b); add(f"{a}+{b}")
    return need


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="all",
                   help="comma-separated centers, or 'all'")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--pairs", default="factorial",
                   help="'factorial' (all cross-class pairs), 'within' (adds "
                        "same-class pairs as the specificity control), or "
                        "explicit 'simclr+sam,mae+swa'")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="runs_gpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ssl-encoder-dir", default="cache",
                   help="expects ssl_encoder_<arch>_hold<fold>_seed<seed>.pt")
    p.add_argument("--dry-run", action="store_true", help="print the plan only")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    import numpy as np
    sp = np.load(os.path.join(a.cache, "hospital_splits.npz"))
    folds = sorted({k.split("__")[0] for k in sp.files})
    if a.folds != "all":
        want = a.folds.split(",")
        missing = [f for f in want if f not in folds]
        assert not missing, f"unknown folds {missing}; available: {folds}"
        folds = want
    seeds = [int(s) for s in a.seeds.split(",")]

    if a.pairs == "factorial":
        pairs = M.factorial_pairs()
    elif a.pairs == "within":
        pairs = M.factorial_pairs()
        pairs += [tuple(sorted(c)) for c in itertools.combinations(M.CONDITION_I, 2)]
        pairs += [tuple(sorted(c)) for c in itertools.combinations(M.CONDITION_II, 2)]
    else:
        pairs = []
        for tok in a.pairs.split(","):
            ms = sorted(M.parse_mechanisms(tok))
            assert len(ms) == 2, f"--pairs entry {tok!r} must name exactly 2 mechanisms"
            pairs.append(tuple(ms))
    arms = arms_for_design(pairs)

    todo = []
    for fold in folds:
        for seed in seeds:
            for arm in arms:
                tag = f"{M.arm_name(M.parse_mechanisms(arm))}__{fold}__seed{seed}"
                if os.path.exists(os.path.join(a.out, tag + ".json")):
                    continue
                todo.append((fold, seed, arm))

    print(f"folds={folds} seeds={seeds}")
    print(f"pairs ({len(pairs)}): {['+'.join(p) for p in pairs]}")
    print(f"distinct arms per fold-seed ({len(arms)}): {arms}")
    print(f"runs outstanding: {len(todo)} of {len(folds)*len(seeds)*len(arms)}")
    if a.dry_run:
        for t in todo[:20]:
            print("  would run:", t)
        if len(todo) > 20:
            print(f"  ... and {len(todo)-20} more")
        return

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for i, (fold, seed, arm) in enumerate(todo, 1):
        enc = os.path.join(a.ssl_encoder_dir,
                           f"ssl_encoder_{a.arch}_hold{fold}_seed{seed}.pt")
        cmd = [sys.executable, "train_gpu.py", "--holdout", fold,
               "--mechanisms", arm, "--seed", str(seed), "--arch", a.arch,
               "--size", str(a.size), "--epochs", str(a.epochs),
               "--bs", str(a.bs), "--lr", str(a.lr), "--cache", a.cache,
               "--out", a.out, "--workers", str(a.workers)]
        if "simclr" in arm:
            cmd += ["--ssl-encoder", enc]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        print(f"\n=== [{i}/{len(todo)}] {arm} hold={fold} seed={seed} "
              f"(elapsed {(time.time()-t0)/3600:.2f} h)", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            # Do not abort the sweep: one arm failing (e.g. a missing SSL
            # encoder) should not cost the remaining runs. The gap is visible
            # because the record is absent, and re-running picks it up.
            print(f"!!! arm failed (exit {r.returncode}); continuing", flush=True)
    print(f"\nsweep done in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
