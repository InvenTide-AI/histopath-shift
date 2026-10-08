"""Sequence the cross-institution chest-radiograph experiment.

Stage 1: 4 SimCLR pre-trainings (pool S/M x seed 0/1), run 2 at a time.
Stage 2: 12 supervised runs (6 arms x 2 seeds), run 3 at a time.

Thread counts are set so the total stays near the 10 available cores; SAM arms
cost roughly 2x a plain arm, so they are interleaved rather than batched.
Any stage already on disk is skipped, so an interrupted run resumes.
"""
import itertools
import os
import subprocess
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = "cache_cxr", "runs_cxr"
SEEDS = [0, 1]
ARMS = ["baseline", "sam", "ssl_S", "ssl_M", "ssl_S_sam", "ssl_M_sam"]


def sh(cmd, log):
    with open(log, "a") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=SRC)


def run_pool(jobs, width):
    """Run (cmd, logfile, done_marker) jobs at most `width` at a time."""
    pend = list(jobs)
    live = []
    while pend or live:
        while pend and len(live) < width:
            cmd, log, marker = pend.pop(0)
            if marker and os.path.exists(marker):
                print(f"skip (exists): {marker}", flush=True)
                continue
            print("launch:", " ".join(cmd[-6:]), flush=True)
            live.append((sh(cmd, log), cmd))
        time.sleep(10)
        for pr, cmd in list(live):
            if pr.poll() is not None:
                live.remove((pr, cmd))
                print(f"finished rc={pr.returncode}: {' '.join(cmd[-6:])}", flush=True)
                if pr.returncode != 0:
                    print("  !! nonzero exit — see log", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # ---- stage 1: pre-training
    jobs = []
    for pool, sd in itertools.product(["S", "M"], SEEDS):
        marker = f"{SRC}/{CACHE}/ssl_encoder_cxr_{pool}_seed{sd}.pt"
        jobs.append(([sys.executable, "-u", f"{SRC}/pretrain_ssl_cxr.py",
                      "--pool", pool, "--seed", str(sd), "--threads", "5"],
                     f"{SRC}/{OUT}/ssl_{pool}_seed{sd}.log", marker))
    print("=== stage 1: SSL pre-training", flush=True)
    run_pool(jobs, width=2)
    print(f"stage 1 done in {time.time() - t0:.0f}s", flush=True)

    # ---- stage 2: supervised arms
    jobs = []
    for arm, sd in itertools.product(ARMS, SEEDS):
        marker = f"{SRC}/{OUT}/{arm}_seed{sd}.json"
        jobs.append(([sys.executable, "-u", f"{SRC}/run_arm_cxr.py",
                      "--arm", arm, "--seed", str(sd), "--threads", "3"],
                     f"{SRC}/{OUT}/{arm}_seed{sd}.log", marker))
    print("=== stage 2: supervised arms", flush=True)
    run_pool(jobs, width=3)
    print(f"ALL DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
