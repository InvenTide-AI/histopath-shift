"""Drive the full NCT-CRC-HE ablation: SSL pre-training then 4 arms x 2 seeds.

Concurrency is deliberately limited. The original Camelyon17 session ran six
jobs at once on the same 10-core/16 GB machine and spent ~16 h per arm; the
per-step cost measured in isolation implies well under an hour, so that run was
memory-thrashing rather than compute-bound. Four concurrent jobs at 2 threads
each keeps the resident set inside RAM.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["baseline", "ssl", "sam", "ssl_sam"]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--data", default="cache")
ap.add_argument("--out", default="runs_crc")
ap.add_argument("--skip-ssl", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def run(cmd, log):
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads),
               MKL_NUM_THREADS=str(a.threads))
    with open(log, "w") as f:
        # -u: without it Python block-buffers stdout when redirected to a file
        # and the logs stay empty for the whole run, hiding progress.
        return subprocess.Popen([sys.executable, "-u"] + cmd, cwd=HERE, env=env,
                                stdout=f, stderr=subprocess.STDOUT)


def wait_all(procs, label):
    bad = 0
    for tag, p in procs:
        rc = p.wait()
        print(f"  {label} {tag} rc={rc}", flush=True)
        bad += rc != 0
    if bad:
        sys.exit(f"{bad} {label} job(s) failed -- see logs in {a.out}")


t0 = time.time()

# ---- Pillar I encoders (one per seed; the SSL arms both consume these) ----
if not a.skip_ssl:
    procs = []
    for s in a.seeds:
        ck = os.path.join(HERE, a.data, f"ssl_encoder_crc_seed{s}.pt")
        if os.path.exists(ck):
            print(f"  ssl seed{s}: encoder exists, skipping", flush=True)
            continue
        procs.append((f"seed{s}", run(
            ["pretrain_ssl_crc.py", "--seed", str(s), "--data", a.data,
             "--threads", str(max(2, 10 // max(len(a.seeds), 1)))],
            os.path.join(a.out, f"ssl_pretrain_seed{s}.log"))))
    wait_all(procs, "ssl")
    print(f"SSL pre-training done ({time.time() - t0:.0f}s)", flush=True)

# ---- 4 arms x N seeds, bounded concurrency ----
todo = [(arm, s) for arm, s in itertools.product(ARMS, a.seeds)
        if not os.path.exists(os.path.join(HERE, a.out, f"{arm}_seed{s}.json"))]
print(f"runs to do: {len(todo)}", flush=True)
running = []
while todo or running:
    while todo and len(running) < a.jobs:
        arm, s = todo.pop(0)
        tag = f"{arm}_seed{s}"
        running.append((tag, run(
            ["run_arm_crc.py", "--arm", arm, "--seed", str(s), "--data", a.data,
             "--out", a.out, "--threads", str(a.threads)],
            os.path.join(a.out, f"{tag}.log"))))
        print(f"  launched {tag}", flush=True)
    time.sleep(5)
    for tag, p in list(running):
        if p.poll() is not None:
            print(f"  finished {tag} rc={p.returncode} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            running.remove((tag, p))

print(f"ALL DONE in {time.time() - t0:.0f}s", flush=True)
