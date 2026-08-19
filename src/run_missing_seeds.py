"""Complete the Camelyon17 4x2 design: run `sam` and `ssl_sam` at seed 1.

The published runs cover baseline/ssl at seeds 0,1 and sam/ssl_sam at seed 0
only, so the four-arm interaction is measurable at seed 0 alone. This script
fills the two gaps using hyperparameters read back from the published run
records (epochs=5, bs=128, lr=0.02, rho=0.05, ntrain=30000) so the new runs are
directly comparable.

Prerequisite: cache/splits.npz and cache/unlabeled.npz rebuilt by
prepare_data.py. The rebuild is verified sample-identical to the published runs
(same seed 20240, same shard set, re-derived selection indices reproduce the
cached label vectors exactly).

Skip-if-exists: both the SSL encoder and each arm's output JSON are checked, so
an interrupted run resumes without repeating finished work.
"""
import os, subprocess, sys, time

SRC = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SRC, "cache")
OUT = os.path.join(os.path.dirname(SRC), "runs_camelyon17")
SEED = 1
EPOCHS, BS, LR, RHO, NTRAIN, THREADS = 5, 128, 0.02, 0.05, 30000, 8
os.makedirs(OUT, exist_ok=True)


def run(cmd, label):
    t0 = time.time()
    print(f"\n=== {label}\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=SRC)
    if p.returncode != 0:
        print(f"FAILED {label} rc={p.returncode}", flush=True)
        sys.exit(p.returncode)
    print(f"=== done {label} in {(time.time()-t0)/3600:.2f} h", flush=True)


# ssl_sam needs the seed-1 SimCLR encoder; sam does not touch it.
enc = os.path.join(DATA, f"ssl_encoder_seed{SEED}.pt")
if os.path.exists(enc):
    print(f"skip pretrain seed{SEED}: {enc} exists", flush=True)
else:
    run([sys.executable, "-u", "pretrain_ssl.py", "--seed", str(SEED),
         "--data", DATA, "--threads", str(THREADS)], f"simclr pretrain seed{SEED}")

for arm in ("sam", "ssl_sam"):
    dst = os.path.join(OUT, f"{arm}_seed{SEED}.json")
    if os.path.exists(dst):
        print(f"skip {arm} seed{SEED}: {dst} exists", flush=True)
        continue
    run([sys.executable, "-u", "run_arm.py", "--arm", arm, "--seed", str(SEED),
         "--epochs", str(EPOCHS), "--bs", str(BS), "--lr", str(LR),
         "--rho", str(RHO), "--ntrain", str(NTRAIN),
         "--data", DATA, "--out", OUT, "--threads", str(THREADS)],
        f"{arm} seed{SEED}")

print("\nboth missing seed-1 arms complete ->", OUT, flush=True)
