"""Run the pool-size test on the CRC stain-shift axis (DESIGN_crc_pool40k.md).

All six arms are run on one internally consistent split set: `baseline`, `sam`,
`ssl`/`ssl_sam` at the original 10k pool, and `ssl`/`ssl_sam` at 40k -- seeds 0
and 1 throughout. The published numbers cannot serve as the comparison because
the splits were rebuilt under a corrected rng derivation and the stain-shift
test set is a different tile sample (see DESIGN_crc_pool40k.md). The 10k arms
are the within-splits reference and also independently replicate the paper's
original negative finding.

Sequential by design -- each arm already uses several threads and the machine
has 10 cores; overlapping arms would make the recorded wall times
uninterpretable. Every step is resumable: pretrain skips an existing encoder,
and run_arm_crc.py skips an arm whose {tag}.json already exists.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "runs_crc_pool40k")
SEEDS = [0, 1]
os.makedirs(OUT, exist_ok=True)

# (encoder tag, pool file) for the two pool conditions. The pools are NESTED
# and composition-matched at 21.96% tumour: p10k is a class-stratified subset
# of p20k, so only size varies. See DESIGN_crc_pool40k.md for why the first
# (non-nested) attempt was discarded.
POOLS = [("_p10k", "crc_unlabeled_p10k.npz"),
         ("_p20k", "crc_unlabeled_p20k.npz")]
SPLITS = "crc_splits_n15k.npz"   # train shrunk to 15k to free tumour headroom

# Arms that never read the unlabeled pool -- run once, not once per pool.
POOL_FREE = ["baseline", "sam"]
POOL_DEP = ["ssl", "ssl_sam"]


def run(cmd, label):
    t0 = time.time()
    print(f"\n=== {label}", flush=True)
    p = subprocess.run(cmd, cwd=HERE)
    if p.returncode != 0:
        sys.exit(f"FAILED ({p.returncode}): {label}")
    print(f"=== {label} done in {time.time() - t0:.0f}s", flush=True)


def arm_done(name, seed):
    return os.path.exists(os.path.join(OUT, f"{name}_seed{seed}.json"))


# ---- SSL pre-training: one encoder per (pool, seed) ----
for tag, pool in POOLS:
    for seed in SEEDS:
        enc = os.path.join(DATA, f"ssl_encoder_crc_seed{seed}{tag}.pt")
        if os.path.exists(enc):
            print(f"skip pretrain seed{seed}{tag}: exists",
                  flush=True)
            continue
        run([sys.executable, "pretrain_ssl_crc.py", "--seed", str(seed),
             "--pool", pool, "--tag", tag, "--data", DATA, "--threads", "8"],
            f"pretrain seed{seed} pool={pool}")

# ---- supervised arms, cheapest first so early failures surface fast ----
for arm in POOL_FREE:
    for seed in SEEDS:
        if arm_done(arm, seed):
            print(f"skip {arm} seed{seed}: done", flush=True)
            continue
        run([sys.executable, "run_arm_crc.py", "--arm", arm, "--seed",
             str(seed), "--splits", SPLITS, "--data", DATA, "--out", OUT,
             "--threads", "8"],
            f"{arm} seed{seed}")

for tag, pool in POOLS:
    suffix = tag
    for arm in POOL_DEP:
        for seed in SEEDS:
            name = f"{arm}{suffix}"
            if arm_done(name, seed):
                print(f"skip {name} seed{seed}: done", flush=True)
                continue
            # --arm selects the training recipe; --run-name distinguishes the
            # two pool conditions in the output filenames.
            run([sys.executable, "run_arm_crc.py", "--arm", arm, "--seed",
                 str(seed), "--enc-tag", tag, "--run-name", name,
                 "--splits", SPLITS, "--data", DATA, "--out", OUT,
                 "--threads", "8"],
                f"{name} seed{seed}")

print("\nall six arms complete ->", OUT, flush=True)
