"""End-to-end GPU smoke test for the extended pipeline.

Builds a tiny synthetic fold cache (2000 train / 500 each val/test), then runs
every arm in run_arm_ext.py -- baseline, sam, ssl, ssl_sam, groupdro, coral,
irm, mixstyle -- for one epoch each on GPU.  Prints wall-time and off-site
AUROC.  Purpose is not to obtain real results but to confirm the whole
train+eval loop works before spending H100 hours on real Camelyon17 data.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np


def make_synthetic_fold(out, n_train=2000, n_val=500, n_test=500,
                        n_sites=3, seed=0):
    """Two-class 64x64 uint8 patches where site 0/1/2 have different
    per-site brightness offsets so a naive classifier can shortcut on site.
    Test set has a *different* site with a different brightness pattern
    than any training site: exactly the acquisition-shift structure the
    paper's arms are meant to close."""
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)

    def batch(n, sites, class0_offset=0.0, class1_offset=0.15, bright=0.0):
        y = rng.integers(0, 2, n).astype(np.int64)
        base = rng.uniform(0.3, 0.6, size=(n, 1, 1, 1))
        offset = np.where(y[:, None, None, None] == 0, class0_offset, class1_offset)
        img = np.clip(base + offset + bright + 0.05 * rng.standard_normal((n, 64, 64, 3)), 0, 1)
        return (img * 255).astype(np.uint8), y

    Xtr, ytr = batch(n_train, n_sites)
    # each train patch belongs to one of the n_sites; small per-site brightness
    str_ = rng.integers(0, n_sites, n_train).astype(np.int64)
    site_shift = np.array([-0.05, 0.0, +0.05])[str_].reshape(-1, 1, 1, 1)
    Xtr = np.clip(Xtr.astype(np.float32) / 255.0 + site_shift, 0, 1)
    Xtr = (Xtr * 255).astype(np.uint8)

    Xiv, yiv = batch(n_val, n_sites)   # id_val, same sites as train
    Xov, yov = batch(n_val, n_sites, bright=+0.10)  # ood_val, different site
    Xot, yot = batch(n_test, n_sites, bright=+0.20)  # ood_test, a different site again

    np.savez_compressed(os.path.join(out, "splits.npz"),
                        Xtr=Xtr, ytr=ytr, str=str_,
                        Xiv=Xiv, yiv=yiv, Xov=Xov, yov=yov, Xot=Xot, yot=yot)
    # small unlabeled pool for SSL
    Xu, _ = batch(1000, n_sites)
    np.savez_compressed(os.path.join(out, "unlabeled.npz"), Xu=Xu)
    print(f"synthetic fold at {out}: train {Xtr.shape}, "
          f"ood_test {Xot.shape}, unlabeled {Xu.shape}")


def run(cmd, env=None):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.returncode, time.time() - t0, r.stdout, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="./data/smoke_cache")
    ap.add_argument("--out_root", default="./data/smoke_runs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None or True else "cpu")
    ap.add_argument("--arms", default="baseline sam ssl ssl_sam groupdro coral irm mixstyle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.force:
        shutil.rmtree(args.cache_root, ignore_errors=True)
        shutil.rmtree(args.out_root, ignore_errors=True)
    os.makedirs(args.out_root, exist_ok=True)
    data = os.path.join(args.cache_root, "fold_smoke")
    if not os.path.exists(os.path.join(data, "splits.npz")):
        make_synthetic_fold(data, seed=args.seed)

    src = os.path.dirname(os.path.abspath(__file__))
    # SSL pre-train (once)
    ssl_ck = os.path.join(data, f"ssl_encoder_seed{args.seed}.pt")
    if not os.path.exists(ssl_ck):
        print("=== SSL pre-training ===")
        script = "pretrain_ssl_gpu_ext.py" if args.device.startswith("cuda") else "pretrain_ssl.py"
        rc, dt, out, err = run([sys.executable, os.path.join(src, script),
                                "--data", data, "--seed", str(args.seed),
                                "--epochs", "2", "--bs", "64", "--device", args.device])
        print(out[-800:]);
        if rc: print(err[-800:]); return 1
        print(f"SSL pretrain done in {dt:.1f}s")

    results = {}
    for arm in args.arms.split():
        print(f"=== {arm} ===")
        rc, dt, out, err = run([sys.executable, os.path.join(src, "run_arm_ext.py"),
                                "--data", data, "--out", args.out_root,
                                "--arm", arm, "--seed", str(args.seed),
                                "--epochs", str(args.epochs), "--bs", str(args.bs),
                                "--ntrain", "0",
                                "--device", args.device])
        print(out[-500:])
        if rc:
            print("STDERR:", err[-800:]); results[arm] = None; continue
        rj = json.load(open(os.path.join(args.out_root, f"{arm}_seed{args.seed}.json")))
        auroc = rj["ood_test"]["auroc"]; results[arm] = (auroc, dt)
        print(f"{arm}: ood_test auroc={auroc:.4f}  wall={dt:.1f}s")

    print("\n=== summary ===")
    print(f"{'arm':<12}{'auroc':<10}{'wall(s)':<10}")
    for arm, r in results.items():
        if r is None:
            print(f"{arm:<12}FAILED")
        else:
            print(f"{arm:<12}{r[0]:<10.4f}{r[1]:<10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
