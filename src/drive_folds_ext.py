#!/usr/bin/env python3
"""Drive the extended site-complete factorial: 10 arms x 5 held-out hospitals x N seeds.

Resumable: skips any (fold, arm, seed) whose results JSON already exists.
Emits one line of JSON per completed run to `progress.jsonl`, so a long sweep
can be monitored with `tail -F progress.jsonl` and analysed while others still
run.  Meant to be invoked by SLURM array tasks -- see slurm_run_sweep.sh.
"""
import argparse, json, os, subprocess, sys, time

CODE = os.environ.get("CODE", os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ARMS = ["baseline", "sam", "ssl", "ssl_sam",
                "groupdro", "coral", "irm", "mixstyle"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sh(cmd, env=None):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        log(f"FAILED: {' '.join(cmd)}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.returncode, time.time() - t0, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=os.environ.get("FOLDS", "0 1 2 3 4"))
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "0 1 2"))
    ap.add_argument("--arms", default=" ".join(DEFAULT_ARMS))
    ap.add_argument("--ntrain", type=int, default=int(os.environ.get("NTRAIN", "30000")))
    ap.add_argument("--cache_root", default=os.environ.get("CACHE_ROOT", "cache"))
    ap.add_argument("--out_root", default=os.environ.get("OUT_ROOT", "runs"))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "8")))
    ap.add_argument("--bs", type=int, default=int(os.environ.get("BS", "128")))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "8")))
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    ap.add_argument("--progress", default="progress.jsonl")
    a = ap.parse_args()

    folds = [int(x) for x in a.folds.split()]
    seeds = [int(x) for x in a.seeds.split()]
    arms = a.arms.split()
    log(f"folds={folds} seeds={seeds} arms={arms}")

    for t in folds:
        data = os.path.join(a.cache_root, f"fold{t}")
        out = os.path.join(a.out_root, f"fold{t}")
        os.makedirs(out, exist_ok=True)
        for s in seeds:
            ssl_ck = os.path.join(data, f"ssl_encoder_seed{s}.pt")
            if any(arm in ("ssl", "ssl_sam") for arm in arms) and not os.path.exists(ssl_ck):
                log(f"fold{t} seed{s}: pre-training SSL encoder")
                ssl_script = "pretrain_ssl_gpu_ext.py" if a.device.startswith("cuda") else "pretrain_ssl.py"
                rc, el, _ = sh([sys.executable, os.path.join(CODE, ssl_script),
                                "--data", data, "--seed", str(s),
                                "--threads", str(a.threads), "--device", a.device])
                log(f"fold{t} seed{s}: SSL {'ok' if rc == 0 else 'FAILED'} ({el/60:.1f} min)")
                if rc: continue
            for arm in arms:
                f_ = os.path.join(out, f"{arm}_seed{s}.json")
                if os.path.exists(f_):
                    log(f"fold{t} {arm} seed{s}: done already"); continue
                log(f"fold{t} {arm} seed{s}: training")
                rc, el, _ = sh([sys.executable, os.path.join(CODE, "run_arm_ext.py"),
                                "--data", data, "--out", out, "--arm", arm, "--seed", str(s),
                                "--ntrain", str(a.ntrain), "--epochs", str(a.epochs),
                                "--bs", str(a.bs), "--threads", str(a.threads),
                                "--device", a.device])
                if rc == 0 and os.path.exists(f_):
                    r = json.load(open(f_))
                    ot = r.get("ood_test", {}).get("auroc")
                    log(f"fold{t} {arm} seed{s}: ood_test_auroc={ot:.4f} "
                        f"(ep {r.get('selected_epoch')}, {el/60:.1f} min)")
                    with open(a.progress, "a") as fh:
                        fh.write(json.dumps({"fold": t, "arm": arm, "seed": s,
                                             "ood_test_auroc": ot,
                                             "wall_min": round(el/60, 1)}) + "\n")
    log("ALL DONE")


if __name__ == "__main__":
    main()
