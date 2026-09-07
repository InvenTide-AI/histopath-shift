#!/usr/bin/env python3
"""Run the four arms on every protocol-matched fold, resumably.

Per (fold, seed): pre-train the SSL encoder once (both SSL arms load
ssl_encoder_seed{seed}.pt from the fold's own cache, so no fold ever sees an encoder
pre-trained on its test hospital), then run baseline, sam, ssl, ssl_sam with the
published settings (--ntrain 30000, run_arm.py defaults elsewhere).

Skips any run whose {arm}_seed{seed}.json already exists, so it can be stopped and
restarted freely.  Appends one line of JSON per completed run to progress.jsonl.
"""
import json, os, subprocess, sys, time

CODE = os.environ.get("CODE", "code")
FOLDS = [int(x) for x in os.environ.get("FOLDS", "2 0 1 3 4").split()]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split()]
ARMS = ["baseline", "sam", "ssl", "ssl_sam"]
THREADS = os.environ.get("THREADS", "10")
NTRAIN = os.environ.get("NTRAIN", "30000")

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def sh(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"FAILED: {' '.join(cmd)}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.returncode, time.time() - t0, r.stdout

for t in FOLDS:
    data, out = f"cache/fold{t}", f"runs/fold{t}"
    os.makedirs(out, exist_ok=True)
    for s in SEEDS:
        if not os.path.exists(os.path.join(data, f"ssl_encoder_seed{s}.pt")):
            log(f"fold{t} seed{s}: pre-training SSL encoder")
            rc, el, _ = sh([sys.executable, f"{CODE}/pretrain_ssl.py", "--data", data,
                            "--seed", str(s), "--threads", THREADS])
            log(f"fold{t} seed{s}: pre-training {'ok' if rc == 0 else 'FAILED'} ({el/60:.1f} min)")
            if rc: continue
        else:
            log(f"fold{t} seed{s}: encoder present, skipping pre-training")
        for arm in ARMS:
            f_ = os.path.join(out, f"{arm}_seed{s}.json")
            if os.path.exists(f_):
                log(f"fold{t} {arm} seed{s}: done already"); continue
            log(f"fold{t} {arm} seed{s}: training")
            rc, el, _ = sh([sys.executable, f"{CODE}/run_arm.py", "--data", data, "--out", out,
                            "--arm", arm, "--seed", str(s), "--ntrain", NTRAIN,
                            "--threads", THREADS])
            if rc == 0 and os.path.exists(f_):
                r = json.load(open(f_))
                ot = r.get("ood_test", {}).get("auroc")
                log(f"fold{t} {arm} seed{s}: ood_test_auroc={ot:.4f} "
                    f"(ep {r.get('selected_epoch')}, {el/60:.1f} min)")
                with open("progress.jsonl", "a") as fh:
                    fh.write(json.dumps({"fold": t, "arm": arm, "seed": s,
                                         "ood_test_auroc": ot, "wall_min": round(el/60, 1)}) + "\n")
            else:
                log(f"fold{t} {arm} seed{s}: FAILED")
log("ALL DONE")
