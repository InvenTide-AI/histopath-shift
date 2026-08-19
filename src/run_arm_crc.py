"""Train one arm x one seed of the NCT-CRC-HE replication and dump metrics.

Usage: python run_arm_crc.py --arm baseline|ssl|sam|ssl_sam --seed 0

Differs from run_arm.py (Camelyon17) in exactly two ways, both forced by the
dataset and both documented in DESIGN_crc.md:
  1. Model selection is on IN-DISTRIBUTION validation, because NCT-CRC-HE has
     no third site to hold out as an OOD validation cohort.
  2. Two shift test sets are scored instead of one: NONORM (stain/acquisition
     shift, primary) and CRC_VAL_HE_7K (patient-disjoint external, secondary).
Everything else -- encoder, augmentation, optimiser, schedule, hyperparameters --
is byte-identical to the Camelyon17 run so the datasets are comparable.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True,
               choices=["baseline", "ssl", "sam", "ssl_sam"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--rho", type=float, default=0.05)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--data", default="cache")
p.add_argument("--out", default="runs_crc")
# Pool-size test (DESIGN_crc_pool40k.md). Default "" reproduces the published
# runs: encoder ssl_encoder_crc_seed{seed}.pt.
p.add_argument("--enc-tag", default="",
               help="suffix of the SSL encoder to load, e.g. _pool40k")
p.add_argument("--splits", default="crc_splits.npz",
               help="labeled splits file; default reproduces published runs")
p.add_argument("--run-name", default=None,
               help="output basename; defaults to --arm (published behaviour). "
                    "Lets two pool conditions of the same arm coexist.")
a = p.parse_args()

torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
rng = np.random.RandomState(a.seed)
os.makedirs(a.out, exist_ok=True)
tag = f"{a.run_name or a.arm}_seed{a.seed}"

# ---- idempotent claim -------------------------------------------------------
# Makes the run safe to launch twice (interrupted driver, resumed session, two
# drivers running at once): an arm already finished is skipped, and an arm
# another process is training is left alone rather than racing it to the same
# output files. O_EXCL makes the claim atomic.
_done = os.path.join(a.out, f"{tag}.json")
_lock = os.path.join(a.out, f"{tag}.lock")
if os.path.exists(_done):
    print(f"[{tag}] already complete, skipping", flush=True)
    sys.exit(0)
try:
    _fd = os.open(_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(_fd, str(os.getpid()).encode())
    os.close(_fd)
except FileExistsError:
    print(f"[{tag}] claimed by another process ({_lock}), skipping", flush=True)
    sys.exit(0)
import atexit
atexit.register(lambda: os.path.exists(_lock) and os.remove(_lock))

def load(k):
    """Prefer the memory-mapped .npy copy in cache/mm.

    Concurrent arms otherwise each decompress a private ~370 MB copy of the
    splits, which exceeds RAM on a 16 GB machine and makes the run thrash.
    Memory-mapping lets every process share one page-cache copy.
    """
    mm = os.path.join(a.data, "mm", f"{k}.npy")
    if os.path.exists(mm):
        return np.load(mm, mmap_mode="r")
    return np.load(os.path.join(a.data, a.splits))[k]


Xtr, ytr = load("Xtr"), np.asarray(load("ytr"))
Xiv, yiv = load("Xiv"), np.asarray(load("yiv"))
Xsh, ysh = load("Xsh"), np.asarray(load("ysh"))   # primary: stain shift (NONORM)
Xex, yex = load("Xex"), np.asarray(load("yex"))   # secondary: external cohort

use_ssl = a.arm in ("ssl", "ssl_sam")
use_sam = a.arm in ("sam", "ssl_sam")

enc = T.Encoder()
ssl_meta = {}
if use_ssl:
    ck = os.path.join(a.data, f"ssl_encoder_crc_seed{a.seed}{a.enc_tag}.pt")
    enc.load_state_dict(torch.load(ck, map_location="cpu"))
    # Record which pool the encoder came from, so the pool conditions are
    # distinguishable from the result files alone.
    _hist = os.path.join(a.data,
                         f"ssl_history_crc_seed{a.seed}{a.enc_tag}.json")
    if os.path.exists(_hist):
        _h = json.load(open(_hist))
        ssl_meta = {k: _h.get(k) for k in ("pool", "n_unlabeled")}
    print(f"[{tag}] loaded SSL encoder {ck} "
          f"(pool={ssl_meta.get('pool')}, n={ssl_meta.get('n_unlabeled')})",
          flush=True)
model = T.Classifier(enc)

loss_fn = nn.CrossEntropyLoss()
if use_sam:
    opt = T.SAM(model.parameters(), torch.optim.SGD, rho=a.rho, lr=a.lr,
                momentum=0.9, weight_decay=1e-4)
    base = opt.base_optimizer
else:
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9,
                          weight_decay=1e-4)
    base = opt
steps_per_epoch = len(Xtr) // a.bs
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    base, T_max=a.epochs * steps_per_epoch)

log, best = [], {"id_val_auroc": -1}
t0 = time.time()
for ep in range(a.epochs):
    model.train()
    tot, nb = 0.0, 0
    for xb, yb in T.iterate(Xtr, ytr, a.bs, shuffle=True, rng=rng):
        x = T.normalize(T.sup_augment(T.to_float(xb)))
        if use_sam:
            l = T.sam_step(model, x, yb, opt, loss_fn)
        else:
            opt.zero_grad()
            l_ = loss_fn(model(x), yb)
            l_.backward()
            opt.step()
            l = l_.item()
        sched.step()
        tot += l
        nb += 1
    # selection on IN-DISTRIBUTION validation (no shifted data involved)
    miv = T.compute_metrics(yiv, T.predict(model, Xiv))
    row = dict(epoch=ep + 1, train_loss=tot / max(nb, 1),
               id_val_auroc=miv["auroc"], id_val_acc=miv["accuracy"],
               lr=base.param_groups[0]["lr"], elapsed=time.time() - t0)
    log.append(row)
    print(f"[{tag}] ep{ep + 1}/{a.epochs} loss={row['train_loss']:.4f} "
          f"id_val_auroc={miv['auroc']:.4f} ({row['elapsed']:.0f}s)", flush=True)
    if miv["auroc"] > best["id_val_auroc"]:
        best = dict(id_val_auroc=miv["auroc"], epoch=ep + 1,
                    state={k: v.clone() for k, v in model.state_dict().items()})

# ---- final evaluation: both shift axes scored ONCE with the selected model ----
model.load_state_dict(best["state"])
torch.save(model.state_dict(), os.path.join(a.out, f"{tag}.pt"))

p_iv = T.predict(model, Xiv)
p_sh = T.predict(model, Xsh)
p_ex = T.predict(model, Xex)
res = dict(arm=a.arm, run_name=(a.run_name or a.arm),
           enc_tag=a.enc_tag,
           ssl_encoder=(os.path.basename(ck) if use_ssl else None),
           n_unlabeled=(ssl_meta.get("n_unlabeled") if use_ssl else None),
           ssl_pool=(ssl_meta.get("pool") if use_ssl else None),
           seed=a.seed, dataset="NCT-CRC-HE",
           selected_epoch=best["epoch"], epochs=a.epochs, lr=a.lr, bs=a.bs,
           rho=(a.rho if use_sam else None), use_ssl=use_ssl, use_sam=use_sam,
           n_train=int(len(Xtr)), wall_s=time.time() - t0,
           selection="id_val (no OOD-val cohort exists in this dataset)",
           id_val=T.compute_metrics(yiv, p_iv),
           shift_nonorm=T.compute_metrics(ysh, p_sh),
           shift_external=T.compute_metrics(yex, p_ex),
           log=log)
json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(a.out, f"{tag}_preds.npz"),
                    p_shift_nonorm=p_sh, y_shift_nonorm=ysh,
                    p_shift_external=p_ex, y_shift_external=yex,
                    p_id_val=p_iv, y_id_val=yiv)
print(f"[{tag}] DONE id_val={res['id_val']['auroc']:.4f} "
      f"nonorm={res['shift_nonorm']['auroc']:.4f} "
      f"external={res['shift_external']['auroc']:.4f}", flush=True)
