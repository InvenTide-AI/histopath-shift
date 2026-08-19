"""Train one arm x one seed of the Camelyon17 ablation and dump metrics.

Usage: python run_arm.py --arm baseline|ssl|sam|ssl_sam --seed 0
Model selection is on ood_val (hospital 1) per the official WILDS protocol;
ood_test (hospital 2) is touched only once, at the end, with the selected model.
"""
import argparse, json, os, sys, time
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "STOP_SPAWNS")) and os.environ.get("ALLOW_RUN2") != "1":
    print("STOP_SPAWNS present - exiting without training"); raise SystemExit(0)
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True, choices=["baseline", "ssl", "sam", "ssl_sam"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=0.02)
p.add_argument("--rho", type=float, default=0.05)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--data", default="cache")
p.add_argument("--out", default="runs")
p.add_argument("--ntrain", type=int, default=0, help="0 = use all")
a = p.parse_args()

torch.set_num_threads(a.threads)
torch.manual_seed(a.seed); np.random.seed(a.seed)
rng = np.random.RandomState(a.seed)
os.makedirs(a.out, exist_ok=True)
tag = f"{a.arm}_seed{a.seed}"

d = np.load(os.path.join(a.data, os.environ.get("SPLITS_FILE", "splits.npz")))
Xtr, ytr = d["Xtr"], d["ytr"]
if a.ntrain and a.ntrain < len(Xtr):
    # class-stratified subsample, identical across arms for a given seed
    sub = np.concatenate([
        np.random.RandomState(777).permutation(np.where(ytr == c)[0])[:a.ntrain // 2]
        for c in (0, 1)])
    sub.sort()
    Xtr, ytr = Xtr[sub], ytr[sub]
Xiv, yiv = d["Xiv"], d["yiv"]
Xov, yov = d["Xov"], d["yov"]
Xot, yot = d["Xot"], d["yot"]

use_ssl = a.arm in ("ssl", "ssl_sam")
use_sam = a.arm in ("sam", "ssl_sam")

# ---- build model, optionally from the SSL-pretrained encoder ----
enc = T.Encoder()
if use_ssl:
    ck = os.path.join(a.data, f"ssl_encoder_seed{a.seed}.pt")
    enc.load_state_dict(torch.load(ck, map_location="cpu"))
    print(f"[{tag}] loaded SSL encoder {ck}", flush=True)
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
sched = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=a.epochs * steps_per_epoch)

log, best = [], {"ood_val_auroc": -1}
t0 = time.time()
for ep in range(a.epochs):
    model.train()
    tot, nb = 0.0, 0
    for xb, yb in T.iterate(Xtr, ytr, a.bs, shuffle=True, rng=rng):
        x = T.normalize(T.sup_augment(T.to_float(xb)))
        if use_sam:
            l = T.sam_step(model, x, yb, opt, loss_fn)
        else:
            opt.zero_grad(); l_ = loss_fn(model(x), yb); l_.backward(); opt.step()
            l = l_.item()
        sched.step(); tot += l; nb += 1
    # model selection on the OOD validation hospital (center 1)
    pv = T.predict(model, Xov)
    mv = T.compute_metrics(yov, pv)
    piv = T.predict(model, Xiv)
    miv = T.compute_metrics(yiv, piv)
    row = dict(epoch=ep + 1, train_loss=tot / max(nb, 1),
               ood_val_auroc=mv["auroc"], ood_val_acc=mv["accuracy"],
               id_val_auroc=miv["auroc"], id_val_acc=miv["accuracy"],
               lr=base.param_groups[0]["lr"], elapsed=time.time() - t0)
    log.append(row)
    print(f"[{tag}] ep{ep+1}/{a.epochs} loss={row['train_loss']:.4f} "
          f"id_val_auroc={miv['auroc']:.4f} ood_val_auroc={mv['auroc']:.4f} "
          f"({row['elapsed']:.0f}s)", flush=True)
    if mv["auroc"] > best["ood_val_auroc"]:
        best = dict(ood_val_auroc=mv["auroc"], epoch=ep + 1,
                    state={k: v.clone() for k, v in model.state_dict().items()})

# ---- final evaluation with the selected checkpoint ----
model.load_state_dict(best["state"])
torch.save(model.state_dict(), os.path.join(a.out, f"{tag}.pt"))

p_ot = T.predict(model, Xot)
p_iv = T.predict(model, Xiv)
p_ov = T.predict(model, Xov)
res = dict(arm=a.arm, seed=a.seed, selected_epoch=best["epoch"],
           epochs=a.epochs, lr=a.lr, bs=a.bs,
           rho=(a.rho if use_sam else None), use_ssl=use_ssl, use_sam=use_sam,
           n_train=int(len(Xtr)), wall_s=time.time() - t0,
           ood_test={f"{k}": v for k, v in T.compute_metrics(yot, p_ot).items()},
           id_val={f"{k}": v for k, v in T.compute_metrics(yiv, p_iv).items()},
           ood_val={f"{k}": v for k, v in T.compute_metrics(yov, p_ov).items()},
           log=log)
json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(a.out, f"{tag}_preds.npz"),
                    p_ood_test=p_ot, y_ood_test=yot,
                    p_id_val=p_iv, y_id_val=yiv)
print(f"[{tag}] DONE ood_test auroc={res['ood_test']['auroc']:.4f} "
      f"acc={res['ood_test']['accuracy']:.4f} f1={res['ood_test']['f1']:.4f} "
      f"ap={res['ood_test']['avg_precision']:.4f}", flush=True)
