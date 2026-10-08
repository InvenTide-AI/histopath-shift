"""Train one extended arm x one seed and dump metrics.

Supersedes run_arm.py by adding the domain-generalization objectives introduced
in src/methods.py: GroupDRO, DeepCORAL, IRM, MixStyle, Macenko stain-normalized
ERM, and a frozen-backbone linear probe.  The four paper arms (baseline, sam,
ssl, ssl_sam) also route through here so the whole 10-arm sweep is a single
executable.  Same fold cache format, same evaluation protocol, same output
schema --- so `analyze_sweep.py` and `check_release.py` continue to work on
the concatenated `results_folds.csv`.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib as T
import methods as M


_NORM = {}


def normalize_dev(x):
    """Device-aware ImageNet normalization; caches mean/std per device."""
    dev = x.device
    if dev not in _NORM:
        _NORM[dev] = (T.IMAGENET_MEAN.to(dev), T.IMAGENET_STD.to(dev))
    m, s = _NORM[dev]
    return (x - m) / s


def sup_augment_dev(x):
    """train_lib.sup_augment is device-agnostic (only flip / rot90). Reused
    verbatim; wrapper kept for symmetry with normalize_dev."""
    return T.sup_augment(x)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   choices=["baseline", "sam", "ssl", "ssl_sam",
                            "groupdro", "coral", "irm", "mixstyle",
                            "macenko", "probe_uni",
                            "fish", "lisa", "erm_henorm",
                            "rotinv", "tia_style"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--coral_lambda", type=float, default=1.0)
    p.add_argument("--irm_lambda", type=float, default=1.0)
    p.add_argument("--irm_anneal_epoch", type=int, default=2)
    p.add_argument("--dro_eta", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default=None,
                   help="cpu, cuda, cuda:0, ... (default: cuda if available)")
    p.add_argument("--data", default="cache")
    p.add_argument("--out", default="runs")
    p.add_argument("--ntrain", type=int, default=0)
    p.add_argument("--backbone_dir", default=None,
                   help="Directory holding a frozen foundation-model checkpoint "
                        "(for probe_uni). Must expose backbone.pt + feature_dim.json.")
    return p.parse_args()


def batch_iter(X, y, s, bs, rng):
    n = len(X)
    idx = np.arange(n); rng.shuffle(idx)
    for k in range(0, n - bs + 1, bs):
        b = idx[k:k + bs]
        xb = torch.from_numpy(X[b])
        yb = torch.from_numpy(y[b])
        sb = None if s is None else torch.from_numpy(s[b]).long()
        yield xb, yb, sb


def main():
    a = parse()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.RandomState(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.arm}_seed{a.seed}"

    d = np.load(os.path.join(a.data, os.environ.get("SPLITS_FILE", "splits.npz")))
    Xtr, ytr = d["Xtr"], d["ytr"]
    Xiv, yiv = d["Xiv"], d["yiv"]
    Xov, yov = d["Xov"], d["yov"]
    Xot, yot = d["Xot"], d["yot"]
    # Site label per training example (required by GroupDRO / CORAL / IRM).
    str_ = d["str"] if "str" in d.files else None
    if a.ntrain and a.ntrain < len(Xtr):
        sub = np.concatenate([
            np.random.RandomState(777).permutation(np.where(ytr == c)[0])[:a.ntrain // 2]
            for c in (0, 1)])
        sub.sort()
        Xtr, ytr = Xtr[sub], ytr[sub]
        if str_ is not None:
            str_ = str_[sub]

    use_ssl = a.arm in ("ssl", "ssl_sam")
    use_sam = a.arm in ("sam", "ssl_sam")

    # ---------------- build encoder / classifier ----------------
    if a.arm == "probe_uni":
        if not a.backbone_dir:
            raise SystemExit("--backbone_dir required for probe_uni")
        feat_dim = json.load(open(os.path.join(a.backbone_dir, "feature_dim.json")))["dim"]
        backbone = torch.load(os.path.join(a.backbone_dir, "backbone.pt"), map_location=dev)
        for p in backbone.parameters(): p.requires_grad_(False)
        head = M.LinearProbe(feat_dim).to(dev)
        # Pre-compute features once; a linear probe does not need augmentation.
        with torch.no_grad():
            def emb(X):
                out = []
                for k in range(0, len(X), 512):
                    xb = torch.from_numpy(X[k:k+512]).to(dev)
                    out.append(backbone(normalize_dev(T.to_float(xb).to(dev))).cpu())
                return torch.cat(out, 0)
            F_tr = emb(Xtr); F_ov = emb(Xov); F_iv = emb(Xiv); F_ot = emb(Xot)
        opt = torch.optim.SGD(head.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        best = {"ood_val_auroc": -1}
        t0 = time.time()
        for ep in range(a.epochs):
            head.train()
            idx = np.arange(len(F_tr)); rng.shuffle(idx)
            for k in range(0, len(idx) - a.bs + 1, a.bs):
                b = idx[k:k+a.bs]
                xb = F_tr[b].to(dev); yb = torch.from_numpy(ytr[b]).to(dev)
                opt.zero_grad(); l = loss_fn(head(xb), yb); l.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                pv = torch.softmax(head(F_ov.to(dev)), 1)[:, 1].cpu().numpy()
            m = T.compute_metrics(yov, pv)
            if m["auroc"] > best["ood_val_auroc"]:
                best = {"ood_val_auroc": m["auroc"], "epoch": ep+1,
                        "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        head.load_state_dict(best["state"]); head.eval()
        with torch.no_grad():
            p_ot = torch.softmax(head(F_ot.to(dev)), 1)[:, 1].cpu().numpy()
            p_iv = torch.softmax(head(F_iv.to(dev)), 1)[:, 1].cpu().numpy()
            p_ov = torch.softmax(head(F_ov.to(dev)), 1)[:, 1].cpu().numpy()
        res = dict(arm=a.arm, seed=a.seed, selected_epoch=best["epoch"],
                   epochs=a.epochs, lr=a.lr, bs=a.bs, wall_s=time.time()-t0,
                   ood_test=T.compute_metrics(yot, p_ot),
                   id_val=T.compute_metrics(yiv, p_iv),
                   ood_val=T.compute_metrics(yov, p_ov))
        json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
        print(f"[{tag}] DONE ood_test auroc={res['ood_test']['auroc']:.4f}", flush=True)
        return

    # Macenko: pre-normalize every tile once against a training-set reference.
    # After this the arm trains exactly as ERM does; the intervention is
    # entirely in the input space.
    # tia_style reuses the Macenko input normalization: the MIDOG-winning
    # pipeline stain-normalizes before its classification stage, and adds the
    # colour and dihedral augmentation applied inside the training loop below.
    if a.arm in ("macenko", "tia_style"):
        print(f"[{tag}] fitting Macenko normalizer...", flush=True)
        ref_idx = np.random.RandomState(a.seed).choice(len(Xtr), size=min(64, len(Xtr)), replace=False)
        ref = Xtr[ref_idx].reshape(-1, 3).copy()
        # Fit on a synthesized "reference tile" made from concatenated samples.
        H = int(np.sqrt(ref.shape[0]))
        ref_tile = ref[:H*H].reshape(H, H, 3)
        norm = M.MacenkoNormalizer().fit(ref_tile)
        def apply_norm(arr):
            out = np.empty_like(arr)
            for i in range(len(arr)):
                try:
                    out[i] = norm.transform(arr[i])
                except Exception:
                    out[i] = arr[i]
            return out
        Xtr = apply_norm(Xtr); Xiv = apply_norm(Xiv)
        Xov = apply_norm(Xov); Xot = apply_norm(Xot)
        print(f"[{tag}] applied Macenko to {len(Xtr)+len(Xiv)+len(Xov)+len(Xot)} tiles", flush=True)

    enc = T.Encoder()
    if a.arm == "mixstyle":
        # Inject a MixStyle layer after the second stage.
        ms = M.MixStyle(p=0.5, alpha=0.1)
        orig_features = enc.features
        def forward(x, _f=orig_features, _ms=ms):
            i = 0
            for layer in _f:
                x = layer(x)
                i += 1
                if i == 5:  # after the second-stage MaxPool
                    x = _ms(x)
            return x
        enc.forward = forward  # type: ignore
    if use_ssl:
        ck = os.path.join(a.data, f"ssl_encoder_seed{a.seed}.pt")
        enc.load_state_dict(torch.load(ck, map_location=dev))
    model = T.Classifier(enc).to(dev)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    if use_sam:
        opt = T.SAM(model.parameters(), torch.optim.SGD, rho=a.rho, lr=a.lr,
                    momentum=0.9, weight_decay=1e-4)
        base = opt.base_optimizer
    else:
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
        base = opt
    steps_per_epoch = len(Xtr) // a.bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=a.epochs * steps_per_epoch)

    dro = M.GroupDROState(n_groups=int(str_.max()+1) if str_ is not None else 1,
                          eta=a.dro_eta) if a.arm == "groupdro" else None

    log, best = [], {"ood_val_auroc": -1}
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb, sb in batch_iter(Xtr, ytr, str_, a.bs, rng):
            x = normalize_dev(sup_augment_dev(T.to_float(xb).to(dev)))
            y = yb.to(dev); s = None if sb is None else sb.to(dev)
            if a.arm == "coral":
                # Need features and logits both -- split classifier for one step.
                feats = model.encoder(x)
                logits = model.fc(feats)
                per = loss_fn(logits, y)
                pen = M.coral_penalty(feats, s if s is not None else torch.zeros_like(y))
                loss = per.mean() + a.coral_lambda * pen
                opt.zero_grad(); loss.backward(); opt.step()
                l = loss.item()
            elif a.arm == "irm":
                logits = model(x)
                per = loss_fn(logits, y)
                pen = M.irm_penalty(logits, y, s if s is not None else torch.zeros_like(y))
                lam = a.irm_lambda if ep >= a.irm_anneal_epoch else 1.0
                loss = per.mean() + lam * pen
                opt.zero_grad(); loss.backward(); opt.step()
                l = loss.item()
            elif a.arm == "groupdro":
                logits = model(x)
                per = loss_fn(logits, y)
                loss = dro.loss(per, s if s is not None else torch.zeros_like(y))
                opt.zero_grad(); loss.backward(); opt.step()
                l = loss.item()
            elif a.arm == "fish":
                # Split the batch by hospital, run Fish's meta-step.
                if s is None:
                    logits = model(x); loss = loss_fn(logits, y).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    l = loss.item()
                else:
                    groups = []
                    for k in torch.unique(s):
                        m = (s == k)
                        if m.sum() >= 2:
                            groups.append((x[m], y[m]))
                    if len(groups) >= 2:
                        l = M.fish_meta_step(model, opt, groups, loss_fn, meta_lr=0.1)
                    else:
                        logits = model(x); loss = loss_fn(logits, y).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                        l = loss.item()
            elif a.arm == "lisa":
                # LISA: same-class inter-domain mix-up.
                if s is not None:
                    x_mix, y_mix = M.lisa_batch(x, y, s, alpha=2.0)
                    logits = model(x_mix)
                    loss = loss_fn(logits, y_mix).mean()
                else:
                    logits = model(x); loss = loss_fn(logits, y).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                l = loss.item()
            elif a.arm == "rotinv":
                # Lafarge & Koelzer (MIDOG 2021): orientation invariance.
                logits = model(M.dihedral(x))
                loss = loss_fn(logits, y).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                l = loss.item()
            elif a.arm == "tia_style":
                # Jahanifar et al. (MIDOG 2021/22 winner), classification
                # stage: inputs are already Macenko-normalized above; here we
                # add the heavy colour jitter and orientation augmentation the
                # entry relied on.
                mean_d, std_d = _NORM.get(
                    x.device, (T.IMAGENET_MEAN.to(x.device), T.IMAGENET_STD.to(x.device)))
                _NORM[x.device] = (mean_d, std_d)
                x_raw = (x * std_d + mean_d).clamp(0, 1)
                x_jit = M.he_jitter(x_raw, strength=0.3)
                x_in = M.dihedral((x_jit - mean_d) / std_d)
                logits = model(x_in)
                loss = loss_fn(logits, y).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                l = loss.item()
            elif a.arm == "erm_henorm":
                # Aggressive H&E jitter applied in the pre-normalized [0,1]
                # space, then normalize. `x` reaching here is already
                # normalized; unnormalize first.
                mean_d, std_d = _NORM.get(x.device, (T.IMAGENET_MEAN.to(x.device), T.IMAGENET_STD.to(x.device)))
                _NORM[x.device] = (mean_d, std_d)
                x_raw = (x * std_d + mean_d).clamp(0, 1)
                x_jit = M.he_jitter(x_raw, strength=0.4)
                x_in = (x_jit - mean_d) / std_d
                logits = model(x_in)
                loss = loss_fn(logits, y).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                l = loss.item()
            elif use_sam:
                # sam_step expects loss_fn(logits, y); wrap our per-example
                # loss_fn to return the mean.
                def _l(logits, target):
                    return loss_fn(logits, target).mean()
                l = T.sam_step(model, x, y, opt, _l)
            else:
                opt.zero_grad()
                loss = loss_fn(model(x), y).mean()
                loss.backward(); opt.step()
                l = loss.item()
            sched.step(); tot += l; nb += 1
        pv = predict(model, Xov, dev)
        mv = T.compute_metrics(yov, pv)
        piv = predict(model, Xiv, dev)
        miv = T.compute_metrics(yiv, piv)
        row = dict(epoch=ep+1, train_loss=tot/max(nb,1),
                   ood_val_auroc=mv["auroc"], id_val_auroc=miv["auroc"],
                   lr=base.param_groups[0]["lr"], elapsed=time.time()-t0)
        log.append(row)
        print(f"[{tag}] ep{ep+1}/{a.epochs} loss={row['train_loss']:.4f} "
              f"id_val_auroc={miv['auroc']:.4f} ood_val_auroc={mv['auroc']:.4f} "
              f"({row['elapsed']:.0f}s)", flush=True)
        if mv["auroc"] > best["ood_val_auroc"]:
            best = dict(ood_val_auroc=mv["auroc"], epoch=ep+1,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best["state"])
    torch.save(model.state_dict(), os.path.join(a.out, f"{tag}.pt"))

    p_ot = predict(model, Xot, dev)
    p_iv = predict(model, Xiv, dev)
    p_ov = predict(model, Xov, dev)
    res = dict(arm=a.arm, seed=a.seed, selected_epoch=best["epoch"],
               epochs=a.epochs, lr=a.lr, bs=a.bs,
               rho=(a.rho if use_sam else None), use_ssl=use_ssl, use_sam=use_sam,
               coral_lambda=(a.coral_lambda if a.arm == "coral" else None),
               irm_lambda=(a.irm_lambda if a.arm == "irm" else None),
               dro_eta=(a.dro_eta if a.arm == "groupdro" else None),
               n_train=int(len(Xtr)), wall_s=time.time()-t0,
               device=str(dev),
               ood_test=T.compute_metrics(yot, p_ot),
               id_val=T.compute_metrics(yiv, p_iv),
               ood_val=T.compute_metrics(yov, p_ov),
               log=log)
    json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, f"{tag}_preds.npz"),
                        p_ood_test=p_ot, y_ood_test=yot,
                        p_id_val=p_iv, y_id_val=yiv)
    print(f"[{tag}] DONE ood_test auroc={res['ood_test']['auroc']:.4f}", flush=True)


@torch.no_grad()
def predict(model, X, dev, bs=512):
    model.eval(); out = []
    for k in range(0, len(X), bs):
        xb = torch.from_numpy(X[k:k+bs])
        p = torch.softmax(model(normalize_dev(T.to_float(xb).to(dev))), 1)[:, 1].cpu().numpy()
        out.append(p)
    return np.concatenate(out)


if __name__ == "__main__":
    main()
