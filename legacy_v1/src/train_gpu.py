"""Leave-one-hospital-out training with a composable mechanism suite.

Differences from the earlier CPU pipeline (run_arm.py), and why
---------------------------------------------------------------
* Backbone: ResNet-50 at 224px instead of a 5-layer CNN at 64px. The earlier
  models were small because the runs were CPU-bound, which left a real
  confound: a mechanism that only pays off at scale (SSL especially) would look
  useless. `--arch smallcnn` reproduces the old setting exactly so the
  comparison is available rather than assumed.
* Splits: leave-one-hospital-out over all 5 centers, patient-disjoint, instead
  of the single WILDS ood split.
* Mechanisms: any subset of {simclr, mae, stainaug, dann} x {sam, swa}.
* AMP + channels_last + cosine schedule, because the sweep is large.

The run record schema is a SUPERSET of the earlier one (same key names for the
shared fields), so existing aggregation and bootstrap code reads these records
unchanged.

Data path
---------
Patches are read from the parquet shards by index, using the arrays in
cache/hospital_splits.npz. Decoding 224px images for 68k patches per fold does
not fit in memory, so images are decoded per batch by a worker DataLoader.
`--cache-decoded` pre-decodes into a memmapped uint8 array when disk allows,
which is much faster on a GPU host where decode becomes the bottleneck.
"""
import argparse
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mechanisms as M


# --------------------------------------------------------------------- models
def build_model(arch, n_classes=2, pretrained=False):
    """Returns (model, feature_dim). The classifier is split off so the
    mechanisms that need features (dann, mae, simclr) can reach them."""
    if arch == "resnet50":
        import torchvision
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=w)
        feat = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat
    if arch == "resnet18":
        import torchvision
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        feat = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat
    if arch == "smallcnn":
        import train_lib as TL
        enc = TL.Encoder()
        with torch.no_grad():
            feat = enc(torch.zeros(1, 3, 64, 64)).shape[1]
        return enc, feat
    raise ValueError(f"unknown arch {arch!r}")


class Net(nn.Module):
    def __init__(self, arch, pretrained=False, n_classes=2):
        super().__init__()
        self.encoder, self.feat_dim = build_model(arch, n_classes, pretrained)
        self.head = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x, return_feat=False):
        z = self.encoder(x)
        if z.dim() > 2:
            z = torch.flatten(F.adaptive_avg_pool2d(z, 1), 1)
        logits = self.head(z)
        return (logits, z) if return_feat else logits


# ----------------------------------------------------------------------- data
class ArrayDataset(torch.utils.data.Dataset):
    """Backed by a pre-decoded uint8 memmap (N,H,W,3). This is the path used on
    the GPU host: JPEG decode of 224px patches saturates the CPU long before the
    GPU is busy, so decoding once up front is what keeps utilisation high."""

    def __init__(self, X, y, size, train, mechs, domains=None, seed=0,
                 two_view=False):
        self.X, self.y, self.size, self.train = X, y, size, train
        self.mechs = mechs
        self.domains = domains
        self.two_view = two_view
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.X)

    def _aug(self, x):
        """One SimCLR view: crop, flips, stain/colour jitter, grayscale."""
        c = int(self.size * (0.6 + 0.4 * torch.rand(1).item()))
        top = int(torch.randint(0, max(1, self.size - c + 1), (1,)).item())
        left = int(torch.randint(0, max(1, self.size - c + 1), (1,)).item())
        v = x[:, top:top + c, left:left + c]
        v = F.interpolate(v[None], size=(self.size, self.size), mode="bilinear",
                          align_corners=False)[0]
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, [2])
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, [1])
        v = M.stain_augment(v[None], sigma=0.35, bias=0.08)[0]
        if torch.rand(1).item() < 0.2:
            v = v.mean(0, keepdim=True).expand_as(v).clone()
        return v.clamp_(0, 1)

    def __getitem__(self, i):
        # np.array (not asarray) forces a writable copy: the cache is opened
        # mmap-mode read-only, and torch.from_numpy on a read-only buffer warns
        # and yields a tensor whose in-place ops are undefined behaviour.
        x = torch.from_numpy(np.array(self.X[i])).permute(2, 0, 1).float().div_(255.0)
        if x.shape[-1] != self.size:
            x = F.interpolate(x[None], size=(self.size, self.size), mode="bilinear",
                              align_corners=False)[0]
        if self.two_view:
            # SimCLR pre-training: two independently augmented views of the same
            # patch, no label. Colour jitter is included here (but not in the
            # supervised path) because stain variation is the invariance the
            # encoder should learn for cross-site transfer.
            return (self._aug(x), self._aug(x))
        if self.train:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, [2])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, [1])
            if "stainaug" in self.mechs:
                x = M.stain_augment(x[None], sigma=0.25, bias=0.05)[0]
        out = [x, int(self.y[i])]
        out.append(int(self.domains[i]) if self.domains is not None else 0)
        return tuple(out)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def norm(x):
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


# ------------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    ps, ys = [], []
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(norm(xb))
        ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics(y, p):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, balanced_accuracy_score)
    yhat = (p >= 0.5).astype(int)
    out = dict(auroc=float(roc_auc_score(y, p)),
               avg_precision=float(average_precision_score(y, p)),
               accuracy=float(accuracy_score(y, yhat)),
               balanced_accuracy=float(balanced_accuracy_score(y, yhat)))
    # 15-bin ECE, matching the earlier pipeline
    bins = np.linspace(0, 1, 16)
    conf = np.maximum(p, 1 - p)
    correct = (yhat == y).astype(float)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    out["ece"] = float(ece)
    return out


def train_one(args, data, device):
    mechs = M.parse_mechanisms(args.mechanisms)
    arm = M.arm_name(mechs)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = Net(args.arch, pretrained=args.pretrained).to(device)
    model = model.to(memory_format=torch.channels_last)

    if "simclr" in mechs and args.ssl_encoder and os.path.exists(args.ssl_encoder):
        sd = torch.load(args.ssl_encoder, map_location="cpu")
        missing = model.encoder.load_state_dict(sd, strict=False)
        print(f"[{arm}] loaded SSL encoder {args.ssl_encoder} ({missing})", flush=True)
    elif "simclr" in mechs:
        raise SystemExit(f"--mechanisms includes simclr but no encoder at {args.ssl_encoder!r}; "
                         "run pretrain_gpu.py first")

    tr_loader, iv_loader, te_loader, n_domains = data
    params = list(model.parameters())
    dom_head = None
    if "dann" in mechs:
        dom_head = M.DomainHead(model.feat_dim, n_domains).to(device)
        params += list(dom_head.parameters())
    mae_dec = None
    if "mae" in mechs:
        mae_dec = M.MAEDecoder(model.feat_dim, out_hw=args.size).to(device)
        params += list(mae_dec.parameters())

    if "sam" in mechs:
        import train_lib as TL
        opt = TL.SAM(params, torch.optim.SGD, rho=args.rho, lr=args.lr,
                     momentum=0.9, weight_decay=args.wd)
        base_opt = opt.base_optimizer
    else:
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
        base_opt = opt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=args.epochs)
    swa = M.SWA(model, start_frac=args.swa_start) if "swa" in mechs else None
    lossf = nn.CrossEntropyLoss()
    total_steps = args.epochs * max(len(tr_loader), 1)

    hist, step, best = [], 0, (-1.0, None)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run, nb = 0.0, 0
        for xb, yb, db in tr_loader:
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            db = db.to(device, non_blocking=True)
            lam = M.dann_lambda(step, total_steps) if dom_head is not None else 0.0

            def closure():
                """Full objective: supervised + optional DANN + optional MAE."""
                with torch.autocast(device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    logits, z = model(norm(xb), return_feat=True)
                    loss = lossf(logits, yb)
                    if dom_head is not None:
                        loss = loss + args.dann_w * lossf(dom_head(z, lam), db)
                    if mae_dec is not None:
                        xin, mask = M.random_block_mask(xb, patch=args.mae_patch,
                                                        ratio=args.mae_ratio)
                        _, zm = model(norm(xin), return_feat=True)
                        loss = loss + args.mae_w * M.mae_loss(mae_dec(zm), xb, mask)
                return loss

            if "sam" in mechs:
                # SAM needs two passes; BN stats frozen on the ascent pass
                import train_lib as TL
                TL.enable_running_stats(model, False)
                closure().backward()
                opt.first_step(zero_grad=True)
                TL.enable_running_stats(model, True)
                loss = closure()
                loss.backward()
                opt.second_step(zero_grad=True)
            else:
                opt.zero_grad(set_to_none=True)
                loss = closure()
                loss.backward()
                opt.step()
            run += float(loss.detach()); nb += 1; step += 1
        sched.step()
        if swa is not None:
            swa.maybe_update(model, ep, args.epochs)

        yv, pv = evaluate(model, iv_loader, device, amp_dtype)
        mv = metrics(yv, pv)
        hist.append(dict(epoch=ep + 1, train_loss=run / max(nb, 1),
                         id_val_auroc=mv["auroc"], elapsed=time.time() - t0))
        print(f"[{arm}|hold{args.holdout}|s{args.seed}] ep{ep+1}/{args.epochs} "
              f"loss={run/max(nb,1):.4f} id_val_auroc={mv['auroc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # model selection on IN-DISTRIBUTION val only: using the held-out
        # hospital would leak the test site into selection
        if mv["auroc"] > best[0]:
            best = (mv["auroc"], {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}, ep + 1)

    # SWA replaces the selected weights and REQUIRES a BN refresh
    if swa is not None and swa.active:
        swa.load_into(model)
        bn_batches = []
        for i, (xb, _, _) in enumerate(tr_loader):
            bn_batches.append(norm(xb.to(device)).to(memory_format=torch.channels_last))
            if i + 1 >= args.swa_bn_batches:
                break
        M.SWA.update_bn(model, bn_batches, device=device)
        sel_epoch = args.epochs
    else:
        model.load_state_dict(best[1])
        sel_epoch = best[2]

    out = dict(arm=arm, mechanisms=sorted(mechs), holdout=args.holdout,
               seed=args.seed, arch=args.arch, size=args.size,
               pretrained=bool(args.pretrained), epochs=args.epochs,
               lr=args.lr, bs=args.bs, wd=args.wd, rho=args.rho,
               selected_epoch=sel_epoch, selection="id_val_auroc",
               swa_used=bool(swa is not None and swa.active),
               n_train=len(tr_loader.dataset), n_id_val=len(iv_loader.dataset),
               n_test=len(te_loader.dataset), wall_s=time.time() - t0,
               log=hist)
    yv, pv = evaluate(model, iv_loader, device, amp_dtype)
    yt, pt = evaluate(model, te_loader, device, amp_dtype)
    out["id_val"] = metrics(yv, pv)
    out["ood_test"] = metrics(yt, pt)   # held-out hospital
    return out, model, (yt, pt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", required=True, help="center held out, e.g. train-center1")
    p.add_argument("--mechanisms", default="baseline",
                   help="e.g. baseline | simclr | sam | simclr+sam | mae+swa")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", default="resnet50",
                   choices=["resnet50", "resnet18", "smallcnn"])
    p.add_argument("--size", type=int, default=224,
                   help="model input resolution (patches are upsampled to this)")
    p.add_argument("--cache-size", type=int, default=0,
                   help="resolution of the decoded cache on disk; defaults to "
                        "--size. Set 96 (native) to avoid storing upsampling.")
    p.add_argument("--pretrained", action="store_true",
                   help="ImageNet init (an additional baseline, not a mechanism)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--dann-w", type=float, default=0.3)
    p.add_argument("--mae-w", type=float, default=1.0)
    p.add_argument("--mae-patch", type=int, default=16)
    p.add_argument("--mae-ratio", type=float, default=0.6)
    p.add_argument("--swa-start", type=float, default=0.6)
    p.add_argument("--swa-bn-batches", type=int, default=50)
    p.add_argument("--ssl-encoder", default="")
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="runs_gpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="cap samples per split (smoke tests)")
    p.add_argument("--dry-run", action="store_true",
                   help="1 epoch on --limit samples; writes no run record")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.dry_run:
        a.epochs, a.limit = 1, (a.limit or 200)

    import pandas as pd
    meta = pd.read_parquet(os.path.join(a.cache, "camelyon17_metadata.parquet"))
    meta["center"] = meta["center"].astype(str)
    sp = np.load(os.path.join(a.cache, "hospital_splits.npz"))
    # Patches are natively 96x96, so the cache is stored at native resolution
    # and upsampled in the dataloader. Caching at the training size would store
    # pure interpolation at (size/96)^2 the disk -- 38 GB instead of 7 GB for
    # 224px -- with no extra information.
    cs = a.cache_size or a.size
    dec = os.path.join(a.cache, f"decoded_{cs}.npy")
    idx_path = os.path.join(a.cache, f"decoded_{cs}_index.json")
    if not (os.path.exists(dec) and os.path.exists(idx_path)):
        raise SystemExit(f"missing {dec} / {idx_path}; run decode_patches.py first")
    X = np.load(dec, mmap_mode="r")
    # The decoded array holds ONLY referenced rows, so a split index is not a
    # direct offset -- it must be mapped through row_ids. Getting this wrong
    # trains on the wrong images without any error, so the mapping is explicit
    # and every lookup is asserted present.
    dec_idx = json.load(open(idx_path))
    row_to_pos = {int(r): i for i, r in enumerate(dec_idx["row_ids"])}
    assert len(X) == len(dec_idx["row_ids"]), "decoded array/index length mismatch"

    dom_codes = {c: i for i, c in enumerate(sorted(meta.center.unique()))}
    loaders = []
    for split, train in (("train", True), ("id_val", False), ("test", False)):
        idx = sp[f"{a.holdout}__{split}"]
        if a.limit:
            # Split indices are class-ordered, so a contiguous head would be
            # single-class and every metric would come back NaN or degenerate.
            # Take a balanced subsample instead, so a smoke test exercises the
            # real metric path.
            yfull = meta.label.to_numpy()[idx]
            rs = np.random.default_rng(a.seed)
            per = max(1, a.limit // 2)
            pick = np.concatenate([
                rs.choice(np.where(yfull == c)[0],
                          min(per, int((yfull == c).sum())), replace=False)
                for c in (0, 1) if (yfull == c).any()])
            idx = idx[np.sort(pick)]
        y = meta.label.to_numpy()[idx]
        dom = np.array([dom_codes[c] for c in meta.center.to_numpy()[idx]])
        missing = [int(r) for r in idx if int(r) not in row_to_pos]
        assert not missing, (f"{len(missing)} split rows absent from the decoded cache "
                             f"(first: {missing[:5]}); re-run decode_patches.py")
        pos = np.array([row_to_pos[int(r)] for r in idx], dtype=np.int64)
        ds = ArrayDataset(_Indexed(X, pos), y, a.size, train,
                          M.parse_mechanisms(a.mechanisms), domains=dom, seed=a.seed)
        loaders.append(torch.utils.data.DataLoader(
            ds, batch_size=a.bs, shuffle=train, num_workers=a.workers,
            pin_memory=(device.type == "cuda"), drop_last=train,
            persistent_workers=a.workers > 0))
    rec, model, (yt, pt) = train_one(a, (*loaders, len(dom_codes)), device)

    print(json.dumps({k: rec[k] for k in ("arm", "holdout", "seed", "arch",
                                          "selected_epoch", "wall_s")}, indent=1))
    print("id_val :", json.dumps(rec["id_val"]))
    print("ood_test:", json.dumps(rec["ood_test"]))
    if a.dry_run or a.limit:
        # A subsampled run must never leave a record: run_sweep.py treats an
        # existing record as "this arm is done", so a smoke-test file would
        # silently substitute for a real result in the analysis.
        print(f"[{'dry-run' if a.dry_run else 'limit=%d' % a.limit}] "
              "no record written (subsampled run)")
        return
    os.makedirs(a.out, exist_ok=True)
    tag = f"{rec['arm']}__{a.holdout}__seed{a.seed}"
    with open(os.path.join(a.out, tag + ".json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    np.savez_compressed(os.path.join(a.out, tag + "_preds.npz"), y=yt, p=pt)
    print("wrote", os.path.join(a.out, tag + ".json"))


class _Indexed:
    """Maps dataset position -> row in the big memmap, without materialising."""

    def __init__(self, X, idx):
        self.X, self.idx = X, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.X[self.idx[i]]


if __name__ == "__main__":
    main()
