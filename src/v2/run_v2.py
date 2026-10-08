"""v2 training driver: load one fold cache ONCE, then run a list of jobs.

Each job = (arm, seed, hp overrides [, tta, pcam]). Fixed step budget (default 1200
steps, bs 128, cosine), evaluation of id_val / ood_val / ood_test at exactly 10
checkpoints round(k * steps / 10), k=1..10 (fp32 evaluation on both tiers),
selection by id / ood / last / oracle (max AUROC, ties -> earliest checkpoint),
run record per CONTRACTS.md B + {tag}_preds.npz + {tag}.pt (ID-selected checkpoint).
Every record carries `cache_fingerprint` (common.cache_fingerprint of the fold cache)
and `status` ('ok' | 'diverged': non-finite loss / predictions -> training stops, the
record keeps the checkpoints so far, non-finite probabilities are scored as 0.5).
A job whose JSON already exists *for the same cache fingerprint* is skipped (resume);
a record / probe-feature file / SSL checkpoint from a different cache is recomputed.

Usage:
  # single job
  python src/v2/run_v2.py --tier C --cohort c17 --fold 0 --arm groupdro --seed 0 --hp eta=0.1
  # with inline TTA and PCam evaluation
  python src/v2/run_v2.py --tier C --cohort c17 --fold 0 --arm erm --seed 0 --tta bnadapt,tent --pcam
  # a plan shard (from make_plan.py)
  python src/v2/run_v2.py --plan slurm/v2/plans/final/C_c17_fold0.json
Paths default to $V2/cache{64,96}/{cohort}/fold{t}, $V2/runs (phase final) or
$V2/runs_tune (phase tune), $V2/ssl, $V2/feats; override with --cache_dir/--out_root/...
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import arms as A  # noqa: E402
from backbones import build_net, n_params, bn_modules  # noqa: E402
import ssl_v2  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ============================================================================ context
class Ctx:
    def __init__(self, a, tier, cohort, fold, phase):
        self.a, self.tier, self.cohort, self.fold_id, self.phase = a, tier, cohort, fold, phase
        self.dev = torch.device(a.device)
        self.path = a.cache_dir or C.fold_dir(a.cache_root, tier, cohort, fold)
        t0 = time.time()
        self.fold = C.Fold(self.path, self.dev)
        self.load_s = time.time() - t0
        self.fp = self.fold.fingerprint
        log(f"loaded {self.path} in {self.load_s:.1f}s: n_tr={self.fold.n('tr')} n_iv={self.fold.n('iv')} "
            f"n_ov={self.fold.n('ov') if self.fold.has_ov else None} n_ot={self.fold.n('ot')} K={self.fold.K} "
            f"P={self.fold.P} on {self.fold.img_device} fingerprint={self.fp[:12]}")
        out_root = a.out_root or os.path.join(C.V2, "runs" if phase == "final" else "runs_tune")
        self.out_dir = os.path.join(out_root, tier, cohort, f"fold{fold}")
        os.makedirs(self.out_dir, exist_ok=True)
        self._pcam = None
        self._feats = {}
        self.code_version = C.code_version()

    def n_u(self):
        return int(np.load(os.path.join(self.path, "Xu.npy"), mmap_mode="r").shape[0])

    def pcam(self):
        if self._pcam is None:
            t0 = time.time()
            crop = 64 if self.fold.P == 64 else None   # tier C: centre-crop 96->64; tier S native 96
            self._pcam = C.load_pcam(self.fold.img_device, crop)
            log(f"loaded PCam test {tuple(self._pcam[0].shape)} in {time.time() - t0:.1f}s")
        return self._pcam


# ============================================================================ helpers
def eval_splits(model, X, ctx, cl):
    """fp32 predictions + metrics on iv / ov / ot. Returns (preds, metrics, n_nonfinite)."""
    out_p, out_m, n_bad = {}, {}, 0
    for sp, name in (("iv", "id_val"), ("ov", "ood_val"), ("ot", "ood_test")):
        if sp not in X:
            out_m[name] = None
            continue
        p = C.predict(model, X[sp], ctx.dev, amp=False, channels_last=cl)
        p, nb = C.sanitize_probs(p)
        n_bad += nb
        out_p[sp] = p
        out_m[name] = C.metrics(ctx.fold.y[sp], p)
    return out_p, out_m, n_bad


def id_val_by_domain(ctx, p_iv):
    """Per-training-domain ID-val AUROC (keyed by train-domain name), so that selection
    criteria restricted to some domains can be computed after the fact."""
    d = ctx.fold.d_np.get("iv")
    if d is None or len(d) != len(p_iv) or (d < 0).any():
        return None
    names = ctx.fold.info().get("train_domains") or []
    y = ctx.fold.y["iv"]
    out = {}
    for k in np.unique(d):
        m = d == k
        two = len(np.unique(y[m])) == 2
        from sklearn.metrics import roc_auc_score
        out[names[k] if k < len(names) else str(int(k))] = dict(
            auroc=float(roc_auc_score(y[m], p_iv[m])) if two else None, n=int(m.sum()))
    return out


def _auc(m):
    return -1.0 if (m is None or m.get("auroc") is None) else m["auroc"]


class Selector:
    """Tracks the id / ood / oracle / last checkpoints (strict improvement -> earliest wins)."""

    def __init__(self, has_ov):
        self.rules = ["id", "ood", "last", "oracle"] if has_ov else ["id", "last", "oracle"]
        self.best = {r: (-2.0, None) for r in ("id", "ood", "oracle")}
        self.p_ot, self.p_all, self.metrics, self.state_id = {}, {}, {}, None

    def update(self, step, mets, p, model):
        """p: dict split -> probabilities at this checkpoint (must contain 'ot')."""
        p_ot = p["ot"]
        keys = {"id": mets["id_val"], "ood": mets["ood_val"], "oracle": mets["ood_test"]}
        for r, m in keys.items():
            if r not in self.rules:
                continue
            v = _auc(m)
            if v > self.best[r][0]:
                self.best[r] = (v, step)
                self.p_ot[r], self.p_all[r] = p_ot, p
                self.metrics[r] = mets["ood_test"]
                if r == "id":
                    self.state_id = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.best["last"] = (None, step)
        self.p_ot["last"], self.p_all["last"], self.metrics["last"] = p_ot, p, mets["ood_test"]

    def selected(self):
        return {r: int(self.best[r][1]) for r in self.rules}

    def preds(self, fold):
        """preds npz content: OOD-test (contract B) + ID-val / OOD-val at every selected
        checkpoint, with labels and ID-val training-domain index."""
        out = {"y_ood_test": fold.y["ot"].astype(np.int8)}
        for r in self.rules:
            out[f"p_ood_test_{r}"] = self.p_ot[r].astype(np.float32)
        for sp, name in (("iv", "id_val"), ("ov", "ood_val")):
            if sp not in self.p_all[self.rules[0]]:
                continue
            out[f"y_{name}"] = fold.y[sp].astype(np.int8)
            for r in self.rules:
                out[f"p_{name}_{r}"] = self.p_all[r][sp].astype(np.float32)
        if "iv" in fold.d_np:
            out["d_id_val"] = fold.d_np["iv"].astype(np.int16)
        return out


def write_record(ctx, tag, rec, preds, state):
    """preds and checkpoint first, JSON last (the JSON marks the job as done); every
    file is written atomically through a unique temp name."""
    base = os.path.join(ctx.out_dir, tag)
    C.atomic_npz(base + "_preds.npz", **preds)
    if state is not None:
        C.atomic_torch({k: v.cpu() for k, v in state.items()}, base + ".pt")
    C.atomic_json(rec, base + ".json")


def base_record(ctx, job, spec, hp, tag, hpkey):
    info = ctx.fold.info()
    return {"schema": "v2", "arm": job["arm"], "tier": ctx.tier, "backbone": spec["backbone"],
            "init": spec["init"], "cohort": ctx.cohort, "fold": ctx.fold_id,
            "test_domain": info["test_domain"], "val_domain": info["val_domain"], "seed": int(job["seed"]),
            "hparams": hp, "phase": job.get("phase", ctx.phase), "input_res": None, "n_params": None,
            "n_train": ctx.fold.n("tr"), "wall_s": None, "checkpoints": [], "selected": {}, "test_at": {},
            "code_version": ctx.code_version, "torch": torch.__version__, "device": str(ctx.dev),
            "tag": tag, "hpkey": tag.split("__")[1], "hpkey_full": hpkey,
            "hp_source": job.get("hp_source", "selected" if job.get("hpkey") == "selected" else "explicit"),
            "method": spec["method"], "cache_dir": ctx.path,
            "cache_fingerprint": ctx.fp, "cache_created": ctx.fold.manifest.get("created"),
            "selection_metric": "auroc", "eval_precision": "fp32", "status": "ok"}


# ============================================================================ fine-tuning job
def run_train_job(ctx, job, spec, hp, tag, hpkey):
    t_job = time.time()
    method, fold, dev = spec["method"], ctx.fold, ctx.dev
    seed = int(job["seed"])
    C.set_seed(seed)
    extra = {}
    n_u = ctx.n_u() if method == "compute_matched" else None
    n_steps = A.total_steps(method, hp, n_u)
    if method == "compute_matched":
        extra["ssl_images_seen"] = A.ssl_images_seen(n_u, hp["ssl_epochs"], hp["ssl_bs"])
    ev_steps = C.eval_steps(n_steps, int(hp["eval_every"]))
    ev_set = set(ev_steps)
    # derived budget values are recorded alongside the hyperparameters (not part of the hpkey)
    hp = dict(hp, total_steps=n_steps)
    if method == "irm":
        hp["irm_anneal_steps"] = int(round(hp["irm_anneal_frac"] * n_steps))
    if method == "compute_matched":
        hp["ssl_images_seen"] = extra["ssl_images_seen"]

    # ---- inputs (Macenko applied identically to every split, incl. test: per group id
    #      (slide / image) if mac_level='group', else per tile)
    X = fold.X
    if method in A.USES_MACENKO:
        from stain import make_macenko_fn
        mkey = ("macenko",) + tuple(hp[k] for k in sorted(hp) if k.startswith("mac_"))
        P = fold.P
        level = hp["mac_level"]
        unit = hp.get("mac_group_unit", "group")
        ref_groups = split_groups(fold, "tr", unit)[0] if hp["mac_ref"] == "train_group" else None
        fn = make_macenko_fn(fold.X["tr"], ref=hp["mac_ref"], n_ref=hp["mac_nref"], io=hp["mac_io"],
                             alpha=hp["mac_alpha"], beta=hp["mac_beta"],
                             min_tissue=int(hp["mac_min_tissue_frac"] * P * P), seed=ctx.fold_id,
                             nonneg=bool(int(hp["mac_nonneg"])), level=level,
                             group_tiles=int(hp["mac_group_tiles"]), group_px=int(hp["mac_group_px"]),
                             group_min_px=int(hp["mac_group_min_px"]),
                             group_maxc=hp.get("mac_group_maxc", "tile_median"), ref_groups=ref_groups)
        fn.group_unit = unit
        X = macenko_splits(fold, mkey, fn)
        extra["macenko"] = dict(HE_ref=fn.HE_ref.tolist(), maxC_ref=fn.maxC_ref.tolist(), ref_info=fn.ref_info,
                                transform_s=X["_time"], mac_level=level,
                                mean_rgb={sp: dict(raw=fold.X[sp].float().mean((0, 1, 2)).tolist(),
                                                   norm=X[sp].float().mean((0, 1, 2)).tolist())
                                          for sp in fold.X})
        if "_stats" in X:
            extra["macenko"]["splits"] = X["_stats"]
        X = {k: v for k, v in X.items() if not k.startswith("_")}

    # ---- model
    ms = dict(p=hp["ms_p"], alpha=hp["ms_alpha"]) if method == "mixstyle" else None
    model = build_net(spec["backbone"], spec["init"], mixstyle=ms)
    if method in A.USES_SSL:
        cfg = dict(epochs=hp["ssl_epochs"], bs=hp["ssl_bs"], lr=hp["ssl_lr"], temp=hp["ssl_temp"])
        pt, js = ssl_v2.ssl_paths(ctx.a.ssl_root, ctx.tier, ctx.cohort, ctx.fold_id, spec["backbone"],
                                  spec["init"], seed)
        st = ssl_v2.checkpoint_status(pt, js, ctx.fp)
        if st != "ok":
            if ctx.a.no_auto_ssl:
                raise FileNotFoundError(f"{pt} ({st})")
            log(f"[{tag}] SSL checkpoint {st} -> pre-training {pt}")
            # isolated RNG: fine-tuning draws exactly the same random numbers whether the
            # checkpoint was pre-trained here or loaded from disk
            with C.rng_isolated():
                ssl_v2.run_and_save(fold.Xu, ctx.a.ssl_root, ctx.tier, ctx.cohort, ctx.fold_id, spec["backbone"],
                                    spec["init"], seed, dev, cfg, log=log, force=True, fingerprint=ctx.fp,
                                    cache_dir=ctx.path)
            extra["ssl_pretrained_in_job"] = True
        info = json.load(open(js))
        for k, v in cfg.items():
            if info["cfg"].get(k) != v:
                raise ValueError(f"SSL checkpoint {pt} was trained with {k}={info['cfg'].get(k)}, job wants {v}")
        model.encoder.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
        extra["ssl"] = {k: info[k] for k in ("steps", "images_seen", "n_unlabeled", "wall_s", "cfg")}
        extra["ssl"]["final_nt_xent"] = info["history"][-1]["nt_xent"]
        extra["ssl"]["path"] = pt
        extra["ssl"]["cache_fingerprint"] = info.get("cache_fingerprint")
    model = model.to(dev)
    cl = bool(hp["channels_last"])   # bf16 training autocast (hp['amp']) lives in the Trainer; eval is fp32
    if cl:
        model = model.to(memory_format=torch.channels_last)
    tr = A.Trainer(method, model, hp, fold.K, n_steps, dev)
    sampler = C.EpochSampler(fold.n("tr"), hp["bs"], seed, fold.img_device)

    rec = base_record(ctx, job, spec, hp, tag, hpkey)
    rec["input_res"] = 224 if model.pre.resize_to else fold.P
    rec["n_params"] = n_params(model)
    sel = Selector(fold.has_ov)
    loss_acc, n_acc = torch.zeros((), device=dev), 0
    t_train = t_eval = 0.0
    t0 = time.time()
    for step in range(1, n_steps + 1):
        idx = sampler.next()
        x = C.to_float(X["tr"][idx], dev)
        idd = idx.to(dev)
        loss = tr.step(x, fold.yt["tr"][idd], fold.d_tr[idd])
        loss_acc += loss.float()
        n_acc += 1
        if step in ev_set:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            t_train += t1 - t0
            p, mets, n_bad = eval_splits(model, X, ctx, cl)
            tl = float(loss_acc) / max(n_acc, 1)
            ck = {"step": step, "train_loss": tl, "lr": tr.lr(), **mets,
                  "id_val_by_domain": id_val_by_domain(ctx, p["iv"])}
            diverged = (not math.isfinite(tl)) or n_bad > 0
            if diverged:
                ck["diverged"], ck["n_nonfinite_preds"] = True, int(n_bad)
            rec["checkpoints"].append(ck)
            sel.update(step, mets, p, model)
            loss_acc, n_acc = torch.zeros((), device=dev), 0
            t0 = time.time()
            t_eval += t0 - t1
            log(f"[{tag}] step {step}/{n_steps} loss={tl:.4f} id={_auc(mets['id_val']):.4f} "
                f"ood={_auc(mets['ood_val']):.4f} test={_auc(mets['ood_test']):.4f} "
                f"(train {t_train:.1f}s eval {t_eval:.1f}s)")
            if diverged:
                rec["status"], rec["diverged_at_step"] = "diverged", step
                log(f"[{tag}] DIVERGED at step {step} (loss={tl}, {n_bad} non-finite predictions) -> stop")
                break
    rec["selected"] = sel.selected()
    rec["test_at"] = {r: sel.metrics[r] for r in sel.rules}
    preds = sel.preds(fold)

    # ---- ID-selected model for TTA / PCam
    model.load_state_dict(sel.state_id)
    model.eval()
    n_done = rec["checkpoints"][-1]["step"]
    timing = dict(train_s=t_train, eval_s=t_eval, n_steps=n_steps, eval_steps=ev_steps,
                  train_s_per_100=100 * t_train / n_done, n_evals=len(rec["checkpoints"]))
    tta_req = job.get("tta") or []
    if tta_req:
        rec["tta"] = {}
        for t in tta_req:
            t1 = time.time()
            m2 = copy.deepcopy(model)
            if not bn_modules(m2):
                log(f"[{tag}] skip TTA {t}: no BatchNorm")
                continue
            if t == "bnadapt":
                p = A.tta_bnadapt(m2, X["ot"], dev, bs=128, seed=seed, amp=False, cl=cl)
            elif t == "tent":
                p = A.tta_tent(m2, X["ot"], dev, bs=128, lr=1e-3, steps=1, seed=seed, amp=False, cl=cl)
            else:
                raise ValueError(t)
            p = C.sanitize_probs(p)[0]
            rec["tta"][t] = C.metrics(fold.y["ot"], p)
            preds[f"p_tta_{t}"] = p.astype(np.float32)
            timing[f"tta_{t}_s"] = time.time() - t1
            log(f"[{tag}] TTA {t}: auroc={rec['tta'][t]['auroc']:.4f} ({timing[f'tta_{t}_s']:.1f}s)")
            del m2
    if job.get("pcam") and ctx.cohort == "c17":
        t1 = time.time()
        Xp, yp = ctx.pcam()
        if method in A.USES_MACENKO:   # same normalisation as every other split (per WSI if level='group')
            Xp, pst = _pcam_macenko(mkey, fn, Xp)
            if pst is not None:
                extra["macenko"]["pcam"] = pst
        p = C.predict(model, Xp, dev, amp=False, channels_last=cl)
        rec["pcam"] = {"id": C.metrics(yp, p)}
        preds["p_pcam_id"] = p.astype(np.float32)
        timing["pcam_s"] = time.time() - t1
        log(f"[{tag}] PCam auroc={rec['pcam']['id']['auroc']:.4f} ({timing['pcam_s']:.1f}s)")
    rec["wall_s"] = time.time() - t_job
    rec["timing"] = timing
    rec["extra"] = extra
    write_record(ctx, tag, rec, preds, sel.state_id)
    sid = rec["selected"]["id"]
    log(f"[{tag}] DONE sel={rec['selected']} test@id auroc={_auc(rec['test_at']['id']):.4f} "
        f"wall={rec['wall_s']:.1f}s (train {100 * t_train / n_done:.2f}s/100 steps); id-step {sid} "
        f"status={rec['status']}")
    return rec


_PCAM_MAC = {}
PCAM_META = os.path.join(C.DATA_ROOT, "pcam", "camelyonpatch_level_2_split_test_meta.csv")


def load_groups(cache_dir, sp):
    """Group ids of split `sp` (g{sp}.npy: patient / slide / image), or None if absent. Never labels."""
    f = os.path.join(cache_dir, f"g{sp}.npy")
    return np.load(f) if os.path.exists(f) else None


def pcam_groups(n):
    """PCam test WSI id per row (the 'wsi' column only; no label columns are read), or None."""
    if not os.path.exists(PCAM_META):
        return None
    import pandas as pd
    head = pd.read_csv(PCAM_META, nrows=0).columns
    if "wsi" not in head:
        return None
    g = pd.read_csv(PCAM_META, usecols=["wsi"])["wsi"].astype(str).to_numpy()
    return g if len(g) == n else None


_SLIDES = {}


def c17_slides(fold, sp, g):
    """Camelyon17 slide id per row of split `sp` (WILDS metadata.csv 'slide' column at the
    rows.npz row ids), or None if unavailable or inconsistent with the cached patient ids `g`.
    Reads only the patient / slide columns (never labels)."""
    m = getattr(fold, "manifest", None) or {}
    if m.get("cohort") != "c17":
        return None
    rp = os.path.join(fold.path, "rows.npz")
    meta = os.path.join(os.path.dirname(m.get("source", "")), "metadata.csv")
    if not os.path.exists(meta):
        meta = os.path.join(C.DATA_ROOT, "wilds", "camelyon17_v1.0", "metadata.csv")
    if not (os.path.exists(rp) and os.path.exists(meta)):
        return None
    rows = np.load(rp)
    if f"rows_{sp}" not in rows.files:
        return None
    r = rows[f"rows_{sp}"].astype(np.int64)
    if meta not in _SLIDES:
        import pandas as pd
        head = list(pd.read_csv(meta, nrows=0).columns)
        if head[1:2] != ["patient"] or head[6:7] != ["slide"]:
            return None
        df = pd.read_csv(meta, usecols=[0, 1, 6])                          # row id, patient, slide
        df.columns = ["row", "patient", "slide"]
        assert (df["row"].to_numpy() == np.arange(len(df))).all()
        _SLIDES[meta] = (df["patient"].to_numpy(np.int64), df["slide"].to_numpy(np.int64))
    pat, sl = _SLIDES[meta]
    if r.max(initial=-1) >= len(sl) or g is None or len(g) != len(r) or not np.array_equal(pat[r], g):
        return None
    return sl[r]


def split_groups(fold, sp, unit="group"):
    """(group id per row of split `sp`, source string) for the group-level Macenko, or (None, None).
    unit 'group'        : the cache's g{sp}.npy (patient / slide / image).
    unit 'slide_domain' : the slide where the cache group is coarser (c17: WILDS slide instead of
                          patient), crossed with the split's domain id where the split has one
                          (canine tr / iv: slide x scanner -- a canine group is a slide imaged on
                          several scanners; MIDOG / c17 groups lie within one domain, unchanged)."""
    n = int(fold.X[sp].shape[0])
    g = load_groups(fold.path, sp)
    if g is None or len(g) != n:
        return None, None
    if unit == "group":
        return g, f"g{sp}.npy"
    if unit != "slide_domain":
        raise ValueError(f"mac_group_unit must be 'group' or 'slide_domain', got {unit!r}")
    src = f"g{sp}.npy"
    s = c17_slides(fold, sp, g)
    if s is not None:
        g, src = s, f"metadata.slide[rows_{sp}]"
    d = (getattr(fold, "d_np", None) or {}).get(sp)
    if d is not None and len(d) == n:
        g = np.char.add(np.char.add(g.astype(str), "@d"), np.asarray(d).astype(str))
        src += f" x d{sp}.npy"
    return g, src


def macenko_splits(fold, mkey, fn):
    """Macenko-normalised copies of every split (cached on the Fold under `mkey`).
    level 'tile': fn(X) per split (unchanged per-tile path). level 'group': fn.group(X, g)
    with the split's group ids (split_groups, unit fn.group_unit, default 'group'); a split
    without group ids falls back to per-tile (recorded).
    Returns {split: X, '_time': s[, '_stats': {split: stats}]}."""
    if fn.level != "group":
        return fold.transformed(mkey, fn)
    if mkey not in fold._transformed:
        t0 = time.time()
        out, stats = {}, {}
        for sp, Xs in fold.X.items():
            g, src = split_groups(fold, sp, getattr(fn, "group_unit", "group"))
            if g is None:
                out[sp] = fn(Xs)
                stats[sp] = dict(level="tile", group_source=None, n_rows=int(Xs.shape[0]))
            else:
                out[sp], st = fn.group(Xs, g)
                stats[sp] = dict(st, group_source=src)
        out["_time"], out["_stats"] = time.time() - t0, stats
        fold._transformed[mkey] = out
    return fold._transformed[mkey]


def _pcam_macenko(mkey, fn, Xp):
    """(normalised PCam, stats or None). Group-level: groups = PCam 'wsi' column if
    available, else per tile (recorded in the stats)."""
    if mkey not in _PCAM_MAC:
        if fn.level != "group":
            _PCAM_MAC[mkey] = (fn(Xp), None)
        else:
            g = pcam_groups(Xp.shape[0])
            if g is None:
                _PCAM_MAC[mkey] = (fn(Xp), dict(level="tile", group_source=None, n_rows=int(Xp.shape[0])))
            else:
                Xn, st = fn.group(Xp, g)
                _PCAM_MAC[mkey] = (Xn, dict(st, group_source="pcam_meta.wsi"))
    return _PCAM_MAC[mkey]


# ============================================================================ probes (D10)
FEAT_VERSION = "fp32-v2"   # bump when the extraction rule changes (old feature files are then ignored)


@torch.no_grad()
def extract(net, X_u8, dev, bs=512):
    """Frozen-backbone features in fp32 (TF32 matmuls; no bf16 autocast)."""
    net.eval()
    out = []
    for k in range(0, X_u8.shape[0], bs):
        x = C.to_float(X_u8[k:k + bs], dev)
        out.append(net.features(x).float())
    return torch.cat(out)


def _feat_ok(f, n_rows, ident):
    """Feature file usable: exists, its sidecar matches `ident` and the row count matches."""
    meta = f[:-4] + ".meta.json"
    if not (os.path.exists(f) and os.path.exists(meta)):
        return False
    try:
        m = json.load(open(meta))
    except (OSError, ValueError):
        return False
    return m.get("ident") == ident and m.get("n_rows") == n_rows


def get_feats(ctx, backbone, init, want_pcam):
    """Probe features, extracted once and cached on disk as float32 (the in-memory copy
    is bit-identical to what a later process loads). Fold features live in a directory
    keyed by the cache fingerprint and carry a sidecar with the fingerprint, so a rebuilt
    cache never reuses stale features. PCam features are shared by all folds (keyed by
    the PCam file and crop). Concurrent writers are safe (unique temp names; atomic
    rename; identical content)."""
    key = (backbone, init)
    if key in ctx._feats and (not want_pcam or "pcam" in ctx._feats[key]["F"]):
        return ctx._feats[key]
    root = os.path.join(ctx.a.feats_root, ctx.tier, ctx.cohort, f"fold{ctx.fold_id}", ctx.fp[:16])
    os.makedirs(root, exist_ok=True)
    stem = os.path.join(root, f"{backbone}_{init}_r224")
    net = None
    feats, t_ext = {}, 0.0
    splits = [sp for sp in ("tr", "iv", "ov", "ot") if sp in ctx.fold.X]
    for sp in splits + (["pcam"] if want_pcam else []):
        if sp == "pcam":
            crop = "c64" if ctx.fold.P == 64 else "c96"
            proot = os.path.join(ctx.a.feats_root, ctx.tier, "pcam")
            os.makedirs(proot, exist_ok=True)
            f = os.path.join(proot, f"{backbone}_{init}_r224_{crop}.npy")
            st = os.stat(C.PCAM_X)
            ident = dict(version=FEAT_VERSION, src=C.PCAM_X, size=st.st_size, crop=crop)
            n_rows = None
        else:
            f = f"{stem}_{sp}.npy"
            ident = dict(version=FEAT_VERSION, cache_fingerprint=ctx.fp, split=sp)
            n_rows = ctx.fold.n(sp)
        if n_rows is None and os.path.exists(f[:-4] + ".meta.json"):
            n_rows = json.load(open(f[:-4] + ".meta.json")).get("n_rows")
        if _feat_ok(f, n_rows, ident):
            feats[sp] = torch.from_numpy(np.load(f)).to(ctx.dev).float()
            continue
        if net is None:
            net = build_net(backbone, init, probe=True).to(ctx.dev)
        t0 = time.time()
        Xs = ctx.pcam()[0] if sp == "pcam" else ctx.fold.X[sp]
        Fm = extract(net, Xs, ctx.dev)
        t_ext += time.time() - t0
        arr = np.ascontiguousarray(Fm.cpu().numpy().astype(np.float32))
        C.atomic_npy(arr, f)
        C.atomic_json(dict(ident=ident, n_rows=int(arr.shape[0]), dim=int(arr.shape[1]), dtype="float32",
                           backbone=backbone, init=init, cache_dir=ctx.path), f[:-4] + ".meta.json")
        feats[sp] = torch.from_numpy(arr).to(ctx.dev)
        log(f"extracted {backbone}_{init} {sp} {tuple(Fm.shape)} in {time.time() - t0:.1f}s -> {f}")
    if net is None:
        net = build_net(backbone, init, probe=True)
    ctx._feats[key] = {"F": feats, "n_backbone": n_params(net), "extract_s": t_ext}
    return ctx._feats[key]


def run_probe_job(ctx, job, spec, hp, tag, hpkey):
    t_job = time.time()
    dev, fold, seed = ctx.dev, ctx.fold, int(job["seed"])
    want_pcam = bool(job.get("pcam")) and ctx.cohort == "c17"
    fe = get_feats(ctx, spec["backbone"], spec["init"], want_pcam)
    Fd = fe["F"]
    C.set_seed(seed)
    mu, sd = Fd["tr"].mean(0, keepdim=True), Fd["tr"].std(0, keepdim=True).clamp_min(1e-6)
    norm = (lambda z: (z - mu) / sd) if hp["feat_std"] else (lambda z: z)
    head = nn.Linear(Fd["tr"].shape[1], 2).to(dev)
    n_steps = int(hp["steps"])
    ev_steps = C.eval_steps(n_steps, int(hp["eval_every"]))
    ev_set = set(ev_steps)
    hp = dict(hp, total_steps=n_steps)
    opt = torch.optim.SGD(head.parameters(), lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, C.cosine_lambda(n_steps))
    sampler = C.EpochSampler(fold.n("tr"), hp["bs"], seed, dev)
    rec = base_record(ctx, job, spec, hp, tag, hpkey)
    rec["input_res"] = 224
    rec["n_params"] = fe["n_backbone"] + n_params(head)
    sel = Selector(fold.has_ov)

    @torch.no_grad()
    def prob(Fm):
        return C.sanitize_probs(torch.softmax(head(norm(Fm)).double(), 1)[:, 1].cpu().numpy())

    loss_acc, n_acc, t0, t_train, t_eval = 0.0, 0, time.time(), 0.0, 0.0
    ytr = fold.yt["tr"]
    for step in range(1, n_steps + 1):
        idx = sampler.next()
        loss = F.cross_entropy(head(norm(Fd["tr"][idx])), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_acc += loss.detach()
        n_acc += 1
        if step in ev_set:
            t1 = time.time()
            t_train += t1 - t0
            pb = {sp: prob(Fd[sp]) for sp in ("iv", "ov", "ot") if sp in Fd}
            p = {sp: v[0] for sp, v in pb.items()}
            n_bad = sum(v[1] for v in pb.values())
            mets = {"id_val": C.metrics(fold.y["iv"], p["iv"]),
                    "ood_val": C.metrics(fold.y["ov"], p["ov"]) if "ov" in p else None,
                    "ood_test": C.metrics(fold.y["ot"], p["ot"])}
            tl = float(loss_acc) / n_acc
            ck = {"step": step, "train_loss": tl, "lr": opt.param_groups[0]["lr"], **mets,
                  "id_val_by_domain": id_val_by_domain(ctx, p["iv"])}
            diverged = (not math.isfinite(tl)) or n_bad > 0
            if diverged:
                ck["diverged"], ck["n_nonfinite_preds"] = True, int(n_bad)
            rec["checkpoints"].append(ck)
            sel.update(step, mets, p, head)
            loss_acc, n_acc = 0.0, 0
            t0 = time.time()
            t_eval += t0 - t1
            if diverged:
                rec["status"], rec["diverged_at_step"] = "diverged", step
                log(f"[{tag}] DIVERGED at step {step} -> stop")
                break
    rec["selected"] = sel.selected()
    rec["test_at"] = {r: sel.metrics[r] for r in sel.rules}
    preds = sel.preds(fold)
    head.load_state_dict(sel.state_id)
    if job.get("tta"):
        log(f"[{tag}] TTA not applicable to frozen probes; ignored")
    if want_pcam:
        _, yp = ctx.pcam()
        p = prob(Fd["pcam"])[0]
        rec["pcam"] = {"id": C.metrics(yp, p)}
        preds["p_pcam_id"] = p.astype(np.float32)
    rec["wall_s"] = time.time() - t_job
    rec["timing"] = dict(train_s=t_train, eval_s=t_eval, n_steps=n_steps, eval_steps=ev_steps,
                         train_s_per_100=100 * t_train / rec["checkpoints"][-1]["step"],
                         feature_extract_s=fe["extract_s"])
    rec["extra"] = {"n_trainable": n_params(head), "feat_dim": int(Fd["tr"].shape[1])}
    state = {"head": head.state_dict(), "feat_mu": mu.cpu(), "feat_sd": sd.cpu()}
    write_record(ctx, tag, rec, preds, {"head." + k: v for k, v in state["head"].items()} |
                 {"feat_mu": state["feat_mu"], "feat_sd": state["feat_sd"]})
    log(f"[{tag}] DONE sel={rec['selected']} test@id auroc={_auc(rec['test_at']['id']):.4f} "
        f"wall={rec['wall_s']:.1f}s (extract {fe['extract_s']:.1f}s)")
    fe["extract_s"] = 0.0   # count extraction once
    return rec


# ============================================================================ main
def job_tag(job, tier):
    """(spec, hp, hpkey_full, tag). The tag's hpkey is the non-default-hp string, except for
    final-phase jobs whose hyperparameters were selected per fold (make_plan sets
    job['hpkey'] = 'selected'): their tag is fold-independent (the full hp string differs
    between folds) and the actual values are in the record's hparams / hpkey_full."""
    spec = A.resolve_arm(job["arm"], tier)
    hp, hpkey = A.merge_hp(A.default_hp(spec["method"], tier, spec["backbone"], spec["init"]), job.get("hp"))
    label = job.get("hpkey") or hpkey
    return spec, hp, hpkey, f"{job['arm']}__{label}__s{int(job['seed'])}"


def _stale_reason(js, ctx, hp):
    """None if the existing record at `js` can be reused, else why not."""
    try:
        old = json.load(open(js))
    except (OSError, ValueError):
        return "unreadable record"
    if old.get("cache_fingerprint") != ctx.fp:
        return f"cache fingerprint {str(old.get('cache_fingerprint'))[:12]} != {ctx.fp[:12]}"
    oh = old.get("hparams") or {}
    diff = sorted(k for k in hp if json.dumps(oh.get(k)) != json.dumps(hp[k]))
    if diff:
        return f"hyperparameters differ: {diff}"
    return None


def run_job(ctx, job):
    spec, hp, hpkey, tag = job_tag(job, ctx.tier)
    js = os.path.join(ctx.out_dir, tag + ".json")
    if os.path.exists(js) and not ctx.a.force:
        why = _stale_reason(js, ctx, hp)
        if why is None:
            log(f"[{tag}] exists -> skip")
            return None
        log(f"[{tag}] existing record is stale ({why}) -> re-running")
    log(f"[{tag}] start: method={spec['method']} backbone={spec['backbone']} init={spec['init']} hp={hp}")
    if spec["method"] == "probe":
        return run_probe_job(ctx, job, spec, hp, tag, hpkey)
    return run_train_job(ctx, job, spec, hp, tag, hpkey)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=None, help="plan JSON {tier,cohort,fold,phase,jobs:[...]}")
    ap.add_argument("--tier", choices=["C", "S"])
    ap.add_argument("--cohort")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--phase", default=None, choices=["tune", "final", "smoke"])
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hp", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--tta", default="", help="comma list: bnadapt,tent")
    ap.add_argument("--pcam", action="store_true")
    ap.add_argument("--cache_root", default=C.V2)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--ssl_root", default=os.path.join(C.V2, "ssl"))
    ap.add_argument("--feats_root", default=os.path.join(C.V2, "feats"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="re-run even if the JSON exists")
    ap.add_argument("--no_auto_ssl", action="store_true", help="fail instead of pre-training a missing SSL ckpt")
    ap.add_argument("--hp_all", nargs="*", default=[], help="key=value applied to every plan job (e.g. steps=60)")
    return ap.parse_args()


def main():
    a = parse()
    plan = json.load(open(a.plan)) if a.plan else {}
    tier = a.tier or plan["tier"]
    cohort = a.cohort or plan["cohort"]
    fold = a.fold if a.fold is not None else plan["fold"]
    phase = a.phase or plan.get("phase", "final")
    if a.plan:
        jobs = plan["jobs"]
    else:
        hp = dict(kv.split("=", 1) for kv in a.hp)
        jobs = [dict(arm=a.arm, seed=a.seed, hp=hp, tta=[t for t in a.tta.split(",") if t], pcam=a.pcam)]
    if a.hp_all:
        extra = dict(kv.split("=", 1) for kv in a.hp_all)
        for j in jobs:
            j["hp"] = {**(j.get("hp") or {}), **extra}
    ctx = Ctx(a, tier, cohort, fold, phase)
    n_fail = 0
    t0 = time.time()
    for i, job in enumerate(jobs):
        job.setdefault("phase", phase)
        log(f"=== job {i + 1}/{len(jobs)}: {job}")
        try:
            run_job(ctx, job)
        except Exception as e:  # keep going; record the failure
            n_fail += 1
            traceback.print_exc()
            with open(os.path.join(ctx.out_dir, "_failures.jsonl"), "a") as f:
                f.write(json.dumps(dict(job=job, error=repr(e), time=time.time())) + "\n")
        if ctx.dev.type == "cuda":
            torch.cuda.empty_cache()
    log(f"all {len(jobs)} jobs done in {time.time() - t0:.0f}s, failures={n_fail}")
    # Hard exit: every record is already written atomically. Two tier-S shards (9287868_0/1) hung for >2 h in
    # interpreter teardown after this line while holding their GPUs, so skip teardown entirely.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
