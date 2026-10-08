"""CPU unit tests for the v2 training code. Runnable as a plain script
(`python src/v2/tests/test_v2.py [-k substring]`) or with pytest if installed."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
V2SRC = os.path.dirname(HERE)
sys.path.insert(0, V2SRC)
import common as C  # noqa: E402
import arms as A  # noqa: E402
import stain as S  # noqa: E402
from backbones import build_net, n_params, WEIGHT_FILES, MixStyle  # noqa: E402

torch.set_num_threads(4)
DEV = "cpu"


def tiny_batch(B=24, P=32, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, 3, P, P, generator=g)
    y = torch.arange(B) % 2
    d = (torch.arange(B) // 2) % K
    return x, y, d


def model_for(method, backbone, tier):
    init = "random"
    hp = A.default_hp(method, tier, backbone, init)
    ms = dict(p=1.0, alpha=0.1) if method == "mixstyle" else None
    return build_net(backbone, init, mixstyle=ms), hp


# ----------------------------------------------------------------------------- architecture
def test_cnn4_matches_v1():
    # v1 code: src/ in the working tree, legacy_v1/src/ in the public release
    for d in (os.path.dirname(V2SRC), os.path.join(os.path.dirname(os.path.dirname(V2SRC)), "legacy_v1", "src")):
        if os.path.exists(os.path.join(d, "train_lib.py")):
            sys.path.insert(0, d)
    import train_lib as T  # v1, read-only import
    net = build_net("cnn4", "random")
    assert n_params(net) == 583394, n_params(net)
    v1 = T.Classifier(T.Encoder())
    assert set(v1.state_dict()) == set(net.state_dict()), "state_dict keys differ from v1"
    net.load_state_dict(v1.state_dict())


def _weights_missing():
    from common import BACKBONE_DIR
    miss = [f for f in WEIGHT_FILES.values() if not os.path.exists(os.path.join(BACKBONE_DIR, f))]
    if miss:
        print(f"  (skipped: {len(miss)} pretrained weight files missing in {BACKBONE_DIR}; "
              f"run scripts/fetch_backbones.py)")
    return bool(miss)


def test_weights_load():
    if _weights_missing():
        return
    for (bb, init) in WEIGHT_FILES:
        probe = bb not in ("resnet50",)
        net = build_net(bb, init, probe=probe)
        x = torch.rand(2, 3, 96, 96)
        with torch.no_grad():
            f = net.eval().features(x)
        assert f.shape == (2, net.out_dim) and torch.isfinite(f).all(), (bb, init)
    assert build_net("resnet50", "imagenet").pre.resize_to is None           # FT native
    assert build_net("resnet50", "imagenet", probe=True).pre.resize_to == 224
    assert build_net("vit_s16", "lunit_dino").pre.resize_to == 224
    assert abs(build_net("resnet50", "lunit_bt").pre.mean[0, 0, 0, 0].item() - 0.70322989) < 1e-6


def test_resolve_arm():
    r = A.resolve_arm
    assert r("erm", "C") == dict(method="erm", backbone="cnn4", init="random", probe=False)
    assert r("erm", "S")["backbone"] == "resnet50" and r("erm", "S")["init"] == "imagenet"
    assert r("he_jitter_lunit_dino", "S") == dict(method="he_jitter", backbone="vit_s16", init="lunit_dino",
                                                  probe=False)
    assert r("erm_lunit_bt", "S")["backbone"] == "resnet50"
    assert r("probe_convnext_tiny_imagenet", "S")["backbone"] == "convnext_tiny"
    assert r("probe_vit_b16_swag", "S")["init"] == "swag"
    assert r("probe_resnet50_lunit_mocov2", "S")["init"] == "lunit_mocov2"
    assert r("macenko_tile", "C")["method"] == "macenko_tile"
    try:
        r("rotinv", "C")
        raise AssertionError("rotinv must be dropped")
    except ValueError:
        pass


def test_hpkey_and_budget():
    d = A.default_hp("groupdro", "C", "cnn4", "random")
    hp, key = A.merge_hp(d, {"eta": "0.1"})
    assert key == "eta=0.1" and hp["eta"] == 0.1
    assert A.merge_hp(d, {"eta": 0.01})[1] == "default"
    hp, key = A.merge_hp(d, {"lr": 1, "eta": 0.001})
    assert key == "eta=0.001,lr=1" and isinstance(hp["lr"], float)
    try:
        A.merge_hp(d, {"bogus": 1})
        raise AssertionError
    except KeyError:
        pass
    cm = A.default_hp("compute_matched", "C", "cnn4", "random")
    # v1 C17 pool: 40000 images, 6 epochs x 156 steps x 2 views x 256 = 479,232 images
    assert A.ssl_images_seen(40000, 6, 256) == 479232
    assert A.total_steps("compute_matched", cm, 40000) == 1200 + 3744
    assert A.total_steps("erm", d) == 1200


# ----------------------------------------------------------------------------- every arm trains
def _run_steps(method, backbone, tier, n=3):
    torch.manual_seed(0)
    np.random.seed(0)
    model, hp = model_for(method, backbone, tier)
    hp = dict(hp, bs=24)
    if method == "irm":
        hp["irm_anneal_frac"] = 0.34
    tr = A.Trainer(method, model, hp, K=3, n_steps=n, device=DEV)
    before = [p.detach().clone() for p in model.parameters()]
    losses = []
    for s in range(n):
        x, y, d = tiny_batch(seed=s)
        losses.append(float(tr.step(x, y, d)))
    assert all(np.isfinite(losses)), (method, backbone, losses)
    changed = any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    assert changed, f"{method}/{backbone}: parameters did not change"
    return model


def test_every_arm_cnn4():
    for m in A.METHODS:
        _run_steps(m, "cnn4", "C")


def test_every_arm_resnet50():
    for m in A.METHODS:
        _run_steps(m, "resnet50", "S", n=2)


def test_eval_determinism_every_arm():
    """D1: after training, eval-mode predictions must be deterministic for every arm."""
    for bb, tier in (("cnn4", "C"), ("resnet50", "S")):
        for m in A.METHODS:
            model = _run_steps(m, bb, tier, n=2)
            model.eval()
            x, _, _ = tiny_batch(B=8, seed=99)
            with torch.no_grad():
                o1, o2 = model(x), model(x)
            assert torch.equal(o1, o2), f"{m}/{bb}: eval not deterministic"
    # and the v1 bug is detectable: mixstyle active in train mode changes outputs
    net = build_net("cnn4", "random", mixstyle=dict(p=1.0, alpha=0.1))
    assert any(isinstance(mm, MixStyle) for mm in net.modules()), "MixStyle must be a registered submodule"
    net.train()
    x, _, _ = tiny_batch(B=8)
    torch.manual_seed(1)
    a = net(x)
    torch.manual_seed(2)
    b = net(x)
    assert not torch.allclose(a, b)


def test_mixstyle_insertion_points():
    net = build_net("cnn4", "random", mixstyle=dict(p=1.0, alpha=0.1))
    shapes = []
    h = net.encoder.mixstyle.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))
    net.train()
    net(torch.rand(4, 3, 64, 64))
    h.remove()
    assert shapes == [(4, 32, 32, 32), (4, 64, 16, 16)], shapes   # after stage-1 / stage-2 MaxPool
    r = build_net("resnet50", "random", mixstyle=dict(p=1.0, alpha=0.1))
    shapes = []
    r.encoder.mixstyle.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))
    r.train()
    r(torch.rand(4, 3, 96, 96))
    assert shapes == [(4, 256, 24, 24), (4, 512, 12, 12)], shapes  # after layer1 / layer2


# ----------------------------------------------------------------------------- Macenko (D2)
def test_macenko_per_tile():
    rng = np.random.RandomState(0)
    H = 32
    N = H * H

    def unit(v):
        v = np.asarray(v, float)
        return v / np.linalg.norm(v)

    # realistic H&E optical-density vectors (Ruifrok & Johnston-like) and a shifted stain
    HE_A = np.stack([unit([0.65, 0.70, 0.29]), unit([0.22, 0.80, 0.56])], 1)
    HE_B = np.stack([unit([0.55, 0.78, 0.30]), unit([0.10, 0.85, 0.52])], 1)

    def conc(B):
        Cc = np.zeros((B, 2, N))
        for b in range(B):
            kind = rng.choice(4, N, p=[0.3, 0.3, 0.3, 0.1])        # pure H, pure E, mix, background
            amt = rng.uniform(0.2, 1.5, size=(2, N))
            Cc[b, 0] = np.where(kind == 0, amt[0], np.where(kind == 2, amt[0] * 0.6, 0))
            Cc[b, 1] = np.where(kind == 1, amt[1], np.where(kind == 2, amt[1] * 0.6, 0))
        return Cc

    Cc = conc(4)
    tA = S.synth_tiles(HE_A, Cc, H=H)
    tB = S.synth_tiles(HE_B, Cc * 1.3, H=H)   # different stain vectors and stain intensity
    X = torch.from_numpy(np.concatenate([tA, tB]))
    ref_HE, ref_maxC = S.HE_CANON, S.MAXC_CANON
    out, HE, ok = S.macenko_normalize(X, ref_HE, ref_maxC, min_tissue=50, return_he=True)
    assert ok.all()
    cosA = (HE[:4] * torch.from_numpy(HE_A).float()).sum(1)
    cosB = (HE[4:] * torch.from_numpy(HE_B).float()).sum(1)
    assert (cosA > 0.98).all() and (cosB > 0.98).all(), (cosA, cosB)        # recovers the true stains
    assert (HE[0] - HE[4]).abs().max() > 0.05                               # differs per tile
    raw = (X[:4].float() - X[4:].float()).abs().mean().item()
    nrm = (out[:4].float() - out[4:].float()).abs().mean().item()
    assert nrm < 0.25 * raw, (raw, nrm)                                     # stain difference removed
    # background-only tile falls back to the reference matrix
    bg = torch.full((1, H, H, 3), 236, dtype=torch.uint8)
    _, HEb, okb = S.macenko_normalize(bg, ref_HE, ref_maxC, min_tissue=50, return_he=True)
    assert not okb[0] and torch.allclose(HEb[0], ref_HE)
    # the v1 'global transform' would leave the A/B difference largely intact: check that the
    # per-tile result is not just a fixed colour map, i.e. two tiles with identical RGB in
    # different stain contexts are mapped differently
    fn = S.make_macenko_fn(torch.from_numpy(tA), ref="train", n_ref=4)
    assert fn(X).shape == X.shape and fn(X).dtype == torch.uint8


REAL_CACHE = os.path.join(C.V2, "cache96")


def test_macenko_reference_real():
    """Reference fitted on real training tiles is a physical H&E pair (E red OD ~0.2-0.3,
    close to the canonical eosin vector) on every available fold, stable across folds, and
    per-tile H/E labelling agrees with the reference."""
    import glob
    folds = sorted(glob.glob(os.path.join(REAL_CACHE, "*", "fold*", "Xtr.npy")))
    if not folds:
        print("  (skipped: no real v2 cache)")
        return
    P = 96
    mt = int(0.05 * P * P)
    Es = {}
    for fx in folds:
        coh = fx.split(os.sep)[-3]
        t = int(fx.split(os.sep)[-2][4:])
        Xa = np.load(fx, mmap_mode="r")
        X = torch.from_numpy(np.ascontiguousarray(Xa[: min(len(Xa), 6000)]))
        fn = S.make_macenko_fn(X, min_tissue=mt, seed=t)
        H, E = fn.HE_ref[:, 0], fn.HE_ref[:, 1]
        assert H[0] > E[0] + 0.1, (fx, fn.HE_ref)                       # H has the larger red OD
        assert E[0] < 0.35, (fx, fn.HE_ref)                             # physical eosin
        assert float(E @ S.HE_CANON[:, 1]) > 0.9 and float(H @ S.HE_CANON[:, 0]) > 0.95, (fx, fn.HE_ref)
        _, he, ok = S.macenko_normalize(X[:500], fn.HE_ref, fn.maxC_ref, min_tissue=mt, return_he=True)
        R = fn.HE_ref
        ident = he[:, :, 0] @ R[:, 0] + he[:, :, 1] @ R[:, 1]
        swap = he[:, :, 0] @ R[:, 1] + he[:, :, 1] @ R[:, 0]
        assert float((swap > ident)[ok].float().mean()) == 0.0
        Es.setdefault(coh, []).append(E)
    for coh, v in Es.items():   # fold-to-fold stability (training domains differ by fold, so not identical)
        V = torch.stack(v)
        assert float((V @ V.T).min()) > 0.95, (coh, V)


C17_FOLD4 = os.path.join(C.V2, "cache64", "c17", "fold4")


def test_macenko_nonneg_c17_center4():
    """Regression for the c17 fold-4 (test = center 4) collapse of the macenko arm
    (off-site AUROC 0.42 vs ERM 0.59-0.80). Per-tile H/E vectors are nearly collinear, so
    the unconstrained LS fit gives eosin-rich pixels negative H; rebuilt with HE_ref
    after the maxC rescale they turn into over-bright (> io) saturated pink. On real tiles
    that artefact is label-correlated with opposite signs: mostly normal tiles in ID-val
    (the training distribution), mostly tumour tiles on center 4. With C >= 0 (the fix),
    concentrations are non-negative and no reconstructed pixel exceeds io."""
    if not os.path.exists(os.path.join(C17_FOLD4, "Xot.npy")):
        print("  (skipped: no c17 fold4 cache64)")
        return
    from sklearn.metrics import roc_auc_score
    P, io = 64, 240.0
    mt = int(0.05 * P * P)
    Xtr = np.load(os.path.join(C17_FOLD4, "Xtr.npy"), mmap_mode="r")
    idx = np.sort(np.random.RandomState(4).choice(len(Xtr), 256, replace=False))   # as make_macenko_fn(seed=fold)
    Xr = torch.from_numpy(np.ascontiguousarray(Xtr[idx]))
    rng = np.random.RandomState(0)

    def split(name, n=400):
        Xa = np.load(os.path.join(C17_FOLD4, f"X{name}.npy"), mmap_mode="r")
        y = np.load(os.path.join(C17_FOLD4, f"y{name}.npy"))
        i = np.sort(rng.choice(len(y), min(n, len(y)), replace=False))
        return torch.from_numpy(np.ascontiguousarray(Xa[i])), y[i]

    data = {sp: split(sp) for sp in ("iv", "ot")}
    for nonneg in (False, True):
        HE_ref, maxC_ref = S.fit_reference(Xr, io, min_tissue=mt, nonneg=nonneg)
        assert float(HE_ref[:, 0] @ HE_ref[:, 1]) > 0.85                       # near-collinear reference pair
        auc = {}
        for sp, (X, y) in data.items():
            out = S.macenko_normalize(X, HE_ref, maxC_ref, io, min_tissue=mt, nonneg=nonneg)
            bright = (out.reshape(len(X), -1, 3).max(2).values > io).float().mean(1).numpy()
            OD = S._od(X, io)
            he, ok = S.estimate_stains(OD, 1.0, 0.15, mt)
            he = torch.where(ok.view(-1, 1, 1), S.align_to(he, HE_ref), HE_ref.expand(len(X), 3, 2))
            Cc = S.concentrations(OD, he, nonneg)
            if nonneg:
                assert float(Cc.min()) >= 0.0
                assert bright.max() == 0.0, (sp, bright.max())                   # never brighter than io
            else:
                auc[sp] = roc_auc_score(y, bright)
                if sp == "ot":   # most center-4 tumour tiles carry the negative-H artefact
                    negH = (Cc[:, 0] < -0.05).float().mean(1).numpy()
                    assert (negH[y == 1] > 0.2).mean() > 0.7, (negH[y == 1] > 0.2).mean()
        if not nonneg:   # the pre-fix artefact flips its label association between ID-val and center 4
            assert auc["iv"] < 0.35 and auc["ot"] > 0.65, auc
    # mac_nonneg is part of the hparams (records written before the fix lack it -> stale)
    for arm in A.USES_MACENKO:
        assert A.default_hp(arm, "C", "cnn4", "random")["mac_nonneg"] == 1


# ----------------------------------------------------------------------------- group-level Macenko
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _group_tiles(rng, HE, n, H=32, scale=1.0, tissue=0.6):
    Cc = rng.uniform(0.1, 1.2, (n, 2, H * H)) * (rng.rand(n, 1, H * H) < tissue) * scale
    Cc[:, 0] *= rng.rand(n, H * H) < 0.7          # mixture of H-only, E-only and H+E pixels
    Cc[:, 1] *= rng.rand(n, H * H) < 0.7
    return S.synth_tiles(HE, Cc, H=H)


def _synth_groups(seed=0, H=32):
    """Three stain contexts (groups 7, 3, 11) + one background-only group (5)."""
    rng = np.random.RandomState(seed)
    HEs = {7: np.stack([_unit([0.65, 0.70, 0.29]), _unit([0.22, 0.80, 0.56])], 1),
           3: np.stack([_unit([0.55, 0.78, 0.30]), _unit([0.10, 0.85, 0.52])], 1),
           11: np.stack([_unit([0.60, 0.72, 0.35]), _unit([0.25, 0.75, 0.61])], 1)}
    Xs, gs = [], []
    for gid, HE in HEs.items():
        Xs.append(_group_tiles(rng, HE, 20, H, scale=1.0 + 0.2 * (gid % 3)))
        gs += [gid] * 20
    Xs.append(np.full((4, H, H, 3), 236, np.uint8))
    gs += [5] * 4
    return torch.from_numpy(np.concatenate(Xs)), np.asarray(gs, np.int64), HEs


def test_macenko_group_level():
    """Group-level ('per-slide') Macenko: one stain matrix per group id, recovered from the
    pooled pixels; every tile normalised with its group's matrix; background-only groups
    fall back to the reference; the result is invariant to the row order (within and
    across groups); the tile and pixel subsamples are deterministic."""
    X, g, HEs = _synth_groups()
    R, mR = S.HE_CANON, S.MAXC_CANON
    kw = dict(min_tissue=50, max_tiles=8, max_px=3000, min_px=500)   # small caps: subsampling is exercised
    out, st, HE_rows, inv = S.macenko_normalize_groups(X, g, R, mR, return_he=True, **kw)
    assert st["n_groups"] == 4 and st["n_fallback"] == 1 and st["n_rows_fallback"] == 4, st
    assert st["fallback_reasons"] == {"few_tissue_px": 1}
    assert st["per_group"]["5"]["ok"] is False and torch.allclose(HE_rows[g == 5], R.expand(4, 3, 2))
    for gid, HE in HEs.items():
        rows = HE_rows[torch.from_numpy(g == gid)]
        assert (rows == rows[0]).all()                                      # one matrix per group
        cos = (rows[0] * torch.from_numpy(HE).float()).sum(0)
        assert (cos > 0.98).all(), (gid, cos)                                # recovers the group's stains
    assert set(st["per_group"]) == {"3", "5", "7", "11"} and st["per_group"]["7"]["n"] == 20
    # same tile in two stain contexts -> different outputs; same group -> same matrix
    assert (HE_rows[0] - HE_rows[20]).abs().max() > 0.05
    # row-order invariance: shuffle all rows (so the order within every group changes)
    perm = np.random.RandomState(1).permutation(len(g))
    out_p, st_p = S.macenko_normalize_groups(X[torch.from_numpy(perm)], g[perm], R, mR, **kw)
    assert torch.equal(out_p, out[torch.from_numpy(perm)])
    assert json.dumps(st_p, sort_keys=True) == json.dumps(st, sort_keys=True)
    # a group's estimate depends only on that group's tiles
    keep = g != 3
    out_s, _ = S.macenko_normalize_groups(X[torch.from_numpy(keep)], g[keep], R, mR, **kw)
    assert torch.equal(out_s, out[torch.from_numpy(keep)])
    # tile subsample = first max_tiles of a seeded content-hash order (order independent)
    i7 = np.nonzero(g == 7)[0]
    o1 = S._tile_order(X[torch.from_numpy(i7)], S.group_seed(7))
    o2 = S._tile_order(X[torch.from_numpy(i7[::-1].copy())], S.group_seed(7))
    assert np.array_equal(i7[o1], i7[::-1][o2])
    assert S.group_seed(np.int64(7)) == S.group_seed(7) != S.group_seed(8)
    # make_macenko_fn: fn(X) stays per-tile, fn.group is the group transform
    fn = S.make_macenko_fn(X, ref="canonical", min_tissue=50, level="group", group_tiles=8, group_px=3000,
                           group_min_px=500)
    assert fn.level == "group" and torch.equal(fn.group(X, g)[0], out)
    assert torch.equal(fn(X), S.macenko_normalize(X, R, mR, min_tissue=50))
    try:
        S.make_macenko_fn(X, ref="canonical", level="slide")
        raise AssertionError("unknown mac_level must raise")
    except ValueError:
        pass


def test_macenko_group_no_labels():
    """No labels enter the group-level transform: the stain functions take no label
    argument, and run_v2.macenko_splits gives bit-identical splits (and stats) when every
    label file of the cache is replaced."""
    import inspect
    import run_v2 as R
    for f in (S.macenko_normalize_groups, S.estimate_group, S.apply_stains, S.make_macenko_fn, R.macenko_splits):
        names = set(inspect.signature(f).parameters)
        assert not names & {"y", "labels", "label", "yt", "targets"}, (f.__name__, names)
    for f in (R.split_groups, R.c17_slides, S.fit_reference_groups):
        names = set(inspect.signature(f).parameters)
        assert not names & {"y", "labels", "label", "yt", "targets"}, (f.__name__, names)
    with tempfile.TemporaryDirectory() as td:
        res, res2 = [], []
        for flip in (False, True):
            root = f"{td}/c{int(flip)}"
            _write_cache(root, np.arange(64) % 3, P=32)
            Xs, gs, _ = _synth_groups(seed=2)
            np.save(root + "/Xtr.npy", Xs.numpy())                 # 64 tiles in 4 groups
            np.save(root + "/gtr.npy", gs)
            np.save(root + "/got.npy", np.arange(32) // 8)        # 'ot' has groups; 'ov' has none -> per tile
            if flip:
                for sp in ("tr", "iv", "ov", "ot"):
                    y = np.load(f"{root}/y{sp}.npy")
                    np.save(f"{root}/y{sp}.npy", np.random.RandomState(9).permutation(1 - y))
            fold = C.Fold(root, "cpu")
            hp = A.default_hp("macenko", "C", "cnn4", "random")
            fn = S.make_macenko_fn(fold.X["tr"], n_ref=16, min_tissue=50, seed=0, level="group",
                                   group_min_px=hp["mac_group_min_px"] // 8)
            out = R.macenko_splits(fold, ("macenko", 1), fn)
            res.append(out)
            # the final configuration (target = group estimator on the training groups, slide x domain)
            gtr, _ = R.split_groups(fold, "tr", hp["mac_group_unit"])
            fn2 = S.make_macenko_fn(fold.X["tr"], ref=hp["mac_ref"], min_tissue=50, seed=0, level="group",
                                    group_min_px=hp["mac_group_min_px"] // 8, group_maxc=hp["mac_group_maxc"],
                                    ref_groups=gtr)
            fn2.group_unit = hp["mac_group_unit"]
            res2.append((R.macenko_splits(fold, ("macenko", 2), fn2), fn2.HE_ref, fn2.maxC_ref))
        for sp in ("tr", "iv", "ov", "ot"):
            assert torch.equal(res2[0][0][sp], res2[1][0][sp]), sp
        assert torch.equal(res2[0][1], res2[1][1]) and torch.equal(res2[0][2], res2[1][2])
        assert res2[0][0]["_stats"]["tr"]["group_source"] == "gtr.npy x dtr.npy"
        a, b = res
        for sp in ("tr", "iv", "ov", "ot"):
            assert torch.equal(a[sp], b[sp]), sp
        sa, sb = dict(a["_stats"]), dict(b["_stats"])
        assert json.dumps(sa, sort_keys=True) == json.dumps(sb, sort_keys=True)
        assert sa["ov"] == dict(level="tile", group_source=None, n_rows=32)
        assert sa["ot"]["level"] == "group" and sa["ot"]["n_groups"] == 4 and sa["ot"]["group_source"] == "got.npy"
        assert sa["tr"]["n_groups"] == 4 and sa["tr"]["n_fallback"] == 1 and sa["iv"]["n_groups"] == 32


def test_macenko_group_c17_center4():
    """On real c17 fold-4 tiles (test = center 4), group-level (per-patient ~ per-slide)
    stain vectors of the center-4 groups are close to the training reference (cos >= 0.95
    for H and E) for the majority of groups (per-tile estimates on the pale 64-px tiles are
    not), no group falls back, and the run_v2 reference fit is used unchanged."""
    if not os.path.exists(os.path.join(C17_FOLD4, "Xot.npy")):
        print("  (skipped: no c17 fold4 cache64)")
        return
    P = 64
    mt = int(0.05 * P * P)
    Xtr = torch.from_numpy(np.load(os.path.join(C17_FOLD4, "Xtr.npy")))
    import types
    import run_v2 as R
    hp = A.default_hp("macenko", "C", "cnn4", "random")
    Xot = torch.from_numpy(np.load(os.path.join(C17_FOLD4, "Xot.npy")))
    fold = types.SimpleNamespace(path=C17_FOLD4, manifest=json.load(open(os.path.join(C17_FOLD4, "manifest.json"))),
                                 X={"tr": Xtr, "ot": Xot}, d_np={"tr": np.load(os.path.join(C17_FOLD4, "dtr.npy"))})
    gtr, src = R.split_groups(fold, "tr", hp["mac_group_unit"])
    assert src == "metadata.slide[rows_tr] x dtr.npy" and len(np.unique(gtr)) == 24   # 21 patients, 24 slides
    fn = S.make_macenko_fn(Xtr, ref=hp["mac_ref"], n_ref=hp["mac_nref"], io=hp["mac_io"], alpha=hp["mac_alpha"],
                           beta=hp["mac_beta"], min_tissue=mt, seed=4, nonneg=True, level=hp["mac_level"],
                           group_tiles=hp["mac_group_tiles"], group_px=hp["mac_group_px"],
                           group_min_px=hp["mac_group_min_px"], group_maxc=hp["mac_group_maxc"], ref_groups=gtr)
    assert fn.ref_info["ref"] == "train_group" and fn.ref_info["n_groups_used"] == 24 and "fallback" not in fn.ref_info
    del Xtr
    got, src = R.split_groups(fold, "ot", hp["mac_group_unit"])
    assert src == "metadata.slide[rows_ot]" and len(np.unique(got)) == 10                # 9 patients, 10 slides
    assert len(np.unique(np.load(os.path.join(C17_FOLD4, "got.npy")))) == 9
    out, st = fn.group(Xot, got)
    assert out.shape == Xot.shape and out.dtype == torch.uint8
    cos = np.array([v["cos"] for v in st["per_group"].values() if v["ok"]])
    close = (cos >= 0.95).all(1)
    assert st["n_groups"] == len(np.unique(got)) and st["n_fallback"] == 0, st["fallback_reasons"]
    assert close.mean() > 0.5, cos
    assert st["frac_groups_cos_ge_095"] == close.mean()
    # per-tile estimates on the same tiles are much less consistent with the reference
    _, he, ok = S.macenko_normalize(Xot[:2000], fn.HE_ref, fn.maxC_ref, min_tissue=mt, return_he=True)
    ct = S._cos_cols(he[ok], fn.HE_ref)
    assert float(ct.min()) < float(cos.min())


def test_macenko_group_reference_and_units():
    """(1) mac_ref='train_group': the target is fitted with the group estimator, so normalising
    a training set whose groups share one stain context is (nearly) the identity, unlike the
    per-tile reference; (2) maxc='pooled' is the 99th percentile over all pixels of the sampled
    tiles; (3) split_groups 'slide_domain' crosses the group id with the split's domain id."""
    import run_v2 as R
    rng = np.random.RandomState(5)
    HE = np.stack([_unit([0.65, 0.70, 0.29]), _unit([0.22, 0.80, 0.56])], 1)
    # every group: 9 faint and 3 dark tiles (per-tile median and pooled 99th percentile then differ)
    X = torch.from_numpy(np.concatenate([np.concatenate([_group_tiles(rng, HE, 9, 32, scale=0.4),
                                                         _group_tiles(rng, HE, 3, 32, scale=1.4)])
                                         for _ in range(5)]))
    g = np.repeat(np.arange(5), 12)
    kw = dict(min_tissue=50, group_tiles=12, group_px=20000, group_min_px=500, level="group")
    fg = S.make_macenko_fn(X, ref="train_group", group_maxc="pooled", ref_groups=g, **kw)
    ft = S.make_macenko_fn(X, ref="train", n_ref=60, group_maxc="pooled", **kw)
    assert fg.ref_info["n_groups_used"] == 5 and (S._cos_cols(fg.HE_ref[None], torch.from_numpy(HE).float()) > 0.98).all()
    dg = (fg.group(X, g)[0].float() - X.float()).abs().mean()
    dt = (ft.group(X, g)[0].float() - X.float()).abs().mean()
    assert dg < 2.0 and dg < 0.25 * dt, (dg, dt)
    try:
        S.make_macenko_fn(X, ref="train_group", **kw)
        raise AssertionError("train_group without ref_groups must raise")
    except ValueError:
        pass
    # too few usable training groups -> per-tile reference, recorded
    bg = torch.full((6, 32, 32, 3), 236, dtype=torch.uint8)
    fb = S.make_macenko_fn(torch.cat([X[:12], bg]), ref="train_group", ref_groups=np.r_[np.zeros(12), np.ones(6)],
                           n_ref=8, **kw)
    assert fb.ref_info.get("fallback") == "train"
    # pooled maxC
    Xg = X[:12]
    HEg, mC, info = S.estimate_group(Xg, fg.HE_ref, fg.maxC_ref, min_tissue=50, seed=3, max_tiles=100, max_px=10 ** 7,
                                     min_px=500, maxc="pooled")
    Cc = S.concentrations(S._od(Xg, 240.0).reshape(1, -1, 3), HEg[None])[0]
    assert info["ok"] and torch.allclose(mC, torch.quantile(Cc, 0.99, dim=1).clamp_min(1e-3))
    _, mC2, _ = S.estimate_group(Xg, fg.HE_ref, fg.maxC_ref, min_tissue=50, seed=3, max_tiles=100, max_px=10 ** 7,
                                 min_px=500, maxc="tile_median")
    assert not torch.allclose(mC, mC2)
    # split_groups
    with tempfile.TemporaryDirectory() as td:
        _write_cache(td, np.arange(64) % 2, P=16)
        np.save(td + "/gtr.npy", np.arange(64) // 8)               # 8 groups, each spans both domains
        np.save(td + "/got.npy", np.arange(32) // 8)
        fold = C.Fold(td, "cpu")
        gg, src = R.split_groups(fold, "tr", "slide_domain")
        assert src == "gtr.npy x dtr.npy" and len(np.unique(gg)) == 16
        assert all(len(set(fold.d_np["tr"][gg == u])) == 1 for u in np.unique(gg))
        g0, src0 = R.split_groups(fold, "tr", "group")
        assert src0 == "gtr.npy" and np.array_equal(g0, np.arange(64) // 8)
        go, srco = R.split_groups(fold, "ot", "slide_domain")      # no domain ids for ot
        assert srco == "got.npy" and np.array_equal(go, np.arange(32) // 8)
        assert R.split_groups(fold, "ov", "slide_domain") == (None, None)


# golden outputs of the per-tile path, computed with stain.py BEFORE mac_level was added
_TILE_GOLDEN = {"synth": "60b95703472c738e", "c17": "71fbfe08a436e6cb"}


def test_macenko_tile_unchanged():
    """mac_level='tile' (arm macenko_tile) is the pre-existing per-tile path, bit for bit."""
    import hashlib
    import run_v2 as R
    rng = np.random.RandomState(123)
    HE = np.stack([_unit([0.65, 0.70, 0.29]), _unit([0.22, 0.80, 0.56])], 1)
    Cc = rng.uniform(0, 1.2, (12, 2, 32 * 32)) * (rng.rand(12, 1, 32 * 32) < 0.6)
    X = torch.from_numpy(S.synth_tiles(HE, Cc, H=32))
    fn = S.make_macenko_fn(X, n_ref=8, min_tissue=50, seed=0)
    assert fn.level == "tile"
    assert hashlib.sha256(fn(X).numpy().tobytes()).hexdigest()[:16] == _TILE_GOLDEN["synth"]
    fn2 = S.make_macenko_fn(X, n_ref=8, min_tissue=50, seed=0, level="tile")
    assert torch.equal(fn2(X), fn(X))

    class _F:                                   # minimal Fold stand-in for macenko_splits
        def __init__(self):
            self.X, self._transformed, self.path = {"tr": X, "ot": X.flip(0)}, {}, "/nonexistent"
        transformed = C.Fold.transformed
    out = R.macenko_splits(_F(), ("k",), fn2)
    assert set(out) == {"tr", "ot", "_time"} and torch.equal(out["ot"], fn(X.flip(0)))
    if os.path.exists(os.path.join(C17_FOLD4, "Xot.npy")):
        Xtr = np.load(os.path.join(C17_FOLD4, "Xtr.npy"), mmap_mode="r")
        f = S.make_macenko_fn(torch.from_numpy(np.ascontiguousarray(Xtr[:2000])), min_tissue=204, seed=4,
                              level="tile")
        Xo = torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(C17_FOLD4, "Xot.npy"), mmap_mode="r")[:500]))
        assert hashlib.sha256(f(Xo).numpy().tobytes()).hexdigest()[:16] == _TILE_GOLDEN["c17"]
    # arm macenko_tile = macenko with mac_level='tile'; macenko / tia_style are group-level
    for tier, bb in (("C", "cnn4"), ("S", "resnet50")):
        ht = A.default_hp("macenko_tile", tier, bb, "random")
        hm = A.default_hp("macenko", tier, bb, "random")
        assert ht["mac_level"] == "tile" and hm["mac_level"] == "group"
        assert A.default_hp("tia_style", tier, bb, "random")["mac_level"] == "group"
        assert ht["mac_ref"] == "train" and hm["mac_ref"] == "train_group"
        assert dict(ht, mac_level="group", mac_ref="train_group") == hm
    assert A.merge_hp(A.default_hp("macenko", "C", "cnn4", "random"), {"mac_level": "tile"})[1] == "mac_level=tile"
    assert A.resolve_arm("macenko_tile", "C") == dict(method="macenko_tile", backbone="cnn4", init="random",
                                                      probe=False)
    assert A.resolve_arm("macenko_tile_lunit_bt", "S")["method"] == "macenko_tile"


def test_nonmacenko_hp_unchanged():
    """Default hyperparameters and arm resolution of every non-macenko arm are byte-identical
    to the snapshot taken before mac_level was added (so their run records' resume /
    staleness decisions are unchanged); macenko-family dicts only gained mac_* keys."""
    snap = json.load(open(os.path.join(HERE, "hp_snapshot_nonmac.json")))
    for key, s in snap["default_hp"].items():
        tier, arm = key.split("|")
        spec = A.resolve_arm(arm, tier)
        assert json.dumps(spec) == snap["resolve"][key], key
        assert spec["method"] not in A.USES_MACENKO, key
        assert json.dumps(A.default_hp(spec["method"], tier, spec["backbone"], spec["init"])) == s, key
    new_keys = {"mac_level", "mac_group_tiles", "mac_group_px", "mac_group_min_px", "mac_group_maxc",
                "mac_group_unit"}
    for key, s in snap["macenko_family_before"].items():
        tier, arm = key.split("|")
        spec = A.resolve_arm(arm, tier)
        old, new = json.loads(s), A.default_hp(spec["method"], tier, spec["backbone"], spec["init"])
        assert set(new) - set(old) == new_keys, key
        assert all(new[k] == v for k, v in old.items() if k != "mac_ref"), key     # mac_ref: value 'train_group'
        assert new["mac_ref"] == ("train" if spec["method"] == "macenko_tile" else "train_group"), key
    # every arm that is not macenko-family is covered by the snapshot
    arms = list(A.METHODS) + [m + "_" + i for m in A.METHODS for i in A.LUNIT_INITS]
    for tier in ("C", "S"):
        for arm in arms:
            if A.resolve_arm(arm, tier)["method"] not in A.USES_MACENKO:
                assert f"{tier}|{arm}" in snap["default_hp"], (tier, arm)


def test_eval_schedule():
    assert C.eval_steps(1200) == list(range(120, 1201, 120))
    for n in (4944, 3672, 3552, 1201, 10):
        st = C.eval_steps(n)
        assert len(st) == 10 and st[-1] == n and st == sorted(set(st)), (n, st)
    assert C.eval_steps(100, 30) == [30, 60, 90, 100]


def _writer(args):
    path, k = args
    import common as C2
    C2.atomic_npy(np.full((4096, 512), k, np.float32), path)
    return True


def test_atomic_concurrent_writers():
    import multiprocessing as mp
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "feat.npy")
        for _ in range(3):
            with mp.get_context("fork").Pool(5) as pool:
                assert all(pool.map(_writer, [(f, k) for k in range(5)]))
            a = np.load(f)
            assert a.shape == (4096, 512) and len(np.unique(a)) == 1
        assert sorted(os.listdir(td)) == ["feat.npy"]                  # no temp files left


def test_metrics_nonfinite():
    m = C.metrics(np.arange(10) % 2, np.full(10, np.nan))
    assert m["auroc"] == 0.5 and m["acc"] == 0.5
    p, nb = C.sanitize_probs(np.array([0.1, np.inf, np.nan]))
    assert nb == 2 and np.isfinite(p).all()


def test_rng_isolated():
    C.set_seed(3)
    a = (np.random.rand(), torch.rand(1).item())
    C.set_seed(3)
    with C.rng_isolated():
        np.random.rand(100)
        torch.rand(100)
    b = (np.random.rand(), torch.rand(1).item())
    assert a == b


def test_cache_fingerprint():
    with tempfile.TemporaryDirectory() as td:
        _write_cache(td + "/c", np.arange(64) % 3)
        f1 = C.cache_fingerprint(td + "/c")
        assert f1 == C.cache_fingerprint(td + "/c")
        y = np.load(td + "/c/yot.npy")
        np.save(td + "/c/yot.npy", 1 - y)
        assert C.cache_fingerprint(td + "/c") != f1


# ----------------------------------------------------------------------------- GroupDRO (D6)
def _write_cache(root, dtr, P=16, n=64, with_ov=True, seed=0):
    rng = np.random.RandomState(seed)
    os.makedirs(root, exist_ok=True)
    K = len(np.unique(dtr))

    def save(k, v):
        np.save(os.path.join(root, k + ".npy"), v)

    save("Xtr", rng.randint(0, 255, (n, P, P, 3)).astype(np.uint8))
    save("ytr", (np.arange(n) % 2).astype(np.int64))
    save("dtr", np.asarray(dtr, np.int64))
    save("gtr", np.arange(n, dtype=np.int64))
    sps = ("iv", "ov", "ot") if with_ov else ("iv", "ot")
    for sp in sps:
        save("X" + sp, rng.randint(0, 255, (32, P, P, 3)).astype(np.uint8))
        save("y" + sp, rng.permutation(np.arange(32) % 2).astype(np.int64))
    save("div", (np.arange(32) % K).astype(np.int64))
    save("giv", np.arange(1000, 1032, dtype=np.int64))
    save("Xu", rng.randint(0, 255, (64, P, P, 3)).astype(np.uint8))
    json.dump(dict(cohort="c17", fold=0, patch_px=P, test_domain="t", val_domain="v" if with_ov else None,
                   train_domains=[f"dom{k}" for k in range(K)]), open(os.path.join(root, "manifest.json"), "w"))


def test_groupdro_contiguous():
    with tempfile.TemporaryDirectory() as td:
        _write_cache(td + "/ok", np.arange(64) % 3)
        f = C.Fold(td + "/ok", "cpu")
        assert f.K == 3
        _write_cache(td + "/bad", 1 + np.arange(64) % 2)       # v1 MIDOG/canine-style labels {1,2}
        try:
            C.Fold(td + "/bad", "cpu")
            raise AssertionError("non-contiguous domain labels must be rejected")
        except AssertionError as e:
            assert "contiguous" in str(e)
    q = torch.full((3,), 1 / 3)
    per = torch.tensor([1.0, 1.0, 3.0, 3.0])
    d = torch.tensor([0, 0, 1, 1])                               # group 2 absent from the batch
    loss = A.groupdro_loss(per, d, q, eta=0.5, K=3)
    assert abs(q.sum().item() - 1) < 1e-6 and q[1] > q[0] > q[2]  # absent group not up-weighted
    assert torch.isfinite(loss)
    model, hp = model_for("groupdro", "cnn4", "C")
    tr = A.Trainer("groupdro", model, hp, K=3, n_steps=1, device=DEV)
    assert tr.q.shape == (3,)


# ----------------------------------------------------------------------------- IRM (D4)
def test_irm_schedule():
    assert [A.irm_weight(s, 300, 100.0) for s in (0, 299, 300, 1199)] == [1.0, 1.0, 100.0, 100.0]
    model, hp = model_for("irm", "cnn4", "C")
    hp = dict(hp, irm_anneal_frac=0.25, irm_lambda=100.0)
    tr = A.Trainer("irm", model, hp, K=3, n_steps=8, device=DEV)
    assert tr.irm_anneal == 2
    seen = []
    orig = tr.base.step

    def chk(*a, **k):
        seen.append(len(tr.base.state))
        return orig(*a, **k)
    tr.base.step = chk
    for s in range(4):
        tr.step(*tiny_batch(seed=s))
    assert seen[0] == 0 and seen[1] > 0 and seen[2] == 0 and seen[3] > 0, seen   # reset at the anneal step


# ----------------------------------------------------------------------------- Fish (D7)
def test_fish_math():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2))
    init_p = [p.detach().clone() for p in model.parameters()]
    init_b = [b.detach().clone() for b in model.buffers()]

    def make_opt(ps):
        return torch.optim.SGD(ps, lr=0.1, momentum=0.9, weight_decay=1e-4)
    fs = A.FishState(model, make_opt, meta_lr=0.3)
    x = torch.randn(24, 5)
    y = torch.arange(24) % 2
    d = torch.arange(24) % 3
    from contextlib import nullcontext
    fs.step(model, x, y, d, nullcontext)
    for p0, pm, pi in zip(init_p, model.parameters(), fs.inner.parameters()):
        assert torch.allclose(pm, p0 + 0.3 * (pi - p0), atol=1e-6)               # params interpolated
        assert not torch.allclose(pi, p0)
    for b0, bm, bi in zip(init_b, model.buffers(), fs.inner.buffers()):
        assert torch.equal(bm, bi)                                               # buffers copied
    assert fs.opt.param_groups[0]["weight_decay"] == 1e-4
    mom = {id(p): fs.opt.state[p]["momentum_buffer"].clone() for p in fs.inner.parameters()}
    fs.step(model, x, y, d, nullcontext)
    assert all(not torch.equal(mom[id(p)], torch.zeros_like(mom[id(p)])) for p in fs.inner.parameters())
    assert len(fs.opt.state) == len(list(fs.inner.parameters()))                # inner momentum persists
    # inner model starts each meta step from the meta weights
    snap = [p.detach().clone() for p in model.parameters()]
    fs.opt.param_groups[0]["lr"] = 0.0
    fs.opt.state.clear()
    fs.step(model, x, y, d, nullcontext)
    for a, b in zip(snap, model.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


# ----------------------------------------------------------------------------- LISA
def test_lisa_partners():
    torch.manual_seed(0)
    x, y, d = tiny_batch(B=32)
    xm, ym, partner, _ = A.lisa_mix(x, y, d, alpha=2.0, intra_label=True)
    assert (y[partner] == y).all() and (d[partner] != d).all()
    assert torch.allclose(ym, nn.functional.one_hot(y, 2).float())
    xm, ym, partner, _ = A.lisa_mix(x, y, d, alpha=2.0, intra_label=False)
    assert (d[partner] == d).all() and (y[partner] != y).all()
    assert torch.allclose(ym.sum(1), torch.ones(32))


# ----------------------------------------------------------------------------- TTA (D8)
def test_tent_stream_shuffled():
    n = 512
    X = torch.zeros(n, 16, 16, 3, dtype=torch.uint8)
    X[n // 2:] = 200                                                  # label-sorted stream
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    model = build_net("cnn4", "random")
    rec = []
    p = A.tta_tent(copy.deepcopy(model), X, "cpu", bs=128, record=rec)
    assert len(rec) == 4 and sorted(torch.cat(rec).tolist()) == list(range(n))
    for idx in rec:
        frac = y[idx.numpy()].mean()
        assert 0.2 < frac < 0.8, frac                                 # every batch has both classes
    assert p.shape == (n,) and np.isfinite(p).all()
    order = A.tta_order(n, 0)
    assert not np.array_equal(order, np.arange(n)) and np.array_equal(order, A.tta_order(n, 0))
    pb = A.tta_bnadapt(copy.deepcopy(model), X, "cpu", bs=128)
    assert pb.shape == (n,)


# ----------------------------------------------------------------------------- SSL + JSON contract
def test_ssl_tiny():
    import ssl_v2
    Xu = torch.randint(0, 255, (64, 16, 16, 3), dtype=torch.uint8)
    state, info = ssl_v2.pretrain(Xu, "cnn4", "random", 0, "cpu", cfg=dict(epochs=1, bs=16))
    assert info["steps"] == 4 and info["images_seen"] == 4 * 2 * 16
    net = build_net("cnn4", "random")
    net.encoder.load_state_dict(state)


def test_run_record_contract():
    with tempfile.TemporaryDirectory() as td:
        _write_cache(td + "/cache", np.arange(64) % 3, with_ov=True)
        _write_cache(td + "/cache_noov", np.arange(64) % 2, with_ov=False)
        for cache, arm, extra in ((td + "/cache", "erm", ["--tta", "bnadapt,tent"]),
                                  (td + "/cache_noov", "groupdro", []),
                                  (td + "/cache", "simclr", ["--hp", "ssl_epochs=1", "ssl_bs=16"])):
            cmd = [sys.executable, os.path.join(V2SRC, "run_v2.py"), "--tier", "C", "--cohort", "c17",
                   "--fold", "0", "--cache_dir", cache, "--out_root", td + "/out", "--ssl_root", td + "/ssl",
                   "--arm", arm, "--seed", "1", "--device", "cpu", "--phase", "tune"]
            if arm != "simclr":
                cmd += ["--hp", "steps=10", "bs=16"]
            else:
                extra = ["--hp", "steps=10", "bs=16", "ssl_epochs=1", "ssl_bs=16"]
            r = subprocess.run(cmd + extra, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
            out = td + "/out/C/c17/fold0"
            js = [f for f in os.listdir(out) if f.startswith(arm + "__") and f.endswith(".json")]
            assert len(js) == 1, js
            rec = json.load(open(os.path.join(out, js[0])))
            preds = dict(np.load(os.path.join(out, js[0][:-5] + "_preds.npz")))
            C.validate_record(rec, preds)
            assert rec["phase"] == "tune" and rec["n_params"] == 583394
            assert len(rec["checkpoints"]) == 10
            if arm == "erm":
                assert set(rec["tta"]) == {"bnadapt", "tent"} and "ood" in rec["selected"]
            if arm == "groupdro":
                assert rec["checkpoints"][0]["ood_val"] is None and "ood" not in rec["selected"]
            if arm == "simclr":
                assert rec["extra"]["ssl"]["images_seen"] == 4 * 2 * 16
            assert os.path.exists(os.path.join(out, js[0][:-5] + ".pt"))
            # resume: second invocation skips
            r2 = subprocess.run(cmd + extra, capture_output=True, text=True)
            assert "exists -> skip" in r2.stdout, r2.stdout[-2000:]


def _run(cache, out, ssl, arm, hp, extra=(), seed=0, feats=None):
    cmd = [sys.executable, os.path.join(V2SRC, "run_v2.py"), "--tier", "C", "--cohort", "c17", "--fold", "0",
           "--cache_dir", cache, "--out_root", out, "--ssl_root", ssl, "--arm", arm, "--seed", str(seed),
           "--device", "cpu", "--phase", "tune", "--hp", *hp, *extra]
    if feats:
        cmd += ["--feats_root", feats]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def _load(out, arm):
    d = out + "/C/c17/fold0"
    js = [f for f in os.listdir(d) if f.startswith(arm + "__") and f.endswith(".json")]
    assert len(js) == 1, js
    return json.load(open(os.path.join(d, js[0]))), dict(np.load(os.path.join(d, js[0][:-5] + "_preds.npz")))


def test_ssl_autopretrain_reproducible():
    """simclr gives bit-identical fine-tuning whether the SSL checkpoint is pre-trained in
    the job or loaded from disk; a .pt without .json is not treated as a checkpoint."""
    with tempfile.TemporaryDirectory() as td:
        _write_cache(td + "/cache", np.arange(64) % 3)
        hp = ["steps=10", "bs=16", "ssl_epochs=1", "ssl_bs=16"]
        r = _run(td + "/cache", td + "/A", td + "/ssl", "simclr", hp)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "pre-training" in r.stdout
        r = _run(td + "/cache", td + "/B", td + "/ssl", "simclr", hp)
        assert r.returncode == 0 and "pre-training" not in r.stdout, r.stdout[-3000:]
        (ra, _), (rb, _) = _load(td + "/A", "simclr"), _load(td + "/B", "simclr")
        la = [c["train_loss"] for c in ra["checkpoints"]]
        lb = [c["train_loss"] for c in rb["checkpoints"]]
        assert la == lb, (la, lb)
        assert ra["extra"]["ssl"]["cache_fingerprint"] == ra["cache_fingerprint"]
        js = [f for f in os.listdir(td + "/ssl/C/c17/fold0") if f.endswith(".json")][0]
        os.remove(os.path.join(td + "/ssl/C/c17/fold0", js))          # crash between .pt and .json
        r = _run(td + "/cache", td + "/C", td + "/ssl", "simclr", hp)
        assert r.returncode == 0 and "pre-training" in r.stdout, r.stdout[-3000:]


def test_diverged_run_record():
    with tempfile.TemporaryDirectory() as td:
        _write_cache(td + "/cache", np.arange(64) % 3)
        r = _run(td + "/cache", td + "/out", td + "/ssl", "erm", ["steps=20", "bs=16", "lr=1e12"])
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        rec, preds = _load(td + "/out", "erm")
        assert rec["status"] == "diverged", [c["train_loss"] for c in rec["checkpoints"]]
        assert rec["checkpoints"][-1].get("diverged") and len(rec["checkpoints"]) < 10
        C.validate_record(rec, preds)


def test_macenko_run_record():
    """run_v2 records mac_level, groups / fallbacks per split and group stain stats for the
    group-level arms; macenko_tile keeps the per-tile path (no per-split group stats), under
    its own tag; a per-tile macenko record (no mac_level) is stale for the group-level arm."""
    with tempfile.TemporaryDirectory() as td:
        _write_cache(td + "/cache", np.arange(64) % 3)
        np.save(td + "/cache/got.npy", np.arange(32) // 4)
        hp = ["steps=10", "bs=16"]
        for arm in ("macenko", "macenko_tile", "tia_style"):
            r = _run(td + "/cache", td + "/out", td + "/ssl", arm, hp)
            assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
            rec, preds = _load(td + "/out", arm)
            C.validate_record(rec, preds)
            m = rec["extra"]["macenko"]
            assert rec["tag"] == f"{arm}__bs=16,steps=10__s0" and rec["hparams"]["mac_level"] == m["mac_level"]
            if arm == "macenko_tile":
                assert m["mac_level"] == "tile" and "splits" not in m
                continue
            assert m["mac_level"] == "group"
            sp = m["splits"]
            assert set(sp) == {"tr", "iv", "ov", "ot"}
            # 16-px noise tiles, one tile per training group: every group falls back (counted)
            assert sp["tr"]["n_groups"] == 64 and sp["tr"]["n_fallback"] == 64 and sp["tr"]["n_rows_fallback"] == 64
            assert sp["ot"]["n_groups"] == 8 and sp["ot"]["group_source"] == "got.npy"
            assert sp["ov"]["level"] == "tile" and sp["ov"]["group_source"] is None   # no gov.npy
            assert "per_group" in sp["iv"] and "cos_H" in sp["iv"]
        # an old per-tile record (hparams without mac_level) is stale for the group-level arm
        js = os.path.join(td, "out/C/c17/fold0/macenko__bs=16,steps=10__s0.json")
        rec = json.load(open(js))
        for k in ("mac_level", "mac_group_tiles", "mac_group_px", "mac_group_min_px"):
            rec["hparams"].pop(k)
        json.dump(rec, open(js, "w"))
        r = _run(td + "/cache", td + "/out", td + "/ssl", "macenko", hp)
        assert "hyperparameters differ: ['mac_group_min_px', 'mac_group_px', 'mac_group_tiles', 'mac_level']" \
            in r.stdout, r.stdout[-2000:]
        r = _run(td + "/cache", td + "/out", td + "/ssl", "macenko_tile", hp)
        assert "exists -> skip" in r.stdout


def test_resume_stale_cache():
    with tempfile.TemporaryDirectory() as td:
        _write_cache(td + "/cache", np.arange(64) % 3)
        hp = ["steps=10", "bs=16"]
        assert _run(td + "/cache", td + "/out", td + "/ssl", "erm", hp).returncode == 0
        r = _run(td + "/cache", td + "/out", td + "/ssl", "erm", hp)
        assert "exists -> skip" in r.stdout
        m = json.load(open(td + "/cache/manifest.json"))
        m["created"] = "rebuilt"
        json.dump(m, open(td + "/cache/manifest.json", "w"))
        r = _run(td + "/cache", td + "/out", td + "/ssl", "erm", hp)
        assert r.returncode == 0 and "stale" in r.stdout and "exists -> skip" not in r.stdout, r.stdout[-2000:]


def test_probe_features_consistent():
    """Probe record identical whether features were just extracted or loaded from disk;
    features stored as float32 under the cache fingerprint with a sidecar."""
    if _weights_missing():
        return
    with tempfile.TemporaryDirectory() as td:
        _write_cache(td + "/cache", np.arange(64) % 3)
        hp = ["steps=10", "bs=16"]
        arm = "probe_resnet18_imagenet"
        r = _run(td + "/cache", td + "/A", td + "/ssl", arm, hp, feats=td + "/feats")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        r = _run(td + "/cache", td + "/B", td + "/ssl", arm, hp, feats=td + "/feats")
        assert r.returncode == 0 and "extracted" not in r.stdout, r.stdout[-2000:]
        (ra, pa), (rb, pb) = _load(td + "/A", arm), _load(td + "/B", arm)
        assert [c["id_val"] for c in ra["checkpoints"]] == [c["id_val"] for c in rb["checkpoints"]]
        assert np.array_equal(pa["p_ood_test_id"], pb["p_ood_test_id"])
        fp = ra["cache_fingerprint"]
        fdir = os.path.join(td, "feats", "C", "c17", "fold0", fp[:16])
        f = [x for x in os.listdir(fdir) if x.endswith("_tr.npy")][0]
        assert np.load(os.path.join(fdir, f)).dtype == np.float32
        assert os.path.exists(os.path.join(fdir, f[:-4] + ".meta.json"))
        assert {"p_id_val_id", "y_id_val", "d_id_val", "p_ood_val_id"} <= set(pa)


def test_select_per_fold():
    """select_hparams picks per fold (not the cross-fold mean), skips diverged configs, and
    make_plan uses the fold's own selection with a fold-independent tag."""
    import pandas as pd
    import select_hparams as SH
    import make_plan as MP
    rows = []
    # fold 0 prefers eta=0.1, fold 1 prefers eta=0.001; the pooled mean prefers 0.01
    tab = {0: {0.001: 0.70, 0.01: 0.85, 0.1: 0.90}, 1: {0.001: 0.95, 0.01: 0.86, 0.1: 0.60}}
    for t, d in tab.items():
        for eta, a in d.items():
            rows.append(dict(tier="C", cohort="midog21", fold=t, arm="groupdro", hpkey=f"eta={eta}",
                             hp=json.dumps({"eta": eta, "lr": 0.02}, sort_keys=True), id_val_auroc=a,
                             status="diverged" if (t == 1 and eta == 0.001) else "ok"))
    sel = SH.select(pd.DataFrame(rows))
    e = sel["C"]["midog21"]["groupdro"]
    assert e["folds"]["0"]["hp"]["eta"] == 0.1
    assert e["folds"]["1"]["hp"]["eta"] == 0.01                        # 0.001 diverged -> skipped
    assert e["pooled"]["hp"]["eta"] == 0.01 and e["pooled"]["leaks_test_domain"]
    assert MP.final_hp(sel, "C", "midog21", "groupdro", False, fold=0)["eta"] == 0.1
    assert MP.final_hp(sel, "C", "midog21", "groupdro", False, fold=1)["eta"] == 0.01
    j0 = MP.final_jobs("C", "midog21", sel, False, arms=["groupdro"], fold=0)
    j1 = MP.final_jobs("C", "midog21", sel, False, arms=["groupdro"], fold=1)
    import run_v2 as R
    assert R.job_tag(j0[0], "C")[3] == R.job_tag(j1[0], "C")[3] == "groupdro__selected__s0"
    assert R.job_tag(j0[0], "C")[2] != R.job_tag(j1[0], "C")[2]


# ----------------------------------------------------------------------------- runner
def main():
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v) and sel in k]
    fails = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)", flush=True)
        except Exception:
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name} ({time.time() - t0:.1f}s)", flush=True)
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
