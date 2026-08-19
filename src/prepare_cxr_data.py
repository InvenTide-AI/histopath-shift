"""Stream NIH ChestX-ray14 and CheXpert into patient-disjoint splits for the
cross-institution effusion experiment.

Task: pleural effusion, binary, frontal views only.
  NIH      -> institution A: train / id_val / id_test        (labelled)
  CheXpert -> institution B: ood_test                        (labelled, scored once)
  pool S   -> unlabelled NIH        (single-site)
  pool M   -> unlabelled NIH + CheXpert (site-spanning)

Patient disjointness is enforced on patient IDs, never on images: both cohorts
carry several images per patient, so an image-level split would leak a patient
across train and test and inflate every arm equally.

CheXpert label convention: blank -> negative (standard), explicit uncertain ->
dropped. Every test set is balanced 50/50 by subsampling so that AUROC is
comparable across cohorts of very different prevalence (NIH ~9%, CheXpert ~41%).

Resumable: each split is checkpointed to cache_cxr/_part_<name>.npz.
"""
import argparse
import collections
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hfio import HttpFile, _retry

NIH = "https://huggingface.co/datasets/arudaev/chest-xray-14-320/resolve/main/data/"
CHX = "https://huggingface.co/datasets/danjacobellis/chexpert/resolve/main/data/"
NIH_SHARDS = [f"train-{i:05d}-of-00012.parquet" for i in range(12)]
CHX_SHARDS = [f"train-{i:05d}-of-00023.parquet" for i in range(23)]
PX = 96
CACHE = "cache_cxr"


def decode(b, px=PX):
    im = Image.open(io.BytesIO(b)).convert("L").resize((px, px), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def nih_shard(fn, want_cols=("image", "labels", "filename")):
    """Return (imgs_u8, y, patient_ids) for one NIH shard."""
    pf = pq.ParquetFile(HttpFile(NIH + fn))
    X, Y, P = [], [], []
    for rg in range(pf.num_row_groups):
        t = _retry(lambda: pf.read_row_group(rg, columns=list(want_cols)))
        labs = t["labels"].to_pylist()
        fns = t["filename"].to_pylist()
        ims = t["image"].to_pylist()
        for im, lab, f in zip(ims, labs, fns):
            toks = {s.strip() for s in str(lab).split("|")}
            X.append(decode(im["bytes"]))
            Y.append(1 if "Effusion" in toks else 0)
            P.append("NIH_" + f.split("_")[0])
    return np.stack(X), np.array(Y, np.int64), np.array(P)


def chx_shard(fn, rg_workers=8):
    """Frontal-only, effusion absent/present only. Returns (imgs, y, pids).

    This release has ~98 row groups of ~100 rows each, so per-request latency
    dominates; row groups are fetched concurrently, each on its own file handle
    (ParquetFile is not thread-safe for concurrent reads).
    """
    cols = ["image", "Path", "Pleural Effusion", "Frontal/Lateral"]
    nrg = pq.ParquetFile(HttpFile(CHX + fn)).num_row_groups

    def rd(rg):
        p = pq.ParquetFile(HttpFile(CHX + fn))
        return _retry(lambda: p.read_row_group(rg, columns=cols))

    X, Y, P = [], [], []
    with ThreadPoolExecutor(rg_workers) as ex:
        tables = list(ex.map(rd, range(nrg)))
    for t in tables:
        eff = t["Pleural Effusion"].to_pylist()
        fl = t["Frontal/Lateral"].to_pylist()
        paths = t["Path"].to_pylist()
        ims = t["image"].to_pylist()
        for im, e, v, p in zip(ims, eff, fl, paths):
            if v != 0:            # 0 = Frontal
                continue
            # 0 unlabeled(blank) -> negative; 1 uncertain -> drop; 2 absent; 3 present
            if e == 1:
                continue
            y = 1 if e == 3 else 0
            parts = p.split("/")
            pid = "CHX_" + (parts[2] if len(parts) > 2 else p)
            X.append(decode(im["bytes"]))
            Y.append(y)
            P.append(pid)
    if not X:
        return np.zeros((0, PX, PX), np.uint8), np.zeros(0, np.int64), np.zeros(0, "<U24")
    return np.stack(X), np.array(Y, np.int64), np.array(P)


def _one(kind, fn):
    """Fetch+decode one shard, checkpointed so an interrupted run resumes."""
    ck = f"{CACHE}/_sh_{kind}_{fn.split('-')[1]}.npz"
    if os.path.exists(ck):
        d = np.load(ck, allow_pickle=True)
        return d["X"], d["y"], d["p"]
    X, y, p = (nih_shard if kind == "nih" else chx_shard)(fn)
    np.savez_compressed(ck, X=X, y=y, p=p)
    return X, y, p


def gather(kind, shards, workers=6):
    Xs, Ys, Ps = [], [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_one, kind, f) for f in shards]
        for i, fu in enumerate(futs):
            X, y, p = fu.result()
            Xs.append(X), Ys.append(y), Ps.append(p)
            print(f"[{kind}] shard {i + 1}/{len(shards)} n={len(y)}", flush=True)
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ps)


def keep_all_pos(X, y, pids, n, rng):
    """Training set: keep every positive, fill to n with random negatives.

    Balancing the training set to 50/50 would discard most negatives; only the
    test sets need balancing, and there only for cross-cohort comparability.
    Residual imbalance is handled by class-weighted loss in run_arm_cxr.py.
    """
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    k = min(len(neg), max(0, n - len(pos)))
    take = np.concatenate([pos, rng.choice(neg, k, replace=False)])
    rng.shuffle(take)
    return X[take], y[take], pids[take]


def balance(X, y, pids, n, rng):
    """Subsample to n total, 50/50 on y, without splitting a patient's images
    across the boundary (patients are already partitioned before this call)."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    k = min(n // 2, len(pos), len(neg))
    take = np.concatenate([rng.choice(pos, k, replace=False),
                           rng.choice(neg, k, replace=False)])
    rng.shuffle(take)
    return X[take], y[take], pids[take]


def split_patients(pids, fracs, rng):
    """Partition unique patients by fraction. Returns list of boolean masks."""
    uniq = np.unique(pids)
    rng.shuffle(uniq)
    cuts = np.cumsum([int(round(f * len(uniq))) for f in fracs[:-1]])
    groups = np.split(uniq, cuts)
    return [np.isin(pids, g) for g in groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_id_val", type=int, default=4000)
    ap.add_argument("--n_id_test", type=int, default=4000)
    ap.add_argument("--n_ood", type=int, default=6000)
    ap.add_argument("--n_pool_each", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    # Only as many shards as the splits need; each NIH shard ~7.2k images,
    # each CheXpert shard ~9.7k rows of which ~44% are usable frontal.
    ap.add_argument("--nih_shards", type=int, default=12)
    ap.add_argument("--chx_shards", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- NIH (institution A) -------------------------------------------------
    pn = f"{CACHE}/_raw_nih.npz"
    if os.path.exists(pn):
        d = np.load(pn, allow_pickle=True)
        Xn, yn, pn_ids = d["X"], d["y"], d["p"]
    else:
        Xn, yn, pn_ids = gather("nih", NIH_SHARDS)
        np.savez_compressed(pn, X=Xn, y=yn, p=pn_ids)
    print(f"NIH total {len(yn)} imgs, {len(np.unique(pn_ids))} patients, "
          f"prev={yn.mean():.3f}", flush=True)

    # patient-disjoint: train / id_val / id_test / unlabelled-pool
    m_tr, m_iv, m_it, m_pool = split_patients(pn_ids, [.55, .12, .12, .21], rng)
    for nm, m in [("train", m_tr), ("id_val", m_iv), ("id_test", m_it), ("poolA", m_pool)]:
        print(f"  {nm}: {m.sum()} imgs, {len(np.unique(pn_ids[m]))} patients", flush=True)

    Xtr, ytr, ptr = keep_all_pos(Xn[m_tr], yn[m_tr], pn_ids[m_tr], a.n_train, rng)
    Xiv, yiv, piv = balance(Xn[m_iv], yn[m_iv], pn_ids[m_iv], a.n_id_val, rng)
    Xit, yit, pit = balance(Xn[m_it], yn[m_it], pn_ids[m_it], a.n_id_test, rng)
    # unlabelled pool A: no balancing, labels never used
    idx = rng.choice(np.where(m_pool)[0],
                     min(a.n_pool_each, int(m_pool.sum())), replace=False)
    XuA, puA = Xn[idx], pn_ids[idx]

    # ---- CheXpert (institution B) -------------------------------------------
    pc = f"{CACHE}/_raw_chx.npz"
    if os.path.exists(pc):
        d = np.load(pc, allow_pickle=True)
        Xc, yc, pc_ids = d["X"], d["y"], d["p"]
    else:
        Xc, yc, pc_ids = gather("chx", CHX_SHARDS)
        np.savez_compressed(pc, X=Xc, y=yc, p=pc_ids)
    print(f"CheXpert total {len(yc)} frontal imgs, {len(np.unique(pc_ids))} patients, "
          f"prev={yc.mean():.3f}", flush=True)

    # OOD test patients must be disjoint from the unlabelled pool-M patients
    m_ood, m_poolB = split_patients(pc_ids, [.65, .35], rng)
    Xood, yood, pood = balance(Xc[m_ood], yc[m_ood], pc_ids[m_ood], a.n_ood, rng)
    idxB = rng.choice(np.where(m_poolB)[0],
                      min(a.n_pool_each, int(m_poolB.sum())), replace=False)
    XuB, puB = Xc[idxB], pc_ids[idxB]

    # ---- leakage assertions -------------------------------------------------
    sets = {"train": ptr, "id_val": piv, "id_test": pit, "ood": pood,
            "poolA": puA, "poolB": puB}
    for k1 in ["train", "id_val", "id_test"]:
        for k2 in ["train", "id_val", "id_test", "poolA"]:
            if k1 >= k2:
                continue
            ov = set(sets[k1]) & set(sets[k2])
            assert not ov, f"patient leak {k1}/{k2}: {list(ov)[:3]}"
    ov = set(pood) & set(puB)
    assert not ov, f"OOD/poolM patient leak: {list(ov)[:3]}"
    assert not (set(ptr) & set(pood)), "cross-cohort id collision"

    np.savez_compressed(
        f"{CACHE}/cxr_splits.npz",
        Xtr=Xtr, ytr=ytr, Xiv=Xiv, yiv=yiv, Xit=Xit, yit=yit,
        Xood=Xood, yood=yood)
    np.savez_compressed(f"{CACHE}/cxr_pool_S.npz", Xu=XuA)
    np.savez_compressed(f"{CACHE}/cxr_pool_M.npz",
                        Xu=np.concatenate([XuA, XuB]))
    manifest = dict(
        px=PX, task="pleural_effusion", view="frontal_only",
        chexpert_label_convention="blank->negative, uncertain dropped",
        n_train=len(ytr), n_id_val=len(yiv), n_id_test=len(yit), n_ood=len(yood),
        n_pool_S=len(XuA), n_pool_M=len(XuA) + len(XuB),
        pool_M_composition=dict(nih=len(XuA), chexpert=len(XuB)),
        prev_train=float(ytr.mean()), prev_ood=float(yood.mean()),
        patients=dict((k, int(len(np.unique(v)))) for k, v in sets.items()),
        patient_disjoint_verified=True)
    json.dump(manifest, open(f"{CACHE}/cxr_manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == "__main__":
    main()
