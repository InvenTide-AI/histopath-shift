#!/usr/bin/env python3
"""Decode one protocol-matched fold into the cache layout run_arm.py expects.

    python materialize_fold_cache.py --fold 0 --shards data --out cache
      -> cache/fold0/splits.npz     Xtr,ytr,Xiv,yiv,Xov,yov,Xot,yot  (uint8 64x64x3)
      -> cache/fold0/unlabeled.npz  Xu

Decoding matches prepare_data.py exactly: centre-crop 96x96 -> 64x64, uint8,
rows returned in ascending global-index order.

--check-only reads just the parquet row counts and label columns (seconds, no
image decode) and asserts that the global-index -> (shard, row) mapping agrees
with camelyon17_metadata.parquet.  RUN THIS FIRST: it is the cheap test that
the index contract holds for your copy of the shards.

NOTE: only --check-only has been exercised in the authoring session; the decode
path could not be run there because the shard host was unreachable.  The label
cross-check below is the guard -- it fails loudly on any mapping mismatch.
"""
import argparse, io, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image

CROP = 64
ROLES = [("train", "Xtr", "ytr"), ("id_val", "Xiv", "yiv"),
         ("ood_val", "Xov", "yov"), ("ood_test", "Xot", "yot")]

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--shards", default="data", help="dir of Camelyon17-WILDS parquet shards")
ap.add_argument("--out", default="cache")
ap.add_argument("--folds", default="protocol_matched_folds.npz")
ap.add_argument("--meta", default="camelyon17_metadata.parquet")
ap.add_argument("--check-only", action="store_true")
a = ap.parse_args()

meta = pd.read_parquet(a.meta)
spec = json.load(open(a.folds.replace(".npz", ".json")))
z = np.load(a.folds)
t = a.fold

# global index -> (shard, row): shards are contiguous blocks in spec["shard_order"]
offs, base = {}, 0
for s in spec["shard_order"]:
    path = os.path.join(a.shards, s)
    if not os.path.exists(path):
        raise SystemExit(f"missing shard {path}")
    n = pq.ParquetFile(path).metadata.num_rows
    offs[s] = (base, n)
    base += n
if base != len(meta):
    raise SystemExit(f"shard rows {base} != metadata rows {len(meta)}; wrong shard set")

want = {r: np.asarray(z[f"t{t}__{r}"]) for r, _, _ in ROLES}
want["unlabeled"] = np.asarray(z[f"t{t}__unlabeled"])
need = np.unique(np.concatenate(list(want.values())))
need_set = set(int(i) for i in need)

img, lab = {}, {}
for s in spec["shard_order"]:
    b, n = offs[s]
    hits = need[(need >= b) & (need < b + n)]
    if not len(hits):
        continue
    pf = pq.ParquetFile(os.path.join(a.shards, s))
    cols = ["label"] if a.check_only else ["image", "label"]
    off = 0
    hset = set(int(i) - b for i in hits)
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        d = batch.to_pydict()
        for i in range(len(d["label"])):
            if off + i in hset:
                g = b + off + i
                lab[g] = int(d["label"][i])
                if not a.check_only:
                    cell = d["image"][i]
                    im = Image.open(io.BytesIO(cell["bytes"] if isinstance(cell, dict) else cell)).convert("RGB")
                    w, h = im.size
                    l_, tp = (w - CROP) // 2, (h - CROP) // 2
                    img[g] = np.asarray(im.crop((l_, tp, l_ + CROP, tp + CROP)), dtype=np.uint8)
        off += len(d["label"])
    print(f"{s}: {len(hits)} rows", flush=True)

# the guard: shard labels must equal metadata labels for every selected row
bad = [g for g in need_set if lab[g] != int(meta.at[g, "label"])]
if bad:
    raise SystemExit(f"index mapping mismatch on {len(bad)} rows (first: {bad[:5]}); "
                     "your shard order or shard set differs from spec['shard_order']")
print(f"index contract verified on {len(need_set)} rows")
if a.check_only:
    raise SystemExit(0)

d_ = os.path.join(a.out, f"fold{t}")
os.makedirs(d_, exist_ok=True)
out = {}
for r, xk, yk in ROLES:
    ix = np.sort(want[r])
    out[xk] = np.stack([img[int(g)] for g in ix])
    out[yk] = np.asarray([lab[int(g)] for g in ix], dtype=np.int64)
    assert abs(out[yk].mean() - 0.5) < 1e-9, f"{r} lost class balance"
np.savez_compressed(os.path.join(d_, "splits.npz"), **out)
np.savez_compressed(os.path.join(d_, "unlabeled.npz"),
                    Xu=np.stack([img[int(g)] for g in np.sort(want["unlabeled"])]))
json.dump({"fold": t, **{k: v for k, v in spec["folds"][f"t{t}"].items()},
           "sizes": {k: int(v.shape[0]) for k, v in out.items() if k.startswith("X")}},
          open(os.path.join(d_, "fold_manifest.json"), "w"), indent=1)
print("wrote", d_, {k: v.shape for k, v in out.items() if k.startswith("X")})
