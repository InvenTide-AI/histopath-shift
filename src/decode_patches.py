"""Decode the patches referenced by the hospital splits into a uint8 memmap.

Why pre-decode
--------------
On a GPU host, JPEG decode of 224px patches saturates the CPU well before the
GPU is busy: a ResNet-50 step at bs=128 is a few tens of ms, while decoding 128
JPEGs is comparable or worse. Decoding once into a memmapped array makes the
sweep GPU-bound instead of decode-bound, which is the difference between hours
and days for 80 runs.

Only the patches actually referenced by cache/hospital_splits.npz are decoded --
the union across folds, deduplicated -- not the whole 456k-patch dataset.

Output
------
cache/decoded_<size>.npy       (N_referenced, size, size, 3) uint8, in ascending
                               order of the original metadata row id.
cache/decoded_<size>_index.json provenance + row_ids, the mapping from array
                               position -> original metadata row.

Only referenced rows are allocated. Allocating the full metadata length would
cost 68.6 GB at 224px versus 34.6 GB for the ~230k rows the splits actually
touch, and would leave half the file as zeros that a bug could silently train
on. Consumers must map original row ids through `row_ids` (train_gpu.py does
this via its remap table) -- a split index is NOT a direct offset into this
array.
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DS = "wltjr1007/Camelyon17-WILDS"
REPO_URL = f"https://huggingface.co/datasets/{DS}/resolve/main/data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache")
    p.add_argument("--data", default="data_wilds",
                   help="directory of wltjr1007 parquet shards")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N rows only")
    a = p.parse_args()

    import pandas as pd
    import pyarrow.parquet as pq
    from PIL import Image

    meta = pd.read_parquet(os.path.join(a.cache, "camelyon17_metadata.parquet"))
    sp = np.load(os.path.join(a.cache, "hospital_splits.npz"))
    need = np.unique(np.concatenate([sp[k] for k in sp.files]))
    if a.limit:
        need = need[:a.limit]
    print(f"patches referenced by splits: {len(need)} of {len(meta)} rows")

    need = np.sort(need)
    pos_of_row = {int(r): i for i, r in enumerate(need)}   # original row -> array pos
    out_path = os.path.join(a.cache, f"decoded_{a.size}.npy")
    est_gb = len(need) * a.size * a.size * 3 / 1e9
    print(f"allocating {est_gb:.1f} GB for {len(need)} rows")
    X = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                  shape=(len(need), a.size, a.size, 3))

    # Row ids are global over the concatenated metadata table, so recover each
    # shard's base offset from the shard column rather than assuming shard order.
    shard_base = {s: int(meta.index[meta.shard == s][0])
                  for s in meta.shard.unique()}
    filled = np.zeros(len(need), dtype=bool)
    done, t0 = 0, time.time()
    for shard, rows in meta.iloc[need].groupby("shard").groups.items():
        path = os.path.join(a.data, shard)
        base = shard_base[shard]
        if os.path.exists(path):
            pf = pq.ParquetFile(path)
        else:
            # No local copy: stream the image column over HTTP range requests.
            # Reading whole row groups sequentially runs ~55 patches/s, versus
            # ~7/s for scattered single-row reads, so the loop below walks row
            # groups in order rather than seeking per patch.
            from hfio import HttpFile
            pf = pq.ParquetFile(HttpFile(f"{REPO_URL}/{shard}"))
        table = pf.read(columns=["image"])
        col = table.column("image").to_pylist()
        for gid in sorted(int(r) for r in rows):
            im = Image.open(io.BytesIO(col[gid - base]["bytes"])).convert("RGB")
            if im.size != (a.size, a.size):
                im = im.resize((a.size, a.size), Image.BILINEAR)
            pos = pos_of_row[gid]
            X[pos] = np.asarray(im, dtype=np.uint8)
            filled[pos] = True
            done += 1
        del table, col
        print(f"  {shard}: {len(rows)} patches ({done}/{len(need)}, "
              f"{time.time()-t0:.0f}s)", flush=True)
    X.flush()
    assert filled.all(), f"{(~filled).sum()} referenced rows were never decoded"
    with open(os.path.join(a.cache, f"decoded_{a.size}_index.json"), "w") as fh:
        json.dump({"size": a.size,
                   "n_decoded": int(len(need)),
                   "metadata_rows": int(len(meta)),
                   "row_ids": [int(r) for r in need],
                   "source": "wltjr1007/Camelyon17-WILDS",
                   "note": ("array position i holds original metadata row row_ids[i]; "
                            "split indices must be mapped through row_ids")},
                  fh, indent=2)
    print("wrote", out_path, f"({os.path.getsize(out_path)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
