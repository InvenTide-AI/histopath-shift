#!/usr/bin/env python3
"""Fetch the Camelyon17-WILDS parquet shards and decode, once, every patch needed by
any of the five protocol-matched folds into a single uint8 store.

Output (in --out):
  patches_union.npy   (N, 64, 64, 3) uint8  -- center 64x64 crop of the 96x96 patch,
                                               matching prepare_data.py
  union_gidx.npy      (N,) int64            -- global metadata row index of each row
Per-fold caches are then a gather over this store (see assemble_fold_cache.py).

Shards are downloaded one at a time and deleted after decoding, so peak extra disk is
one shard (~0.5 GB). Labels/centers of every decoded row are asserted against the
metadata table, which is what makes the global-index contract safe to rely on.
"""
import argparse, io, json, os, sys, time, urllib.request
import numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image

DS = "wltjr1007/Camelyon17-WILDS"
CROP = 64          # prepare_data.py: center 64x64 of 96x96
RAW = 96

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="protocol_matched_folds.npz")
    ap.add_argument("--metadata", default="camelyon17_metadata.parquet")
    ap.add_argument("--out", default="union_store")
    ap.add_argument("--keep-shards", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lock = os.path.join(a.out, "build.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        sys.exit(f"another build holds {lock} (pid {open(lock).read().strip()}); "
                 f"remove it if that process is gone")
    import atexit; atexit.register(lambda: os.path.exists(lock) and os.remove(lock))

    md = pd.read_parquet(a.metadata)
    Z = np.load(a.folds)
    union = np.unique(np.concatenate([Z[k] for k in Z.files]))
    N = len(union)
    log(f"union: {N} rows across {md.loc[union,'shard'].nunique()} shards")

    gidx_path = os.path.join(a.out, "union_gidx.npy")
    np.save(gidx_path, union.astype(np.int64))
    pos = {int(g): i for i, g in enumerate(union)}          # global row -> store row

    arr_path = os.path.join(a.out, "patches_union.npy")
    done_path = os.path.join(a.out, "shards_done.json")
    if os.path.exists(arr_path):
        X = np.lib.format.open_memmap(arr_path, mode="r+")
        assert X.shape == (N, CROP, CROP, 3), X.shape
    else:
        X = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8,
                                      shape=(N, CROP, CROP, 3))
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()

    # shard blocks are contiguous in metadata row order (verified); recover each block
    blocks = (md.reset_index().rename(columns={"index": "gidx"})
                .groupby("shard").gidx.agg(["min", "max", "count"])
                .sort_values("min"))
    o = (RAW - CROP) // 2
    t0 = time.time()
    for si, (shard, row) in enumerate(blocks.iterrows(), 1):
        if shard in done:
            log(f"[{si}/{len(blocks)}] {shard}: already done"); continue
        need = union[(union >= row["min"]) & (union <= row["max"])]
        url = f"https://huggingface.co/datasets/{DS}/resolve/main/data/{shard}"
        loc = os.path.join(a.out, shard)
        if not os.path.exists(loc):
            urllib.request.urlretrieve(url, loc)
        pf = pq.ParquetFile(loc)
        assert pf.metadata.num_rows == row["count"], (shard, pf.metadata.num_rows, row["count"])
        needset = set(need.tolist())
        base, n_dec = int(row["min"]), 0
        for rg in range(pf.metadata.num_row_groups):
            nrows = pf.metadata.row_group(rg).num_rows
            g0 = base
            base += nrows
            hits = [g for g in range(g0, g0 + nrows) if g in needset]
            if not hits:
                continue
            tb = pf.read_row_group(rg, columns=["image", "label", "center"])
            imgs, labs, cens = tb.column("image"), tb.column("label"), tb.column("center")
            for g in hits:
                j = g - g0
                cell = imgs[j].as_py()
                b = cell["bytes"] if isinstance(cell, dict) else cell
                im = Image.open(io.BytesIO(b)).convert("RGB")
                assert im.size == (RAW, RAW), im.size
                X[pos[g]] = np.asarray(im, dtype=np.uint8)[o:o + CROP, o:o + CROP]
                assert labs[j].as_py() == int(md.at[g, "label"]), ("label", g)
                assert cens[j].as_py() == int(md.at[g, "center"]), ("center", g)
                n_dec += 1
        X.flush()
        if not a.keep_shards:
            try:
                os.remove(loc)
            except FileNotFoundError:
                pass          # a concurrent/resumed run already removed it
        done.add(shard)
        json.dump(sorted(done), open(done_path, "w"))
        el = time.time() - t0
        log(f"[{si}/{len(blocks)}] {shard}: decoded {n_dec}/{len(need)} "
            f"| elapsed {el/60:.1f} min | eta {el/si*(len(blocks)-si)/60:.1f} min")
    log(f"done: {arr_path} {os.path.getsize(arr_path)/1e9:.2f} GB")

if __name__ == "__main__":
    main()
