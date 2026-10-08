"""Decode Camelyon17-WILDS parquet shards to 64x64 uint8 arrays.

The WILDS label is defined by the CENTRAL 32x32 region of each 96x96 patch,
so we center-crop to 64x64 (preserving the full label-bearing region plus
context) rather than resizing the whole patch. Subsampling is seeded and
recorded in data_manifest.json.
"""
import glob, io, json, os, sys
import numpy as np, pyarrow.parquet as pq
from PIL import Image

OUT = "cache"; os.makedirs(OUT, exist_ok=True)
RES, CROP = 64, 64
rng = np.random.RandomState(20240)

# per-split cap (CPU budget); None = take all
PLAN = {
    "id_train":        ("Xtr", "ytr", 60000),
    "id_val":          ("Xiv", "yiv", 12000),
    "ood_val":         ("Xov", "yov", 12000),
    "ood_test":        ("Xot", "yot", 25000),
    "unlabeled_train": ("Xu",  None,  40000),
}


def decode(buf):
    im = Image.open(io.BytesIO(buf)).convert("RGB")
    w, h = im.size
    l, t = (w - CROP) // 2, (h - CROP) // 2
    return np.asarray(im.crop((l, t, l + CROP, t + CROP)), dtype=np.uint8)


def read_selected(files, want_idx):
    """Decode only the rows in `want_idx` (global row indices across `files`,
    in file order). Returns images in ascending global-index order."""
    want = set(int(i) for i in want_idx)
    imgs, labs, base = [], [], 0
    for f in files:
        pf = pq.ParquetFile(f)
        n = pf.metadata.num_rows
        if not any(base <= i < base + n for i in want):
            base += n; continue
        off = 0
        for batch in pf.iter_batches(batch_size=512):
            d = batch.to_pydict()
            col, ys = d["image"], d.get("label")
            for i in range(len(col)):
                g = base + off + i
                if g in want:
                    cell = col[i]
                    imgs.append(decode(cell["bytes"] if isinstance(cell, dict) else cell))
                    labs.append(int(ys[i]) if ys is not None else -1)
            off += len(col)
        base += n
    return np.stack(imgs), np.asarray(labs, dtype=np.int64)


manifest = {"source": "huggingface.co/datasets/jxie/camelyon17 (WILDS official splits)",
            "license": "CC0 (Camelyon17)", "crop": f"center {CROP}x{CROP} of 96x96",
            "label": "tumor in central 32x32 region", "subsample_seed": 20240,
            "hospitals": {"id_train": "0,3,4", "id_val": "0,3,4",
                          "ood_val": "1", "ood_test": "2 (held out)",
                          "unlabeled_train": "0,3,4 (no test-hospital leakage)"},
            "splits": {}}

for split, (xk, yk, cap) in PLAN.items():
    files = sorted(glob.glob(f"data/{split}-*.parquet"))
    if not files:
        print(f"SKIP {split}: no shards"); continue

    # parquet rows are LABEL-SORTED, so sequential reads return one class only.
    # Read the label column first (cheap, columnar) and sample class-stratified.
    all_lab = np.concatenate([pq.read_table(f, columns=["label"])["label"].to_numpy()
                              for f in files])
    n_total = len(all_lab)
    if yk:
        per = cap // 2 if cap else None
        sel = []
        for c in (0, 1):
            idx_c = np.where(all_lab == c)[0]
            take = len(idx_c) if per is None else min(per, len(idx_c))
            sel.append(rng.choice(idx_c, size=take, replace=False))
        sel = np.sort(np.concatenate(sel))
    else:
        take = n_total if cap is None else min(cap, n_total)
        sel = np.sort(rng.choice(n_total, size=take, replace=False))

    X, y = read_selected(files, sel)
    rec = {"n_used": int(len(X)), "n_available_in_shards": int(n_total),
           "shards_read": [os.path.basename(f) for f in files], "cap": cap,
           "sampling": "class-stratified random" if yk else "uniform random"}
    if yk:
        rec["class_balance"] = {"neg": int((y == 0).sum()), "pos": int((y == 1).sum()),
                                "pos_frac": float(y.mean())}
        np.savez_compressed(os.path.join(OUT, f"_{split}.npz"), X=X, y=y)
    else:
        np.savez_compressed(os.path.join(OUT, "unlabeled.npz"), Xu=X)
    manifest["splits"][split] = rec
    print(f"{split}: {X.shape} {rec.get('class_balance','')}", flush=True)

# merge labelled splits into one file with the keys run_arm.py expects
lab = {}
for split, (xk, yk, _) in PLAN.items():
    p = os.path.join(OUT, f"_{split}.npz")
    if yk and os.path.exists(p):
        z = np.load(p); lab[xk] = z["X"]; lab[yk] = z["y"]
if lab:
    np.savez_compressed(os.path.join(OUT, "splits.npz"), **lab)
    print("wrote cache/splits.npz:", {k: v.shape for k, v in lab.items()})
json.dump(manifest, open(os.path.join(OUT, "data_manifest.json"), "w"), indent=1)
