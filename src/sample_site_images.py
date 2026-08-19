"""Fetch a per-hospital image sample for the model-free site-shift analysis.

The shift analysis needs pixels, but it does not need all 456k patches: stain
statistics and site separability are stable at a few thousand patches per site.
Downloading every shard to measure colour differences would cost ~10 GB and
hours for numbers that converge at ~3k patches.

Centers are distributed across shards very unevenly (center 1 lives almost
entirely in two validation shards, center 2 entirely in the test shards), so
this reads only the row groups that contain the requested sites, via HTTP range
requests -- never a whole shard.

Output: cache/site_sample_<size>.npz with images, labels, center, patient, row.
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hfio import HttpFile  # noqa: E402

# hfio.BASE points at the CRC dataset used by the earlier experiment; the
# Camelyon17 mirror (the one carrying center/patient metadata) is a different
# repo, so its base is defined here rather than by mutating hfio.
DS = "wltjr1007/Camelyon17-WILDS"
REPO_URL = f"https://huggingface.co/datasets/{DS}/resolve/main/data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache")
    p.add_argument("--per-site", type=int, default=3000)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image

    meta = pd.read_parquet(os.path.join(a.cache, "camelyon17_metadata.parquet"))
    meta["center"] = meta["center"].astype(str)
    sites = sorted(meta.center.unique())
    rng = np.random.default_rng(a.seed)

    # Choose, per site, the single shard holding the most of that site's rows --
    # one shard per site keeps the number of HTTP openings minimal. Sampling
    # within a shard still spans multiple patients (checked below).
    ct = pd.crosstab(meta.shard, meta.center)
    plan = {s: ct[s].idxmax() for s in sites}
    print("shard per site:", json.dumps(plan, indent=1), flush=True)

    imgs, labs, cens, pats, rows = [], [], [], [], []
    t0 = time.time()
    for s, shard in plan.items():
        sub = meta[(meta.center == s) & (meta.shard == shard)]
        take = rng.choice(sub.index.to_numpy(),
                          min(a.per_site, len(sub)), replace=False)
        take = np.sort(take)
        base = int(meta.index[meta.shard == shard][0])
        local = take - base

        f = HttpFile(f"{REPO_URL}/{shard}")
        pf = pq.ParquetFile(f)
        # map global row -> (row_group, offset) so only needed groups are read
        starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows
                                  for i in range(pf.num_row_groups)])
        want = {}
        for lr, gr in zip(local, take):
            g = int(np.searchsorted(starts, lr, side="right") - 1)
            want.setdefault(g, []).append((int(lr - starts[g]), int(gr)))
        got = 0
        for g in sorted(want):
            tab = pf.read_row_group(g, columns=["image"])
            col = tab.column("image").to_pylist()
            for off, gr in want[g]:
                im = Image.open(io.BytesIO(col[off]["bytes"])).convert("RGB")
                if im.size != (a.size, a.size):
                    im = im.resize((a.size, a.size), Image.BILINEAR)
                imgs.append(np.asarray(im, dtype=np.uint8))
                r = meta.loc[gr]
                labs.append(int(r.label)); cens.append(s)
                pats.append(int(r.patient)); rows.append(int(gr))
                got += 1
            del tab, col
        print(f"  site {s}: {got} patches from {shard} "
              f"({len(want)} row groups, {time.time()-t0:.0f}s)", flush=True)

    X = np.stack(imgs)
    out = os.path.join(a.cache, f"site_sample_{a.size}.npz")
    np.savez_compressed(out, images=X, label=np.array(labs),
                        center=np.array(cens), patient=np.array(pats),
                        row=np.array(rows))
    df = pd.DataFrame(dict(center=cens, patient=pats, label=labs))
    print("\nsample composition:")
    print(df.groupby("center").agg(n=("label", "size"),
                                   patients=("patient", "nunique"),
                                   pos_frac=("label", "mean")).round(3).to_string())
    print("wrote", out, f"({os.path.getsize(out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
