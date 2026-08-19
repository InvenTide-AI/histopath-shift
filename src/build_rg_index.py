"""Index NCT-CRC-HE parquet shards at ROW-GROUP granularity.

Each shard is ~488 MB with 33 row groups of 100 rows. Row groups are
individually range-readable, so sampling at row-group granularity means bytes
downloaded == bytes decoded. The shards are label-sorted, so most row groups
hold a single tissue class; this index records the exact class composition of
every row group so splits can be allocated disjointly without guessing.

Writes crc_rg_index_<prefix>.json:
    {shard: [[rg_idx, {class_id: count}], ...], ...}
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hfio import BASE, HttpFile, shard_names

prefix = sys.argv[1] if len(sys.argv) > 1 else "NCT_CRC_HE_100K"
out = f"crc_rg_index_{prefix}.json"
index = json.load(open(out)) if os.path.exists(out) else {}

for f in shard_names(prefix):
    if f in index:
        continue
    pf = pq.ParquetFile(HttpFile(BASE + f))
    lab = pf.read(columns=["label"])["label"].to_numpy()
    md = pf.metadata
    sizes = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
    bounds = np.cumsum([0] + sizes)
    rows = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        u, c = np.unique(lab[a:b], return_counts=True)
        rows.append([i, {str(int(k)): int(v) for k, v in zip(u, c)}])
    index[f] = rows
    json.dump(index, open(out, "w"))
    print(f"{f[:38]} rgs={len(rows)} rows={md.num_rows}", flush=True)

print("shards indexed:", len(index), "->", out)
