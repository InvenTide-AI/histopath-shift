import collections
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hfio import BASE, HttpFile, shard_names

NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
prefix = sys.argv[1] if len(sys.argv) > 1 else "NCT_CRC_HE_100K"
sh = shard_names(prefix)
per, tot = {}, collections.Counter()
for i, f in enumerate(sh):
    pf = pq.ParquetFile(HttpFile(BASE + f))
    lab = pf.read(columns=["label"])["label"].to_numpy()
    c = collections.Counter(lab.tolist())
    per[f] = c
    tot.update(c)
    print(i, pf.metadata.num_rows, {NAMES[k]: v for k, v in sorted(c.items())},
          flush=True)
json.dump({f: {str(k): int(v) for k, v in c.items()} for f, c in per.items()},
          open(f"crc_shard_labels_{prefix}.json", "w"), indent=1)
print("TOTAL", {NAMES[k]: v for k, v in sorted(tot.items())})
