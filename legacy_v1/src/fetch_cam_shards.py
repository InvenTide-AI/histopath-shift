"""Download the Camelyon17-WILDS parquet shards named in data_manifest.json."""
import os, sys, urllib.request, time
BASE = "https://huggingface.co/datasets/jxie/camelyon17/resolve/main/"
SHARDS = [
 "data/id_train-00000-of-00003-da7f777dde05cdc6.parquet",
 "data/id_train-00001-of-00003-1e758f9bb82bcfa1.parquet",
 "data/id_train-00002-of-00003-c624aac83e3fe9f4.parquet",
 "data/id_val-00000-of-00001-2016d95c986aa53e.parquet",
 "data/ood_val-00000-of-00001-5c0c47fc6c1fd0d1.parquet",
 "data/ood_test-00000-of-00001-4e2cffbdce8ec122.parquet",
 "data/unlabeled_train-00000-of-00005-244d727dba43c080.parquet",
 "data/unlabeled_train-00001-of-00005-a57074f7af1b59fa.parquet",
]
os.makedirs("data", exist_ok=True)
for s in SHARDS:
    dst = os.path.join("data", os.path.basename(s))
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print("skip", os.path.basename(s), flush=True); continue
    t0 = time.time()
    try:
        urllib.request.urlretrieve(BASE + s, dst + ".part")
        os.replace(dst + ".part", dst)
        print(f"ok {os.path.basename(s)} {os.path.getsize(dst)/1e6:.0f} MB {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL {os.path.basename(s)}: {type(e).__name__} {str(e)[:160]}", flush=True)
        sys.exit(1)
print("all shards present", flush=True)
