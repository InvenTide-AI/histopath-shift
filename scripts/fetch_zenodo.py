"""Download a Zenodo record's files in parallel, resumably.

Zenodo through the site egress proxy delivers roughly 1 MB/s on a single
connection, so all usable throughput comes from running many connections at
once. Each file lands in a .part and is renamed only on success, so re-running
resumes instead of restarting.

Usage:  python src/fetch_zenodo.py <record_id> <dest_dir> [name_filter]
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

WORKERS = int(os.environ.get("ZEN_WORKERS", "12"))


def record_files(rid):
    d = json.loads(urllib.request.urlopen(
        "https://zenodo.org/api/records/%s" % rid, timeout=60).read())
    return {f["key"]: f["links"]["self"] for f in d["files"]}


def fetch(item):
    key, url, dest = item
    dst = os.path.join(dest, key)
    if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
        return key, "skip"
    part = dst + ".part"
    for attempt in range(6):
        try:
            urllib.request.urlretrieve(url, part)
            os.replace(part, dst)
            return key, "ok"
        except Exception as e:
            if attempt == 5:
                return key, "FAIL %s %s" % (type(e).__name__, str(e)[:60])
            time.sleep(3 * (attempt + 1))


def main():
    rid, dest = sys.argv[1], sys.argv[2]
    filt = sys.argv[3] if len(sys.argv) > 3 else ""
    os.makedirs(dest, exist_ok=True)
    fs = record_files(rid)
    json.dump(fs, open(os.path.join(dest, "filelist.json"), "w"))
    todo = [(k, v, dest) for k, v in sorted(fs.items()) if filt in k]
    print("record %s: %d files match %r, %d workers -> %s"
          % (rid, len(todo), filt, WORKERS, dest), flush=True)
    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for key, status in ex.map(fetch, todo):
            done += 1
            if status.startswith("FAIL"):
                print("  %s %s" % (key, status), flush=True)
            if done % 20 == 0 or done == len(todo):
                have = sum(os.path.getsize(os.path.join(dest, f))
                           for f in os.listdir(dest) if not f.endswith(".part"))
                el = time.time() - t0
                print("  %3d/%d, %.1f GB, %.0f s, %.1f MB/s aggregate"
                      % (done, len(todo), have / 1e9, el, have / 1e6 / max(el, 1)),
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
