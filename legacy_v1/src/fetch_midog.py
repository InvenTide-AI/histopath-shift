"""Download the MIDOG 2021 training regions from Zenodo, in parallel, resumable.

Zenodo via the site proxy gives ~1 MB/s per connection, so throughput here comes
entirely from running many connections at once. Each file is fetched to a .part
and renamed on success, so re-running the script resumes rather than restarts.
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

DEST = "./data/midog"
FILELIST = os.path.join(DEST, "filelist.json")
WORKERS = int(os.environ.get("MIDOG_WORKERS", "12"))


def want(keys_arg):
    """Which image numbers to fetch. Default: all 200."""
    if keys_arg == "annotated":
        return ["%03d.tiff" % i for i in range(1, 151)]     # 3 annotated scanners
    if keys_arg == "gt450":
        return ["%03d.tiff" % i for i in range(151, 201)]   # unannotated pool
    return ["%03d.tiff" % i for i in range(1, 201)]


def fetch(item):
    key, url = item
    dst = os.path.join(DEST, key)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        return key, "skip", 0
    part = dst + ".part"
    for attempt in range(6):
        try:
            t0 = time.time()
            urllib.request.urlretrieve(url, part)
            os.replace(part, dst)
            return key, "ok", time.time() - t0
        except Exception as e:
            if attempt == 5:
                return key, "FAIL %s %s" % (type(e).__name__, str(e)[:60]), 0
            time.sleep(3 * (attempt + 1))


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else "all"
    fs = json.load(open(FILELIST))
    todo = [(k, fs[k]) for k in want(sel) if k in fs]
    print("fetching %d files with %d workers -> %s" % (len(todo), WORKERS, DEST),
          flush=True)
    done = n_ok = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for key, status, dt in ex.map(fetch, todo):
            done += 1
            if status == "ok":
                n_ok += 1
            if status.startswith("FAIL"):
                print("  %s %s" % (key, status), flush=True)
            if done % 10 == 0 or done == len(todo):
                have = sum(os.path.getsize(os.path.join(DEST, f))
                           for f in os.listdir(DEST) if f.endswith(".tiff"))
                el = time.time() - t0
                print("  %3d/%d done, %.1f GB on disk, %.0f s elapsed, %.1f MB/s aggregate"
                      % (done, len(todo), have / 1e9, el, have / 1e6 / max(el, 1)),
                      flush=True)
    print("complete: %d newly downloaded" % n_ok, flush=True)


if __name__ == "__main__":
    main()
