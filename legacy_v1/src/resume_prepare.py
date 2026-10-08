"""Resume prepare_crc_data.py under a stall watchdog.

The HF CDN intermittently accepts a connection and then delivers no bytes.
HttpFile has a 90s socket timeout, but a stalled read inside pyarrow's
buffered reader can sit far longer than that, so a wall-clock watchdog is
needed on top. prepare_crc_data.py checkpoints each split to
cache/_part_<name>.npz, so killing and relaunching loses at most one split.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TARGET = os.path.join(CACHE, "crc_splits.npz")
STALL_S = 420          # no new bytes in cache/ for this long -> restart
ATTEMPT_CAP_S = 3600   # hard cap per attempt
MAX_ATTEMPTS = 12


def cache_state():
    """(total bytes, newest mtime) across cache/ -- progress fingerprint."""
    tot, newest = 0, 0.0
    for f in os.listdir(CACHE) if os.path.isdir(CACHE) else []:
        p = os.path.join(CACHE, f)
        if os.path.isfile(p):
            st = os.stat(p)
            tot += st.st_size
            newest = max(newest, st.st_mtime)
    return tot, newest


for attempt in range(1, MAX_ATTEMPTS + 1):
    if os.path.exists(TARGET):
        print(f"target exists: {TARGET}", flush=True)
        break
    print(f"\n=== attempt {attempt}", flush=True)
    p = subprocess.Popen(
        [sys.executable, "prepare_crc_data.py", "--out", "cache"],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    t0 = time.time()
    last_bytes, last_change = cache_state()[0], time.time()
    while p.poll() is None:
        time.sleep(20)
        b, _ = cache_state()
        if b != last_bytes:
            last_bytes, last_change = b, time.time()
            print(f"  progress: cache={b / 1e6:.0f} MB "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
        stalled = time.time() - last_change
        if stalled > STALL_S:
            print(f"  STALL {stalled:.0f}s with no new bytes -> killing",
                  flush=True)
            p.kill()
            break
        if time.time() - t0 > ATTEMPT_CAP_S:
            print("  attempt cap reached -> killing", flush=True)
            p.kill()
            break
    out = p.stdout.read() if p.stdout else ""
    tail = [l for l in out.splitlines() if l.strip()][-6:]
    print("\n".join("  " + l[:160] for l in tail), flush=True)
    if os.path.exists(TARGET):
        print(f"\nSPLITS COMPLETE on attempt {attempt}", flush=True)
        break
    time.sleep(15)
else:
    sys.exit(f"gave up after {MAX_ATTEMPTS} attempts; cache has "
             f"{cache_state()[0] / 1e6:.0f} MB")

print("cache contents:", flush=True)
for f in sorted(os.listdir(CACHE)):
    print(f"  {f}  {os.path.getsize(os.path.join(CACHE, f)) / 1e6:.1f} MB",
          flush=True)
