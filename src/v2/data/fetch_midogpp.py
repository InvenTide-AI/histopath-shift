"""Download MIDOG++ (Aubreville et al., Sci Data 10:484, 2023) from figshare collection 6615571.

Steps (each step is resumable; re-running the script continues where it stopped):
 1. list all articles of the collection            -> <dest>/api/articles.json
 2. list the files of every article (with backoff)  -> <dest>/api/<article_id>.json
 3. build name -> {size, url, article}              -> <dest>/filelist.json
 4. fetch MIDOG++.json (annotations, figshare file 41265615)
 5. for every image that carries >= 1 annotation:
      - if it is one of the MIDOG 2021 images already at <midog21>/NNN.tiff with
        identical file name, byte size (vs the figshare API) and width/height
        (vs MIDOG++.json), symlink it instead of downloading;
      - else download with `curl -C -` (resumable), N concurrent streams, and
        verify the final byte size AND the figshare md5 (supplied_md5, else
        computed_md5); a file with the right size but a wrong md5 is deleted and
        downloaded again from scratch.
 6. md5 of every downloaded (non-symlink) image re-checked at the end; symlinked
    MIDOG 2021 files cannot match the figshare md5 (the figshare copies carry
    52 extra tag bytes) and are verified by the pixel-data range comparison.
 7. write <dest>/download_report.json (incl. md5 results)

Usage: python fetch_midogpp.py [--workers 8] [--dry_run]
       python fetch_midogpp.py --md5_only   # (re)check md5 of the files on disk, update the report
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

COLLECTION = 6615571
ANN_URL = "https://ndownloader.figshare.com/files/41265615"
API = "https://api.figshare.com/v2"


def get_json(url, tries=12):
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # 403 = rate limit on figshare, back off
            last = e
            time.sleep(min(60, 2 * 2 ** k))
    raise RuntimeError("failed %s: %s" % (url, last))


def list_articles(dest):
    p = os.path.join(dest, "api", "articles.json")
    if os.path.exists(p):
        return json.load(open(p))
    # NB: paginating with page_size=100 returns duplicates and misses articles
    # (unstable ordering); one page of 1000 returns all 506 unique articles.
    arts = get_json("%s/collections/%d/articles?page_size=1000&page=1" % (API, COLLECTION))
    meta = get_json("%s/collections/%d" % (API, COLLECTION))
    assert len({x["id"] for x in arts}) == meta["articles_count"], (len(arts), meta["articles_count"])
    json.dump(arts, open(p, "w"))
    return arts


def list_files(dest, arts):
    out = {}
    ids = sorted({a["id"] for a in arts})

    def one(aid):
        p = os.path.join(dest, "api", "%d.json" % aid)
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if isinstance(d, list):
                    return aid, d
            except Exception:
                pass
        d = get_json("%s/articles/%d/files" % (API, aid))
        json.dump(d, open(p, "w"))
        return aid, d

    with cf.ThreadPoolExecutor(3) as ex:
        for aid, d in ex.map(one, ids):
            for f in d:
                if f["name"] in out and out[f["name"]]["size"] != f["size"]:
                    print("WARNING duplicate name with different size:", f["name"])
                out[f["name"]] = {"size": f["size"], "url": f["download_url"],
                                  "article": aid, "md5": f.get("supplied_md5") or f.get("computed_md5")}
    return out


def md5_file(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def curl(item):
    name, url, size, dst = item[:4]
    md5 = item[4] if len(item) > 4 else None
    for attempt in range(8):
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            if md5 is None or md5_file(dst) == md5:
                return name, "ok"
            os.remove(dst)        # right size, wrong content (e.g. bad resume) -> fresh download
            continue
        if os.path.exists(dst) and os.path.getsize(dst) > size:
            os.remove(dst)
        r = subprocess.run(["curl", "-sSL", "--fail", "-C", "-", "--retry", "5",
                            "--retry-delay", "5", "-o", dst, url],
                           capture_output=True, text=True)
        if r.returncode not in (0, 33):   # 33 = range not supported -> restart
            time.sleep(5 * (attempt + 1))
        elif r.returncode == 33 and os.path.exists(dst):
            os.remove(dst)
    ok = os.path.exists(dst) and os.path.getsize(dst) == size
    if ok and md5 is not None and md5_file(dst) != md5:
        return name, "FAIL md5"
    return name, "ok" if ok else "FAIL size %s vs %d" % (
        os.path.getsize(dst) if os.path.exists(dst) else None, size)


def md5_check(dest, fl, names, workers):
    """md5 of every non-symlink file in `names` vs the figshare md5 -> report dict."""
    todo = [n for n in names if os.path.exists(os.path.join(dest, n)) and not os.path.islink(os.path.join(dest, n))]

    def one(n):
        exp = fl.get(n, {}).get("md5")
        got = md5_file(os.path.join(dest, n))
        return n, exp, got

    out = {"checked": 0, "ok": 0, "mismatch": [], "no_reference_md5": [],
           "skipped_symlinks_pixel_verified": sorted(n for n in names if os.path.islink(os.path.join(dest, n)))}
    with cf.ThreadPoolExecutor(workers) as ex:
        for n, exp, got in ex.map(one, todo):
            out["checked"] += 1
            if not exp:
                out["no_reference_md5"].append(n)
            elif exp == got:
                out["ok"] += 1
            else:
                out["mismatch"].append([n, exp, got])
    out["all_ok"] = not out["mismatch"] and not out["no_reference_md5"]
    return out


def res_tags(img):
    """X/YResolution + unit from TIFF tags -> microns per pixel (None if absent)."""
    t = img.tag_v2
    xr, yr, unit = t.get(282), t.get(283), t.get(296, 2)
    if not xr:
        return None
    per_um = {2: 25400.0, 3: 10000.0}.get(int(unit))
    return {"xres": float(xr), "yres": float(yr), "unit": int(unit),
            "mpp_x": per_um / float(xr) if per_um else None,
            "mpp_y": per_um / float(yr) if per_um else None}


def range_get(url, lo, hi):
    for k in range(6):
        r = subprocess.run(["curl", "-sSL", "--fail", "-r", "%d-%d" % (lo, hi), url],
                           capture_output=True)
        if r.returncode == 0 and len(r.stdout) == hi - lo + 1:
            return r.stdout
        time.sleep(3 * (k + 1))
    raise RuntimeError("range get failed %s %d-%d" % (url, lo, hi))


def verify_reuse(old, url, size, im, chunk=1 << 20):
    """A local MIDOG 2021 TIFF is reused iff it has the same name, width/height
    (vs MIDOG++.json) and image layout, and its strip bytes equal the figshare
    file at the start, middle and end of the pixel data (HTTP range requests).
    The figshare copies are 52 bytes larger: they add X/YResolution and
    ResolutionUnit tags; pixel data are identical (full check on a sample)."""
    import io
    loc = Image.open(old)
    hdr = range_get(url, 0, 65535)
    try:
        rem = Image.open(io.BytesIO(hdr))
        rt = rem.tag_v2
    except Exception as e:
        return False, {"why": "remote header unparsable %s" % e}
    lt = loc.tag_v2
    info = {"local_size": os.path.getsize(old), "remote_size": size,
            "remote_res": res_tags(rem), "local_res": res_tags(loc)}
    if loc.size != (im["width"], im["height"]) or rem.size != loc.size or rem.mode != loc.mode:
        info["why"] = "size/mode mismatch %s %s %s" % (loc.size, rem.size, (im["width"], im["height"]))
        return False, info
    lo_off, ro_off = lt.get(273), rt.get(273)
    lo_cnt, ro_cnt = lt.get(279), rt.get(279)
    if lt.get(259) != rt.get(259) or tuple(lo_cnt) != tuple(ro_cnt) or len(lo_off) != 1:
        info["why"] = "layout mismatch"
        return False, info
    n = int(lo_cnt[0])
    for rel in (0, n // 2, n - chunk):
        with open(old, "rb") as fh:
            fh.seek(int(lo_off[0]) + rel)
            a_ = fh.read(chunk)
        b_ = range_get(url, int(ro_off[0]) + rel, int(ro_off[0]) + rel + chunk - 1)
        if a_ != b_:
            info["why"] = "strip bytes differ at %d" % rel
            return False, info
    info["checked_chunks"] = 3
    return True, info


def full_pixel_check(old, url, size, tmpdir):
    import numpy as np
    os.makedirs(tmpdir, exist_ok=True)
    dst = os.path.join(tmpdir, os.path.basename(old))
    curl((os.path.basename(old), url, size, dst))
    A = np.asarray(Image.open(old).convert("RGB"))
    B = np.asarray(Image.open(dst).convert("RGB"))
    eq = bool(A.shape == B.shape and np.array_equal(A, B))
    os.remove(dst)
    return eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "midogpp"))
    ap.add_argument("--midog21", default=os.path.join(os.environ.get("HISTOPATH_DATA", "data"), "midog"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--full_check", default="001.tiff,060.tiff,120.tiff")
    ap.add_argument("--md5_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.dest, "api"), exist_ok=True)
    if a.md5_only:
        fl = json.load(open(os.path.join(a.dest, "filelist.json")))
        rp = os.path.join(a.dest, "download_report.json")
        report = json.load(open(rp))
        names = [m["file_name"] for m in json.load(open(os.path.join(a.dest, "image_meta.json")))]
        t0 = time.time()
        report["md5"] = md5_check(a.dest, fl, names, a.workers)
        report["md5"]["seconds"] = round(time.time() - t0, 1)
        json.dump(report, open(rp, "w"), indent=1)
        m = report["md5"]
        print("md5: checked %d ok %d mismatch %d no_ref %d symlinks(skipped) %d -> %s" % (
            m["checked"], m["ok"], len(m["mismatch"]), len(m["no_reference_md5"]),
            len(m["skipped_symlinks_pixel_verified"]), "ALL OK" if m["all_ok"] else "FAIL"), flush=True)
        sys.exit(0 if m["all_ok"] else 1)

    arts = list_articles(a.dest)
    print("articles: %d (%d unique)" % (len(arts), len({x['id'] for x in arts})), flush=True)
    fl = list_files(a.dest, arts)
    json.dump(fl, open(os.path.join(a.dest, "filelist.json"), "w"), indent=0)
    print("files listed: %d" % len(fl), flush=True)

    annp = os.path.join(a.dest, "MIDOG++.json")
    if not os.path.exists(annp):
        subprocess.run(["curl", "-sSL", "-o", annp, ANN_URL], check=True)
    ann = json.load(open(annp))
    n_ann = {}
    for x in ann["annotations"]:
        n_ann[x["image_id"]] = n_ann.get(x["image_id"], 0) + 1
    wanted = [im for im in ann["images"] if n_ann.get(im["id"], 0) > 0]
    print("annotated images: %d of %d" % (len(wanted), len(ann["images"])), flush=True)

    report = {"reused_midog21": [], "downloaded": [], "missing_in_figshare": [],
              "failed": [], "reuse_rejected": []}
    reuse_info = {}
    todo = []
    for im in wanted:
        name = im["file_name"]
        dst = os.path.join(a.dest, name)
        if name not in fl:
            report["missing_in_figshare"].append(name)
            continue
        size = fl[name]["size"]
        old = os.path.join(a.midog21, name)
        if os.path.exists(old) and not os.path.lexists(dst):
            ok, info = verify_reuse(old, fl[name]["url"], size, im)
            if ok:
                if not a.dry_run:
                    os.symlink(old, dst)
                report["reused_midog21"].append(name)
                reuse_info[name] = info
                continue
            report["reuse_rejected"].append([name, info])
        if os.path.islink(dst):
            report["reused_midog21"].append(name)
            if name not in reuse_info:
                reuse_info[name] = verify_reuse(os.path.realpath(dst), fl[name]["url"], size, im)[1]
            continue
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            report["downloaded"].append(name)
            continue
        todo.append((name, fl[name]["url"], size, dst, fl[name].get("md5")))
    tot = sum(t[2] for t in todo)
    print("reused from MIDOG21: %d; to download: %d files, %.1f GB; missing in figshare: %s"
          % (len(report["reused_midog21"]), len(todo), tot / 1e9, report["missing_in_figshare"]),
          flush=True)
    if a.dry_run:
        return
    t0 = time.time(); done = 0; got = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        sizes = {t[0]: t[2] for t in todo}
        for name, st in ex.map(curl, todo):
            done += 1
            if st == "ok":
                report["downloaded"].append(name); got += sizes[name]
            else:
                report["failed"].append([name, st]); print("  FAIL", name, st, flush=True)
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                print("  %d/%d  %.1f GB  %.0fs  %.1f MB/s" % (done, len(todo), got / 1e9, el,
                                                             got / 1e6 / max(el, 1)), flush=True)
    # final verification of every wanted image
    bad = []
    for im in wanted:
        p = os.path.join(a.dest, im["file_name"])
        exp = fl.get(im["file_name"], {}).get("size")
        if os.path.islink(p):
            exp = reuse_info.get(im["file_name"], {}).get("local_size")
        if not os.path.exists(p) or os.path.getsize(p) != exp:
            bad.append(im["file_name"])
    # full pixel comparison for a few reused images (one per MIDOG21 scanner)
    report["full_pixel_check"] = {}
    for name in a.full_check.split(","):
        if name in reuse_info:
            report["full_pixel_check"][name] = full_pixel_check(
                os.path.realpath(os.path.join(a.dest, name)), fl[name]["url"], fl[name]["size"],
                os.path.join(a.dest, "_cmp"))
    print("full pixel checks:", report["full_pixel_check"], flush=True)
    # per-image metadata incl. resolution tags (from figshare header for reused files)
    meta = []
    for im in wanted:
        p = os.path.join(a.dest, im["file_name"])
        if im["file_name"] in reuse_info:
            r = reuse_info[im["file_name"]]["remote_res"]
            src = "midog21_symlink"
        else:
            r = res_tags(Image.open(p)) if os.path.exists(p) else None
            src = "figshare"
        meta.append(dict(file_name=im["file_name"], id=im["id"], tumor_type=im["tumor_type"],
                         width=im["width"], height=im["height"], n_annotations=n_ann[im["id"]],
                         source=src, resolution=r))
    json.dump(meta, open(os.path.join(a.dest, "image_meta.json"), "w"), indent=0)
    report["reuse_info"] = reuse_info
    report["md5"] = md5_check(a.dest, fl, [im["file_name"] for im in wanted], a.workers)
    print("md5: checked %d ok %d mismatch %s" % (report["md5"]["checked"], report["md5"]["ok"],
                                                 report["md5"]["mismatch"]), flush=True)
    if not report["md5"]["all_ok"]:
        bad += [m[0] for m in report["md5"]["mismatch"]] + report["md5"]["no_reference_md5"]
    report["final_size_mismatch_or_missing"] = bad
    report["n_wanted"] = len(wanted)
    json.dump(report, open(os.path.join(a.dest, "download_report.json"), "w"), indent=1)
    print("done. wanted=%d reused=%d downloaded=%d failed=%d bad=%d"
          % (len(wanted), len(report["reused_midog21"]), len(report["downloaded"]),
             len(report["failed"]), len(bad)), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
