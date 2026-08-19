"""Seekable read-only file over HTTP range requests, for reading parquet
footers and single columns without downloading whole shards."""
import http.client
import io
import time
import urllib.error
import urllib.request

BASE = "https://huggingface.co/datasets/1aurent/NCT-CRC-HE/resolve/main/data/"


def _retry(fn, tries=10, base_delay=3.0):
    """The CDN redirect intermittently yields a 502/504 from the egress proxy;
    these are transient and succeed on retry."""
    for a in range(tries):
        try:
            return fn()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                http.client.HTTPException) as e:
            # http.client.IncompleteRead: the CDN accepts the range request and
            # then truncates the body. It is an HTTPException, not an OSError,
            # so it escaped this handler and killed the whole prepare run.
            code = getattr(e, "code", None)
            if code is not None and code not in (408, 429, 500, 502, 503, 504):
                raise
            if a == tries - 1:
                raise
            time.sleep(base_delay * (a + 1))


class HttpFile(io.RawIOBase):
    def __init__(self, url, timeout=90):
        self.url, self.timeout, self._pos = url, timeout, 0
        r = _retry(lambda: urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=timeout))
        self.size = int(r.headers["Content-Length"])

    def seek(self, o, whence=0):
        self._pos = o if whence == 0 else (
            self._pos + o if whence == 1 else self.size + o)
        return self._pos

    def tell(self):
        return self._pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            n = self.size - self._pos
        if n <= 0:
            return b""
        end = min(self._pos + n, self.size) - 1
        want = end - self._pos + 1
        # The CDN sometimes returns fewer bytes than the range asked for (or
        # truncates mid-body). pyarrow treats a short read as EOF and fails on
        # a corrupt footer, so re-request the remainder until the range is
        # fully satisfied.
        chunks, got = [], 0
        while got < want:
            lo = self._pos + got
            req = urllib.request.Request(
                self.url, headers={"Range": f"bytes={lo}-{end}"})
            part = _retry(
                lambda: urllib.request.urlopen(req, timeout=self.timeout).read())
            if not part:
                break
            chunks.append(part)
            got += len(part)
        b = b"".join(chunks)
        self._pos += len(b)
        return b

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def shard_names(prefix, api="https://huggingface.co/api/datasets/1aurent/NCT-CRC-HE"):
    import json
    r = json.load(urllib.request.urlopen(api, timeout=60))
    return sorted(s["rfilename"].split("/")[-1] for s in r["siblings"]
                  if s["rfilename"].startswith(f"data/{prefix}-"))
