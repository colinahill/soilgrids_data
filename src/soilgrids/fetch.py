"""HTTP access to the ISRIC WebDAV tree.

ISRIC is a public academic server that measures at ~2.6 MB/s aggregate and does
not go faster with more connections (8 and 32 keep-alive connections give the
same throughput), so this module is built to be polite and durable rather than
fast: a bounded keep-alive pool, exponential backoff, an optional rate limit,
and ETag verification so a source replaced mid-run is caught rather than mixed
into the store.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

DEFAULT_WORKERS = 8  # measured: more buys nothing
MAX_ATTEMPTS = 6
RETRY_BASE_SECONDS = 2.0
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class SourceChanged(RuntimeError):
    """The object's ETag no longer matches the manifest: ISRIC replaced it.

    Not retryable. `latest/` is a mutable path, so this is the signal that the
    run would otherwise silently mix two snapshots.
    """


class FetchError(RuntimeError):
    """The object could not be fetched after MAX_ATTEMPTS."""


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    url: str
    size: int
    etag: str | None
    last_modified: str | None


class _RateLimiter:
    """Simple global request-rate cap, shared across worker threads."""

    def __init__(self, per_second: float | None):
        self._interval = 1.0 / per_second if per_second else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            due = max(now, self._next)
            self._next = due + self._interval
        if due > now:
            time.sleep(due - now)


class Fetcher:
    """A keep-alive HTTP client for the source tree.

    Use as a context manager; the underlying pool is shared by every worker
    thread, which is what keeps connections warm across thousands of tiles.
    """

    def __init__(
        self,
        *,
        workers: int = DEFAULT_WORKERS,
        rate_limit: float | None = None,
        timeout: float = 120.0,
    ):
        self.workers = max(1, workers)
        self._limiter = _RateLimiter(rate_limit)
        self._client = httpx.Client(
            limits=httpx.Limits(max_connections=self.workers, max_keepalive_connections=self.workers),
            timeout=httpx.Timeout(timeout, connect=20.0),
            follow_redirects=True,
            headers={"User-Agent": "soilgrids-data/0.1 (+https://github.com/colinahill/soilgrids_data)"},
        )

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- primitives ---------------------------------------------------------

    def _request(self, method: str, url: str, **kw) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._limiter.wait()
            try:
                resp = self._client.request(method, url, **kw)
                if resp.status_code in RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"{resp.status_code} {resp.reason_phrase}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in RETRY_STATUS:
                    raise FetchError(f"{method} {url} failed: HTTP {status}") from None
                last = exc
                if attempt == MAX_ATTEMPTS:
                    break
                delay = RETRY_BASE_SECONDS * 2 ** (attempt - 1) * (0.5 + random.random())
                log.warning(
                    "%s %s attempt %d/%d failed (%s); retrying in %.1fs", method, url, attempt, MAX_ATTEMPTS, exc, delay
                )
                time.sleep(delay)
        raise FetchError(f"{method} {url} failed after {MAX_ATTEMPTS} attempts: {last}") from last

    def head(self, url: str) -> ObjectInfo:
        r = self._request("HEAD", url)
        return ObjectInfo(
            url=url,
            size=int(r.headers.get("content-length", -1)),
            etag=r.headers.get("etag"),
            last_modified=r.headers.get("last-modified"),
        )

    def get(self, url: str, *, expect_etag: str | None = None, expect_size: int | None = None) -> bytes:
        """Fetch a whole object, verifying it is the one the manifest recorded."""
        r = self._request("GET", url)
        etag = r.headers.get("etag")
        if expect_etag and etag and etag != expect_etag:
            raise SourceChanged(f"{url}: ETag is {etag}, manifest recorded {expect_etag}")
        body = r.content
        if expect_size is not None and len(body) != expect_size:
            raise SourceChanged(f"{url}: {len(body)} bytes, manifest recorded {expect_size}")
        return body

    def get_range(self, url: str, start: int, length: int) -> bytes:
        """Fetch a byte range -- used to read a tile's header without its pixels."""
        r = self._request("GET", url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        return r.content

    def get_text(self, url: str) -> str:
        return self._request("GET", url).text

    def map(self, fn, items, *, workers: int | None = None):
        """Run ``fn`` over ``items`` on the shared pool, preserving order."""
        with ThreadPoolExecutor(max_workers=workers or self.workers) as pool:
            return list(pool.map(fn, items))


# -- WebDAV directory listings ---------------------------------------------

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<td>(.*?)</td>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_SIZE_RE = re.compile(r"^([\d,]+)\s*Bytes$")


def parse_listing(html: str) -> dict[str, int | None]:
    """Parse an ISRIC WsgiDAV directory listing into {name: size or None}.

    Directories get ``None``. Used to cross-check the VRT inventory: a tile
    present on disk but missing from a VRT would otherwise be dropped silently.
    """
    out: dict[str, int | None] = {}
    for body in _ROW_RE.findall(html):
        cells = [_TAG_RE.sub("", c).strip() for c in _CELL_RE.findall(body)]
        if len(cells) != 4:
            continue
        name, kind, size, _modified = cells
        if not name:
            continue
        if kind == "Directory":
            out[name.rstrip("/")] = None
        elif m := _SIZE_RE.match(size):
            out[name] = int(m.group(1).replace(",", ""))
        else:
            out[name] = None
    return out
