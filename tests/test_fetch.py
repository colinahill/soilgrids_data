"""fetch.py: retries, rate limiting, and source-drift detection.

ISRIC is a public academic server that throttles per connection, so this layer is
built to be polite and durable rather than fast. The behaviour that matters is
what it does when things go wrong.
"""

from __future__ import annotations

import time

import httpx
import pytest

from soilgrids import fetch


def _fetcher(handler, **kw) -> fetch.Fetcher:
    f = fetch.Fetcher(workers=2, **kw)
    f._client = httpx.Client(transport=httpx.MockTransport(handler))
    return f


def test_get_returns_the_body():
    with _fetcher(lambda r: httpx.Response(200, content=b"abc")) as f:
        assert f.get("https://x/y") == b"abc"


@pytest.mark.parametrize("status", sorted(fetch.RETRY_STATUS))
def test_transient_statuses_are_retried_then_succeed(monkeypatch, status):
    monkeypatch.setattr(fetch, "RETRY_BASE_SECONDS", 0.0)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, content=b"ok") if len(calls) > 2 else httpx.Response(status)

    with _fetcher(handler) as f:
        assert f.get("https://x/y") == b"ok"
    assert len(calls) == 3


def test_permanent_statuses_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    with _fetcher(handler) as f, pytest.raises(fetch.FetchError, match="HTTP 404"):
        f.get("https://x/y")
    assert len(calls) == 1, "a 404 must not be retried"


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(fetch, "RETRY_BASE_SECONDS", 0.0)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    with _fetcher(handler) as f, pytest.raises(fetch.FetchError, match="after 6 attempts"):
        f.get("https://x/y")
    assert len(calls) == fetch.MAX_ATTEMPTS


def test_transport_errors_are_retried(monkeypatch):
    monkeypatch.setattr(fetch, "RETRY_BASE_SECONDS", 0.0)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectTimeout("boom")
        return httpx.Response(200, content=b"ok")

    with _fetcher(handler) as f:
        assert f.get("https://x/y") == b"ok"


def test_etag_mismatch_raises_source_changed():
    """`latest/` is mutable: a replaced object must not be mixed into the store."""
    handler = lambda r: httpx.Response(200, content=b"abc", headers={"etag": '"new"'})  # noqa: E731
    with _fetcher(handler) as f, pytest.raises(fetch.SourceChanged, match="ETag is"):
        f.get("https://x/y", expect_etag='"old"')


def test_size_mismatch_raises_source_changed():
    with _fetcher(lambda r: httpx.Response(200, content=b"abc")) as f:
        with pytest.raises(fetch.SourceChanged, match="3 bytes, manifest recorded 99"):
            f.get("https://x/y", expect_size=99)
        assert f.get("https://x/y", expect_size=3) == b"abc", "a match must pass"


def test_matching_etag_passes():
    handler = lambda r: httpx.Response(200, content=b"abc", headers={"etag": '"same"'})  # noqa: E731
    with _fetcher(handler) as f:
        assert f.get("https://x/y", expect_etag='"same"') == b"abc"


def test_source_changed_is_never_retried(monkeypatch):
    monkeypatch.setattr(fetch, "RETRY_BASE_SECONDS", 0.0)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, content=b"abc", headers={"etag": '"new"'})

    with _fetcher(handler) as f, pytest.raises(fetch.SourceChanged):
        f.get("https://x/y", expect_etag='"old"')
    assert len(calls) == 1


def test_head_reports_size_etag_and_modified():
    handler = lambda r: httpx.Response(  # noqa: E731
        200, headers={"content-length": "406513", "etag": '"e"', "last-modified": "Sat, 11 Apr 2020 17:24:06 GMT"}
    )
    with _fetcher(handler) as f:
        info = f.head("https://x/y")
    assert (info.size, info.etag) == (406513, '"e"')
    assert info.last_modified.startswith("Sat, 11 Apr")


def test_rate_limiter_paces_requests():
    with _fetcher(lambda r: httpx.Response(200, content=b"x"), rate_limit=20.0) as f:
        t0 = time.monotonic()
        for _ in range(5):
            f.get("https://x/y")
        elapsed = time.monotonic() - t0
    assert elapsed >= 0.15, f"5 requests at 20/s should take >=0.2s, took {elapsed:.3f}s"


def test_map_preserves_order():
    with _fetcher(lambda r: httpx.Response(200, content=b"x")) as f:
        assert f.map(lambda i: i * 2, range(6)) == [0, 2, 4, 6, 8, 10]
