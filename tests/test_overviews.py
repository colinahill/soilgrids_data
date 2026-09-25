"""The overview kernel must be fill-aware, and the fill must never default.

The fill-value trap is the reason this module drives topozarr's kernel directly,
so it gets a regression test with a hand-computed reference.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from topozarr.engine import block_reduce

from soilgrids import config

FILL = config.FILL_VALUE  # NaN: the store holds decoded float32


def _reference_mean(a: np.ndarray, factor: int = 2) -> np.ndarray:
    """NaN-aware mean of factor x factor blocks (a true mean: no truncation)."""
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    blocks = (
        a[:h, :w]
        .reshape(h // factor, factor, w // factor, factor)
        .transpose(0, 2, 1, 3)
        .reshape(h // factor, w // factor, factor * factor)
    )
    valid = ~np.isnan(blocks)
    n = valid.sum(axis=-1)
    total = np.where(valid, blocks, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan).astype("float32")


@pytest.fixture
def coastline() -> np.ndarray:
    """A field with a wavy nodata boundary, so many 2x2 blocks straddle it.

    A straight diagonal only makes ~16 mixed blocks in a 60x60 grid; a real
    coastline wanders, and the mixed blocks are exactly the case that the fill
    value has to be right for.
    """
    yy, xx = np.mgrid[0:60, 0:60]
    base = (30.0 + 10 * np.sin(xx / 7.0) + 5 * np.cos(yy / 5.0)).astype("float32")
    boundary = 30 + 9 * np.sin(xx / 1.5) + 5 * np.cos(yy / 2.0)
    return np.where(xx + yy > boundary, base, np.nan).astype("float32")


def test_kernel_mean_is_fill_aware(coastline):
    got = block_reduce(np.ascontiguousarray(coastline[None]), (1, 2, 2), "mean", FILL, True)[0]
    np.testing.assert_allclose(got, _reference_mean(coastline), rtol=1e-6, equal_nan=True)


def test_mixed_blocks_actually_exist_in_the_fixture(coastline):
    blocks = coastline[:60, :60].reshape(30, 2, 30, 2).transpose(0, 2, 1, 3).reshape(30, 30, 4)
    n = (~np.isnan(blocks)).sum(axis=-1)
    assert ((n > 0) & (n < 4)).sum() > 30, "fixture must straddle the boundary in many blocks"


def test_nan_is_self_identifying_so_the_fill_argument_no_longer_matters(coastline):
    """Recorded deliberately: float32/NaN removes the trap that int16 storage had.

    With a -32768 sentinel, omitting fill_value made the kernel average the
    sentinel as data (-16207 where the answer was 352). NaN carries its own
    "no data" meaning, so skipna handles it either way. fill_value is still
    passed, because the all-fill shard skip compares against dst.fill_value,
    but a mistake there can no longer corrupt a coastline.
    """
    good = block_reduce(np.ascontiguousarray(coastline[None]), (1, 2, 2), "mean", FILL, True)[0]
    unset = block_reduce(np.ascontiguousarray(coastline[None]), (1, 2, 2), "mean", None, True)[0]
    np.testing.assert_array_equal(good, unset)
    blocks = coastline.reshape(30, 2, 30, 2).transpose(0, 2, 1, 3).reshape(30, 30, 4)
    n = (~np.isnan(blocks)).sum(axis=-1)
    mixed = (n > 0) & (n < 4)
    assert mixed.sum() > 30
    assert not np.isnan(good[mixed]).any(), "mixed land/nodata cells must survive"


def test_float32_dtype_is_preserved_and_the_mean_is_not_truncated(coastline):
    got = block_reduce(np.ascontiguousarray(coastline[None]), (1, 2, 2), "mean", FILL, True)
    assert got.dtype == np.dtype("float32")
    block = np.array([[[10.0, 20.0], [30.0, 41.0]]], dtype="float32")
    assert block_reduce(block, (1, 2, 2), "mean", FILL, True)[0, 0, 0] == np.float32(25.25)


def test_all_fill_blocks_stay_fill():
    a = np.full((20, 20), np.nan, dtype="float32")
    got = block_reduce(np.ascontiguousarray(a[None]), (1, 2, 2), "mean", FILL, True)[0]
    assert np.all(np.isnan(got))


def test_downsample_writes_nothing_for_an_all_fill_source(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from topozarr.engine import downsample_level

    root = zarr.open_group(zarr.storage.LocalStore(str(tmp_path / "z")), mode="a")
    kw = dict(chunks=(5, 5), shards=(20, 20), dtype="float32", fill_value=FILL)
    src = root.create_array("s", shape=(40, 40), **kw)
    dst = root.create_array("d", shape=(20, 20), **kw)
    untouched = root.create_array("u", shape=(20, 20), **kw)
    with ThreadPoolExecutor(2) as pool:
        for f in downsample_level(src, dst, stride=(2, 2), method="mean", fill_value=FILL, executor=pool):
            f.result()
    assert np.all(np.isnan(dst[:]))
    # compare against an array that was never written: metadata counts either way,
    # so the only question is whether any chunk objects landed
    assert dst.nbytes_stored() == untouched.nbytes_stored(), "all-fill shards must not be written"


def test_downsample_writes_only_the_shards_that_have_data(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from topozarr.engine import downsample_level

    root = zarr.open_group(zarr.storage.LocalStore(str(tmp_path / "z2")), mode="a")
    kw = dict(chunks=(5, 5), shards=(20, 20), dtype="float32", fill_value=FILL)
    src = root.create_array("s", shape=(80, 80), **kw)
    src[:20, :20] = 40.0  # one output shard's worth of real data
    dst = root.create_array("d", shape=(40, 40), **kw)
    with ThreadPoolExecutor(2) as pool:
        for f in downsample_level(src, dst, stride=(2, 2), method="mean", fill_value=FILL, executor=pool):
            f.result()
    assert np.all(dst[:10, :10] == 40.0)
    assert np.all(np.isnan(dst[10:, :])) and np.all(np.isnan(dst[:, 10:]))


class _FlakyWrites:
    """A dst array whose first ``failures`` writes raise, like the Source Coop
    gateway's unparseable error responses."""

    def __init__(self, arr, failures: int, message: str = "object store error service error: error parsing XML"):
        self._arr, self.failures, self.message, self.attempts = arr, failures, message, 0

    def __getattr__(self, name):
        return getattr(self._arr, name)

    def __setitem__(self, key, value):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError(self.message)
        self._arr[key] = value


def _one_shard_arrays(tmp_path):
    root = zarr.open_group(zarr.storage.LocalStore(str(tmp_path / "z")), mode="a")
    kw = dict(chunks=(5, 5), shards=(20, 20), dtype="float32", fill_value=FILL)
    src = root.create_array("s", shape=(40, 40), **kw)
    src[:] = 40.0
    return src, root.create_array("d", shape=(20, 20), **kw)


def test_reduce_region_retries_a_transient_write_failure(tmp_path, monkeypatch):
    from soilgrids import materialize, overviews

    monkeypatch.setattr(materialize, "RETRY_BASE_SECONDS", 0.0)
    src, dst = _one_shard_arrays(tmp_path)
    flaky = _FlakyWrites(dst, failures=2)
    assert overviews._reduce_region(src, flaky, (slice(0, 20), slice(0, 20)), (2, 2))
    assert flaky.attempts == 3
    assert np.all(dst[:] == 40.0)


def test_reduce_region_stops_at_once_on_expired_credentials(tmp_path, monkeypatch):
    from soilgrids import materialize, overviews

    monkeypatch.setattr(materialize, "RETRY_BASE_SECONDS", 0.0)
    src, dst = _one_shard_arrays(tmp_path)
    flaky = _FlakyWrites(dst, failures=99, message="ExpiredToken: the provided token has expired")
    with pytest.raises(materialize.CredentialsExpired):
        overviews._reduce_region(src, flaky, (slice(0, 20), slice(0, 20)), (2, 2))
    assert flaky.attempts == 1
