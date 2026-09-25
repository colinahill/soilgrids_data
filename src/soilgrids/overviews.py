"""Phase 4: multiscale overview pyramids.

Driven through ``topozarr.engine.downsample_level`` rather than
``topozarr.create_pyramid``. The kernel takes explicit src/dst zarr arrays and an
explicit ``fill_value``, which buys three things over the high-level API:

* **the fill value is passed as an argument**, never sniffed from
  ``encoding["_FillValue"]``. This mattered enormously while the store held
  scaled Int16: with a -32768 sentinel, the high-level API resolved the fill to
  ``None`` and averaged the sentinel as data -- measured at -16207 where the
  correct value was 352 -- besides declaring the level's fill as 0 (a legal soil
  value) and writing every all-ocean shard. Now that the store holds float32/NaN
  the danger is gone (NaN is self-identifying, so ``skipna`` handles it either
  way, verified), but the explicit argument is kept: ``skip_empty`` compares
  against ``dst.fill_value``, and being explicit costs nothing;
* **no fusion path**, so memory is bounded by the region size by construction
  rather than by how much RAM happens to be free;
* **our layout**: ``create_pyramid`` imposes ordinal level groups (``0/``,
  ``1/``) at the store root and wants native at ``0/<var>``, which would mean
  duplicating or relocating 158 GB. Here native stays at the group root and the
  levels are factor-named children.

Each level is reduced from the level above (a chained stride-2 mean), streamed in
shard-aligned regions on a thread pool. Because level shards are depth-1, the
work proceeds one (property, depth) layer at a time; ~1.6 MB per in-flight
region.

The region loop is ours rather than ``downsample_level``'s, for one reason:
``downsample_level`` walks every region of the destination grid, and only
discovers a window is empty after reading it. Land is 24.2 % of this grid, so on
a full store that is ~72 % wasted reads (70 488 regions for the 2x level alone),
and on a partial or development store it is ~99 %. Passing the materialized cells
in lets the loop visit only regions that can contain data. The reduction itself
is still topozarr's Rust kernel.

``boundary="trim"`` semantics mean level shapes are floor(parent/2), so the chain
discards at most one coarse pixel per level at the far right/bottom edge.

The pass is checkpointable at region granularity. The region ordering is a pure
function of (cells, factors), so a resume token is just "level f, first n regions
of that ordering" plus a fingerprint of the scope -- a few dozen bytes on the
group attrs rather than a list of 37 000 region keys. A scope change (different
``--cells``) changes the fingerprint, and the run restarts rather than silently
skipping work it never did.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import zarr
from icechunk import Session
from topozarr.engine import block_reduce

from . import config, grid
from .materialize import retry_transient

log = logging.getLogger(__name__)

OVERVIEW_ATTR = "overviews_built"
# Resume token for an interrupted run: {"key", "factor", "done", "total"}. It
# lives on the FIRST LEVEL'S ARRAY, not on the group that carries
# OVERVIEW_ATTR, because attrs are one object per node and icechunk cannot merge
# two writers' updates to one node (ZarrMetadataDoubleUpdate is unsolvable by
# any conflict solver). materialize writes its cell ledger to the group attrs
# throughout a multi-hour backfill, so a token there would make every checkpoint
# of a concurrent overviews run unrebaseable. On the level array, nothing else
# writes it, and the checkpoint rebases cleanly over phase 3.
PROGRESS_ATTR = "overviews_progress"


@dataclass
class LevelStats:
    factor: int
    shape: tuple[int, ...]
    regions: int = 0
    written: int = 0
    skipped: int = 0  # already done before this run, per the resume token


@dataclass
class OverviewStats:
    levels: list[LevelStats] = field(default_factory=list)
    total_regions: int = 0
    regions_done: int = 0  # this run only; excludes regions skipped by a resume
    shards_written: int = 0  # live, so a checkpoint message mid-level is not 0


def _stride(ndim: int) -> tuple[int, ...]:
    return (1, 2, 2) if ndim == 3 else (2, 2)


def _regions(
    shape: tuple[int, ...], shards: tuple[int, int], ndim: int, only_cells: set[str] | None, factor: int
) -> Iterator[tuple[slice, ...]]:
    """Shard-aligned destination regions, restricted to where data can exist.

    One region per (depth, shard) so writes never straddle a storage object and
    two workers never touch one. ``only_cells`` are native tileSG cells; a cell
    maps to a level window by dividing by ``factor``.

    Takes shape/shards rather than the array so the whole ladder can be planned
    (and counted, for progress and for a resume token) before anything is opened
    -- array handles do not survive a checkpoint commit, plans do.
    """
    sy, sx = shards
    h, w = shape[-2], shape[-1]
    depths = range(shape[0]) if ndim == 3 else [None]

    keep: set[tuple[int, int]] | None = None
    if only_cells is not None:
        n = grid.cell_px()
        keep = set()
        for cell in only_cells:
            row, col = (int(v) for v in cell.split("-"))
            y0, y1 = row * n // factor, -(-((row + 1) * n) // factor)
            x0, x1 = col * n // factor, -(-((col + 1) * n) // factor)
            for by in range(y0 // sy, min(-(-y1 // sy), -(-h // sy))):
                for bx in range(x0 // sx, min(-(-x1 // sx), -(-w // sx))):
                    keep.add((by, bx))

    blocks = sorted(keep) if keep is not None else [(by, bx) for by in range(-(-h // sy)) for bx in range(-(-w // sx))]
    for d in depths:
        for by, bx in blocks:
            ys = slice(by * sy, min((by + 1) * sy, h))
            xs = slice(bx * sx, min((bx + 1) * sx, w))
            if ys.start >= ys.stop or xs.start >= xs.stop:
                continue
            yield (d, ys, xs) if d is not None else (ys, xs)


def _plan(spec: config.PropertySpec, factors: tuple[int, ...], only_cells: set[str] | None) -> dict[int, list[tuple]]:
    """The full region ordering for every level, before any array is opened.

    Deterministic in (cells, factors): ``build_property`` checks each level's
    real shape against the same ``grid.overview_shapes`` this is built from, so a
    store that disagrees raises there rather than being quietly mis-planned.
    """
    shapes = grid.overview_shapes(factors=factors)
    enc = config.ENCODING
    shards = (enc.shard_y, enc.shard_x)
    plan = {}
    for factor in factors:
        shape = (spec.depth_extent, *shapes[factor]) if spec.ndim == 3 else shapes[factor]
        plan[factor] = list(_regions(shape, shards, spec.ndim, only_cells, factor))
    return plan


def _scope_key(spec: config.PropertySpec, factors: tuple[int, ...], only_cells: set[str] | None) -> str:
    """Fingerprint of what a run was asked to do; a resume token is only valid for its own scope."""
    payload = json.dumps(
        {
            "property": spec.name,
            "factors": list(factors),
            "cells": sorted(only_cells) if only_cells is not None else None,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class _Reporter:
    """Region-level progress.

    The unit is the region, not the level. Every region reduces one parent
    window into one shard, so regions are near-uniform work, while levels differ
    by 4x each: on a 0..8 level bar the first tick is ~73 % of the run. A tqdm
    bar on a TTY; periodic log lines otherwise, because this pass is normally
    run detached against a remote store, where a redrawn bar is 37 000 log lines.
    """

    def __init__(self, name: str, total: int, *, enabled: bool = True, log_every: int = 0) -> None:
        self.name, self.total = name, total
        self.done = 0
        self.factor = 0
        self.bar = None
        self.log_every = 0
        self.started = time.monotonic()
        self.last_logged = 0
        if not enabled:
            return
        if sys.stderr.isatty():
            from tqdm import tqdm

            self.bar = tqdm(total=total, desc=f"overviews {name}", unit="region", smoothing=0.05)
        else:
            self.log_every = log_every or max(total // 50, 1)

    def start_level(self, factor: int, regions: int, skipped: int) -> None:
        self.factor = factor
        if self.bar is not None:
            self.bar.set_description(f"overviews {self.name} {factor}x")
        if skipped:
            self.advance(skipped, silent=True)

    def advance(self, n: int = 1, *, silent: bool = False) -> None:
        self.done += n
        if self.bar is not None:
            self.bar.update(n)
            return
        if silent or not self.log_every or self.done - self.last_logged < self.log_every:
            return
        self.last_logged = self.done
        elapsed = time.monotonic() - self.started
        rate = self.done / elapsed if elapsed else 0.0
        log.info(
            "%s: %dx level, %d/%d regions (%.1f%%), %.1f region/s, ETA %s",
            self.name,
            self.factor,
            self.done,
            self.total,
            100.0 * self.done / max(self.total, 1),
            rate,
            _hms((self.total - self.done) / rate if rate else 0.0),
        )

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _reduce_region(src: zarr.Array, dst: zarr.Array, region, stride: tuple[int, ...]) -> bool:
    """Reduce one destination region from ``src``. Returns True if it was written."""
    spatial = region[-2:]
    lead = region[:-2]
    in_sel = tuple(
        slice(s.start * f, min(s.stop * f, n)) for s, f, n in zip(spatial, stride[-2:], src.shape[-2:], strict=True)
    )
    where = f"{dst.path} {[(s.start, s.stop) if isinstance(s, slice) else s for s in region]}"
    block = retry_transient(lambda: np.ascontiguousarray(src[(*lead, *in_sel)]), what=f"read for {where}")
    if not np.any(~np.isnan(block)):
        return False  # all-fill: never written, so it costs nothing
    shaped = block if block.ndim == len(stride) else block[np.newaxis]
    out = block_reduce(shaped, stride, config.OVERVIEW_RESAMPLING, config.FILL_VALUE, True)
    value = out if block.ndim == len(stride) else out[0]

    def write():
        dst[region] = value

    retry_transient(write, what=f"write {where}")
    return True


def build_property(
    session: Session,
    spec: config.PropertySpec,
    *,
    factors: tuple[int, ...] | None = None,
    workers: int | None = None,
    progress: bool = True,
    progress_every: int = 0,
    only_cells: set[str] | None = None,
    commit_every: int = 0,
    checkpoint: Callable[[Session, OverviewStats], Session] | None = None,
) -> tuple[Session, OverviewStats]:
    """Build every overview level for one property. Commits nothing itself.

    ``only_cells`` restricts work to the native cells that have been
    materialized, which is what keeps the pass proportional to the data rather
    than to the grid.

    ``checkpoint(session, stats) -> Session`` is called every ``commit_every``
    regions (and at every level boundary); it must commit the session it is given
    and return a fresh one. The resume token is written to the group attrs
    *before* the commit, so it is never ahead of the data it describes. Returns
    the final session alongside the stats, so the caller commits the tail.
    """
    factors = config.OVERVIEW_FACTORS if factors is None else factors
    shapes = grid.overview_shapes(factors=factors)
    enc = config.ENCODING
    plan = _plan(spec, factors, only_cells)
    key = _scope_key(spec, factors, only_cells)
    stats = OverviewStats(total_regions=sum(len(r) for r in plan.values()))

    resume_at, resume_done = 0, 0
    state = progress_state(session, spec, factors)
    if state and state.get("key") == key and state.get("factor") in factors:
        resume_at = factors.index(int(state["factor"]))
        resume_done = min(int(state.get("done", 0)), len(plan[factors[resume_at]]))
        log.info(
            "%s: resuming at the %dx level, %d of its %d regions already done (%d of %d overall)",
            spec.name,
            factors[resume_at],
            resume_done,
            len(plan[factors[resume_at]]),
            sum(len(plan[f]) for f in factors[:resume_at]) + resume_done,
            stats.total_regions,
        )
    elif state:
        log.warning("%s: ignoring a resume token written for a different scope; starting from the top", spec.name)

    reporter = _Reporter(spec.name, stats.total_regions, enabled=progress, log_every=progress_every)
    try:
        for i, factor in enumerate(factors):
            regions = plan[factor]
            group = zarr.open_group(session.store, path=spec.group, mode="r+")
            dst = group[f"{factor}x"][spec.name]
            want = (spec.depth_extent, *shapes[factor]) if spec.ndim == 3 else shapes[factor]
            if tuple(dst.shape) != want:
                raise RuntimeError(
                    f"{spec.group}/{factor}x/{spec.name}: shape {tuple(dst.shape)} != expected {want}; "
                    f"re-run init-store (level shapes are floor(parent/2), not ceil(native/factor))"
                )
            if tuple(dst.shards or dst.chunks)[-2:] != (enc.shard_y, enc.shard_x):
                raise RuntimeError(
                    f"{spec.group}/{factor}x/{spec.name}: shard grid {tuple(dst.shards or dst.chunks)} does not "
                    f"match EncodingSpec ({enc.shard_y}, {enc.shard_x}); the region plan would not be shard-aligned"
                )
            if not config.is_fill(dst.fill_value):
                raise RuntimeError(
                    f"{spec.group}/{factor}x/{spec.name}: fill_value is {dst.fill_value}, expected "
                    f"NaN; a wrong fill silently corrupts every coastline"
                )

            # levels below the resume point are complete; the one at it restarts
            # mid-ordering, which is safe because the ordering is deterministic
            pos = len(regions) if i < resume_at else (resume_done if i == resume_at else 0)
            level = LevelStats(factor=factor, shape=tuple(dst.shape), regions=len(regions), skipped=pos)
            reporter.start_level(factor, len(regions), pos)
            src = group[spec.name] if i == 0 else group[f"{factors[i - 1]}x"][spec.name]
            stride = _stride(spec.ndim)
            batch = commit_every if (commit_every and checkpoint is not None) else len(regions)

            while pos < len(regions):
                todo = regions[pos : pos + max(batch, 1)]

                # bind src/dst/stride as defaults: a bare closure over the loop
                # variables would reduce every level from whatever `src` ended up
                # as, and would hold handles from a session a checkpoint retired
                def work(region, _src=src, _dst=dst, _stride=stride):
                    return _reduce_region(_src, _dst, region, _stride)

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for ok in pool.map(work, todo):
                        level.written += bool(ok)
                        stats.shards_written += bool(ok)
                        reporter.advance()
                pos += len(todo)
                stats.regions_done += len(todo)
                if checkpoint is not None and commit_every:
                    record_progress(session, spec, factors, key=key, factor=factor, done=pos, total=stats.total_regions)
                    session = checkpoint(session, stats)
                    group = zarr.open_group(session.store, path=spec.group, mode="r+")
                    src = group[spec.name] if i == 0 else group[f"{factors[i - 1]}x"][spec.name]
                    dst = group[f"{factor}x"][spec.name]

            stats.levels.append(level)
            log.info(
                "%s: %dx level written (%s, %d of %d regions had data%s)",
                spec.name,
                factor,
                tuple(dst.shape),
                level.written,
                len(regions),
                f", {level.skipped} already done" if level.skipped else "",
            )
    finally:
        reporter.close()
    clear_progress(session, spec, factors)
    return session, stats


def _progress_array(session_or_store, spec: config.PropertySpec, factors: tuple[int, ...], mode: str = "r"):
    """The node the resume token lives on: the first level's array for this property."""
    store_ = session_or_store.store if hasattr(session_or_store, "store") else session_or_store
    return zarr.open_array(store_, path=f"{spec.group}/{factors[0]}x/{spec.name}", mode=mode)


def progress_state(session_or_store, spec: config.PropertySpec, factors: tuple[int, ...] | None = None) -> dict | None:
    """The resume token for one property, or None."""
    factors = config.OVERVIEW_FACTORS if factors is None else factors
    try:
        arr = _progress_array(session_or_store, spec, factors)
    except (KeyError, FileNotFoundError, zarr.errors.NodeNotFoundError):
        return None
    return arr.attrs.get(PROGRESS_ATTR)


def record_progress(
    session: Session,
    spec: config.PropertySpec,
    factors: tuple[int, ...],
    *,
    key: str,
    factor: int,
    done: int,
    total: int,
) -> None:
    arr = _progress_array(session, spec, factors, mode="r+")
    arr.attrs[PROGRESS_ATTR] = {"key": key, "factor": int(factor), "done": int(done), "total": int(total)}


def clear_progress(session: Session, spec: config.PropertySpec, factors: tuple[int, ...]) -> None:
    """Drop a finished property's resume token. A no-op when there is none, so a
    run that never checkpointed does not dirty the session just to say so."""
    arr = _progress_array(session, spec, factors, mode="r+")
    if PROGRESS_ATTR not in arr.attrs:
        return
    del arr.attrs[PROGRESS_ATTR]


def built(session_or_store, group: str) -> set[str]:
    try:
        gp = zarr.open_group(getattr(session_or_store, "store", session_or_store), path=group, mode="r")
    except (KeyError, FileNotFoundError):
        return set()
    return set(gp.attrs.get(OVERVIEW_ATTR, []))


def mark_built(session: Session, group: str, prop: str) -> None:
    gp = zarr.open_group(session.store, path=group, mode="r+")
    gp.attrs[OVERVIEW_ATTR] = sorted(set(gp.attrs.get(OVERVIEW_ATTR, [])) | {prop})
