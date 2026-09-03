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
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import zarr
from icechunk import Session
from topozarr.engine import block_reduce

from . import config, grid

log = logging.getLogger(__name__)

OVERVIEW_ATTR = "overviews_built"


@dataclass
class LevelStats:
    factor: int
    shape: tuple[int, ...]
    regions: int = 0
    written: int = 0


@dataclass
class OverviewStats:
    levels: list[LevelStats] = field(default_factory=list)


def _stride(ndim: int) -> tuple[int, ...]:
    return (1, 2, 2) if ndim == 3 else (2, 2)


def _regions(dst: zarr.Array, ndim: int, only_cells: set[str] | None, factor: int) -> Iterator[tuple[slice, ...]]:
    """Shard-aligned destination regions, restricted to where data can exist.

    One region per (depth, shard) so writes never straddle a storage object and
    two workers never touch one. ``only_cells`` are native tileSG cells; a cell
    maps to a level window by dividing by ``factor``.
    """
    shards = tuple(dst.shards or dst.chunks)
    sy, sx = shards[-2], shards[-1]
    h, w = dst.shape[-2], dst.shape[-1]
    depths = range(dst.shape[0]) if ndim == 3 else [None]

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


def _reduce_region(src: zarr.Array, dst: zarr.Array, region, stride: tuple[int, ...]) -> bool:
    """Reduce one destination region from ``src``. Returns True if it was written."""
    spatial = region[-2:]
    lead = region[:-2]
    in_sel = tuple(
        slice(s.start * f, min(s.stop * f, n)) for s, f, n in zip(spatial, stride[-2:], src.shape[-2:], strict=True)
    )
    block = np.ascontiguousarray(src[(*lead, *in_sel)])
    if not np.any(~np.isnan(block)):
        return False  # all-fill: never written, so it costs nothing
    shaped = block if block.ndim == len(stride) else block[np.newaxis]
    out = block_reduce(shaped, stride, config.OVERVIEW_RESAMPLING, config.FILL_VALUE, True)
    dst[region] = out if block.ndim == len(stride) else out[0]
    return True


def build_property(
    session: Session,
    spec: config.PropertySpec,
    *,
    factors: tuple[int, ...] | None = None,
    workers: int | None = None,
    progress: bool = True,
    only_cells: set[str] | None = None,
) -> OverviewStats:
    """Build every overview level for one property. Commits nothing.

    ``only_cells`` restricts work to the native cells that have been
    materialized, which is what keeps the pass proportional to the data rather
    than to the grid.
    """
    factors = config.OVERVIEW_FACTORS if factors is None else factors
    shapes = grid.overview_shapes(factors=factors)
    group = zarr.open_group(session.store, path=spec.group, mode="r+")
    stats = OverviewStats()

    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=len(factors), desc=f"overviews {spec.name}", unit="level")

    src = group[spec.name]
    for factor in factors:
        expected = shapes[factor]
        dst = group[f"{factor}x"][spec.name]
        want = (spec.depth_extent, *expected) if spec.ndim == 3 else expected
        if tuple(dst.shape) != want:
            raise RuntimeError(
                f"{spec.group}/{factor}x/{spec.name}: shape {tuple(dst.shape)} != expected {want}; "
                f"re-run init-store (level shapes are floor(parent/2), not ceil(native/factor))"
            )
        if not config.is_fill(dst.fill_value):
            raise RuntimeError(
                f"{spec.group}/{factor}x/{spec.name}: fill_value is {dst.fill_value}, expected "
                f"NaN; a wrong fill silently corrupts every coastline"
            )
        regions = list(_regions(dst, spec.ndim, only_cells, factor))
        stride = _stride(spec.ndim)
        written = 0

        # bind src/dst/stride as defaults: a bare closure over the loop variables
        # would reduce every level from whatever `src` ended up as
        def work(region, _src=src, _dst=dst, _stride=stride):
            return _reduce_region(_src, _dst, region, _stride)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ok in pool.map(work, regions):
                written += bool(ok)
        stats.levels.append(LevelStats(factor=factor, shape=tuple(dst.shape), regions=len(regions), written=written))
        log.info(
            "%s: %dx level written (%s, %d of %d regions had data)",
            spec.name,
            factor,
            tuple(dst.shape),
            written,
            len(regions),
        )
        if bar is not None:
            bar.update(1)
        src = dst  # chain: the next level reduces this one
    if bar is not None:
        bar.close()
    return stats


def built(session_or_store, group: str) -> set[str]:
    try:
        gp = zarr.open_group(getattr(session_or_store, "store", session_or_store), path=group, mode="r")
    except (KeyError, FileNotFoundError):
        return set()
    return set(gp.attrs.get(OVERVIEW_ATTR, []))


def mark_built(session: Session, group: str, prop: str) -> None:
    gp = zarr.open_group(session.store, path=group, mode="r+")
    gp.attrs[OVERVIEW_ATTR] = sorted(set(gp.attrs.get(OVERVIEW_ATTR, [])) | {prop})
