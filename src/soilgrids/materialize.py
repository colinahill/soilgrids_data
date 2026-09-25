"""Phase 3: fill the native arrays from the ISRIC tile tree.

The unit of work is one **(property, cell)**: one ``tileSG`` cell (1800 px) of
one property, across all of its depths. That is the smallest unit that holds
whole source tiles for both subtilings (16 of 450 px, or 9 of 600 px) *and* whole
shards (4x4), so every tile is fetched exactly once, no shard is ever
read-modify-written, and two workers never touch one storage object.

A cell buffer is 38.9 MB for six depths, so memory stays flat no matter how much
of the grid a run covers. Raw tiles are never written to disk.

Each tile is placed using **its own GeoTIFF tie-point**, asserted against the
manifest row: the file is the ground truth at write time, not the VRT.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import zarr
from icechunk import Session

from . import config, grid, store, tiff
from .fetch import Fetcher, SourceChanged

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
RETRY_BASE_SECONDS = 2.0

# progress/provenance attrs, written on the owning group
DONE_ATTR = "materialized_cells"
COMPLETE_ATTR = "materialized_properties"
COARSE_ATTR = "coarse_source_fills"


class CredentialsExpired(RuntimeError):
    """Object-store credentials are dead; stop at the last checkpoint.

    Raised instead of retrying: every remaining shard would fail the same way,
    so a long backfill should stop and tell the operator to re-authenticate.
    """


class PlacementMismatch(RuntimeError):
    """A tile's own tie-point/shape disagrees with the manifest."""


@dataclass
class CellStats:
    cells_done: int = 0
    tiles_fetched: int = 0
    tiles_missing: int = 0
    shards_written: int = 0
    shards_empty: int = 0
    bytes_fetched: int = 0
    anomalies: list[str] = field(default_factory=list)
    # "{relpath}#{depth}" -> canonical pixels actually filled from a coarse source
    coarse_pixels: dict[str, int] = field(default_factory=dict)


def retry_transient[T](operation: Callable[[], T], *, what: str) -> T:
    """Run ``operation``, retrying transient object-store errors with backoff.

    S3-compatible gateways return responses the AWS SDK cannot classify as
    retryable, so neither it nor icechunk retries them. Expired credentials are
    the exception: they would fail every remaining write too.
    """
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return operation()
        except SourceChanged:
            raise
        except Exception as exc:
            if store.is_credentials_failure(exc):
                raise CredentialsExpired(
                    f"{what}: credentials rejected ({exc}). Run `source-coop login` "
                    f"(or refresh creds.json) and re-run; the last checkpoint is intact."
                ) from None
            last = exc
            if attempt == MAX_ATTEMPTS:
                break
            delay = RETRY_BASE_SECONDS * 2 ** (attempt - 1) * (0.5 + random.random())
            log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs", what, attempt, MAX_ATTEMPTS, exc, delay)
            time.sleep(delay)
    raise RuntimeError(f"{what} failed after {MAX_ATTEMPTS} attempts: {last}") from last


def index_manifest(manifest, spec: config.PropertySpec) -> dict[tuple[int, int], list[dict]]:
    """Manifest rows for one property, grouped by cell, in raster order."""
    rows = manifest.filter(manifest["property"] == spec.name).to_dicts()
    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["cell_row"], r["cell_col"])].append(r)
    return dict(sorted(by_cell.items()))


def decode_values(data: np.ndarray, spec: config.PropertySpec) -> np.ndarray:
    """Source Int16 in mapped units -> float32 in conventional units, NaN nodata.

    The only transform this pipeline applies to pixel values, and it is exactly
    invertible: ``rint(stored * conversion_factor)`` recovers every one of the
    65 535 Int16 values at both scale factors (verified), which is what lets
    `validate` stay an exact check rather than a tolerance.
    """
    out = data.astype("float32") / np.float32(spec.conversion_factor)
    out[data == config.SOURCE_NODATA] = np.float32("nan")
    return out


def encode_values(values: np.ndarray, spec: config.PropertySpec) -> np.ndarray:
    """The inverse of decode_values, for comparing the store against source bytes."""
    out = np.rint(values.astype("float64") * spec.conversion_factor)
    out[np.isnan(values)] = config.SOURCE_NODATA
    return out.astype(config.SOURCE_DTYPE)


def place_tile(
    buf: np.ndarray,
    row: dict,
    header: tiff.TiffTile,
    data: np.ndarray,
    cell: tuple[int, int],
    depth_index: int | None,
    spec: config.PropertySpec | None = None,
) -> tuple[int, int, int, int]:
    """Place one tile into the cell buffer, verifying it against the manifest.

    Returns the cell-local ``(y0, y1, x0, x1)`` it wrote, which is what lets the
    caller record 250 m *coverage* -- a distinct thing from 250 m *values*, and
    the one that decides where a coarse source may go.

    The tile's own tie-point decides where it goes. Ragged (coastline-clipped)
    tiles need no special case: 2 039 of the 2 801 sit at an arbitrary offset
    inside their subtile, and their tie-point says exactly where.

    ``spec`` converts the source integers into the store's physical units; omit
    it only when the caller has already decoded ``data``.
    """
    if header.nodata is not None and header.nodata != config.SOURCE_NODATA:
        raise PlacementMismatch(f"{row['url']}: nodata {header.nodata}, expected {config.SOURCE_NODATA}")
    # A tile coarser than 250 m covers ratio^2 canonical pixels per source pixel,
    # so placing it 1:1 would silently shrink it (measured: tileSG-014-022_3-4 is
    # 750 m, and its tie-point IS on the lattice, so nothing else here catches
    # it). Coarse sources belong to resolve_coarse/apply_coarse.
    if header.pixel_size != (config.GRID.pixel_size, config.GRID.pixel_size):
        raise PlacementMismatch(
            f"{row['url']}: native pixel size {header.pixel_size} is not the canonical "
            f"{config.GRID.pixel_size} m; a coarser source must go through the gap-fill "
            f"path, never a direct placement"
        )
    if (header.height, header.width) != (row["height"], row["width"]):
        raise PlacementMismatch(
            f"{row['url']}: file is {header.height}x{header.width}, manifest says {row['height']}x{row['width']}"
        )
    x_off, y_off = grid.xy_to_pixel(*header.tiepoint)
    if (x_off, y_off) != (row["x_off"], row["y_off"]):
        raise PlacementMismatch(
            f"{row['url']}: tie-point places it at ({x_off}, {y_off}), manifest says "
            f"({row['x_off']}, {row['y_off']}) -- the source grid has changed"
        )
    n = grid.cell_px()
    cx, cy = grid.cell_pixel_offset(*cell)
    dy, dx = y_off - cy, x_off - cx
    if not (dy >= 0 and dy + header.height <= n and dx >= 0 and dx + header.width <= n):
        raise PlacementMismatch(
            f"{row['url']} at ({x_off}, {y_off}) size {header.width}x{header.height} "
            f"does not fit cell {cell} ({cx}, {cy})"
        )
    values = data if spec is None else decode_values(data, spec)
    target = buf if depth_index is None else buf[depth_index]
    target[dy : dy + header.height, dx : dx + header.width] = values
    return dy, dy + header.height, dx, dx + header.width


@dataclass(frozen=True, slots=True)
class CoarseFill:
    """One upstream tile published below 250 m, expanded onto the canonical grid.

    ``values`` is already decoded to the store's units and nearest-expanded to
    ``width`` x ``height`` canonical pixels, so applying it is a clipped copy.
    """

    relpath: str
    depth: str
    ratio_x: int  # native pixel size / 250 m, per axis; the two need not agree
    ratio_y: int
    x_off: int
    y_off: int
    width: int  # canonical pixels, = source width * ratio
    height: int
    values: np.ndarray
    depth_index: int | None

    @property
    def key(self) -> str:
        return f"{self.relpath}#{self.depth}"


def resolve_coarse(
    fetcher: Fetcher, spec: config.PropertySpec, rows: list[dict], stats: CellStats
) -> tuple[dict[tuple[int, int], list[CoarseFill]], set[tuple[str, str]]]:
    """Fetch and expand the few sources ISRIC publishes coarser than 250 m.

    Three tiles are measurably coarse (500 m, 750 m and 15 500 m), and the
    15 500 m one appears in all eleven properties. Their values are ordinary
    SoilGrids predictions, not corruption: depth profiles are monotone and
    sand+silt+clay sums to 100 %, so they are kept rather than dropped -- for
    tileSG-014-022_3-4 nothing else covers the window at all.

    They are never allowed to overwrite a full-resolution pixel. ``apply_coarse``
    writes only where the cell buffer is still fill, which makes the outcome
    independent of tile arrival order -- unlike the layer VRT, where whichever
    source GDAL happens to list last wins the overlap, so a 15.5 km aggregate
    paints over real 250 m data in one place and is painted over in another.

    Returns ``(fills by cell, (relpath, depth) keys to keep out of the
    direct-placement path)``. The key is per depth, not per tile: nothing
    measured has one depth of a tile coarse and another at 250 m, but a
    relpath-only skip would silently drop the 250 m depth if that ever changed.

    An expanded window can straddle a cell boundary (010-049_1-1 spans four), so
    each fill is offered to every cell it touches and clipped there.
    """
    candidates = [r for r in rows if r.get("resampled_in_vrt")]
    if not candidates:
        return {}, set()
    by_cell: dict[tuple[int, int], list[CoarseFill]] = defaultdict(list)
    coarse_keys: set[tuple[str, str]] = set()
    depth_of = {d: i for i, d in enumerate(spec.depths)}
    for row in candidates:
        body = retry_transient(
            lambda r=row: fetcher.get(r["url"], expect_etag=r.get("etag"), expect_size=r.get("size")),
            what=f"GET {row['relpath']} (coarse candidate)",
        )
        header, data = tiff.decode(body)
        px, py = header.pixel_size
        win = grid.source_window(header.tiepoint, header.pixel_size, header.width, header.height)
        if not win.coarse:
            # The VRT stretches it but the file really is 250 m on both axes.
            # No measured tile does this, but the VRT is not the authority here,
            # so defer to the file and leave it to the normal placement path.
            continue
        coarse_keys.add((row["relpath"], row["depth"]))
        if header.nodata is not None and header.nodata != config.SOURCE_NODATA:
            raise PlacementMismatch(f"{row['url']}: nodata {header.nodata}, expected {config.SOURCE_NODATA}")
        if abs(win.x_off - row["x_off"]) > 1 or abs(win.y_off - row["y_off"]) > 1:
            raise PlacementMismatch(
                f"{row['url']}: the file's tie-point puts its {win.ratio_x}x{win.ratio_y} window at "
                f"({win.x_off}, {win.y_off}), manifest says ({row['x_off']}, {row['y_off']}) "
                f"-- the source grid has changed"
            )
        # Within that one pixel, prefer the manifest: it carries the VRT's own
        # DstRect, which is where every GDAL reader places this tile. A coarse
        # tie-point is not on the lattice to begin with (010-049_1-1 sits exactly
        # half a pixel off), so rounding it here would only disagree with the
        # published layer for no gain.
        x_off, y_off = int(row["x_off"]), int(row["y_off"])
        depth_index = None
        if spec.ndim == 3:
            if row["depth"] not in depth_of:
                raise PlacementMismatch(
                    f"{row['relpath']}: manifest depth {row['depth']!r} is not one of "
                    f"{spec.name}'s depths {list(spec.depths)}"
                )
            depth_index = depth_of[row["depth"]]
        values = decode_values(data, spec)
        values = np.repeat(np.repeat(values, win.ratio_y, axis=0), win.ratio_x, axis=1)
        fill = CoarseFill(
            relpath=row["relpath"],
            depth=row["depth"],
            ratio_x=win.ratio_x,
            ratio_y=win.ratio_y,
            x_off=x_off,
            y_off=y_off,
            width=win.width,
            height=win.height,
            values=values,
            depth_index=depth_index,
        )
        for c in grid.cells_touching(x_off, y_off, win.width, win.height):
            by_cell[c].append(fill)
        stats.anomalies.append(
            f"{row['relpath']} ({row['depth']}): native {px:g}x{py:g} m = {win.ratio_x}x{win.ratio_y} "
            f"canonical pixels; expanded to {win.width}x{win.height} at ({x_off}, {y_off}), "
            f"written only where no 250 m data exists"
        )
    return dict(by_cell), coarse_keys


def apply_coarse(buf: np.ndarray, fill: CoarseFill, cell: tuple[int, int], covered: np.ndarray) -> int:
    """Copy one coarse fill into the pixels no 250 m tile reached.

    ``covered`` marks where a full-resolution tile's raster landed, which is NOT
    the same as where it holds a value. A 250 m nodata is a statement -- the
    fine-resolution soil mask excluded that pixel, typically water or ice -- and
    it is a more precise one than a 15.5 km aggregate can make, since at that
    size a mostly-land cell carries a value straight across a lake. So coverage,
    not fill, is the test: a coarse source may only speak where the 250 m product
    says nothing at all.

    A tile that failed to fetch counts as covered too, so a transient error
    surfaces as a hole and an anomaly rather than being papered over.

    Returns the pixels actually written. The test is a property of the cell, not
    of arrival order, so the result is the same however the tiles interleave.
    """
    n = grid.cell_px()
    cx, cy = grid.cell_pixel_offset(*cell)
    x0, x1 = max(fill.x_off, cx), min(fill.x_off + fill.width, cx + n)
    y0, y1 = max(fill.y_off, cy), min(fill.y_off + fill.height, cy + n)
    if x1 <= x0 or y1 <= y0:
        return 0
    src = fill.values[y0 - fill.y_off : y1 - fill.y_off, x0 - fill.x_off : x1 - fill.x_off]
    target = buf if fill.depth_index is None else buf[fill.depth_index]
    seen = covered if fill.depth_index is None else covered[fill.depth_index]
    sel = (slice(y0 - cy, y1 - cy), slice(x0 - cx, x1 - cx))
    view = target[sel]  # a basic-slicing view, so writes go through to buf
    gap = ~seen[sel] & ~np.isnan(src)
    view[gap] = src[gap]
    # two coarse sources have never been measured to overlap, but if one ever
    # did, the second must not overwrite the first
    seen[sel] |= gap
    return int(gap.sum())


def fill_cell(
    fetcher: Fetcher,
    arr: zarr.Array,
    spec: config.PropertySpec,
    cell: tuple[int, int],
    rows: list[dict],
    stats: CellStats,
    coarse: list[CoarseFill] | None = None,
    skip_keys: frozenset[tuple[str, str]] | set[tuple[str, str]] = frozenset(),
) -> None:
    """Fetch one cell's tiles for every depth, assemble, and write whole shards.

    ``coarse`` holds any sub-250 m sources overlapping this cell; they are
    applied last and only into pixels no full-resolution tile reached.
    """
    n = grid.cell_px()
    coarse = coarse or []
    rows = [r for r in rows if (r["relpath"], r["depth"]) not in skip_keys]
    depth_of = {d: i for i, d in enumerate(spec.depths)}
    shape = (spec.depth_extent, n, n) if spec.ndim == 3 else (n, n)
    buf = np.full(shape, config.FILL_VALUE, dtype=config.DTYPE)
    # where a 250 m tile's raster landed, whatever value it carries. Only needed
    # when a coarse source overlaps this cell, and it costs 19 MB alongside the
    # 39 MB buffer, so it is not allocated for the ~99.9 % of cells with none.
    covered = np.zeros(shape, bool) if coarse else None

    def mark(depth_index: int | None, y0: int, y1: int, x0: int, x1: int) -> None:
        if covered is None:
            return
        seen = covered if depth_index is None else covered[depth_index]
        seen[max(y0, 0) : min(y1, n), max(x0, 0) : min(x1, n)] = True

    def fetch_one(row: dict) -> tuple[dict, bytes | None]:
        try:
            body = retry_transient(
                lambda: fetcher.get(row["url"], expect_etag=row.get("etag"), expect_size=row.get("size")),
                what=f"GET {row['relpath']}",
            )
            return row, body
        except SourceChanged:
            raise
        except RuntimeError as exc:
            # a tile the VRT lists but the server will not serve: record and
            # leave the pixels at fill rather than aborting the whole property
            stats.anomalies.append(f"{row['url']}: {exc}")
            return row, None

    cell_x, cell_y = grid.cell_pixel_offset(*cell)
    for row, body in fetcher.map(fetch_one, rows):
        if body is None:
            stats.tiles_missing += 1
            # a tile the server would not serve still counts as 250 m coverage:
            # a fetch error must stay a hole, not become a coarse guess
            dy, dx = row["y_off"] - cell_y, row["x_off"] - cell_x
            mark(
                depth_of.get(row["depth"]) if spec.ndim == 3 else None,
                dy,
                dy + row["height"],
                dx,
                dx + row["width"],
            )
            continue
        stats.tiles_fetched += 1
        stats.bytes_fetched += len(body)
        header, data = tiff.decode(body)
        if header.rows_per_strip != row["rows_per_strip"]:
            stats.anomalies.append(
                f"{row['relpath']}: RowsPerStrip {header.rows_per_strip}, VRT said {row['rows_per_strip']}"
            )
        depth_index = None
        if spec.ndim == 3:
            # not depth_of.get(): a None depth index would make place_tile write
            # into the depth axis of the 3-D buffer and silently corrupt the cell
            if row["depth"] not in depth_of:
                raise PlacementMismatch(
                    f"{row['relpath']}: manifest depth {row['depth']!r} is not one of "
                    f"{spec.name}'s depths {list(spec.depths)}"
                )
            depth_index = depth_of[row["depth"]]
        mark(depth_index, *place_tile(buf, row, header, data, cell, depth_index, spec))

    # last, and only where no 250 m tile reached at all: a coarse aggregate must
    # never displace a full-resolution prediction, nor contradict a 250 m nodata
    for fill in coarse:
        filled = apply_coarse(buf, fill, cell, covered)
        if filled:
            stats.coarse_pixels[fill.key] = stats.coarse_pixels.get(fill.key, 0) + filled

    for y0, y1, x0, x1 in grid.shard_windows(*cell):
        by, bx = y0 - cell_y, x0 - cell_x  # the window's position inside the buffer
        block = buf[..., by : by + (y1 - y0), bx : bx + (x1 - x0)]
        if not np.any(~np.isnan(block)):
            stats.shards_empty += 1  # all-fill: never written, so it costs nothing
            continue
        sel = (slice(None), slice(y0, y1), slice(x0, x1)) if spec.ndim == 3 else (slice(y0, y1), slice(x0, x1))

        def write(_sel=sel, _block=block):
            arr[_sel] = _block

        retry_transient(write, what=f"write shard {cell} at ({y0}, {x0})")
        stats.shards_written += 1
    stats.cells_done += 1


def done_cells(session_or_store, group: str, prop: str) -> set[str]:
    """Cells already filled for one property, from the group's provenance attrs."""
    try:
        gp = zarr.open_group(getattr(session_or_store, "store", session_or_store), path=group, mode="r")
    except (KeyError, FileNotFoundError):
        return set()
    return set(gp.attrs.get(DONE_ATTR, {}).get(prop, []))


def record_done(session: Session, group: str, prop: str, cells: Iterable[str]) -> None:
    gp = zarr.open_group(session.store, path=group, mode="r+")
    done = dict(gp.attrs.get(DONE_ATTR, {}))
    done[prop] = sorted(set(done.get(prop, [])) | set(cells))
    gp.attrs[DONE_ATTR] = done


def record_coarse(session: Session, group: str, prop: str, records: list[dict]) -> None:
    """Publish which windows came from a source coarser than 250 m.

    Kept in the group attrs rather than a per-pixel mask: it is a handful of
    windows out of 9.5 billion pixels, and a reader needs to know they are a
    500 m-15.5 km aggregate, not where every one of them is.
    """
    if not records:
        return
    gp = zarr.open_group(session.store, path=group, mode="r+")
    cur = dict(gp.attrs.get(COARSE_ATTR, {}))
    merged = {(r["source"], r["depth"]): r for r in cur.get(prop, [])}
    for r in records:
        merged[(r["source"], r["depth"])] = r
    cur[prop] = [merged[k] for k in sorted(merged)]
    gp.attrs[COARSE_ATTR] = cur


def complete_properties(session_or_store, group: str) -> set[str]:
    """Properties a full-scope run has finished, from the group's provenance attrs."""
    try:
        gp = zarr.open_group(getattr(session_or_store, "store", session_or_store), path=group, mode="r")
    except (KeyError, FileNotFoundError):
        return set()
    return set(gp.attrs.get(COMPLETE_ATTR, []))


def mark_complete(session: Session, group: str, prop: str) -> None:
    gp = zarr.open_group(session.store, path=group, mode="r+")
    gp.attrs[COMPLETE_ATTR] = sorted(set(gp.attrs.get(COMPLETE_ATTR, [])) | {prop})


def materialize_property(
    session: Session,
    fetcher: Fetcher,
    spec: config.PropertySpec,
    manifest,
    *,
    cells: set[tuple[int, int]] | None = None,
    overwrite: bool = False,
    commit_every: int = 0,
    checkpoint: Callable[[Session, CellStats], Session] | None = None,
    progress: bool = True,
) -> tuple[Session, CellStats]:
    """Fill one property. Commits nothing itself; the caller owns commit policy.

    ``checkpoint(session, stats) -> Session`` is called every ``commit_every``
    cells and must commit the session it is given and return a fresh one.
    Returns the final session alongside the stats so the caller commits the tail.
    """
    stats = CellStats()
    by_cell = index_manifest(manifest, spec)
    already = set() if overwrite else done_cells(session, spec.group, spec.name)

    # The resume check normally cannot run before `resolve_coarse`, because an
    # expanded coarse window can reach cells whose own manifest rows say nothing
    # about it, and those cells have to exist in by_cell for the filter to see
    # them as pending. A property a full-scope run already finished is the one
    # case where it can: every cell the expansion reached is in the ledger by
    # definition, so there is nothing left for the fetches to discover. Without
    # this, re-running the command GETs every tile the VRT resampled -- 194
    # across the eleven properties -- only to log "nothing to do".
    if (
        not overwrite
        and spec.name in complete_properties(session, spec.group)
        and all(f"{r}-{c}" in already for r, c in by_cell if cells is None or (r, c) in cells)
    ):
        log.info("%s: nothing to do (%d cells already filled, property marked complete)", spec.name, len(already))
        return session, stats

    # resolved once per property, before the subset and resume filters run
    coarse_by_cell, coarse_keys = resolve_coarse(fetcher, spec, [r for rs in by_cell.values() for r in rs], stats)
    for c in coarse_by_cell:
        by_cell.setdefault(c, [])
    by_cell = dict(sorted(by_cell.items()))
    coarse_index = {f.key: f for fs in coarse_by_cell.values() for f in fs}
    if cells is not None:
        by_cell = {k: v for k, v in by_cell.items() if k in cells}
    todo = [(c, rows) for c, rows in by_cell.items() if f"{c[0]}-{c[1]}" not in already]
    if not todo:
        log.info("%s: nothing to do (%d cells already filled)", spec.name, len(already))
        return session, stats

    log.info(
        "%s: %d of %d cells to fill (%d already done), %d workers",
        spec.name,
        len(todo),
        len(by_cell),
        len(already),
        fetcher.workers,
    )
    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=len(todo), desc=f"materialize {spec.name}", unit="cell")

    arr = zarr.open_group(session.store, path=spec.group, mode="r+")[spec.name]
    pending: list[str] = []
    for cell, rows in todo:
        fill_cell(
            fetcher,
            arr,
            spec,
            cell,
            rows,
            stats,
            coarse=coarse_by_cell.get(cell, []),
            skip_keys=coarse_keys,
        )
        pending.append(f"{cell[0]}-{cell[1]}")
        if bar is not None:
            bar.update(1)
            bar.set_postfix(shards=stats.shards_written, empty=stats.shards_empty, missing=stats.tiles_missing)
        if commit_every and checkpoint is not None and len(pending) >= commit_every:
            record_done(session, spec.group, spec.name, pending)
            # the callback receives the live session: a closure over the caller's
            # variable would commit a stale one after the first checkpoint
            session = checkpoint(session, stats)
            arr = zarr.open_group(session.store, path=spec.group, mode="r+")[spec.name]
            pending.clear()
    if pending:
        record_done(session, spec.group, spec.name, pending)
    record_coarse(
        session,
        spec.group,
        spec.name,
        [
            {
                "source": f.relpath,
                "depth": f.depth,
                "native_pixel_size_m": [
                    f.ratio_x * config.GRID.pixel_size,
                    f.ratio_y * config.GRID.pixel_size,
                ],
                "x_off": f.x_off,
                "y_off": f.y_off,
                "width": f.width,
                "height": f.height,
                "pixels_filled": n_px,
            }
            for key, n_px in sorted(stats.coarse_pixels.items())
            if (f := coarse_index.get(key)) is not None
        ],
    )
    if bar is not None:
        bar.close()
    return session, stats


def cells_in_bbox(bbox: tuple[float, float, float, float]) -> set[tuple[int, int]]:
    """Cells overlapping a lon/lat bbox (min_lon, min_lat, max_lon, max_lat).

    IGH is interrupted, so a lon/lat box maps to a ragged set of projected
    positions; the box is sampled densely and every cell it touches is kept.
    """
    from pyproj import Transformer

    from . import metadata

    tf = Transformer.from_crs("EPSG:4326", metadata.crs(), always_xy=True)
    lon0, lat0, lon1, lat1 = bbox
    lons = np.linspace(lon0, lon1, 200)
    lats = np.linspace(lat0, lat1, 200)
    lo, la = np.meshgrid(lons, lats)
    xs, ys = tf.transform(lo.ravel(), la.ravel())
    g = config.GRID
    n = grid.cell_px()
    out: set[tuple[int, int]] = set()
    ok = np.isfinite(xs) & np.isfinite(ys)
    ix = ((np.asarray(xs)[ok] - g.x_min) / g.pixel_size).astype(int)
    iy = ((g.y_max - np.asarray(ys)[ok]) / g.pixel_size).astype(int)
    keep = (ix >= 0) & (ix < g.width) & (iy >= 0) & (iy < g.height)
    for cx, cy in zip(ix[keep] // n, iy[keep] // n, strict=True):
        out.add((int(cy), int(cx)))
    return out
