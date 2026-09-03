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
) -> None:
    """Place one tile into the cell buffer, verifying it against the manifest.

    The tile's own tie-point decides where it goes. Ragged (coastline-clipped)
    tiles need no special case: 2 039 of the 2 801 sit at an arbitrary offset
    inside their subtile, and their tie-point says exactly where.

    ``spec`` converts the source integers into the store's physical units; omit
    it only when the caller has already decoded ``data``.
    """
    if header.nodata is not None and header.nodata != config.SOURCE_NODATA:
        raise PlacementMismatch(f"{row['url']}: nodata {header.nodata}, expected {config.SOURCE_NODATA}")
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


def fill_cell(
    fetcher: Fetcher,
    arr: zarr.Array,
    spec: config.PropertySpec,
    cell: tuple[int, int],
    rows: list[dict],
    stats: CellStats,
) -> None:
    """Fetch one cell's tiles for every depth, assemble, and write whole shards."""
    n = grid.cell_px()
    depth_of = {d: i for i, d in enumerate(spec.depths)}
    shape = (spec.depth_extent, n, n) if spec.ndim == 3 else (n, n)
    buf = np.full(shape, config.FILL_VALUE, dtype=config.DTYPE)

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

    for row, body in fetcher.map(fetch_one, rows):
        if body is None:
            stats.tiles_missing += 1
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
        place_tile(buf, row, header, data, cell, depth_index, spec)

    cell_x0, cell_y0 = grid.cell_pixel_offset(*cell)
    for y0, y1, x0, x1 in grid.shard_windows(*cell):
        by, bx = y0 - cell_y0, x0 - cell_x0  # the window's position inside the buffer
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
    by_cell = index_manifest(manifest, spec)
    if cells is not None:
        by_cell = {k: v for k, v in by_cell.items() if k in cells}
    already = set() if overwrite else done_cells(session, spec.group, spec.name)
    todo = [(c, rows) for c, rows in by_cell.items() if f"{c[0]}-{c[1]}" not in already]
    stats = CellStats()
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
        fill_cell(fetcher, arr, spec, cell, rows, stats)
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
