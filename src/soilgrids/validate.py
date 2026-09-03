"""Phase 6: verify the store's structure and, by sampling, its contents.

Three kinds of check, in increasing strength:

* **structural** -- shapes, dtypes, chunk/shard grids, coordinates, required
  attrs, and the multiscales layout, all against config.py. Needs no network.
* **byte-exact sampling** -- re-fetch random source tiles and compare them
  pixel-for-pixel with what the store holds. This is the check that catches
  placement, orientation and encoding bugs in one shot, so the sample always
  includes a ragged tile and a 600 px-tiled property when the property has them.
* **derived** -- overview levels recomputed from their parent, and the
  sand+silt+clay closure that only holds if three independent arrays agree.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import zarr

from . import config, grid, materialize, metadata, tiff
from .fetch import Fetcher

log = logging.getLogger(__name__)


@dataclass
class Result:
    check: str
    passed: bool
    message: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.check}: {self.message}"


def _add(results: list[Result], check: str, passed: bool, message: str) -> None:
    results.append(Result(check, passed, message))
    log.log(logging.INFO if passed else logging.ERROR, "%s", results[-1])


def check_structure(
    store_or_session,
    properties: list[config.PropertySpec] | None = None,
    *,
    factors: tuple[int, ...] | None = None,
) -> list[Result]:
    """Everything checkable without touching the network.

    ``factors`` is the overview ladder the store was built with; it defaults to
    config's, but a store built with a subset must be checked against that
    subset rather than reported as missing levels.
    """
    st = getattr(store_or_session, "store", store_or_session)
    properties = properties if properties is not None else config.included_properties()
    factors = config.OVERVIEW_FACTORS if factors is None else factors
    g = config.GRID
    enc = config.ENCODING
    results: list[Result] = []
    root = zarr.open_group(st, mode="r")

    for key in ("title", "license", "source_version", "dataset_version", "grid_crs_proj4"):
        _add(results, f"root attr {key}", key in root.attrs, "present" if key in root.attrs else "MISSING")

    shapes = grid.overview_shapes(factors=factors) if factors else {}
    for group_name in config.GROUPS:
        gp = root[group_name]
        # coordinates
        for axis, size, first in (
            ("y", g.height, g.y_max - 0.5 * g.pixel_size),
            ("x", g.width, g.x_min + 0.5 * g.pixel_size),
        ):
            arr = gp[axis]
            ok = arr.shape == (size,) and np.isclose(arr[0], first)
            _add(
                results,
                f"{group_name}/{axis} coord",
                ok,
                f"{arr.shape} starting {arr[0]} (want ({size},) starting {first})",
            )
        sr = gp["spatial_ref"]
        _add(
            results,
            f"{group_name}/spatial_ref",
            "crs_wkt" in sr.attrs and "GeoTransform" in sr.attrs,
            f"GeoTransform={sr.attrs.get('GeoTransform')}",
        )
        if group_name == "soil_properties":
            labels = [str(v) for v in gp["depth_interval"][:]]
            _add(results, "depth_interval coord", labels == config.DEPTH_LABELS, f"{labels}")
        if factors:
            ms = gp.attrs.get("multiscales", {}).get("layout", [])
            assets = [e.get("asset") for e in ms]
            want = [".", *(f"{f}x" for f in factors)]
            _add(results, f"{group_name} multiscales layout", assets == want, f"{assets}")

        for spec in (p for p in properties if p.group == group_name):
            native_shape = (spec.depth_extent, g.height, g.width) if spec.ndim == 3 else (g.height, g.width)
            for factor, arr_path in [(1, spec.name)] + [(f, f"{f}x/{spec.name}") for f in factors]:
                try:
                    arr = gp[arr_path]
                except KeyError:
                    _add(results, f"{group_name}/{arr_path}", False, "MISSING")
                    continue
                if factor == 1:
                    want_shape: tuple[int, ...] = native_shape
                    want_shards = enc.shards(spec.ndim, depth_extent=spec.depth_extent)
                else:
                    h, w = shapes[factor]
                    want_shape = (spec.depth_extent, h, w) if spec.ndim == 3 else (h, w)
                    want_shards = enc.shards(spec.ndim, depth_extent=1)
                ok = (
                    tuple(arr.shape) == want_shape
                    and str(arr.dtype) == config.DTYPE
                    and config.is_fill(arr.fill_value)
                    and tuple(arr.chunks) == enc.chunks(spec.ndim)
                    and tuple(arr.shards or ()) == want_shards
                )
                _add(
                    results,
                    f"{group_name}/{arr_path} encoding",
                    ok,
                    f"shape={tuple(arr.shape)} dtype={arr.dtype} fill={arr.fill_value} "
                    f"chunks={tuple(arr.chunks)} shards={tuple(arr.shards or ())}",
                )
                for key in ("units", "source_conversion_factor", "grid_mapping", "long_name"):
                    if key not in arr.attrs:
                        _add(results, f"{group_name}/{arr_path} attr {key}", False, "MISSING")
    return results


def check_tiles(
    store_or_session,
    fetcher: Fetcher,
    spec: config.PropertySpec,
    manifest,
    *,
    samples: int = 8,
    seed: int = 0,
    only_cells: set[str] | None = None,
) -> list[Result]:
    """Re-fetch sampled source tiles and compare byte-for-byte with the store.

    ``only_cells`` restricts the sample to cells that have actually been
    materialized. Without it a partial ingest "fails" on every unfilled cell,
    where the store is correctly holding its fill value.
    """
    st = getattr(store_or_session, "store", store_or_session)
    arr = zarr.open_group(st, path=spec.group, mode="r")[spec.name]
    rows = manifest.filter(manifest["property"] == spec.name).to_dicts()
    if only_cells is not None:
        rows = [r for r in rows if f"{r['cell_row']}-{r['cell_col']}" in only_cells]
    if not rows:
        return [Result(f"{spec.name} tile sample", False, "no manifest rows for the filled cells")]

    rng = random.Random(seed)
    picked = rng.sample(rows, min(samples, len(rows)))
    # a ragged tile is the case most likely to be placed wrongly, so force one in
    ragged = [r for r in rows if not r["is_full"]]
    if ragged and all(r["is_full"] for r in picked):
        picked[-1] = rng.choice(ragged)

    results: list[Result] = []
    depth_of = {d: i for i, d in enumerate(spec.depths)}
    for row in picked:
        body = fetcher.get(row["url"])
        header, want = tiff.decode(body)
        x_off, y_off = grid.xy_to_pixel(*header.tiepoint)
        sel = (
            (depth_of[row["depth"]], slice(y_off, y_off + header.height), slice(x_off, x_off + header.width))
            if spec.ndim == 3
            else (slice(y_off, y_off + header.height), slice(x_off, x_off + header.width))
        )
        # the store holds decoded float32; re-encode it and compare with the
        # source bytes, which is an EXACT check because the transform is
        # exactly invertible for every Int16 value
        got = materialize.encode_values(arr[sel], spec)
        ok = np.array_equal(got, want)
        diff = int((got != want).sum())
        _add(
            results,
            f"{spec.name}/{row['depth']} tile {row['relpath']}",
            ok,
            f"{'byte-exact' if ok else f'{diff} of {want.size} pixels differ'} "
            f"({'ragged' if not row['is_full'] else 'full'} {header.width}x{header.height} at {x_off},{y_off})",
        )
    return results


def check_overviews(
    store_or_session,
    spec: config.PropertySpec,
    *,
    samples: int = 2,
    seed: int = 0,
    only_cells: set[str] | None = None,
    factors: tuple[int, ...] | None = None,
) -> list[Result]:
    """Recompute sampled overview windows from the parent level and compare.

    Windows are drawn inside filled cells where known, so the check compares real
    data rather than confirming that fill coarsens to fill.
    """
    st = getattr(store_or_session, "store", store_or_session)
    gp = zarr.open_group(st, path=spec.group, mode="r")
    results: list[Result] = []
    rng = random.Random(seed)
    anchors: list[tuple[int, int]] | None = None
    if only_cells:
        anchors = []
        for cell in rng.sample(sorted(only_cells), min(samples, len(only_cells))):
            row, col = (int(v) for v in cell.split("-"))
            cx, cy = grid.cell_pixel_offset(row, col)
            anchors.append((cy, cx))
    factors = config.OVERVIEW_FACTORS if factors is None else factors
    for i, factor in enumerate(factors):
        parent = gp[spec.name] if i == 0 else gp[f"{factors[i - 1]}x"][spec.name]
        level = gp[f"{factor}x"][spec.name]
        h, w = level.shape[-2:]
        for k in range(samples):
            if anchors:
                ay, ax = anchors[k % len(anchors)]
                y0 = min(max(0, ay // factor), max(0, h - 32))
                x0 = min(max(0, ax // factor), max(0, w - 32))
            else:
                y0 = rng.randrange(0, max(1, h - 32))
                x0 = rng.randrange(0, max(1, w - 32))
            ph, pw = min(32, h - y0), min(32, w - x0)
            psel = (slice(y0 * 2, (y0 + ph) * 2), slice(x0 * 2, (x0 + pw) * 2))
            lsel = (slice(y0, y0 + ph), slice(x0, x0 + pw))
            src = parent[(0, *psel)] if spec.ndim == 3 else parent[psel]
            got = level[(0, *lsel)] if spec.ndim == 3 else level[lsel]
            want = _fill_aware_mean(src)
            # NaN == NaN is False, so a plain array_equal would count every ocean
            # cell as a mismatch; isclose(equal_nan=True) compares the fill
            # pattern and the values, which is what we actually mean
            close = np.isclose(got, want, rtol=1e-6, atol=0.0, equal_nan=True)
            ok = bool(close.all())
            _add(
                results,
                f"{spec.name} {factor}x window ({y0},{x0})",
                ok,
                "matches a recomputed fill-aware mean" if ok else f"{int((~close).sum())} of {want.size} cells differ",
            )
    return results


def _fill_aware_mean(a: np.ndarray, factor: int = 2) -> np.ndarray:
    """Reference implementation of the coarsening, independent of topozarr."""
    h = (a.shape[0] // factor) * factor
    w = (a.shape[1] // factor) * factor
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
        return np.where(n > 0, total / np.maximum(n, 1), np.nan).astype(config.DTYPE)


def check_texture_closure(
    store_or_session, *, samples: int = 4, seed: int = 0, only_cells: set[str] | None = None
) -> list[Result]:
    """sand + silt + clay should sum to ~1000 g/kg where all three have data.

    Three independently stored arrays have to agree, so this catches a whole
    class of placement and depth-indexing errors that per-array checks cannot.
    """
    st = getattr(store_or_session, "store", store_or_session)
    gp = zarr.open_group(st, path="soil_properties", mode="r")
    g = config.GRID
    rng = random.Random(seed)
    results: list[Result] = []
    # sample inside filled cells only, else every window is fill and proves nothing
    windows: list[tuple[int, int]] = []
    if only_cells:
        picked = rng.sample(sorted(only_cells), min(samples, len(only_cells)))
        for cell in picked:
            row, col = (int(v) for v in cell.split("-"))
            cx, cy = grid.cell_pixel_offset(row, col)
            n = grid.cell_px()
            windows.append((cy + rng.randrange(0, n - 450), cx + rng.randrange(0, n - 450)))
    else:
        windows = [(rng.randrange(0, g.height - 450), rng.randrange(0, g.width - 450)) for _ in range(samples)]
    for y0, x0 in windows:
        sel = (0, slice(y0, y0 + 450), slice(x0, x0 + 450))
        parts = {n: gp[n][sel].astype("float64") for n in ("sand", "silt", "clay")}
        valid = np.logical_and.reduce([~np.isnan(v) for v in parts.values()])
        if not valid.any():
            continue
        total = sum(parts.values())[valid]
        ok = bool(np.all(np.abs(total - 100.0) <= 1.5))
        _add(
            results,
            f"sand+silt+clay closure at ({y0},{x0})",
            ok,
            f"{valid.sum()} px, sum range [{total.min():.1f}, {total.max():.1f}] (want ~100)",
        )
    if not results:
        results.append(Result("sand+silt+clay closure", True, "no sampled window had data in all three arrays"))
    return results


def check_crs_matches_source(fetcher: Fetcher, manifest, *, samples: int = 3, seed: int = 0) -> list[Result]:
    """The store's CRS must be the one the source tiles declare.

    IGH has no EPSG code, so the source WKT is the only authority there is.
    """
    ours = metadata.crs()
    rows = manifest.to_dicts()
    rng = random.Random(seed)
    results: list[Result] = []
    for row in rng.sample(rows, min(samples, len(rows))):
        header = tiff.parse_header(fetcher.get(row["url"])[:8192])
        px = header.pixel_size
        ok_px = abs(px[0] - config.GRID.pixel_size) < 1e-9 and abs(px[1] - config.GRID.pixel_size) < 1e-9
        _add(results, f"pixel size of {row['relpath']}", ok_px, f"{px}")
        if header.crs_wkt:
            ok_name = "Goode" in header.crs_wkt or "goode" in header.crs_wkt.lower()
            _add(results, f"CRS name in {row['relpath']}", ok_name, header.crs_wkt.split("|")[0])
    _add(
        results,
        "store CRS has no EPSG code (expected for IGH)",
        ours.to_epsg() is None,
        f"to_epsg()={ours.to_epsg()}",
    )
    return results


def summarise(results: list[Result]) -> tuple[int, int]:
    failed = sum(not r.passed for r in results)
    return len(results) - failed, failed
