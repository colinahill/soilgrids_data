"""Phase 1: enumerate and verify the ISRIC source tree.

Division of authority, which matters because the two disagree:

* the per-layer **VRT is the inventory** -- which tiles exist, their shape, and
  where the VRT places them. It is one 6 MB request per layer instead of ~1 130
  directory listings, and it is the only cheap source of placement for the 22 %
  of tiles that are clipped to a coastline at an arbitrary offset inside their
  subtile;
* each tile's **own GeoTIFF tie-point is the placement** at write time. The VRT
  is not trusted to write with: it contains at least one resampled source (a 1x2
  tile stretched to 62x124 with a fractional DstRect), so materialize re-derives
  the window from the file and asserts it against the manifest.

Everything measured here lands in ``work/{version}/source_manifest.parquet`` with
each tile's ETag, so a re-run detects ISRIC replacing data under the mutable
``latest/`` path instead of silently mixing two snapshots.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from . import config, grid, tiff
from .fetch import Fetcher, parse_listing

log = logging.getLogger(__name__)

MANIFEST_NAME = "source_manifest.parquet"
REPORT_NAME = "source_report.json"


class SourceError(RuntimeError):
    """The source tree is not what config.py says it is. Nothing may proceed."""


@dataclass(frozen=True, slots=True)
class VrtSource:
    """One ``<ComplexSource>`` of a layer VRT."""

    relpath: str  # e.g. tileSG-018-027/tileSG-018-027_1-1.tif
    width: int  # the tile's own raster size
    height: int
    block_y: int  # RowsPerStrip as the VRT reports it
    dst_x: float  # placement in VRT pixel space
    dst_y: float
    dst_w: int
    dst_h: int

    @property
    def resampled(self) -> bool:
        """Does the VRT stretch this source rather than place it 1:1?"""
        return (self.width, self.height) != (self.dst_w, self.dst_h)


@dataclass(frozen=True, slots=True)
class Conflict:
    """Two tiles of one layer claiming the same pixel window."""

    window: tuple[int, int]
    keep: str
    drop: str


def _prefer(a: str, b: str, window: tuple[int, int], tile_px: int) -> tuple[str, str]:
    """Choose deterministically between two tiles claiming one window.

    A tile whose *name* matches the window wins: that is the semantically correct
    name for the position. Failing that the lexicographically smaller path wins,
    so the choice never depends on VRT or thread ordering.
    """

    def named_for_window(relpath: str) -> bool:
        try:
            return grid.parse_tile_name(relpath.split("/")[1], tile_px).pixel_offset == window
        except ValueError:
            return False

    a_ok, b_ok = named_for_window(a), named_for_window(b)
    if a_ok != b_ok:
        return (a, b) if a_ok else (b, a)
    return (a, b) if a <= b else (b, a)


@dataclass
class LayerInventory:
    """Everything phase 1 learned about one (property, depth) layer."""

    property: str
    depth: str
    vrt_url: str
    vrt_size: int
    vrt_width: int
    vrt_height: int
    vrt_x_min: float
    vrt_y_max: float
    dtype: str
    nodata: float | None
    metadata: dict[str, str]
    sources: list[VrtSource]
    anomalies: list[str] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)


def _text(el, path: str) -> str | None:
    found = el.find(path)
    return None if found is None else (found.text or "").strip()


def parse_vrt(xml: str) -> tuple[dict, list[VrtSource]]:
    """Parse a SoilGrids layer VRT into (header, sources)."""
    root = ElementTree.fromstring(xml)
    if root.tag != "VRTDataset":
        raise SourceError(f"not a VRT: root element <{root.tag}>")
    gt = _text(root, "GeoTransform")
    if not gt:
        raise SourceError("VRT has no GeoTransform")
    x_min, px, _, y_max, _, npy = (float(v) for v in gt.split(","))
    band = root.find("VRTRasterBand")
    if band is None:
        raise SourceError("VRT has no VRTRasterBand")
    header = {
        "width": int(root.attrib["rasterXSize"]),
        "height": int(root.attrib["rasterYSize"]),
        "x_min": x_min,
        "y_max": y_max,
        "pixel_size": px,
        "pixel_size_y": -npy,
        "srs": _text(root, "SRS"),
        "dtype": band.attrib.get("dataType", ""),
        "nodata": float(nd) if (nd := _text(band, "NoDataValue")) else None,
        "metadata": {
            mdi.attrib["key"]: (mdi.text or "").strip() for mdi in root.findall("./Metadata/MDI") if "key" in mdi.attrib
        },
    }

    sources: list[VrtSource] = []
    for src in band.iter():
        if src.tag not in ("ComplexSource", "SimpleSource"):
            continue
        fn = _text(src, "SourceFilename")
        props = src.find("SourceProperties")
        dst = src.find("DstRect")
        if fn is None or props is None or dst is None:
            raise SourceError(f"VRT source is missing SourceFilename/SourceProperties/DstRect: {fn!r}")
        relpath = fn.split("/", 2)[-1] if fn.startswith("./") else fn
        relpath = relpath.rsplit("/", 2)[-2] + "/" + relpath.rsplit("/", 1)[-1] if "/" in relpath else relpath
        sources.append(
            VrtSource(
                relpath=relpath,
                width=int(props.attrib["RasterXSize"]),
                height=int(props.attrib["RasterYSize"]),
                block_y=int(props.attrib["BlockYSize"]),
                dst_x=float(dst.attrib["xOff"]),
                dst_y=float(dst.attrib["yOff"]),
                dst_w=int(dst.attrib["xSize"]),
                dst_h=int(dst.attrib["ySize"]),
            )
        )
    if not sources:
        raise SourceError("VRT lists no sources")
    return header, sources


def vrt_to_canonical(header: dict) -> tuple[int, int]:
    """Pixel offset of the VRT's own origin on the canonical grid.

    Each property's VRT is a tight bounding box of its own tiles, so the offset
    differs per layer (sand starts 351 px east of the canonical origin, bdod
    348 px).
    """
    g = config.GRID
    if abs(header["pixel_size"] - g.pixel_size) > 1e-9 or abs(header["pixel_size_y"] - g.pixel_size) > 1e-9:
        raise SourceError(f"VRT pixel size {header['pixel_size']} != canonical {g.pixel_size}")
    return grid.xy_to_pixel(header["x_min"], header["y_max"])


def vrt_url(spec: config.PropertySpec, depth: str) -> str:
    return f"{config.SOURCE_BASE_URL}/{spec.name}/{spec.layer_dir(depth)}.vrt"


def fetch_vrt(fetcher: Fetcher, spec: config.PropertySpec, depth: str) -> str:
    """Download one layer VRT (~6 MB).

    ISRIC throttles per connection at roughly 40 kB/s, so one VRT takes ~2.5
    minutes on its own and the only way to go faster is to run layers
    concurrently -- which is why the caller fetches these in parallel rather
    than looping.
    """
    return fetcher.get_text(vrt_url(spec, depth))


def inventory_layer(
    fetcher: Fetcher | None, spec: config.PropertySpec, depth: str, xml: str | None = None
) -> LayerInventory:
    """Verify one layer's VRT, returning its tile inventory.

    Pass ``xml`` to use an already-downloaded VRT (the parallel path); otherwise
    a fetcher is required and the VRT is downloaded here.
    """
    url = vrt_url(spec, depth)
    if xml is None:
        if fetcher is None:
            raise SourceError("inventory_layer needs either a fetcher or the VRT text")
        xml = fetch_vrt(fetcher, spec, depth)
    header, sources = parse_vrt(xml)
    anomalies: list[str] = []

    if header["dtype"] != "Int16":
        raise SourceError(f"{url}: dataType {header['dtype']}, expected Int16")
    if header["nodata"] != float(config.SOURCE_NODATA):
        raise SourceError(f"{url}: NoDataValue {header['nodata']}, expected {config.SOURCE_NODATA}")
    expected = dict(config.EXPECTED_VRT_METADATA) | {"Outputs_version": spec.outputs_version}
    checks = [(k, header["metadata"].get(k), v) for k, v in expected.items()]
    # Transformation is compared normalised: upstream spells "no transform" as an
    # absent element, an empty one, or literal text, and a change between those
    # spellings is not drift.
    checks.append(
        (
            "Transformation",
            config.normalise_transformation(header["metadata"].get("Transformation")),
            config.normalise_transformation(spec.transformation),
        )
    )
    for key, got, want in checks:
        if got != want:
            raise SourceError(
                f"{url}: {key} is {got!r}, expected {want!r} -- either ISRIC has replaced the data "
                f"under the mutable `latest/` path, or config.PROPERTIES[{spec.name!r}] records the "
                f"wrong provenance. Re-audit before ingesting."
            )

    vrt_to_canonical(header)  # validates the pixel size and lattice alignment
    seen: set[str] = set()
    for src in sources:
        if src.relpath in seen:
            anomalies.append(f"duplicate source {src.relpath}")
        seen.add(src.relpath)
        if src.resampled:
            anomalies.append(
                f"{src.relpath}: VRT resamples {src.width}x{src.height} to {src.dst_w}x{src.dst_h}; "
                f"placement will come from the file's tie-point"
            )
        if src.block_y > src.height:
            anomalies.append(f"{src.relpath}: RowsPerStrip {src.block_y} exceeds height {src.height}")

    return LayerInventory(
        property=spec.name,
        depth=depth,
        vrt_url=url,
        vrt_size=len(xml),
        vrt_width=header["width"],
        vrt_height=header["height"],
        vrt_x_min=header["x_min"],
        vrt_y_max=header["y_max"],
        dtype=header["dtype"],
        nodata=header["nodata"],
        metadata=header["metadata"],
        sources=sources,
        anomalies=anomalies,
    )


def tile_rows(inv: LayerInventory, spec: config.PropertySpec) -> list[dict]:
    """Manifest rows for one layer: inventory + placement + consistency checks.

    Placement comes from the VRT (which reflects each file's own tie-point).
    Disagreement with the tile-name lattice is recorded as an anomaly; two tiles
    claiming one window is fatal.
    """
    anomalies = inv.anomalies
    g = config.GRID
    dx, dy = grid.xy_to_pixel(inv.vrt_x_min, inv.vrt_y_max)
    full = spec.tile_px
    rows: list[dict] = []
    claimed: dict[tuple[int, int], str] = {}
    for src in inv.sources:
        cell, name = src.relpath.split("/")
        ref = grid.parse_tile_name(name, full)
        if ref.cell != cell:
            raise SourceError(f"{inv.vrt_url}: source {src.relpath} sits in the wrong cell directory")
        x_off = round(src.dst_x) + dx
        y_off = round(src.dst_y) + dy
        nx, ny = ref.pixel_offset
        is_full = (src.width, src.height) == (full, full)
        if is_full and (x_off, y_off) != (nx, ny):
            # Measured upstream reality: bdod/0_5's tileSG-015-023_3-3.tif is a
            # full tile whose own GeoTIFF tie-point puts it one subtile west of
            # where its name implies -- it duplicates _3-2's window, and GDAL
            # dropped _3-2 from the VRT as a result. The VRT and the file AGREE;
            # it is the filename that is misleading, and names were never the
            # authority for placement (2 039 ragged tiles already sit at
            # arbitrary offsets). So this is recorded, not fatal.
            anomalies.append(
                f"{src.relpath}: full tile placed at ({x_off}, {y_off}) but its name implies "
                f"({nx}, {ny}); trusting the tie-point, which materialize re-asserts"
            )
        conflict_with = claimed.get((x_off, y_off))
        fits_x = x_off >= 0 and x_off + src.width <= g.width
        fits_y = y_off >= 0 and y_off + src.height <= g.height
        if not (fits_x and fits_y):
            raise SourceError(
                f"{inv.vrt_url}: {src.relpath} at ({x_off}, {y_off}) size {src.width}x{src.height} "
                f"does not fit the canonical grid {g.width}x{g.height}"
            )
        if conflict_with is not None:
            # Two tiles for one window. Never write both: the winner would depend
            # on which worker finished last. Pick deterministically here and let
            # resolve_conflicts verify the two really are interchangeable.
            #
            # Measured: nitrogen/0_5's tileSG-017-054 _1-1 and _1-2 are
            # byte-identical 450x450 rasters sharing a tie-point -- _1-2 is a
            # duplicate file with a wrong name, so either choice gives the same
            # store. bdod's _3-2/_3-3 pair differ in 89.5 % of pixels, which is a
            # different situation entirely and must not be resolved silently.
            keep, drop = _prefer(conflict_with, src.relpath, (x_off, y_off), full)
            inv.conflicts.append(Conflict(window=(x_off, y_off), keep=keep, drop=drop))
            if keep == conflict_with:
                continue
            rows[:] = [r for r in rows if r["relpath"] != conflict_with]
        claimed[(x_off, y_off)] = src.relpath
        rows.append(
            {
                "property": spec.name,
                "depth": inv.depth,
                "cell": cell,
                "subtile": f"{ref.r}-{ref.c}",
                "url": f"{config.SOURCE_BASE_URL}/{spec.name}/{spec.layer_dir(inv.depth)}/{src.relpath}",
                "relpath": src.relpath,
                "width": src.width,
                "height": src.height,
                "x_off": x_off,
                "y_off": y_off,
                "cell_row": ref.row,
                "cell_col": ref.col,
                "is_full": is_full,
                "rows_per_strip": src.block_y,
                "resampled_in_vrt": src.resampled,
                "size": None,
                "etag": None,
                "modified": None,
            }
        )
    return rows


def listing_targets(
    spec: config.PropertySpec, depth: str, rows: list[dict], *, samples: int, seed: int = 0
) -> list[tuple[str, str]]:
    """(cell, listing_url) pairs to cross-check for one layer."""
    cells = sorted({r["cell"] for r in rows})
    if not cells:
        return []
    picked = random.Random(seed).sample(cells, min(samples, len(cells)))
    base = spec.layer_url(depth)
    return [(cell, f"{base}/{cell}/") for cell in picked]


def check_listing(rows: list[dict], cell: str, html: str) -> tuple[list[str], list[str]]:
    """Compare one cell's real directory listing against the VRT inventory.

    Returns ``(disk_only, vrt_only)`` tile names. Also records the real file
    sizes in place.

    ``disk_only`` is not cosmetic. Measured on ``bdod/0_5``: 10 tiles across 9 of
    1 129 cells are on disk but absent from the VRT, and 8 of those hold data in
    windows nothing else covers -- about 26 000 km2, dominated by one full
    600 px tile. GDAL drops them when building the VRT (one has an off-lattice
    tie-point, one duplicates another tile's window), and the rest go with them.
    ``repair_from_disk`` puts back the ones that can be placed.
    """
    listing = parse_listing(html)
    in_cell = [r for r in rows if r["cell"] == cell]
    in_vrt = {r["relpath"].split("/")[1] for r in in_cell}
    on_disk = {n for n, size in listing.items() if n.endswith(".tif") and size is not None}
    for r in in_cell:
        if (size := listing.get(r["relpath"].split("/")[1])) is not None:
            r["size"] = size
    return sorted(on_disk - in_vrt), sorted(in_vrt - on_disk)


def repair_from_disk(
    fetcher: Fetcher,
    spec: config.PropertySpec,
    depth: str,
    rows: list[dict],
    disk_only: dict[str, list[str]],
) -> tuple[list[dict], list[str]]:
    """Add manifest rows for tiles the VRT omitted, placed by their own tie-point.

    Each candidate's header is fetched (a 4 kB range request; there are ~10 per
    layer) and classified:

    * **off-lattice** -- the tie-point is not on the 250 m grid, so the tile
      cannot be placed at all. Recorded and skipped; this is why GDAL rejected it.
    * **window already covered** -- keep the VRT's tile and record both names.
      Deterministic by construction: the VRT always wins, so a re-run gives the
      same store.
    * **window free** -- add it. This is the data the VRT would have lost.
    """
    claimed = {(r["x_off"], r["y_off"]): r["relpath"] for r in rows}
    g = config.GRID
    base = spec.layer_url(depth)
    jobs = [(cell, name) for cell, names in sorted(disk_only.items()) for name in names]
    if not jobs:
        return [], []

    def header_of(job):
        cell, name = job
        return job, fetcher.get_range(f"{base}/{cell}/{name}", 0, 4096)

    extra: list[dict] = []
    notes: list[str] = []
    for (cell, name), buf in fetcher.map(header_of, jobs):
        relpath = f"{cell}/{name}"
        try:
            header = tiff.parse_header(buf)
        except tiff.TiffError as exc:
            notes.append(f"{relpath}: absent from the VRT and unreadable ({exc}); skipped")
            continue
        fx = (header.tiepoint[0] - g.x_min) / g.pixel_size
        fy = (g.y_max - header.tiepoint[1]) / g.pixel_size
        if abs(fx - round(fx)) > 1e-6 or abs(fy - round(fy)) > 1e-6:
            notes.append(
                f"{relpath}: absent from the VRT and its tie-point is off the 250 m lattice "
                f"({fx:.1f}, {fy:.1f} px); cannot be placed, skipped"
            )
            continue
        x_off, y_off = round(fx), round(fy)
        if (prev := claimed.get((x_off, y_off))) is not None:
            notes.append(
                f"{relpath}: absent from the VRT and duplicates the window of {prev} at "
                f"({x_off}, {y_off}); keeping the VRT's tile"
            )
            continue
        fits = x_off >= 0 and y_off >= 0 and x_off + header.width <= g.width and y_off + header.height <= g.height
        if not fits:
            notes.append(f"{relpath}: absent from the VRT and does not fit the canonical grid; skipped")
            continue
        ref = grid.parse_tile_name(name, spec.tile_px)
        claimed[(x_off, y_off)] = relpath
        extra.append(
            {
                "property": spec.name,
                "depth": depth,
                "cell": cell,
                "subtile": f"{ref.r}-{ref.c}",
                "url": f"{base}/{relpath}",
                "relpath": relpath,
                "width": header.width,
                "height": header.height,
                "x_off": x_off,
                "y_off": y_off,
                "cell_row": ref.row,
                "cell_col": ref.col,
                "is_full": (header.width, header.height) == (spec.tile_px, spec.tile_px),
                "rows_per_strip": header.rows_per_strip,
                "resampled_in_vrt": False,
                "size": None,
                "etag": None,
                "modified": None,
            }
        )
        notes.append(
            f"{relpath}: absent from the VRT but placeable at ({x_off}, {y_off}); "
            f"recovered {header.width}x{header.height} px"
        )
    return extra, notes


def resolve_conflicts(fetcher: Fetcher, spec: config.PropertySpec, depth: str, conflicts: list[Conflict]) -> list[str]:
    """Verify each dropped tile was genuinely interchangeable with the kept one.

    Byte-identical duplicates are harmless: either choice yields the same store,
    so the deterministic pick in ``tile_rows`` stands. Tiles that *differ* in one
    window are a real data-quality question a pipeline must not decide silently,
    so those raise.
    """
    notes: list[str] = []
    base = spec.layer_url(depth)

    def load(relpath: str) -> np.ndarray:
        return tiff.decode(fetcher.get(f"{base}/{relpath}"))[1]

    for c in conflicts:
        kept, dropped = load(c.keep), load(c.drop)
        if kept.shape == dropped.shape and np.array_equal(kept, dropped):
            notes.append(f"{c.drop}: byte-identical duplicate of {c.keep} at window {c.window}; kept {c.keep}")
            continue
        detail = (
            "different shapes"
            if kept.shape != dropped.shape
            else f"{int((kept != dropped).sum())} of {kept.size} px differ"
        )
        raise SourceError(
            f"{spec.name}/{depth}: {c.keep} and {c.drop} both claim window {c.window} but their "
            f"pixels DIFFER ({detail}). A pipeline must not choose between two different rasters "
            f"for one window; this layer needs a manual decision."
        )
    return notes


def etag_targets(rows: list[dict], *, samples: int, seed: int = 0) -> list[dict]:
    """The manifest rows to HEAD for ETag/size verification."""
    return random.Random(seed).sample(rows, min(samples, len(rows)))


def record_etag(row: dict, info) -> None:
    row["size"], row["etag"], row["modified"] = info.size, info.etag, info.last_modified


# Declared, not inferred: size/etag/modified are populated only for the sampled
# rows, so schema inference sees Null in the first rows and then fails on the
# first real value.
MANIFEST_SCHEMA: dict[str, object] = {}


def _manifest_schema():
    import polars as pl

    return {
        "property": pl.String,
        "depth": pl.String,
        "cell": pl.String,
        "subtile": pl.String,
        "url": pl.String,
        "relpath": pl.String,
        "width": pl.Int32,
        "height": pl.Int32,
        "x_off": pl.Int32,
        "y_off": pl.Int32,
        "cell_row": pl.Int16,
        "cell_col": pl.Int16,
        "is_full": pl.Boolean,
        "rows_per_strip": pl.Int32,
        "resampled_in_vrt": pl.Boolean,
        "size": pl.Int64,
        "etag": pl.String,
        "modified": pl.String,
    }


def write_manifest(rows: list[dict], report: dict, work_dir: Path) -> tuple[Path, Path]:
    import polars as pl

    out = Path(work_dir) / config.SOURCE_VERSION
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / MANIFEST_NAME
    pl.DataFrame(rows, schema=_manifest_schema()).write_parquet(manifest)
    report_path = out / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return manifest, report_path


def read_manifest(work_dir: Path):
    import polars as pl

    path = Path(work_dir) / config.SOURCE_VERSION / MANIFEST_NAME
    if not path.exists():
        raise SourceError(f"no source manifest at {path}; run `make inspect-source` first")
    return pl.read_parquet(path)
