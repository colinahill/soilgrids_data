"""Canonical-grid arithmetic: tile names, pixel windows, and the work unit.

The ISRIC tile lattice is the authority for placement (see config.GridSpec).
Two facts drive the decomposition:

* subtiles are 450 px for nine properties and 600 px for bdod/phh2o;
* a native shard is 450 px.

lcm(450, 600, 450) = 1800 px = one ``tileSG`` cell, so the unit of work is ONE
CELL of one property: it contains whole source tiles (16 of 450 px, or 9 of
600 px) and whole shards (4x4) for every property, needs a 38.9 MB buffer for
all six depths, and fetches each source tile exactly once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config

TILE_NAME_RE = re.compile(r"^tileSG-(\d{3})-(\d{3})_(\d)-(\d)\.tif$")


def cell_px() -> int:
    """One tileSG cell in canonical pixels (1800 for the real grid).

    A function, not a constant, so tests can shrink the grid via config.GRID.
    """
    return round(config.GRID.cell_m / config.GRID.pixel_size)


@dataclass(frozen=True, slots=True)
class TileRef:
    """One source GeoTIFF: its name, its cell, and where it lands on the grid."""

    row: int  # tileSG cell row, 0 at the top
    col: int  # tileSG cell column, 0 at the far west
    r: int  # subtile row within the cell, 1-based
    c: int  # subtile column within the cell, 1-based
    tile_px: int

    @property
    def cell(self) -> str:
        return f"tileSG-{self.row:03d}-{self.col:03d}"

    @property
    def name(self) -> str:
        return f"{self.cell}_{self.r}-{self.c}.tif"

    @property
    def relpath(self) -> str:
        return f"{self.cell}/{self.name}"

    @property
    def origin_xy(self) -> tuple[float, float]:
        """Upper-left corner in projected metres (the GeoTIFF tie-point)."""
        g = config.GRID
        step = self.tile_px * g.pixel_size
        return (
            g.x_min + self.col * g.cell_m + (self.c - 1) * step,
            g.y_max - self.row * g.cell_m - (self.r - 1) * step,
        )

    @property
    def pixel_offset(self) -> tuple[int, int]:
        """(x_off, y_off) of the tile's upper-left pixel on the canonical grid."""
        x, y = self.origin_xy
        return xy_to_pixel(x, y)


def parse_tile_name(name: str, tile_px: int) -> TileRef:
    m = TILE_NAME_RE.match(name)
    if not m:
        raise ValueError(f"not a SoilGrids tile name: {name!r}")
    row, col, r, c = (int(g) for g in m.groups())
    per_side = 1800 // tile_px
    if not (1 <= r <= per_side and 1 <= c <= per_side):
        raise ValueError(f"{name}: subtile {r}-{c} is outside a {per_side}x{per_side} cell")
    return TileRef(row=row, col=col, r=r, c=c, tile_px=tile_px)


def xy_to_pixel(x: float, y: float) -> tuple[int, int]:
    """Projected metres -> canonical pixel offset, refusing anything off-lattice.

    Used at write time on each tile's own tie-point: the file is the ground
    truth, not the manifest, so a source that moved gets caught here.
    """
    g = config.GRID
    fx = (x - g.x_min) / g.pixel_size
    fy = (g.y_max - y) / g.pixel_size
    ix, iy = round(fx), round(fy)
    if abs(fx - ix) > 1e-6 or abs(fy - iy) > 1e-6:
        raise ValueError(
            f"tie-point ({x}, {y}) is not on the canonical {g.pixel_size} m lattice "
            f"(offset {fx}, {fy}); the source grid has changed"
        )
    if not (0 <= ix <= g.width and 0 <= iy <= g.height):
        raise ValueError(f"tie-point ({x}, {y}) -> pixel ({ix}, {iy}) is outside the canonical grid")
    return ix, iy


def cells() -> list[tuple[int, int]]:
    """Every (row, col) cell of the canonical grid, in raster order."""
    g = config.GRID
    return [(r, c) for r in range(g.max_tile_row + 1) for c in range(g.max_tile_col + 1)]


def cell_pixel_offset(row: int, col: int) -> tuple[int, int]:
    """(x_off, y_off) of a cell's upper-left pixel."""
    n = cell_px()
    return col * n, row * n


def subtiles(tile_px: int) -> list[tuple[int, int]]:
    """The 1-based (r, c) subtile positions inside one cell."""
    per_side = cell_px() // tile_px
    return [(r, c) for r in range(1, per_side + 1) for c in range(1, per_side + 1)]


def cell_tiles(row: int, col: int, tile_px: int) -> list[TileRef]:
    """Every candidate tile of one cell (some will not exist upstream)."""
    return [TileRef(row=row, col=col, r=r, c=c, tile_px=tile_px) for r, c in subtiles(tile_px)]


def shard_windows(row: int, col: int, enc: config.EncodingSpec | None = None) -> list[tuple[int, int, int, int]]:
    """Whole-shard windows inside one cell, as (y0, y1, x0, x1) canonical pixels.

    Writes are shard-aligned so concurrent workers never co-write one storage
    object and no shard is ever read-modify-written.
    """
    enc = enc or config.ENCODING
    n = cell_px()
    if n % enc.shard_y or n % enc.shard_x:
        raise ValueError(f"cell {n} px is not a whole number of {enc.shard_y}x{enc.shard_x} shards")
    x0c, y0c = cell_pixel_offset(row, col)
    out = []
    for dy in range(0, n, enc.shard_y):
        for dx in range(0, n, enc.shard_x):
            out.append((y0c + dy, y0c + dy + enc.shard_y, x0c + dx, x0c + dx + enc.shard_x))
    return out


def overview_shapes(
    width: int | None = None, height: int | None = None, factors: tuple[int, ...] | None = None
) -> dict[int, tuple[int, int]]:
    """Level shapes for the overview ladder, as {factor: (height, width)}.

    Chained stride-2 coarsening with ``boundary="trim"`` semantics, so each
    level is floor(parent / 2) rather than floor(native / factor); the two
    differ once the grid stops dividing evenly (160200 and 59400 divide by 2, 4
    and 8 only).
    """
    g = config.GRID
    w = g.width if width is None else width
    h = g.height if height is None else height
    factors = factors or config.OVERVIEW_FACTORS
    shapes: dict[int, tuple[int, int]] = {}
    cur_h, cur_w, cur_f = h, w, 1
    while cur_f < max(factors):
        cur_h, cur_w, cur_f = cur_h // 2, cur_w // 2, cur_f * 2
        if cur_f in factors:
            shapes[cur_f] = (cur_h, cur_w)
    return shapes


def level_grid(factor: int, shape: tuple[int, int]) -> dict[str, object]:
    """Origin/pixel-size/coords for one overview level (same origin as native)."""
    import numpy as np

    g = config.GRID
    px = g.pixel_size * factor
    h, w = shape
    return {
        "pixel_size": px,
        "x": g.x_min + (np.arange(w) + 0.5) * px,
        "y": g.y_max - (np.arange(h) + 0.5) * px,
        "geotransform": f"{g.x_min} {px} 0.0 {g.y_max} 0.0 {-px}",
    }
