"""grid.py: tile names, placement, and the cell work unit."""

from __future__ import annotations

import pytest

from soilgrids import config, grid

# Tie-points measured from the real tiles (see docs/data-reference.md).
MEASURED = [
    ("tileSG-018-027_1-1.tif", 450, (-7_887_500.0, 500_750.0)),
    ("tileSG-018-027_2-4.tif", 450, (-7_550_000.0, 388_250.0)),
    ("tileSG-028-080_2-4.tif", 450, (16_300_000.0, -4_111_750.0)),
    ("tileSG-018-027_1-1.tif", 600, (-7_887_500.0, 500_750.0)),
]


@pytest.mark.parametrize("name,tile_px,tiepoint", MEASURED)
def test_name_to_tiepoint_matches_the_measured_geotiffs(name, tile_px, tiepoint):
    assert grid.parse_tile_name(name, tile_px).origin_xy == tiepoint


@pytest.mark.parametrize("name,tile_px,tiepoint", MEASURED)
def test_tiepoint_round_trips_through_the_pixel_lattice(name, tile_px, tiepoint):
    ref = grid.parse_tile_name(name, tile_px)
    assert grid.xy_to_pixel(*tiepoint) == ref.pixel_offset


def test_tile_names_and_relpaths():
    ref = grid.TileRef(row=18, col=27, r=2, c=4, tile_px=450)
    assert ref.cell == "tileSG-018-027"
    assert ref.name == "tileSG-018-027_2-4.tif"
    assert ref.relpath == "tileSG-018-027/tileSG-018-027_2-4.tif"


def test_bad_tile_names_are_rejected():
    with pytest.raises(ValueError, match="not a SoilGrids tile name"):
        grid.parse_tile_name("sand_0-5cm_mean.tif", 450)
    with pytest.raises(ValueError, match="outside a 4x4 cell"):
        grid.parse_tile_name("tileSG-000-000_5-1.tif", 450)
    with pytest.raises(ValueError, match="outside a 3x3 cell"):
        grid.parse_tile_name("tileSG-000-000_4-1.tif", 600)


def test_off_lattice_tiepoint_is_refused():
    # half a pixel off: a silently shifted grid is the failure mode that matters
    x, y = grid.parse_tile_name("tileSG-018-027_1-1.tif", 450).origin_xy
    with pytest.raises(ValueError, match="not on the canonical"):
        grid.xy_to_pixel(x + 125.0, y)
    with pytest.raises(ValueError, match="outside the canonical grid"):
        grid.xy_to_pixel(config.GRID.x_min - 250.0, y)


def test_subtilings_cover_a_cell_for_both_families():
    assert len(grid.subtiles(450)) == 16
    assert len(grid.subtiles(600)) == 9
    for tile_px in (450, 600):
        offs = {grid.TileRef(0, 0, r, c, tile_px).pixel_offset for r, c in grid.subtiles(tile_px)}
        assert len(offs) == len(grid.subtiles(tile_px)), "subtiles must not overlap"
        area = len(offs) * tile_px**2
        assert area == grid.cell_px() ** 2, "subtiles must tile the cell exactly"


def test_cells_cover_the_grid_exactly():
    cells = grid.cells()
    assert len(cells) == 33 * 89 == 2937
    g = config.GRID
    x, y = grid.cell_pixel_offset(g.max_tile_row, g.max_tile_col)
    assert x + grid.cell_px() == g.width
    assert y + grid.cell_px() == g.height


def test_shard_windows_tile_a_cell_without_overlap():
    wins = grid.shard_windows(18, 27)
    assert len(wins) == 16
    n = grid.cell_px()
    x0c, y0c = grid.cell_pixel_offset(18, 27)
    covered = sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in wins)
    assert covered == n * n
    assert min(w[0] for w in wins) == y0c and min(w[2] for w in wins) == x0c
    assert max(w[1] for w in wins) == y0c + n and max(w[3] for w in wins) == x0c + n


def test_shard_windows_are_aligned_to_the_array_shard_grid():
    enc = config.ENCODING
    for y0, y1, x0, x1 in grid.shard_windows(7, 11):
        assert y0 % enc.shard_y == 0 and x0 % enc.shard_x == 0
        assert (y1 - y0, x1 - x0) == (enc.shard_y, enc.shard_x)


def test_overview_shapes_use_trim_not_ceil():
    shapes = grid.overview_shapes()
    assert shapes[2] == (29_700, 80_100)
    assert shapes[8] == (7_425, 20_025)
    # 160200/16 = 10012.5 and 59400/16 = 3712.5: trim floors, ceil would not
    assert shapes[16] == (3_712, 10_012)
    assert shapes[256] == (232, 625)
    for f in config.OVERVIEW_FACTORS:
        h, w = shapes[f]
        assert h <= -(-config.GRID.height // f) and w <= -(-config.GRID.width // f)


def test_overview_shapes_are_a_strict_stride_2_chain():
    shapes = grid.overview_shapes()
    prev = (config.GRID.height, config.GRID.width)
    for f in config.OVERVIEW_FACTORS:
        assert shapes[f] == (prev[0] // 2, prev[1] // 2)
        prev = shapes[f]


def test_overview_levels_share_the_native_origin():
    shapes = grid.overview_shapes()
    for f in (2, 16, 256):
        lg = grid.level_grid(f, shapes[f])
        assert lg["pixel_size"] == 250.0 * f
        assert lg["x"][0] == config.GRID.x_min + 0.5 * 250.0 * f
        assert lg["y"][0] == config.GRID.y_max - 0.5 * 250.0 * f


def test_shrunken_grid_keeps_the_same_invariants(small_grid):
    assert grid.cell_px() == 90
    assert len(grid.cells()) == 4
    assert len(grid.shard_windows(0, 0)) == 4
    for tile_px, per_side in ((45, 2), (30, 3)):
        assert len(grid.subtiles(tile_px)) == per_side**2
        offs = {grid.TileRef(0, 0, r, c, tile_px).pixel_offset for r, c in grid.subtiles(tile_px)}
        assert len(offs) * tile_px**2 == 90 * 90
