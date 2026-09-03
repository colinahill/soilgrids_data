"""Cell assembly and placement: the code path that decides where a pixel lands.

Placement is the failure mode with no symptoms -- a store full of plausible
numbers in the wrong places -- so these tests use the real geometry (450 px and
600 px subtilings, ragged tiles at arbitrary offsets) at toy scale.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from soilgrids import config, grid, materialize, tiff

from .conftest import NODATA, build_tile


def _row(ref: grid.TileRef, width: int, height: int, x_off: int, y_off: int, depth: str = "0_5") -> dict:
    return {
        "property": "sand",
        "depth": depth,
        "cell": ref.cell,
        "subtile": f"{ref.r}-{ref.c}",
        "url": f"https://example/{ref.relpath}",
        "relpath": ref.relpath,
        "width": width,
        "height": height,
        "x_off": x_off,
        "y_off": y_off,
        "cell_row": ref.row,
        "cell_col": ref.col,
        "is_full": (width, height) == (ref.tile_px, ref.tile_px),
        "rows_per_strip": 9,
        "resampled_in_vrt": False,
        "size": None,
        "etag": None,
        "modified": None,
    }


def _place(buf, ref, values, x_off, y_off, cell, depth_index=0, tiepoint=None):
    h, w = values.shape
    if tiepoint is None:
        g = config.GRID
        tiepoint = (g.x_min + x_off * g.pixel_size, g.y_max - y_off * g.pixel_size)
    buf_tile = build_tile(values, tiepoint)
    header, data = tiff.decode(buf_tile)
    materialize.place_tile(buf, _row(ref, w, h, x_off, y_off), header, data, cell, depth_index)


def test_full_subtiles_tile_a_cell_exactly(small_grid):
    """16 full tiles must fill a cell with no gap and no overlap."""
    n = grid.cell_px()
    buf = np.full((1, n, n), NODATA, dtype="int16")
    tile_px = 45
    for i, (r, c) in enumerate(grid.subtiles(tile_px)):
        ref = grid.TileRef(0, 0, r, c, tile_px)
        x_off, y_off = ref.pixel_offset
        _place(buf, ref, np.full((tile_px, tile_px), i, "int16"), x_off, y_off, (0, 0))
    assert not (buf == NODATA).any(), "the cell must be completely covered"
    assert sorted(np.unique(buf).tolist()) == list(range(len(grid.subtiles(tile_px))))


def test_600px_family_also_tiles_a_cell(small_grid):
    n = grid.cell_px()
    buf = np.full((1, n, n), NODATA, dtype="int16")
    tile_px = 30
    for i, (r, c) in enumerate(grid.subtiles(tile_px)):
        ref = grid.TileRef(0, 0, r, c, tile_px)
        _place(buf, ref, np.full((tile_px, tile_px), i + 1, "int16"), *ref.pixel_offset, (0, 0))
    assert not (buf == NODATA).any()
    assert len(np.unique(buf)) == 9


def test_ragged_tile_lands_at_its_own_offset_not_the_subtile_origin(small_grid):
    """2 039 of 2 801 ragged tiles sit at an arbitrary offset inside their subtile."""
    n = grid.cell_px()
    buf = np.full((1, n, n), NODATA, dtype="int16")
    ref = grid.TileRef(0, 0, 2, 2, 45)
    sx, sy = ref.pixel_offset
    x_off, y_off = sx + 7, sy + 11  # deliberately not the subtile origin
    _place(buf, ref, np.full((13, 9), 300, "int16"), x_off, y_off, (0, 0))
    assert (buf[0] == 300).sum() == 13 * 9
    ys, xs = np.where(buf[0] == 300)
    assert (ys.min(), xs.min()) == (y_off, x_off), "must honour the tile's own tie-point"
    assert (ys.max(), xs.max()) == (y_off + 12, x_off + 8)


def test_placement_refuses_a_tiepoint_that_disagrees_with_the_manifest(small_grid):
    n = grid.cell_px()
    buf = np.full((1, n, n), NODATA, dtype="int16")
    ref = grid.TileRef(0, 0, 1, 1, 45)
    g = config.GRID
    x_off, y_off = ref.pixel_offset
    # tie-point one pixel east of what the manifest row claims
    bad = (g.x_min + (x_off + 1) * g.pixel_size, g.y_max - y_off * g.pixel_size)
    with pytest.raises(materialize.PlacementMismatch, match="tie-point places it at"):
        _place(buf, ref, np.zeros((45, 45), "int16"), x_off, y_off, (0, 0), tiepoint=bad)


def test_placement_refuses_a_shape_that_disagrees_with_the_manifest(small_grid):
    n = grid.cell_px()
    buf = np.full((1, n, n), NODATA, dtype="int16")
    ref = grid.TileRef(0, 0, 1, 1, 45)
    x_off, y_off = ref.pixel_offset
    values = np.zeros((40, 45), "int16")
    g = config.GRID
    tile = build_tile(values, (g.x_min + x_off * g.pixel_size, g.y_max - y_off * g.pixel_size))
    header, data = tiff.decode(tile)
    row = _row(ref, 45, 45, x_off, y_off)  # manifest claims a full tile
    with pytest.raises(materialize.PlacementMismatch, match="file is 40x45"):
        materialize.place_tile(buf, row, header, data, (0, 0), 0)


def test_placement_refuses_a_tile_that_overflows_its_cell(small_grid):
    n = grid.cell_px()
    buf = np.full((1, n, n), NODATA, dtype="int16")
    ref = grid.TileRef(0, 0, 2, 2, 45)
    sx, sy = ref.pixel_offset
    with pytest.raises(materialize.PlacementMismatch, match="does not fit cell"):
        _place(buf, ref, np.zeros((45, 45), "int16"), sx + 40, sy + 40, (0, 0))


def test_placement_refuses_wrong_nodata(small_grid):
    n = grid.cell_px()
    buf = np.full((1, n, n), NODATA, dtype="int16")
    ref = grid.TileRef(0, 0, 1, 1, 45)
    x_off, y_off = ref.pixel_offset
    g = config.GRID
    tile = build_tile(
        np.zeros((45, 45), "int16"),
        (g.x_min + x_off * g.pixel_size, g.y_max - y_off * g.pixel_size),
        nodata=-9999,
    )
    header, data = tiff.decode(tile)
    with pytest.raises(materialize.PlacementMismatch, match="nodata -9999"):
        materialize.place_tile(buf, _row(ref, 45, 45, x_off, y_off), header, data, (0, 0), 0)


def test_depths_land_on_their_own_slice(small_grid):
    n = grid.cell_px()
    buf = np.full((6, n, n), NODATA, dtype="int16")
    ref = grid.TileRef(0, 0, 1, 1, 45)
    for i, _depth in enumerate(config.DEPTH_LABELS):
        _place(buf, ref, np.full((45, 45), 100 + i, "int16"), *ref.pixel_offset, (0, 0), depth_index=i)
    for i in range(6):
        assert buf[i, 0, 0] == 100 + i, "each depth must occupy only its own slice"


def test_all_fill_shards_are_never_written(small_grid, tmp_path):
    """Ocean is 76 % of the grid; writing it would cost storage for nothing."""
    enc = config.ENCODING
    n = grid.cell_px()
    root = zarr.open_group(zarr.storage.LocalStore(str(tmp_path / "z")), mode="a")
    arr = root.create_array(
        "sand",
        shape=(6, config.GRID.height, config.GRID.width),
        chunks=enc.chunks(3),
        shards=enc.shards(3, depth_extent=6),
        dtype="int16",
        fill_value=NODATA,
    )
    empty_before = arr.nbytes_stored()

    stats = materialize.CellStats()
    buf = np.full((6, n, n), NODATA, dtype="int16")
    buf[:, :10, :10] = 500  # data in one shard only
    for y0, y1, x0, x1 in grid.shard_windows(0, 0):
        cy, cx = y0, x0
        block = buf[..., cy : cy + (y1 - y0), cx : cx + (x1 - x0)]
        if not np.any(block != NODATA):
            stats.shards_empty += 1
            continue
        arr[:, y0:y1, x0:x1] = block
        stats.shards_written += 1
    assert stats.shards_written == 1
    assert stats.shards_empty == len(grid.shard_windows(0, 0)) - 1
    assert arr.nbytes_stored() > empty_before
    assert np.all(arr[0, 45:90, 0:45] == NODATA)


def test_bbox_maps_to_the_expected_cells():
    """A High Plains box must land in the cells the real run used."""
    cells = materialize.cells_in_bbox((-104.0, 37.0, -95.0, 43.0))
    assert cells == {(8, 19), (8, 20), (9, 18), (9, 19), (9, 20)}


def test_bbox_ignores_positions_outside_the_grid():
    # Antarctica is outside the SoilGrids extent entirely
    assert materialize.cells_in_bbox((-60.0, -85.0, -50.0, -80.0)) == set()


def test_unknown_depth_is_refused_not_written_into_the_depth_axis(small_grid):
    """A None depth index would index the 3-D buffer's first axis and corrupt it."""
    import numpy as np

    from soilgrids.fetch import Fetcher  # noqa: F401  (type only)

    sand = config.PROPERTIES["sand"]
    ref = grid.TileRef(0, 0, 1, 1, sand.tile_px)
    x_off, y_off = ref.pixel_offset
    g = config.GRID
    row = _row(ref, sand.tile_px, sand.tile_px, x_off, y_off, depth="0_30")  # not a sand depth
    tile = build_tile(
        np.zeros((sand.tile_px, sand.tile_px), "int16"),
        (g.x_min + x_off * g.pixel_size, g.y_max - y_off * g.pixel_size),
    )

    class OneTile:
        workers = 1

        def get(self, url, **kw):
            return tile

        def map(self, fn, items, **kw):
            return [fn(i) for i in items]

    n = grid.cell_px()
    arr = zarr.open_group(zarr.storage.MemoryStore(), mode="a").create_array(
        "sand", shape=(6, n, n), chunks=(5, 5, 5), dtype="int16", fill_value=NODATA
    )
    with pytest.raises(materialize.PlacementMismatch, match="is not one of sand's depths"):
        materialize.fill_cell(OneTile(), arr, sand, (0, 0), [row], materialize.CellStats())
