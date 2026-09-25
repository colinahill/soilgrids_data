"""Sources ISRIC publishes coarser than 250 m.

Three upstream tiles are measurably below the canonical resolution -- 500 m,
750 m and 15 500 m -- and the 15 500 m one (tileSG-010-049_1-1) appears in all
eleven properties. Their values are ordinary predictions, so they are kept, but
they must never displace a full-resolution pixel and must never be placed 1:1.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from soilgrids import config, grid, materialize, tiff

from .conftest import FILL, NODATA, build_tile
from .test_materialize import _row


def _coarse_tile(values, x_off, y_off, ratio, *, half_pixel=False):
    """A tile whose pixels are `ratio` canonical pixels, as (rx, ry) or one int."""
    g = config.GRID
    rx, ry = (ratio, ratio) if isinstance(ratio, int) else ratio
    x = g.x_min + (x_off + (0.5 if half_pixel else 0)) * g.pixel_size
    y = g.y_max - (y_off + (0.5 if half_pixel else 0)) * g.pixel_size
    return build_tile(
        np.asarray(values, "int16"),
        (x, y),
        pixel_size=(g.pixel_size * rx, g.pixel_size * ry),
        rows_per_strip=1,
    )


# ---------------------------------------------------------------------------
# grid.source_window
# ---------------------------------------------------------------------------


def test_a_250m_tile_is_not_coarse_and_keeps_the_strict_lattice_check():
    g = config.GRID
    x, y = grid.parse_tile_name("tileSG-018-027_1-1.tif", 450).origin_xy
    win = grid.source_window((x, y), (g.pixel_size, g.pixel_size), 450, 450)
    assert not win.coarse
    assert (win.ratio_x, win.ratio_y, win.width, win.height) == (1, 1, 450, 450)
    with pytest.raises(ValueError, match="not on the canonical"):
        grid.source_window((x + 125.0, y), (g.pixel_size, g.pixel_size), 450, 450)


def test_a_square_coarse_tile_expands_on_both_axes():
    """The measured 15 500 m case: 1x2 source -> 62x124 canonical pixels."""
    g = config.GRID
    win = grid.source_window((g.x_min + 1000 * g.pixel_size, g.y_max - 500 * g.pixel_size), (15_500.0, 15_500.0), 1, 2)
    assert (win.ratio_x, win.ratio_y, win.width, win.height) == (62, 62, 62, 124)
    assert win.coarse


def test_an_anisotropic_tile_expands_only_on_its_coarse_axis():
    """Measured: tileSG-017-086_2-1 is 250 m across and 23 750 m down.

    A 28x2 source becomes 28x190 -- full resolution in x, 95 canonical pixels
    per source pixel in y -- which is exactly the VRT's DstRect for it.
    """
    g = config.GRID
    win = grid.source_window((g.x_min + 1000 * g.pixel_size, g.y_max - 500 * g.pixel_size), (250.0, 23_750.0), 28, 2)
    assert (win.ratio_x, win.ratio_y) == (1, 95)
    assert (win.width, win.height) == (28, 190)
    assert win.coarse


def test_a_full_resolution_axis_is_still_held_to_the_lattice():
    """Anisotropy relaxes only the coarse axis: x is still 250 m, so it must land."""
    g = config.GRID
    y = g.y_max - 500 * g.pixel_size
    with pytest.raises(ValueError, match="x offset"):
        grid.source_window((g.x_min + 1000.5 * g.pixel_size, y), (250.0, 4250.0), 7, 2)


def test_a_coarse_tiepoint_may_sit_off_the_lattice():
    """tileSG-010-049_1-1 sits exactly half a canonical pixel off, by measurement.

    `xy_to_pixel` refuses that -- correctly, for a 250 m tile -- so a coarse tile
    has to round instead, which is the whole reason source_window exists.
    """
    g = config.GRID
    x = g.x_min + 1000.5 * g.pixel_size
    y = g.y_max - 500.5 * g.pixel_size
    with pytest.raises(ValueError, match="not on the canonical"):
        grid.xy_to_pixel(x, y)
    win = grid.source_window((x, y), (15_500.0, 15_500.0), 1, 2)
    assert (win.x_off, win.y_off) == (1000, 500)
    assert (win.fx, win.fy) == (1000.5, 500.5)


def test_a_fractional_pixel_size_is_refused():
    with pytest.raises(ValueError, match="not a whole multiple"):
        grid.source_window((config.GRID.x_min, config.GRID.y_max), (375.0, 375.0), 2, 2)


def test_cells_touching_covers_a_window_that_straddles_four_cells(small_grid):
    n = grid.cell_px()
    assert grid.cells_touching(0, 0, n, n) == [(0, 0)]
    # the measured shape of 010-049_1-1: a window overlapping four cells at once
    assert set(grid.cells_touching(n - 2, n - 2, 4, 4)) == {(0, 0), (0, 1), (1, 0), (1, 1)}


# ---------------------------------------------------------------------------
# the placement guard
# ---------------------------------------------------------------------------


def test_place_tile_refuses_a_coarse_tile_outright(small_grid):
    """The 750 m case (tileSG-014-022_3-4) is ON the lattice, so only an explicit
    pixel-size check catches it; without one it is silently written 3x too small.
    """
    n = grid.cell_px()
    buf = np.full((1, n, n), FILL, dtype=config.DTYPE)
    ref = grid.TileRef(0, 0, 1, 1, 45)
    header, data = tiff.decode(_coarse_tile([[100, 200], [300, 400]], 4, 4, 3))
    with pytest.raises(materialize.PlacementMismatch, match="not the canonical"):
        materialize.place_tile(buf, _row(ref, 2, 2, 4, 4), header, data, (0, 0), 0, config.PROPERTIES["sand"])


# ---------------------------------------------------------------------------
# gap-fill semantics
# ---------------------------------------------------------------------------


def _fill(values, x_off, y_off, ratio, depth_index=0):
    spec = config.PROPERTIES["sand"]
    rx, ry = (ratio, ratio) if isinstance(ratio, int) else ratio
    _, data = tiff.decode(_coarse_tile(values, x_off, y_off, (rx, ry)))
    expanded = materialize.decode_values(data, spec)
    expanded = np.repeat(np.repeat(expanded, ry, axis=0), rx, axis=1)
    return materialize.CoarseFill(
        relpath="tileSG-000-000/tileSG-000-000_1-1.tif",
        depth="0_5",
        ratio_x=rx,
        ratio_y=ry,
        x_off=x_off,
        y_off=y_off,
        width=expanded.shape[1],
        height=expanded.shape[0],
        values=expanded,
        depth_index=depth_index,
    )


def _blank(n):
    """An empty cell: nothing written, and no 250 m tile has reached anywhere."""
    return np.full((1, n, n), FILL, dtype=config.DTYPE), np.zeros((1, n, n), bool)


def test_coarse_values_land_where_no_250m_tile_reached(small_grid):
    n = grid.cell_px()
    buf, covered = _blank(n)
    written = materialize.apply_coarse(buf, _fill([[100, 200]], 4, 4, 3), (0, 0), covered)
    assert written == 18  # a 1x2 source at ratio 3 covers 3x6 canonical pixels
    assert np.allclose(buf[0, 4:7, 4:7], 10.0)  # 100 / conversion_factor 10
    assert np.allclose(buf[0, 4:7, 7:10], 20.0)
    assert np.isnan(buf[0, 4:7, 10]).all(), "nothing outside the window may be touched"


def test_full_resolution_data_is_never_overwritten(small_grid):
    n = grid.cell_px()
    buf, covered = _blank(n)
    buf[0, 4:6, 4:6] = 99.0  # a full-resolution tile got here first
    covered[0, 4:6, 4:6] = True
    written = materialize.apply_coarse(buf, _fill([[100, 200]], 4, 4, 3), (0, 0), covered)
    assert written == 18 - 4
    assert np.all(buf[0, 4:6, 4:6] == 99.0), "the 250 m prediction must survive"
    assert np.allclose(buf[0, 6, 4:7], 10.0), "the rest of the window still fills"


def test_a_250m_nodata_blocks_the_coarse_fill(small_grid):
    """Coverage, not fill, is the test.

    A 250 m nodata says the fine-resolution soil mask excluded that pixel -- a
    more precise statement than a coarse aggregate can make, since at 15.5 km a
    mostly-land cell carries a value straight across a lake. So a covered pixel
    stays empty even though it holds no value.
    """
    n = grid.cell_px()
    buf, covered = _blank(n)
    covered[0, 4:6, 4:6] = True  # a tile reached here, and was nodata
    written = materialize.apply_coarse(buf, _fill([[100, 200]], 4, 4, 3), (0, 0), covered)
    assert written == 18 - 4
    assert np.isnan(buf[0, 4:6, 4:6]).all(), "a 250 m nodata must not be contradicted"
    assert np.allclose(buf[0, 6, 4:7], 10.0)


def test_gap_filling_does_not_depend_on_arrival_order(small_grid):
    """The VRT resolves this overlap by source order; the store must not."""
    n = grid.cell_px()
    fill = _fill([[100, 200]], 4, 4, 3)
    before, covered = _blank(n)
    before[0, 4:6, 4:6] = 99.0
    covered[0, 4:6, 4:6] = True
    materialize.apply_coarse(before, fill, (0, 0), covered)
    # the same full-resolution tile, arriving after the coarse source instead
    after, covered2 = _blank(n)
    covered2[0, 4:6, 4:6] = True
    materialize.apply_coarse(after, fill, (0, 0), covered2)
    after[0, 4:6, 4:6] = 99.0
    np.testing.assert_array_equal(np.nan_to_num(before, nan=-1), np.nan_to_num(after, nan=-1))


def test_nodata_in_a_coarse_source_fills_nothing(small_grid):
    n = grid.cell_px()
    buf, covered = _blank(n)
    assert materialize.apply_coarse(buf, _fill([[NODATA, NODATA]], 4, 4, 3), (0, 0), covered) == 0
    assert np.isnan(buf).all()


def test_a_window_is_clipped_to_each_cell_it_touches(small_grid):
    """A straddling window must be split across cells, losing no pixel and
    double-writing none."""
    n = grid.cell_px()
    x0 = y0 = n - 3
    fill = _fill([[100, 200]], x0, y0, 3)  # 3x6 window starting 3 px inside cell (0,0)
    touched = grid.cells_touching(x0, y0, fill.width, fill.height)
    assert set(touched) == {(0, 0), (0, 1)}
    total = 0
    for cell in touched:
        buf, covered = _blank(n)
        total += materialize.apply_coarse(buf, fill, cell, covered)
    assert total == fill.width * fill.height


# ---------------------------------------------------------------------------
# the wiring: resolve -> fan out across cells -> gap-fill -> provenance
# ---------------------------------------------------------------------------


class _Source:
    """Serves full-resolution tiles plus one coarse tile, like the real tree."""

    workers = 2

    def __init__(self, rows, coarse_row=None, coarse_values=None, ratio=1):
        g = config.GRID
        self.by_url = {}
        for r in rows:
            values = (200 + (np.mgrid[0 : r["height"], 0 : r["width"]][0] * 7)).astype("int16")
            tp = (g.x_min + r["x_off"] * g.pixel_size, g.y_max - r["y_off"] * g.pixel_size)
            self.by_url[r["url"]] = build_tile(values, tp, rows_per_strip=r["rows_per_strip"])
        if coarse_row is not None:
            self.by_url[coarse_row["url"]] = _coarse_tile(
                coarse_values, coarse_row["x_off"], coarse_row["y_off"], ratio
            )

    def get(self, url, *, expect_etag=None, expect_size=None):
        return self.by_url[url]

    def map(self, fn, items, *, workers=None):
        return [fn(i) for i in items]


@pytest.fixture
def repo(tmp_path):
    import icechunk

    return icechunk.Repository.create(icechunk.local_filesystem_storage(str(tmp_path / "store.icechunk")))


def test_a_coarse_tile_fills_gaps_across_cells_and_is_recorded(small_grid, repo):
    """The measured shape of tileSG-010-049_1-1, at toy scale.

    Its name puts it in one cell, its window lands in another and straddles a
    boundary, and half of that window is already covered by full-resolution
    tiles. Only the uncovered half may be written -- and the store must say so.
    """
    import zarr

    from soilgrids import template

    from .test_end_to_end import _manifest, _rows

    sand = config.PROPERTIES["sand"]
    rows = _rows(sand, [(0, 0)])  # cell (0,0) fully covered at 250 m; (0,1) empty
    coarse = dict(
        rows[0],
        depth="0_5",
        cell="tileSG-000-001",
        subtile="1-1",
        url="fake://sand/0_5/coarse",
        relpath="tileSG-000-001/tileSG-000-001_1-1.tif",
        width=2,
        height=2,
        x_off=87,
        y_off=10,  # 6x6 window at (87,10) -> straddles cells (0,0) and (0,1)
        cell_row=0,
        cell_col=1,
        is_full=False,
        rows_per_strip=1,
        resampled_in_vrt=True,
    )
    manifest = _manifest([*rows, coarse])
    src = _Source(rows, coarse, [[100, 200], [300, 400]], 3)

    session = repo.writable_session("main")
    template.init_store(session, [sand], factors=(2,))
    session.commit("init")

    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, manifest, progress=False)
    session.commit("materialize")

    # the coarse window reaches a cell with no manifest rows of its own, so that
    # cell has to have been scheduled at all
    assert stats.cells_done == 2
    assert stats.coarse_pixels == {"tileSG-000-001/tileSG-000-001_1-1.tif#0_5": 18}

    ro = repo.readonly_session("main")
    gp = zarr.open_group(ro.store, path=sand.group, mode="r")
    arr = gp[sand.name]
    # the half of the window inside cell (0,0) was already full-resolution data
    assert not np.isnan(arr[0, 10:16, 87:90]).any()
    assert not np.allclose(arr[0, 10:16, 87:90], 10.0), "250 m data must not be displaced"
    # the half in the empty cell is the expanded coarse source: col 1 of a 2x2
    np.testing.assert_allclose(arr[0, 10:13, 90:93], 20.0)
    np.testing.assert_allclose(arr[0, 13:16, 90:93], 40.0)
    # ... and nothing outside the window moved
    assert np.isnan(arr[0, 10:16, 93]).all()

    recorded = gp.attrs[materialize.COARSE_ATTR][sand.name]
    assert len(recorded) == 1
    assert recorded[0]["source"] == "tileSG-000-001/tileSG-000-001_1-1.tif"
    assert recorded[0]["native_pixel_size_m"] == [750.0, 750.0]
    assert (recorded[0]["width"], recorded[0]["height"]) == (6, 6)
    assert recorded[0]["pixels_filled"] == 18


def test_a_250m_file_flagged_by_the_vrt_stays_on_the_normal_path(small_grid, repo):
    """The file, not the VRT, decides whether a tile is coarse.

    No measured tile is flagged `resampled_in_vrt` while being 250 m on both
    axes -- every one of the six is genuinely coarse on at least one axis -- but
    the VRT is not the authority, so a 250 m file must be placed normally and
    byte-exactly rather than diverted into the gap-fill path.
    """
    from soilgrids import template

    from .test_end_to_end import _manifest, _rows

    sand = config.PROPERTIES["sand"]
    rows = _rows(sand, [(0, 0)])
    for r in rows:
        r["resampled_in_vrt"] = True  # the VRT says resampled; every file is 250 m
    manifest = _manifest(rows)
    src = _Source(rows)

    session = repo.writable_session("main")
    template.init_store(session, [sand], factors=(2,))
    session.commit("init")
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, manifest, progress=False)

    assert stats.coarse_pixels == {}, "a 250 m file must never be treated as coarse"
    assert stats.tiles_fetched == len(rows), "every tile still goes through place_tile"


def test_a_tile_that_failed_to_fetch_still_blocks_the_coarse_fill(small_grid, repo, monkeypatch):
    """A fetch error must stay a hole and an anomaly, not become a coarse guess."""
    from soilgrids import template

    monkeypatch.setattr(materialize, "MAX_ATTEMPTS", 1)  # no backoff: the failure is the point

    from .test_end_to_end import _manifest, _rows

    sand = config.PROPERTIES["sand"]
    rows = _rows(sand, [(0, 0)])
    dead = {r["url"] for r in rows if r["depth"] == "0_5"}
    coarse = dict(
        rows[0],
        depth="0_5",
        url="fake://sand/0_5/coarse",
        relpath="tileSG-000-000/tileSG-000-000_9-9.tif",
        width=2,
        height=2,
        x_off=10,
        y_off=10,
        is_full=False,
        rows_per_strip=1,
        resampled_in_vrt=True,
    )
    manifest = _manifest([*rows, coarse])

    src = _Source(rows, coarse, [[100, 200], [300, 400]], 3)
    unreachable = {u: src.by_url.pop(u) for u in dead}

    def get(url, *, expect_etag=None, expect_size=None):
        if url in unreachable:
            raise RuntimeError("503 from the source tree")
        return src.by_url[url]

    src.get = get

    session = repo.writable_session("main")
    template.init_store(session, [sand], factors=(2,))
    session.commit("init")
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, manifest, progress=False)

    assert stats.tiles_missing == len(dead)
    assert stats.coarse_pixels == {}, "a missing tile's window is covered, not coarse-fillable"


def test_one_coarse_depth_does_not_suppress_a_250m_depth_of_the_same_tile(small_grid, repo):
    """The skip is keyed by (relpath, depth), not by relpath.

    Nothing measured has one depth of a tile coarse and another at 250 m, but a
    relpath-only skip would drop the 250 m depth silently -- placed by neither
    path, with no error.
    """
    from soilgrids import template

    from .test_end_to_end import _manifest, _rows

    sand = config.PROPERTIES["sand"]
    rows = _rows(sand, [(0, 0)])
    ref = grid.TileRef(0, 0, 1, 1, sand.tile_px)
    twin = [r for r in rows if r["relpath"] == ref.relpath]
    assert len(twin) == len(sand.depths)
    for r in twin:  # the VRT flags every depth of this tile...
        r["resampled_in_vrt"] = True

    src = _Source(rows)
    # ... but only depth 0_5 is actually coarse; the rest stay 250 m
    coarse_row = next(r for r in twin if r["depth"] == "0_5")
    src.by_url[coarse_row["url"]] = _coarse_tile([[100, 200], [300, 400]], 10, 10, 3)
    coarse_row.update(width=2, height=2, x_off=10, y_off=10, is_full=False, rows_per_strip=1)

    session = repo.writable_session("main")
    template.init_store(session, [sand], factors=(2,))
    session.commit("init")
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, _manifest(rows), progress=False)
    session.commit("materialize")

    assert list(stats.coarse_pixels) == [f"{ref.relpath}#0_5"]
    # every other depth of that tile must still have been placed at 250 m
    arr = zarr.open_group(repo.readonly_session("main").store, path=sand.group, mode="r")[sand.name]
    x0, y0 = ref.pixel_offset
    for di, depth in enumerate(sand.depths):
        if depth == "0_5":
            continue
        block = arr[di, y0 : y0 + sand.tile_px, x0 : x0 + sand.tile_px]
        assert not np.isnan(block).any(), f"{depth} was placed by neither path"
