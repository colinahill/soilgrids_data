"""init -> materialize -> overviews -> validate, on a shrunken grid, no network.

The shape of the real pipeline at toy scale: a synthetic source tree in memory,
both subtilings, a ragged tile, all-ocean cells that must stay unwritten, and a
resumed run that must be idempotent.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from soilgrids import config, grid, materialize, overviews, template, tiff, validate

from .conftest import NODATA, build_tile


@pytest.fixture
def repo(tmp_path):
    import icechunk

    return icechunk.Repository.create(icechunk.local_filesystem_storage(str(tmp_path / "store.icechunk")))


def _synthetic_field(h: int, w: int, y0: int = 0, x0: int = 0) -> np.ndarray:
    """A deterministic field, so the same pixel always has the same value."""
    yy, xx = np.mgrid[y0 : y0 + h, x0 : x0 + w]
    return (200 + (yy * 7 + xx * 3) % 500).astype("int16")


class FakeSource:
    """Serves synthetic tiles for the manifest rows, like the real WebDAV tree."""

    def __init__(self, rows, *, ocean_cells=frozenset(), ragged=None):
        self.by_url = {}
        self.workers = 4
        g = config.GRID
        for r in rows:
            if (r["cell_row"], r["cell_col"]) in ocean_cells:
                values = np.full((r["height"], r["width"]), NODATA, "int16")
            else:
                values = _synthetic_field(r["height"], r["width"], r["y_off"], r["x_off"])
            tp = (g.x_min + r["x_off"] * g.pixel_size, g.y_max - r["y_off"] * g.pixel_size)
            self.by_url[r["url"]] = build_tile(values, tp, rows_per_strip=r["rows_per_strip"])

    def get(self, url, *, expect_etag=None, expect_size=None):
        return self.by_url[url]

    def map(self, fn, items, *, workers=None):
        return [fn(i) for i in items]


def _rows(spec, cells):
    """Manifest rows for whole cells: every subtile present and full."""
    out = []
    for row, col in cells:
        for r, c in grid.subtiles(spec.tile_px):
            ref = grid.TileRef(row, col, r, c, spec.tile_px)
            x_off, y_off = ref.pixel_offset
            for depth in spec.depths:
                out.append(
                    {
                        "property": spec.name,
                        "depth": depth,
                        "cell": ref.cell,
                        "subtile": f"{r}-{c}",
                        "url": f"fake://{spec.name}/{depth}/{ref.relpath}",
                        "relpath": ref.relpath,
                        "width": spec.tile_px,
                        "height": spec.tile_px,
                        "x_off": x_off,
                        "y_off": y_off,
                        "cell_row": row,
                        "cell_col": col,
                        "is_full": True,
                        "rows_per_strip": spec.rows_per_strip,
                        "resampled_in_vrt": False,
                        "size": None,
                        "etag": None,
                        "modified": None,
                    }
                )
    return out


def _manifest(rows):
    import polars as pl

    from soilgrids import catalog

    return pl.DataFrame(rows, schema=catalog._manifest_schema())


def test_full_cycle_with_both_subtilings(small_grid, repo):
    sand = config.PROPERTIES["sand"]  # 45 px tiles, 2x2 per cell
    phh2o = config.PROPERTIES["phh2o"]  # 30 px tiles, 3x3 per cell
    cells = [(0, 0), (0, 1)]
    rows = _rows(sand, cells) + _rows(phh2o, cells)
    manifest = _manifest(rows)
    src = FakeSource(rows)

    session = repo.writable_session("main")
    created = template.init_store(session, [sand, phh2o], factors=(2, 4))
    assert len(created) == 2 * (1 + 2)
    session.commit("init")

    for spec in (sand, phh2o):
        session = repo.writable_session("main")
        session, stats = materialize.materialize_property(session, src, spec, manifest, progress=False)
        assert stats.cells_done == 2
        assert stats.tiles_missing == 0
        materialize.mark_complete(session, spec.group, spec.name)
        session.commit(f"materialize {spec.name}")

        session = repo.writable_session("main")
        cellset = materialize.done_cells(session, spec.group, spec.name)
        assert cellset == {"0-0", "0-1"}
        overviews.build_property(session, spec, factors=(2, 4), progress=False, only_cells=cellset)
        overviews.mark_built(session, spec.group, spec.name)
        session.commit(f"overviews {spec.name}")

    # both properties must agree pixel-for-pixel with the synthetic source
    ro = repo.readonly_session("main")
    gp = zarr.open_group(ro.store, path="soil_properties", mode="r")
    for spec in (sand, phh2o):
        arr = gp[spec.name]
        for row in [r for r in rows if r["property"] == spec.name][:6]:
            header, want = tiff.decode(src.by_url[row["url"]])
            di = list(spec.depths).index(row["depth"])
            got = arr[di, row["y_off"] : row["y_off"] + header.height, row["x_off"] : row["x_off"] + header.width]
            # the store holds decoded physical units; re-encoding must recover the
            # source integers EXACTLY, which is what keeps validate an exact check
            np.testing.assert_array_equal(materialize.encode_values(got, spec), want)
            np.testing.assert_allclose(got, want / spec.conversion_factor, rtol=1e-6)

    # structure must pass its own checks
    results = validate.check_structure(ro, [sand, phh2o], factors=(2, 4))
    _passed, failed = validate.summarise(results)
    assert failed == 0, [r for r in results if not r.passed]


def test_overview_matches_an_independent_mean(small_grid, repo):
    sand = config.PROPERTIES["sand"]
    rows = _rows(sand, [(0, 0)])
    manifest = _manifest(rows)
    src = FakeSource(rows)
    session = repo.writable_session("main")
    template.init_store(session, [sand], factors=(2,))
    session.commit("init")
    session = repo.writable_session("main")
    session, _ = materialize.materialize_property(session, src, sand, manifest, progress=False)
    overviews.build_property(session, sand, factors=(2,), progress=False, only_cells={"0-0"})
    session.commit("fill")

    ro = repo.readonly_session("main")
    gp = zarr.open_group(ro.store, path="soil_properties", mode="r")
    n = grid.cell_px()
    native = gp["sand"][0, :n, :n]
    got = gp["2x"]["sand"][0, : n // 2, : n // 2]
    np.testing.assert_allclose(got, validate._fill_aware_mean(native), rtol=1e-6, equal_nan=True)


def test_ocean_cells_are_never_written(small_grid, repo):
    sand = config.PROPERTIES["sand"]
    rows = _rows(sand, [(0, 0), (1, 1)])
    manifest = _manifest(rows)
    src = FakeSource(rows, ocean_cells={(1, 1)})
    session = repo.writable_session("main")
    template.init_store(session, [sand], factors=())
    session.commit("init")
    session = repo.writable_session("main")
    arr_before = zarr.open_group(session.store, path="soil_properties", mode="r")["sand"].nbytes_stored()
    session, stats = materialize.materialize_property(session, src, sand, manifest, progress=False)
    assert stats.cells_done == 2
    n_shards = len(grid.shard_windows(0, 0))
    assert stats.shards_written == n_shards, "only the land cell's shards"
    assert stats.shards_empty == n_shards, "the ocean cell's shards are all skipped"
    session.commit("fill")
    ro = repo.readonly_session("main")
    arr = zarr.open_group(ro.store, path="soil_properties", mode="r")["sand"]
    n = grid.cell_px()
    assert np.all(np.isnan(arr[0, n : 2 * n, n : 2 * n]))
    assert arr.nbytes_stored() > arr_before


def test_rerun_is_idempotent_and_resumes(small_grid, repo):
    sand = config.PROPERTIES["sand"]
    rows = _rows(sand, [(0, 0), (0, 1)])
    manifest = _manifest(rows)
    src = FakeSource(rows)
    session = repo.writable_session("main")
    template.init_store(session, [sand], factors=())
    session.commit("init")

    # first run: only one cell
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, manifest, cells={(0, 0)}, progress=False)
    assert stats.cells_done == 1
    session.commit("partial")

    # second run over everything must do only the cell that is left
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, manifest, progress=False)
    assert stats.cells_done == 1, "the finished cell must be skipped"
    session.commit("rest")
    assert materialize.done_cells(repo.readonly_session("main"), sand.group, sand.name) == {"0-0", "0-1"}

    # a third run must do nothing at all
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, manifest, progress=False)
    assert stats.cells_done == 0


def test_init_store_is_additive_and_refuses_an_encoding_change(small_grid, repo, monkeypatch):
    sand = config.PROPERTIES["sand"]
    silt = config.PROPERTIES["sand"].model_copy(update={"name": "silt"})
    session = repo.writable_session("main")
    assert len(template.init_store(session, [sand], factors=())) == 1
    session.commit("one")

    session = repo.writable_session("main")
    assert template.init_store(session, [sand], factors=()) == [], "already present"
    created = template.init_store(session, [sand, silt], factors=())
    assert created == ["soil_properties/silt"], "additive: only the new array"
    session.commit("two")

    # 15 divides both subtilings, so validate_consistency passes and the
    # EncodingMismatch check is what fires
    monkeypatch.setattr(config, "ENCODING", config.EncodingSpec(chunk_y=15, chunk_x=15, shard_y=45, shard_x=45))
    session = repo.writable_session("main")
    with pytest.raises(template.EncodingMismatch, match="Shard-aligned writes"):
        template.init_store(session, [sand], factors=())


def test_consumer_reads_conventional_units_with_nan_for_nodata(small_grid, repo):
    """xarray must decode to physical units and NaN without any consumer effort."""
    import xarray as xr

    sand = config.PROPERTIES["sand"]
    rows = _rows(sand, [(0, 0)])
    src = FakeSource(rows)
    session = repo.writable_session("main")
    template.init_store(session, [sand], factors=())
    session.commit("init")
    session = repo.writable_session("main")
    session, _ = materialize.materialize_property(session, src, sand, _manifest(rows), progress=False)
    session.commit("fill")

    st = repo.readonly_session("main").store
    ds = xr.open_zarr(st, group="soil_properties", consolidated=False)

    assert ds.sand.dtype == np.dtype("float32"), "no float64 upcast on read"
    assert ds.sand.attrs["units"] == sand.conventional_units

    # the stored value is the source integer divided by the conversion factor,
    # and re-encoding recovers the source integer exactly
    src_int = tiff.decode(src.by_url[rows[0]["url"]])[1]
    y0, x0 = rows[0]["y_off"], rows[0]["x_off"]
    stored = ds.sand[0, y0 : y0 + src_int.shape[0], x0 : x0 + src_int.shape[1]].values
    np.testing.assert_allclose(stored, src_int / sand.conversion_factor, rtol=1e-6)
    np.testing.assert_array_equal(materialize.encode_values(stored, sand), src_int)

    # an unfilled cell reads as NaN, never as a sentinel a consumer could mistake
    n = grid.cell_px()
    assert np.isnan(ds.sand[0, n, n].values)
