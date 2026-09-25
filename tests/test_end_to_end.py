"""init -> materialize -> overviews -> validate, on a shrunken grid, no network.

The shape of the real pipeline at toy scale: a synthetic source tree in memory,
both subtilings, a ragged tile, all-ocean cells that must stay unwritten, and a
resumed run that must be idempotent.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from soilgrids import config, grid, materialize, overviews, store, template, tiff, validate

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


def _filled(repo, spec, cells, factors):
    """init + materialize `cells`, committed; returns a fresh writable session."""
    session = repo.writable_session("main")
    template.init_store(session, [spec], factors=factors)
    session.commit("init")
    rows = _rows(spec, cells)
    session = repo.writable_session("main")
    session, _ = materialize.materialize_property(session, FakeSource(rows), spec, _manifest(rows), progress=False)
    materialize.mark_complete(session, spec.group, spec.name)
    session.commit("materialize")
    return repo.writable_session("main")


def test_only_cells_leaves_the_other_cells_untouched(small_grid, repo):
    """A subset run must coarsen the subset and nothing else: the point of
    --cells is a trial run that does not silently write half the grid."""
    sand = config.PROPERTIES["sand"]
    session = _filled(repo, sand, [(0, 0), (1, 1)], (2,))
    session, stats = overviews.build_property(session, sand, factors=(2,), progress=False, only_cells={"0-0"})
    session.commit("overviews 0-0")

    n = grid.cell_px() // 2
    gp = zarr.open_group(repo.readonly_session("main").store, path="soil_properties", mode="r")
    assert not np.isnan(gp["2x"]["sand"][0, :n, :n]).all(), "the requested cell must be filled"
    assert np.isnan(gp["2x"]["sand"][0, n:, n:]).all(), "a cell outside --cells must stay untouched"
    assert stats.total_regions == sum(level.regions for level in stats.levels)


def test_checkpointing_matches_one_shot_and_clears_its_token(small_grid, repo):
    sand = config.PROPERTIES["sand"]
    session = _filled(repo, sand, [(0, 0), (1, 1)], (2, 4))
    commits = []

    def checkpoint(live, stats):
        commits.append(live.commit(f"ckpt {stats.regions_done}"))
        return repo.writable_session("main")

    session, stats = overviews.build_property(
        session,
        sand,
        factors=(2, 4),
        progress=False,
        only_cells={"0-0", "1-1"},
        commit_every=4,
        checkpoint=checkpoint,
    )
    assert len(commits) > 1, "commit_every=4 must produce several checkpoints"
    assert stats.regions_done == stats.total_regions
    # the token is a resume aid, not a record: a finished property must not keep one
    assert overviews.progress_state(session, sand, (2, 4)) is None
    session.commit("tail")

    gp = zarr.open_group(repo.readonly_session("main").store, path="soil_properties", mode="r")
    n = grid.cell_px()
    native = gp["sand"][0, :n, :n]
    np.testing.assert_allclose(
        gp["2x"]["sand"][0, : n // 2, : n // 2], validate._fill_aware_mean(native), rtol=1e-6, equal_nan=True
    )


def test_a_killed_run_resumes_from_its_token(small_grid, repo):
    """The kill is simulated the way a real one lands: after a checkpoint has
    committed, so the token on disk describes data that is actually there."""
    sand = config.PROPERTIES["sand"]
    session = _filled(repo, sand, [(0, 0), (1, 1)], (2, 4))

    calls = []

    def dying_checkpoint(live, stats):
        live.commit(f"ckpt {stats.regions_done}")
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt("killed")
        return repo.writable_session("main")

    with pytest.raises(KeyboardInterrupt):
        overviews.build_property(
            session,
            sand,
            factors=(2, 4),
            progress=False,
            only_cells={"0-0", "1-1"},
            commit_every=4,
            checkpoint=dying_checkpoint,
        )

    token = overviews.progress_state(repo.readonly_session("main"), sand, (2, 4))
    assert token is not None and token["done"] > 0

    session = repo.writable_session("main")
    session, stats = overviews.build_property(session, sand, factors=(2, 4), progress=False, only_cells={"0-0", "1-1"})
    skipped = sum(level.skipped for level in stats.levels)
    assert skipped == 8, "the resumed run must skip exactly the regions the token accounted for"
    assert stats.regions_done + skipped == stats.total_regions
    session.commit("resumed")

    gp = zarr.open_group(repo.readonly_session("main").store, path="soil_properties", mode="r")
    n = grid.cell_px()
    for y0, x0 in ((0, 0), (n, n)):
        native = gp["sand"][0, y0 : y0 + n, x0 : x0 + n]
        got = gp["2x"]["sand"][0, y0 // 2 : (y0 + n) // 2, x0 // 2 : (x0 + n) // 2]
        np.testing.assert_allclose(got, validate._fill_aware_mean(native), rtol=1e-6, equal_nan=True)


def test_a_token_from_a_different_scope_is_ignored(small_grid, repo):
    """Resuming against a different --cells set would skip regions that were
    never built, so the fingerprint has to invalidate the token."""
    sand = config.PROPERTIES["sand"]
    session = _filled(repo, sand, [(0, 0), (1, 1)], (2,))
    overviews.record_progress(session, sand, (2,), key="not-this-scope", factor=2, done=6, total=12)
    session.commit("stale token")

    session = repo.writable_session("main")
    session, stats = overviews.build_property(session, sand, factors=(2,), progress=False, only_cells={"0-0", "1-1"})
    assert sum(level.skipped for level in stats.levels) == 0
    assert stats.regions_done == stats.total_regions


def test_a_concurrent_writer_does_not_kill_a_checkpointed_run(small_grid, repo):
    """Phase 3 on one property while phase 4 runs on another.

    icechunk's branch commit is optimistic, so ours fails the moment anything
    else lands -- even though the two touched different arrays. This is the run
    that died at its first checkpoint against the live store. The rebase only
    works because the resume token sits on the level array: on the group, it
    would collide with materialize's cell ledger as an unsolvable
    ZarrMetadataDoubleUpdate.
    """
    sand = config.PROPERTIES["sand"]
    session = _filled(repo, sand, [(0, 0), (1, 1)], (2, 4))

    def competing_checkpoint(live, stats):
        other = repo.writable_session("main")
        gp = zarr.open_group(other.store, path=sand.group, mode="r+")
        ledger = dict(gp.attrs.get(materialize.DONE_ATTR, {}))
        ledger["silt"] = sorted({*ledger.get("silt", []), f"9-{stats.regions_done}"})
        gp.attrs[materialize.DONE_ATTR] = ledger
        other.commit("materialize silt: checkpoint")
        store.commit_with_rebase(live, f"overviews ckpt {stats.regions_done}")
        return repo.writable_session("main")

    session, stats = overviews.build_property(
        session,
        sand,
        factors=(2, 4),
        progress=False,
        only_cells={"0-0", "1-1"},
        commit_every=4,
        checkpoint=competing_checkpoint,
    )
    store.commit_with_rebase(session, "overviews tail")

    gp = zarr.open_group(repo.readonly_session("main").store, path="soil_properties", mode="r")
    assert stats.regions_done == stats.total_regions
    # neither writer clobbered the other
    assert len(gp.attrs[materialize.DONE_ATTR]["silt"]) > 1, "the competing writer's ledger must survive"
    assert set(gp.attrs[materialize.DONE_ATTR]["sand"]) == {"0-0", "1-1"}
    n = grid.cell_px()
    np.testing.assert_allclose(
        gp["2x"]["sand"][0, : n // 2, : n // 2],
        validate._fill_aware_mean(gp["sand"][0, :n, :n]),
        rtol=1e-6,
        equal_nan=True,
    )


def test_a_plain_commit_is_what_fails_under_a_concurrent_writer(small_grid, repo):
    """Guards the claim above: without the rebase this is a hard failure."""
    import icechunk

    sand = config.PROPERTIES["sand"]
    _filled(repo, sand, [(0, 0)], (2,))
    ours = repo.writable_session("main")
    zarr.open_group(ours.store, path=sand.group, mode="r+")["2x"]["sand"][0, :10, :10] = 1.0
    theirs = repo.writable_session("main")
    zarr.open_group(theirs.store, path=sand.group, mode="r+").attrs["unrelated"] = 1
    theirs.commit("someone else")
    with pytest.raises(icechunk.ConflictError):
        ours.commit("ours")
    assert store.commit_with_rebase(ours, "ours, rebased")


class CountingSource(FakeSource):
    """FakeSource that records every fetch, to prove the resume path makes none."""

    def __init__(self, rows):
        super().__init__(rows)
        self.gets = 0

    def get(self, url, *, expect_etag=None, expect_size=None):
        self.gets += 1
        return super().get(url, expect_etag=expect_etag, expect_size=expect_size)


def _fill_and_flag(repo, spec, cells, *, mark: bool):
    """Materialize `cells` with every row flagged `resampled_in_vrt`, so a re-run
    that reaches `resolve_coarse` has to fetch all of them."""
    rows = _rows(spec, cells)
    for r in rows:
        r["resampled_in_vrt"] = True
    manifest = _manifest(rows)
    session = repo.writable_session("main")
    template.init_store(session, [spec], factors=())
    session.commit("init")
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, CountingSource(rows), spec, manifest, progress=False)
    assert stats.cells_done == len(cells)
    if mark:
        materialize.mark_complete(session, spec.group, spec.name)
    session.commit("materialize")
    return manifest, rows


def test_a_finished_property_re_runs_without_touching_the_source(small_grid, repo):
    """`resolve_coarse` runs before the resume filter, so re-running a finished
    property used to GET every tile the VRT resampled -- 194 across the eleven
    properties on the real manifest -- only to log "nothing to do"."""
    sand = config.PROPERTIES["sand"]
    manifest, rows = _fill_and_flag(repo, sand, [(0, 0), (1, 1)], mark=True)

    src = CountingSource(rows)
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, manifest, progress=False)
    assert stats.cells_done == 0
    assert src.gets == 0, "a complete property must not fetch anything to decide it is complete"
    # a subset of a complete property is complete too
    session, stats = materialize.materialize_property(session, src, sand, manifest, cells={(0, 0)}, progress=False)
    assert (stats.cells_done, src.gets) == (0, 0)


def test_an_unfinished_property_still_resolves_its_coarse_sources(small_grid, repo):
    """The guard is the completion attr, not the cell ledger alone: an expanded
    coarse window can reach a cell with no manifest rows, and only a full-scope
    run that finished proves such a cell was already scheduled."""
    sand = config.PROPERTIES["sand"]
    manifest, rows = _fill_and_flag(repo, sand, [(0, 0), (1, 1)], mark=False)

    src = CountingSource(rows)
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, manifest, progress=False)
    assert stats.cells_done == 0  # every cell is in the ledger, so still no work
    assert src.gets == len(rows), "without the completion attr the candidates must still be resolved"


def test_overwrite_ignores_the_completion_shortcut(small_grid, repo):
    sand = config.PROPERTIES["sand"]
    manifest, rows = _fill_and_flag(repo, sand, [(0, 0)], mark=True)

    src = CountingSource(rows)
    session = repo.writable_session("main")
    session, stats = materialize.materialize_property(session, src, sand, manifest, overwrite=True, progress=False)
    assert stats.cells_done == 1 and src.gets > 0


def test_overviews_skips_a_built_property_unless_asked_to_rebuild(small_grid, repo, tmp_path, monkeypatch):
    """A re-run after `bdod` finished used to start its pyramid again from region
    0: the resume token is cleared on completion, and nothing checked the mark."""
    from soilgrids import cli

    sand = config.PROPERTIES["sand"]
    _fill_and_flag(repo, sand, [(0, 0)], mark=True)
    session = repo.writable_session("main")
    overviews.mark_built(session, sand.group, sand.name)
    session.commit("overviews sand")

    calls = []

    def fake_build(session, spec, **kw):
        calls.append((spec.name, kw.get("only_cells")))
        return session, overviews.OverviewStats(total_regions=0)

    monkeypatch.setattr(overviews, "build_property", fake_build)
    uri = str(tmp_path / "store.icechunk")

    cli.build_overviews(store_uri=uri, properties="sand")
    assert calls == []
    cli.build_overviews(store_uri=uri, properties="sand", rebuild=True)
    assert calls == [("sand", {"0-0"})]
    # an explicit --cells subset is a request to build those cells, built or not
    cli.build_overviews(store_uri=uri, properties="sand", cells="0-0")
    assert calls[-1] == ("sand", {"0-0"}) and len(calls) == 2
