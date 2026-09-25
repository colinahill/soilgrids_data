"""The soilgrids CLI: eight ordered, idempotent phases.

Every phase is safe to re-run: it skips work that is already done and current.
Store-writing phases default to a local dev store, so a bare `make materialize`
can never touch the published product; pass --source-coop-account to publish.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from . import catalog, config, materialize, metadata, overviews, remote, store, template, validate
from .fetch import Fetcher, SourceChanged

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
log = logging.getLogger("soilgrids")

WorkDir = Annotated[Path, typer.Option("--work-dir", help="Where manifests and reports live.")]
StoreOpt = Annotated[str | None, typer.Option("--store", help="Local path or s3://bucket/prefix.")]
AccountOpt = Annotated[str | None, typer.Option("--source-coop-account", help="Publish to this Source Coop account.")]
CredsOpt = Annotated[str | None, typer.Option("--credentials-file", help="Source Coop JSON credential export.")]
PropsOpt = Annotated[str | None, typer.Option("--properties", help="Comma-separated subset, default all.")]
WorkersOpt = Annotated[int, typer.Option("--workers", help="Concurrent requests (measured: >8 buys nothing).")]


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_storage(store_uri: str | None, account: str | None, credentials_file: str | None):
    if account:
        return store.source_coop_storage(account, credentials_file=credentials_file)
    return store.storage_from_uri(store_uri or "./soilgrids_store_local", credentials_file=credentials_file)


def _parse_properties(properties: str | None) -> list[config.PropertySpec]:
    if not properties:
        return config.included_properties()
    out = []
    for name in (n.strip() for n in properties.split(",") if n.strip()):
        if name not in config.PROPERTIES:
            raise typer.BadParameter(f"unknown property {name!r}; known: {', '.join(sorted(config.PROPERTIES))}")
        out.append(config.PROPERTIES[name])
    return out


def _die(message: str) -> None:
    log.error("%s", message)
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# 1. inspect-source
# ---------------------------------------------------------------------------


@app.command()
def inspect_source(
    work_dir: WorkDir = Path("work"),
    properties: PropsOpt = None,
    workers: WorkersOpt = 8,
    listing_samples: Annotated[
        int,
        typer.Option(
            "--listing-samples",
            help="Cell listings cross-checked per layer; 0 = every cell (needed for a real build).",
        ),
    ] = 20,
    etag_samples: Annotated[int, typer.Option("--etag-samples", help="Tiles HEADed per layer for ETag/size.")] = 8,
    rate_limit: Annotated[float | None, typer.Option("--rate-limit", help="Max requests/second.")] = None,
    verbose: bool = False,
) -> None:
    """Phase 1: verify the source tree and write the tile manifest.

    Runs in explicit parallel passes rather than layer by layer: ISRIC throttles
    per connection at roughly 40 kB/s, so a 6 MB VRT takes ~2.5 minutes on its
    own and 61 of them sequentially would take over two hours. Concurrency is the
    only lever the server leaves.
    """
    _setup_logging(verbose)
    config.validate_consistency()
    specs = _parse_properties(properties)
    layers = [(s, d) for s in specs for d in s.depths]
    log.info(
        "inspecting %d layers of SoilGrids v%s with %d concurrent requests",
        len(layers),
        config.SOURCE_VERSION,
        workers,
    )

    report: dict = {
        "source_version": config.SOURCE_VERSION,
        "layers": {},
        "anomalies": [],
        "listing_issues": [],
        "recovered": [],
        "conflicts": [],
    }
    with Fetcher(workers=workers, rate_limit=rate_limit) as fetcher:
        # pass 1: the VRTs, all at once (the only slow part)
        log.info("pass 1/3: downloading %d layer VRTs (~6 MB each)", len(layers))
        try:
            xmls = fetcher.map(lambda ld: catalog.fetch_vrt(fetcher, *ld), layers)
        except Exception as exc:
            _die(f"VRT download failed: {exc}")

        # pass 2: parse and cross-check the lattice (local, fast)
        inventories: list[catalog.LayerInventory] = []
        rows_by_layer: list[list[dict]] = []
        conflicts_by_layer: dict[int, list[catalog.Conflict]] = {}
        for i, ((spec, depth), xml) in enumerate(zip(layers, xmls, strict=True)):
            key = f"{spec.name}/{depth}"
            try:
                inv = catalog.inventory_layer(None, spec, depth, xml=xml)
                layer_rows = catalog.tile_rows(inv, spec)
            except (catalog.SourceError, SourceChanged, ValueError) as exc:
                _die(f"{key}: {exc}")
            inventories.append(inv)
            rows_by_layer.append(layer_rows)
            report["layers"][key] = {
                "vrt_url": inv.vrt_url,
                "tiles": len(layer_rows),
                "full_tiles": sum(r["is_full"] for r in layer_rows),
                "ragged_tiles": sum(not r["is_full"] for r in layer_rows),
                "cells": len({r["cell"] for r in layer_rows}),
                "vrt_extent": [inv.vrt_width, inv.vrt_height],
                "provenance": {
                    k: inv.metadata.get(k) for k in (*config.EXPECTED_VRT_METADATA, "Outputs_version", "Transformation")
                },
            }
            report["anomalies"].extend(f"{key}: {a}" for a in inv.anomalies)
            if inv.conflicts:
                conflicts_by_layer[i] = inv.conflicts
            log.info(
                "%s: %d tiles (%d full, %d ragged) in %d cells",
                key,
                len(layer_rows),
                report["layers"][key]["full_tiles"],
                report["layers"][key]["ragged_tiles"],
                report["layers"][key]["cells"],
            )

        # two tiles claiming one window: verify the dropped one was interchangeable
        if conflicts_by_layer:
            total = sum(len(v) for v in conflicts_by_layer.values())
            log.info("verifying %d duplicate-window conflict(s)", total)
            for i, conflicts in sorted(conflicts_by_layer.items()):
                spec, depth = layers[i]
                try:
                    notes = catalog.resolve_conflicts(fetcher, spec, depth, conflicts)
                except catalog.SourceError as exc:
                    _die(str(exc))
                report["conflicts"].extend(f"{spec.name}/{depth}: {n}" for n in notes)
                log.info("%s/%s: %d duplicate window(s) resolved", spec.name, depth, len(notes))

        # pass 3: sampled directory listings and ETags, all at once
        listing_jobs = [
            (i, cell, url)
            for i, ((spec, depth), rows) in enumerate(zip(layers, rows_by_layer, strict=True))
            for cell, url in catalog.listing_targets(spec, depth, rows, samples=listing_samples or 10**9)
        ]
        etag_jobs = [
            (i, row) for i, rows in enumerate(rows_by_layer) for row in catalog.etag_targets(rows, samples=etag_samples)
        ]
        log.info("pass 3/3: %d cell listings and %d tile HEADs", len(listing_jobs), len(etag_jobs))

        def do_listing(job):
            i, cell, url = job
            return i, cell, catalog.check_listing(rows_by_layer[i], cell, fetcher.get_text(url))

        disk_only: dict[int, dict[str, list[str]]] = {}
        for i, cell, (extra_names, vrt_only) in fetcher.map(do_listing, listing_jobs):
            spec, depth = layers[i]
            key = f"{spec.name}/{depth}"
            if extra_names:
                disk_only.setdefault(i, {})[cell] = extra_names
            if vrt_only:
                report["listing_issues"].append(f"{key}: {cell}: in the VRT but not on disk: {vrt_only}")

        # Tiles on disk that the VRT omits are real: measured 10 per bdod layer,
        # 8 of them holding ~26,000 km2 that nothing else covers. Put back the
        # ones that can be placed.
        if disk_only:
            log.info(
                "recovering tiles the VRTs omit: %d candidates across %d layers",
                sum(len(v) for d in disk_only.values() for v in d.values()),
                len(disk_only),
            )
            for i, cells in sorted(disk_only.items()):
                spec, depth = layers[i]
                extra, notes = catalog.repair_from_disk(fetcher, spec, depth, rows_by_layer[i], cells)
                rows_by_layer[i].extend(extra)
                report["recovered"].extend(f"{spec.name}/{depth}: {n}" for n in notes)
                if extra:
                    log.info("%s/%s: recovered %d tile(s) absent from the VRT", spec.name, depth, len(extra))

        def do_head(job):
            _i, row = job
            catalog.record_etag(row, fetcher.head(row["url"]))

        fetcher.map(do_head, etag_jobs)

    rows = [r for layer_rows in rows_by_layer for r in layer_rows]
    report["total_tiles"] = len(rows)
    report["grid"] = metadata.region_note()
    manifest, report_path = catalog.write_manifest(rows, report, work_dir)
    log.info("wrote %s (%d rows) and %s", manifest, len(rows), report_path)
    if report["listing_issues"]:
        for m in report["listing_issues"][:10]:
            log.error("%s", m)
        _die(
            f"{len(report['listing_issues'])} listing cross-check issues; see {report_path}. "
            f"A tile on disk but absent from a VRT would be dropped from the product."
        )
    if report["conflicts"]:
        log.info("%d duplicate window(s) resolved; see %s", len(report["conflicts"]), report_path)
    if report["recovered"]:
        log.info("%d tiles recovered from disk; see %s", len(report["recovered"]), report_path)
    if report["anomalies"]:
        log.warning("%d anomalies recorded in %s (non-fatal)", len(report["anomalies"]), report_path)
    if listing_samples:
        log.warning(
            "listings were SAMPLED (%d cells/layer): tiles the VRT omits can still be missing. "
            "Use --listing-samples 0 for a real build.",
            listing_samples,
        )
    log.info("source verified: %d tiles across %d layers", len(rows), len(layers))


# ---------------------------------------------------------------------------
# 2. init-store
# ---------------------------------------------------------------------------


@app.command()
def init_store(
    store_uri: StoreOpt = None,
    account: AccountOpt = None,
    credentials_file: CredsOpt = None,
    properties: PropsOpt = None,
    no_overviews: Annotated[bool, typer.Option("--no-overviews", help="Skip the pyramid groups.")] = False,
    verbose: bool = False,
) -> None:
    """Phase 2: create (or additively extend) the store structure. Idempotent."""
    _setup_logging(verbose)
    storage = _resolve_storage(store_uri, account, credentials_file)
    repo = store.open_repo(storage, create=True)
    session = repo.writable_session("main")
    try:
        created = template.init_store(session, _parse_properties(properties), factors=() if no_overviews else None)
    except template.EncodingMismatch as exc:
        _die(str(exc))
    if not created:
        log.info("structure is already complete; nothing to create")
        return
    snapshot = session.commit(f"init structure ({len(created)} arrays) for SoilGrids v{config.SOURCE_VERSION}")
    log.info("created %d arrays; snapshot %s", len(created), snapshot)


# ---------------------------------------------------------------------------
# 3. materialize
# ---------------------------------------------------------------------------


@app.command(name="materialize")
def materialize_cmd(
    store_uri: StoreOpt = None,
    account: AccountOpt = None,
    credentials_file: CredsOpt = None,
    properties: PropsOpt = None,
    work_dir: WorkDir = Path("work"),
    bbox: Annotated[str | None, typer.Option("--bbox", help="min_lon,min_lat,max_lon,max_lat subset.")] = None,
    cells: Annotated[str | None, typer.Option("--cells", help="Explicit 'row-col,row-col' cell list.")] = None,
    workers: WorkersOpt = 8,
    commit_every: Annotated[int, typer.Option("--commit-every", help="Checkpoint-commit every N cells.")] = 16,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Refill cells already recorded as done.")] = False,
    rate_limit: Annotated[float | None, typer.Option("--rate-limit", help="Max requests/second.")] = None,
    verbose: bool = False,
) -> None:
    """Phase 3: fill the native arrays, one (property, cell) at a time. Resumable."""
    _setup_logging(verbose)
    manifest = catalog.read_manifest(work_dir)
    specs = _parse_properties(properties)
    subset: set[tuple[int, int]] | None = None
    if bbox:
        try:
            subset = materialize.cells_in_bbox(tuple(float(v) for v in bbox.split(",")))  # type: ignore[arg-type]
        except (ValueError, TypeError):
            raise typer.BadParameter("--bbox wants min_lon,min_lat,max_lon,max_lat") from None
        log.info("bbox %s -> %d cells", bbox, len(subset))
    if cells:
        explicit = {tuple(int(v) for v in c.split("-")) for c in cells.split(",") if c.strip()}
        subset = explicit if subset is None else subset & explicit  # type: ignore[assignment]

    storage = _resolve_storage(store_uri, account, credentials_file)
    repo = store.open_repo(storage)
    with Fetcher(workers=workers, rate_limit=rate_limit) as fetcher:
        for spec in specs:
            session = repo.writable_session("main")

            def checkpoint(live_session, stats, _spec=spec):
                # rebasing, because phase 4 may be coarsening another property
                # onto the same branch while this backfill runs
                snap = store.commit_with_rebase(
                    live_session, f"materialize {_spec.name}: {stats.cells_done} cells, {stats.shards_written} shards"
                )
                log.info("checkpoint %s (%d cells)", snap, stats.cells_done)
                return repo.writable_session("main")

            try:
                session, stats = materialize.materialize_property(
                    session,
                    fetcher,
                    spec,
                    manifest,
                    cells=subset,
                    overwrite=overwrite,
                    commit_every=commit_every,
                    checkpoint=checkpoint,
                )
            except materialize.CredentialsExpired as exc:
                _die(str(exc))
            except (SourceChanged, materialize.PlacementMismatch) as exc:
                _die(f"{spec.name}: {exc}")
            if stats.cells_done == 0:
                continue
            if subset is None:
                materialize.mark_complete(session, spec.group, spec.name)
            # the tail can be empty when the cell count is an exact multiple of
            # commit_every: the last checkpoint already committed everything and
            # handed back a fresh session
            if not session.has_uncommitted_changes:
                log.info(
                    "%s: %d cells, %d shards written, %d empty, %.2f GB fetched; all committed at the last checkpoint",
                    spec.name,
                    stats.cells_done,
                    stats.shards_written,
                    stats.shards_empty,
                    stats.bytes_fetched / 1e9,
                )
                for a in stats.anomalies[:10]:
                    log.warning("%s", a)
                continue
            snapshot = store.commit_with_rebase(
                session,
                f"materialize {spec.name}: {stats.cells_done} cells, {stats.shards_written} shards, "
                f"{stats.bytes_fetched / 1e9:.2f} GB fetched",
            )
            log.info(
                "%s: %d cells, %d shards written, %d empty, %d tiles missing, %.2f GB fetched; snapshot %s",
                spec.name,
                stats.cells_done,
                stats.shards_written,
                stats.shards_empty,
                stats.tiles_missing,
                stats.bytes_fetched / 1e9,
                snapshot,
            )
            for a in stats.anomalies[:10]:
                log.warning("%s", a)


# ---------------------------------------------------------------------------
# 4. overviews
# ---------------------------------------------------------------------------


def _mark_overviews_built(repo, spec: config.PropertySpec, tries: int = 5) -> None:
    """Set the completion attr in its own session, after the data is committed.

    It lands on the group that ``materialize`` also writes its cell ledger to, so
    a concurrent phase-3 run can make it unrebaseable. Re-reading and re-applying
    on a fresh session IS the merge: the attr is a set union, so whoever commits
    second includes what the first one wrote. Kept out of the data commit so a
    lost race costs a retry, never the pyramid.
    """
    for attempt in range(1, tries + 1):
        session = repo.writable_session("main")
        overviews.mark_built(session, spec.group, spec.name)
        if not session.has_uncommitted_changes:
            return
        try:
            store.commit_with_rebase(session, f"overviews {spec.name}: mark built")
            return
        except store.ConcurrentWriter:
            if attempt == tries:
                raise
            log.warning("%s: mark-built lost a race with another writer; retrying (%d/%d)", spec.name, attempt, tries)


@app.command(name="overviews")
def build_overviews(
    store_uri: StoreOpt = None,
    account: AccountOpt = None,
    credentials_file: CredsOpt = None,
    properties: PropsOpt = None,
    cells: Annotated[
        str | None, typer.Option("--cells", help="Explicit 'row-col,row-col' subset of the materialized cells.")
    ] = None,
    workers: Annotated[int | None, typer.Option("--workers", help="Region threads; default auto.")] = None,
    commit_every: Annotated[
        int, typer.Option("--commit-every", help="Checkpoint-commit every N regions. 0 = one commit per property.")
    ] = 0,
    progress_every: Annotated[
        int, typer.Option("--progress-every", help="Regions between progress log lines when stderr is not a TTY.")
    ] = 0,
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Rebuild properties already marked built, e.g. after adding cells."),
    ] = False,
    verbose: bool = False,
) -> None:
    """Phase 4: build the multiscale pyramid, one property at a time. Resumable.

    Properties already marked built are skipped unless ``--rebuild`` is given or
    ``--cells`` names an explicit subset.
    """
    _setup_logging(verbose)
    subset: set[str] | None = None
    if cells:
        try:
            subset = {"{}-{}".format(*(int(v) for v in c.strip().split("-"))) for c in cells.split(",") if c.strip()}
        except (ValueError, TypeError):
            raise typer.BadParameter("--cells wants 'row-col,row-col', e.g. 8-20,9-20") from None
    storage = _resolve_storage(store_uri, account, credentials_file)
    repo = store.open_repo(storage)
    for spec in _parse_properties(properties):
        session = repo.writable_session("main")
        if subset is None and not rebuild and spec.name in overviews.built(session, spec.group):
            log.info("%s: overviews already built; skipping (--rebuild to build again)", spec.name)
            continue
        done = materialize.done_cells(session, spec.group, spec.name)
        if not done:
            log.warning("%s: no cells materialized; nothing to coarsen", spec.name)
            continue
        cellset = done if subset is None else done & subset
        if subset is not None:
            # a cell that was never materialized has nothing to coarsen, and
            # silently coarsening a different set than the one asked for would
            # also invalidate the resume token's scope fingerprint
            missing = sorted(subset - done)
            if missing:
                log.warning(
                    "%s: %d of the requested cells are not materialized, skipping them (%s%s)",
                    spec.name,
                    len(missing),
                    ", ".join(missing[:5]),
                    ", ..." if len(missing) > 5 else "",
                )
            if not cellset:
                continue
        log.info("%s: coarsening within %d materialized cells", spec.name, len(cellset))

        def checkpoint(live_session, stats, _spec=spec):
            snap = store.commit_with_rebase(
                live_session,
                f"overviews {_spec.name}: checkpoint at {stats.regions_done} regions, {stats.shards_written} shards",
            )
            log.info("checkpoint %s (%d/%d regions)", snap, stats.regions_done, stats.total_regions)
            return repo.writable_session("main")

        try:
            session, stats = overviews.build_property(
                session,
                spec,
                workers=workers,
                only_cells=cellset,
                commit_every=commit_every,
                checkpoint=checkpoint if commit_every else None,
                progress_every=progress_every,
            )
        except store.ConcurrentWriter as exc:
            _die(f"{spec.name}: {exc}")
        except materialize.CredentialsExpired as exc:
            _die(str(exc))
        if session.has_uncommitted_changes:
            snapshot = store.commit_with_rebase(
                session, f"overviews {spec.name}: {len(stats.levels)} levels, {stats.shards_written} shards"
            )
            log.info(
                "%s: %d levels (%s); snapshot %s",
                spec.name,
                len(stats.levels),
                ", ".join(f"{s.factor}x{s.shape[-2:]}" for s in stats.levels),
                snapshot,
            )
        else:
            log.info(
                "%s: %d levels, %d shards written; all committed at the last checkpoint",
                spec.name,
                len(stats.levels),
                stats.shards_written,
            )
        # a partial run must not claim the property: `release` gates on this attr
        if subset is None:
            _mark_overviews_built(repo, spec)
        else:
            log.info("%s: built over a %d-cell subset; not marking the property complete", spec.name, len(cellset))


# ---------------------------------------------------------------------------
# 5. status
# ---------------------------------------------------------------------------


@app.command()
def status(
    store_uri: StoreOpt = None,
    account: AccountOpt = None,
    credentials_file: CredsOpt = None,
    work_dir: WorkDir = Path("work"),
    verbose: bool = False,
) -> None:
    """Show the property x cell completion matrix."""
    _setup_logging(verbose)

    repo = store.open_repo(_resolve_storage(store_uri, account, credentials_file))
    ro = repo.readonly_session("main")
    try:
        manifest = catalog.read_manifest(work_dir)
        expected = {
            name: len(
                {(r["cell_row"], r["cell_col"]) for r in manifest.filter(manifest["property"] == name).to_dicts()}
            )
            for name in config.PROPERTIES
        }
    except catalog.SourceError:
        expected = {}
        log.warning("no manifest; showing store-side progress only")

    print(f"{'property':10s} {'cells done':>12s} {'expected':>9s} {'native':>8s} {'overviews':>16s}")
    for spec in config.included_properties():
        done = len(materialize.done_cells(ro, spec.group, spec.name))
        exp = expected.get(spec.name, 0)
        complete = spec.name in materialize.complete_properties(ro, spec.group)
        state = "complete" if complete else ("partial" if done else "-")
        # the completion attr is only written when a property finishes, so an
        # in-flight or interrupted run shows as its resume token instead of "-".
        # A token on top of "built" means a rebuild is part-way through, which is
        # worth seeing: the pyramid is a mix of two runs until it finishes.
        built = spec.name in overviews.built(ro, spec.group)
        token = overviews.progress_state(ro, spec)
        if token:
            ovr = f"{'built+' if built else ''}{token['factor']}x {token['done']}/{token['total']}"
        else:
            ovr = "built" if built else "-"
        print(f"{spec.name:10s} {done:12d} {exp if exp else '?':>9} {state:>8s} {ovr:>16s}")


# ---------------------------------------------------------------------------
# 6. validate
# ---------------------------------------------------------------------------


@app.command(name="validate")
def validate_cmd(
    store_uri: StoreOpt = None,
    account: AccountOpt = None,
    credentials_file: CredsOpt = None,
    properties: PropsOpt = None,
    work_dir: WorkDir = Path("work"),
    samples: Annotated[int, typer.Option("--samples", help="Source tiles re-fetched per property.")] = 8,
    structure_only: Annotated[bool, typer.Option("--structure-only", help="Skip network checks.")] = False,
    workers: WorkersOpt = 8,
    verbose: bool = False,
) -> None:
    """Phase 6: verify structure, and by sampling, contents."""
    _setup_logging(verbose)
    repo = store.open_repo(_resolve_storage(store_uri, account, credentials_file))
    ro = repo.readonly_session("main")
    specs = _parse_properties(properties)
    results = validate.check_structure(ro, specs)

    if not structure_only:
        manifest = catalog.read_manifest(work_dir)
        with Fetcher(workers=workers) as fetcher:
            results += validate.check_crs_matches_source(fetcher, manifest)
            filled: dict[str, set[str]] = {}
            for spec in specs:
                cells = materialize.done_cells(ro, spec.group, spec.name)
                filled[spec.name] = cells
                if not cells:
                    log.info("%s: no cells filled yet, skipping content checks", spec.name)
                    continue
                log.info("%s: sampling within %d filled cells", spec.name, len(cells))
                results += validate.check_tiles(ro, fetcher, spec, manifest, samples=samples, only_cells=cells)
                results += validate.check_coarse_fills(ro, fetcher, spec, manifest, only_cells=cells)
                if spec.name in overviews.built(ro, spec.group):
                    results += validate.check_overviews(ro, spec, only_cells=cells)
        texture = {"sand", "silt", "clay"}
        if texture <= {s.name for s in specs}:
            shared = set.intersection(*(filled.get(n, set()) for n in texture))
            if shared:
                results += validate.check_texture_closure(ro, only_cells=shared)
            else:
                log.info("no cell has all of sand/silt/clay filled; skipping the closure check")

    passed, failed = validate.summarise(results)
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# 7. release
# ---------------------------------------------------------------------------


@app.command()
def release(
    store_uri: StoreOpt = None,
    account: AccountOpt = None,
    credentials_file: CredsOpt = None,
    suffix: Annotated[str, typer.Option("--suffix", help="Appended to the tag for a corrected re-release.")] = "",
    verbose: bool = False,
) -> None:
    """Phase 7: tag the release. Refuses while anything is incomplete."""
    _setup_logging(verbose)
    import zarr

    repo = store.open_repo(_resolve_storage(store_uri, account, credentials_file))
    ro = repo.readonly_session("main")
    missing: list[str] = []
    for spec in config.included_properties():
        gp = zarr.open_group(ro.store, path=spec.group, mode="r")
        if spec.name not in set(gp.attrs.get(materialize.COMPLETE_ATTR, [])):
            missing.append(f"{spec.name} (native incomplete)")
        elif spec.name not in overviews.built(ro, spec.group):
            missing.append(f"{spec.name} (overviews not built)")
    if missing:
        _die("refusing to release; incomplete: " + ", ".join(missing))

    tag = f"soilgrids-{config.SOURCE_VERSION}{suffix}"
    snapshot = repo.lookup_branch("main")
    try:
        repo.create_tag(tag, snapshot_id=snapshot)
    except Exception as exc:  # icechunk raises a generic error for a taken name
        _die(f"could not create tag {tag}: {exc}. Tag names are burned once used; pass --suffix -r2.")
    log.info("tagged %s at %s", tag, snapshot)

    tagged = repo.readonly_session(tag=tag)
    results = validate.check_structure(tagged)
    passed, failed = validate.summarise(results)
    print(f"re-read through tag {tag}: {passed} passed, {failed} failed")
    if failed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# 8. housekeeping
# ---------------------------------------------------------------------------


@app.command()
def info(
    store_uri: StoreOpt = None,
    account: AccountOpt = None,
    credentials_file: CredsOpt = None,
) -> None:
    """Show store structure, tags, and recent snapshots."""
    _setup_logging()
    import zarr

    repo = store.open_repo(_resolve_storage(store_uri, account, credentials_file))
    ro = repo.readonly_session("main")
    root = zarr.open_group(ro.store, mode="r")
    print(f"store: {store.STORE_SUBPATH}  (SoilGrids v{config.SOURCE_VERSION})")
    coord_names = {"x", "y", "spatial_ref"}
    for group_name in config.GROUPS:
        gp = root[group_name]
        arrays = [n for n in sorted(gp.array_keys()) if n not in coord_names and not n.startswith("depth")]
        print(f"\n{group_name}/  ({len(arrays)} data arrays, {len(sorted(gp.group_keys()))} overview levels)")
        for name in arrays:
            a = gp[name]
            print(f"  {name:10s} {tuple(a.shape)!s:26s} chunks={tuple(a.chunks)} shards={tuple(a.shards or ())}")
    print("\ntags:", ", ".join(sorted(repo.list_tags())) or "(none)")
    print("recent snapshots:")
    for i, s in enumerate(repo.ancestry(branch="main")):
        if i >= 8:
            break
        print(f"  {s.id}  {s.written_at:%Y-%m-%d %H:%M}  {s.message[:80]}")


@app.command()
def garbage_collect(
    store_uri: StoreOpt = None,
    account: AccountOpt = None,
    credentials_file: CredsOpt = None,
    older_than_hours: Annotated[int, typer.Option("--older-than-hours")] = 1,
) -> None:
    """Reclaim objects orphaned by checkpoint commits. Never while writing."""
    _setup_logging()
    from datetime import UTC, datetime, timedelta

    repo = store.open_repo(_resolve_storage(store_uri, account, credentials_file))
    cutoff = datetime.now(UTC) - timedelta(hours=older_than_hours)
    summary = repo.garbage_collect(cutoff)
    print(f"reclaimed objects older than {cutoff:%Y-%m-%d %H:%M} UTC:\n{summary}")


@app.command()
def publish_readme(
    account: Annotated[str, typer.Option("--source-coop-account")] = store.SOURCE_COOP_ACCOUNT,
    credentials_file: CredsOpt = None,
) -> None:
    """Upload product/README.md as the Source Coop landing page."""
    _setup_logging()
    remote.upload_readme(account, credentials_file=credentials_file)


@app.command()
def upload_audit(
    account: Annotated[str, typer.Option("--source-coop-account")] = store.SOURCE_COOP_ACCOUNT,
    credentials_file: CredsOpt = None,
    work_dir: WorkDir = Path("work"),
) -> None:
    """Upload the source manifest and reports to audit/{version}/."""
    _setup_logging()
    d = Path(work_dir) / config.SOURCE_VERSION
    files = sorted(p for p in d.glob("*") if p.is_file())
    if not files:
        _die(f"nothing to upload under {d}")
    remote.upload_audit(account, files, config.SOURCE_VERSION, credentials_file=credentials_file)


@app.command()
def clean_remote_store(
    account: Annotated[str, typer.Option("--source-coop-account")] = store.SOURCE_COOP_ACCOUNT,
    credentials_file: CredsOpt = None,
    workers: WorkersOpt = 8,
) -> None:
    """DESTRUCTIVE: delete every object in the published store prefix."""
    _setup_logging()
    if not sys.stdin.isatty():
        _die("refusing to wipe a published store without a terminal")
    s3 = remote.client(credentials_file)
    keys, total = remote.store_keys(account, credentials_file=credentials_file, s3=s3)
    if not keys:
        print("nothing to delete")
        return
    target = f"s3://{account}/{store.PRODUCT_NAME}/{store.STORE_SUBPATH}"
    print(f"about to delete {len(keys)} objects ({total / 1e9:.2f} GB) under {target}")
    if input("type 'delete' to continue: ").strip() != "delete":
        _die("aborted")
    if input(f"type the store path back ({store.STORE_SUBPATH}): ").strip() != store.STORE_SUBPATH:
        _die("aborted")
    remote.delete_keys(account, keys, credentials_file=credentials_file, workers=workers, s3=s3)


@app.command()
def show_config() -> None:
    """Print the frozen structural spec (what the store will look like)."""
    g = config.GRID
    print(
        json.dumps(
            {
                "source_version": config.SOURCE_VERSION,
                "dataset_version": config.DATASET_VERSION,
                "grid": {
                    "width": g.width,
                    "height": g.height,
                    "pixel_size": g.pixel_size,
                    "origin": [g.x_min, g.y_max],
                    "proj4": g.proj4,
                },
                "layers": len(config.layers()),
                "chunks": config.ENCODING.chunks(3),
                "shards_native": config.ENCODING.shards(3, depth_extent=6),
                "shards_overview": config.ENCODING.shards(3, depth_extent=1),
                "overview_factors": list(config.OVERVIEW_FACTORS),
                "properties": {
                    p.name: {
                        "group": p.group,
                        "tile_px": p.tile_px,
                        "mapped_units": p.mapped_units,
                        "conversion_factor": p.conversion_factor,
                    }
                    for p in config.included_properties()
                },
            },
            indent=2,
        )
    )
