# ISRIC SoilGrids v2.0 (250 m) to Icechunk

End-to-end pipeline turning the [ISRIC SoilGrids v2.0](https://soilgrids.org)
GeoTIFF tile tree into an analysis-ready Icechunk Zarr v3 store on
[Source Cooperative](https://source.coop): eleven soil properties on their native
250 m Interrupted Goode Homolosine grid, **byte-identical to the source** — no
resampling, no reprojection — with a multiscale overview pyramid for map-scale
reads.

- **`docs/data-reference.md`** — the authoritative data/product specification
  (grid, tiling, encoding, units, overviews, pipeline, versioning). Every number
  in it was measured from the source, not read from documentation.
- **`docs/future-variables.md`** — what is *not* included (the other four
  statistics, WRB classes, landmask, aggregates, a 4326 reprojection), why, and
  what each would need.
- **`product/README.md`** — the Source Coop landing page, with a copy-paste
  consumer snippet.

## Setup

```bash
uv sync        # install the locked environment
make test      # full test suite: synthetic fixtures, no network, ~1s
```

Nothing needs to be downloaded up front: the source is read over HTTPS and raw
tiles are never written to disk.

## Pipeline

Seven ordered commands. Every one is idempotent — a re-run skips work that is
already done and current — and the long one is checkpointed and resumable.

Each `make` target wraps the `soilgrids` CLI (`uv run soilgrids --help`). All
store-writing targets default to the local dev store
(`STORE=./soilgrids_store_local`); set `ACCOUNT=chill` to target the published
Source Coop product instead, so a bare `make materialize` can never touch it.

### 1. `make inspect-source`

Downloads the 61 layer VRTs and verifies the source is what `config.py` says it
is: the tile-lattice placement formula against every full tile, directory
listings against the VRT inventory, the model provenance in each VRT's metadata,
dtype, nodata and pixel size. Writes `work/2.0.0/source_manifest.parquet` — one
row per tile, with its ETag — and `source_report.json`. Nothing downstream runs
until this passes.

**Use `--listing-samples 0` for a real build** (`make inspect-source
LISTING_SAMPLES=0`). ISRIC's VRTs are incomplete: measured on `bdod/0_5`, 10
tiles are on disk but absent from the VRT, and 8 of them hold ~26,000 km² that
nothing else covers — one is a full 600 px tile. Exhaustive listing is the only
way to find them, and `repair_from_disk` puts back the ones that can be placed.
See `docs/data-reference.md` §3.

Runs in explicit parallel passes, because **ISRIC throttles per connection at
roughly 40 kB/s**: one 6 MB VRT takes ~2.5 minutes alone, but eight at once take
the same 2.5 minutes. Concurrency is the only lever the server leaves.

### 2. `make init-store`

Creates the icechunk repository at the versioned path
(`v{DATASET_VERSION}.icechunk`) and writes the complete empty structure from
config: both groups, coordinates and CRS, 11 native arrays and 88 overview
arrays — metadata only, since fill-value chunks occupy no storage (the whole
empty store is ~2 kB). Idempotent and **additive**: on an existing store it
creates only what is missing. It refuses outright if an existing array was
created under a different `EncodingSpec`, because shard-aligned writes make a
mixed shard grid unsafe.

### 3. `make materialize [PROPERTIES=…] [BBOX=…] [WORKERS=n] [COMMIT_EVERY=n]`

The long pole. The unit of work is one **(property, cell)**: one 1800 px
`tileSG` cell across all six depths. That is the smallest unit holding whole
source tiles for both subtilings (16 of 450 px, or 9 of 600 px) *and* whole
shards (4x4), so every tile is fetched exactly once and no shard is ever
read-modify-written. A cell buffer is 38.9 MB, so memory is flat however much of
the grid a run covers.

Each tile is placed by **its own GeoTIFF tie-point**, asserted against the
manifest: the file is the ground truth at write time, not the VRT. Ragged
coastline tiles need no special case — 2 039 of the 2 801 sit at an arbitrary
offset inside their subtile, and their tie-point says where. All-fill shards are
never written.

Checkpoint-commits every `COMMIT_EVERY` cells, recording completed cells in group
attrs, so a killed run resumes without re-downloading. `BBOX=min_lon,min_lat,max_lon,max_lat`
lands a useful subset early.

### 4. `make overviews [PROPERTIES=…] [CELLS=…] [WORKERS=n] [COMMIT_EVERY=n] [PROGRESS_EVERY=n]`

Builds the `2x` … `256x` pyramid, one property at a time, with topozarr's Rust
kernel driven directly (see `docs/data-reference.md` §6 for why — the high-level
API would silently average nodata into every coastline). Restricted to the cells
that were actually materialized, which keeps the pass proportional to the data
rather than to the grid: 15 seconds for a 5-cell subset instead of 15 minutes.

The unit of work is one **(level, depth, shard)** region, and the pass is
geometric: the `2x` level is ~73 % of the regions and every later level a quarter
of the one before it. So the flags are all in those terms — `CELLS` intersects
the materialized set for a trial run (a subset run deliberately does *not* mark
the property complete, so `release` still refuses it), `COMMIT_EVERY` checkpoints
every N regions, and progress is reported per region rather than per level.

Checkpointing writes a resume token — `{level, regions done, scope fingerprint}`,
a few dozen bytes on the first level's array — *before* each commit, so a killed
run restarts at the region it reached instead of redoing the level. The
fingerprint covers the cell set and the factor ladder: change either and the
token is ignored rather than skipping work that was never done. `make status`
shows the token while a run is in flight or interrupted.

**This phase can run while `materialize` is still backfilling another property.**
icechunk's branch commit is optimistic — any commit that lands first makes yours
fail, even when the two touched different arrays — so every phase-3 and phase-4
commit rebases (`store.commit_with_rebase`). The one thing a rebase cannot merge
is two writers updating a single node's attrs, which is why the resume token
lives on the level array and not on the group that `materialize` writes its cell
ledger to; the completion attr does share that group, so it is set by its own
small retried commit after the data is safe.

### 5. `make status` / `make validate [PROPERTIES=…]`

`status` prints the property × cell completion matrix. `validate` runs structural
checks (shapes, dtypes, chunk/shard grids, coordinates, attrs, multiscales
layout) and then the checks that actually prove the data:

- **byte-exact sampling** — random source tiles re-fetched and compared
  pixel-for-pixel, always including a ragged tile;
- **overview windows** recomputed against an independent fill-aware mean;
- **sand + silt + clay ≈ 1000 g/kg**, which only holds if three independently
  stored arrays agree.

### 6. `make release`

The publication gate: refuses while any property is incomplete or missing
overviews, then creates the immutable release tag (`soilgrids-2.0.0`), reopens
the store through the tag and re-reads every array. Tags are never deleted or
reused — a corrected re-release gets `--suffix -r2`.

### Publishing extras

```bash
make publish-readme ACCOUNT=chill   # upload product/README.md as the landing page
make upload-audit ACCOUNT=chill     # upload the manifest and reports to audit/2.0.0/
make info                           # store structure, tags, recent snapshots
make garbage-collect                # reclaim objects orphaned by checkpoint commits
make clean-local-store              # rm -rf the local dev store
make clean-remote-store             # DESTRUCTIVE wipe of the published store; confirms twice
```

`make help` lists every target and variable.

## Budget

The full `mean` scope is **352 GB of transfer** at the source's measured
throughput — roughly 38 hours, ~3.5 h per property — producing **~177 GB of native
arrays plus ~90 GB of overviews (~267 GB)**. Those come from ratios measured on real
data, not a guess: float32 native is 1.28x the int16 equivalent, and the 2x overview
ladder is 51 % of native (averaging produces high-cardinality floats, so it compresses
less well than the int16 path did).

Setting `config.OVERVIEW_FACTORS = (4, 16, 64, 256)` cuts the ladder to ~18 GB
(~195 GB total) at the cost of tiling smoothness.

With `ACCOUNT=` set, shards stream straight to S3 and nothing accumulates
locally. A full local build needs ~190 GB.

## Updating for a new ISRIC release

Update `SOURCE_VERSION` and `EXPECTED_VRT_METADATA` in `config.py`, bump the
minor `DATASET_VERSION` (new store path), and run the pipeline top to bottom.
`inspect-source` will refuse if the provenance in the VRTs does not match what
config expects, which is also how it detects ISRIC replacing data under the
mutable `latest/` path mid-release.

Changing `EncodingSpec` (chunk or shard shape) is a different kind of change: it
rewrites every array, so it needs a fresh store path and a full re-materialize.
Bump the major `DATASET_VERSION` to publish alongside the old layout. A re-chunk
reads the published store, not ISRIC, so it costs an internal copy rather than
another 38-hour download.
