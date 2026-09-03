# ISRIC SoilGrids v2.0 -> Icechunk Zarr: data reference

The authoritative specification of what this pipeline produces. Every number here
was **measured** from the source tree, not read from documentation; where the two
disagree, this file records the measurement.

- Source: `https://files.isric.org/soilgrids/latest/data/`
- Source version: SoilGrids **2.0.0** (`Code_version v2.0.0`, WoSIS
  `Data stream 7`, quantile regression forests). `Outputs_version` and
  `Transformation` are **per property** -- see §8
- Dataset version: **0.1.0** -> store path `v0.1.0.icechunk`
- Licence: CC-BY 4.0 (as the source is)

## 1. Product

One Icechunk Zarr v3 store: eleven soil-property arrays on the native 250 m
Interrupted Goode Homolosine grid, `mean` statistic only, plus an eight-level
overview pyramid.

```
/                              root attrs: provenance, licence, citation, grid
  soil_properties/             dims (depth_interval[6], y, x)  int16, fill -32768
    bdod cec cfvo clay nitrogen ocd phh2o sand silt soc
    2x/ 4x/ 8x/ 16x/ 32x/ 64x/ 128x/ 256x
  profile_properties/          dims (y, x)
    ocs                        organic carbon stocks, fixed 0-30 cm
    2x/ ... 256x
```

`ocs` is 2-D in its own group because 0-30 cm is not one of the six standard
intervals; putting it on the depth coordinate would leave five all-fill slices
and a `sel(depth_interval="0_30")` that silently returns nothing for every other
property.

### Versioning

A new ISRIC release bumps the minor `DATASET_VERSION` and builds a fresh
`v{x.y.z}.icechunk` path (this dataset has no time dimension: releases replace,
they do not append). Breaking structural changes -- re-chunk, re-grid, changed
semantics -- bump major. Old paths stay readable. Release tags are immutable and
never reused; a corrected re-release gets `-r2`.

## 2. The canonical grid

ISRIC publishes **no monolithic raster** at 250 m: `data/{property}/` holds only
per-layer `.vrt` files and tile directories. The per-property VRT bounding boxes
also *disagree* (sand 159 246 x 58 034 at x0 -19 949 750; bdod and soc 159 243 at
-19 949 000), because each is a tight box around that property's own tiles. So no
VRT defines the grid.

The **tile lattice** does. A tile named `tileSG-{row}-{col}_{r}-{c}.tif` has its
upper-left corner at

```
x = -20_037_500 + col*450_000 + (c-1)*tile_px*250
y =   8_600_750 - row*450_000 - (r-1)*tile_px*250
```

verified against the GeoTIFF tie-points of **all 10 169 full tiles** of
`sand_0-5cm_mean`. Rows 0-32 and columns 0-88 exist, which fixes the extent:

| | |
|---|---|
| Shape | **160 200 x 59 400** px (356 x 132 tiles of 450 px; 267 x 99 of 600 px) |
| Pixel size | 250 m, square, north-up |
| Origin (UL edge) | (-20 037 500, 8 600 750) |
| Extent | x [-20 037 500, 20 012 500], y [-6 249 250, 8 600 750] |
| CRS | `+proj=igh +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs` |
| Land | 24.2 % (2.31 Gpx), from the VRT footprint |

### The projection

Interrupted Goode Homolosine has **no EPSG code** (`CRS.to_epsg()` is `None`).
pyproj handles it exactly: forward/inverse round-trips to nanometre precision on
real field locations. The store carries the CRS as `crs_wkt` on a scalar
`spatial_ref` variable plus the proj4 string in the root attrs.

**No lat/lon coordinates are stored.** Latitude *is* an exact function of `y`
alone in IGH (verified: zero spread across 100 x-positions on five rows, because
both the sinusoidal and Mollweide halves map `y` from phi only and the
interruptions displace `x` only) -- but longitude depends on `y` and on the lobe,
so a 1-D `lon` would be wrong and a 2-D one is 38 GB, a quarter of the dataset.
Consumers transform at selection time.

Roughly 4 % of columns in a mid-latitude row fall in no-man's-land between lobes
and are nodata. Goode's cuts are routed through ocean, so a 6x6 px field window
is contiguous in practice (0.015 deg span even beside the 100W cut); a window
spanning more than ~100 km may straddle a cut.

## 3. Source tiling (per property)

| tile | properties | subtiles/cell | RowsPerStrip | full-tile size | tiles/layer |
|---|---|---|---|---|---|
| 450 px | `cec cfvo clay nitrogen ocd ocs sand silt soc` | 4x4 | 9 | 406 513 B (`ocs` 406 511) | 12 970 (10 169 full, 2 801 ragged) in 1 131 cells |
| 600 px | `bdod phh2o` | 3x3 | 6 | 721 911 / 721 913 B | 7 692 (5 732 full, 1 960 ragged) in 1 129 cells |

Tile counts are measured and identical across the six depths of a property.

Both divide the 1800 px (450 km) `tileSG` cell and the canonical grid exactly.

Tiles are **uncompressed** (`Compression=1`), little-endian, single-band `Int16`,
`SampleFormat=2`, nodata `-32768`, striped with no partial strip on a full tile,
and the strips are **byte-contiguous** -- so a whole tile is one contiguous range
at the end of the file (`offset = filesize - W*H*2`).

**2 801 of 12 970 tiles per layer are ragged** (1 960 of 7 692 for the 600 px
family): clipped to the coastline, 2 091
distinct shapes (401x179, 87x77, ...), and **2 039 of them sit at an arbitrary
offset inside their subtile** rather than at its origin. This is why the VRT is
needed as the inventory and why placement is re-derived from each file's own
tie-point at write time.

At least one VRT source is *resampled* rather than placed 1:1 (a 1x2 tile
stretched to 62x124 with a fractional `DstRect`), which is the other reason the
VRT is not trusted for placement.

### Upstream inconsistencies (measured, and handled)

The VRT is the inventory, but it is not complete, and tile *names* are not
authoritative for placement. Measured exhaustively on `bdod/0_5` (7 681 VRT
sources, 1 129 cells):

| finding | count | handling |
|---|---|---|
| tiles on disk but absent from the VRT | 10 in 9 cells | `catalog.repair_from_disk` |
| -- of those, in a window nothing else covers | **8 (~26 000 km2)** | recovered into the manifest |
| -- of those, tie-point off the 250 m lattice | 1 | recorded, skipped (unplaceable) |
| -- of those, duplicating another tile's window | 1 | VRT's tile wins, deterministically |
| full tiles whose VRT placement != their name | 1 | recorded as an anomaly, tie-point wins |
| windows claimed twice *within* the VRT | 0 in bdod, 1 in nitrogen | resolved if byte-identical, else fatal |

The worked example is cell `tileSG-015-023`. On disk, `_3-2` and `_3-3` are both
full 600 px tiles carrying the **same** tie-point; their pixels differ in 89.5 % of
cells but their nodata masks are byte-identical, so they really do describe the same
window (correlation +0.75 in place, -0.13 as an eastern neighbour -- the name
`_3-3` is simply wrong). GDAL hit the collision when building the VRT, kept `_3-3`,
and dropped `_3-2` *and* `_3-1` -- and `_3-1` covers a full 22 500 km2 window that
nothing else covers.

Two consequences for the pipeline:

* **`--listing-samples 0` (every cell) is required for a real build.** Sampling
  20 cells per layer finds these only by luck. Exhaustive listing costs ~1 130
  requests per layer, a few minutes each, versus a 38 h ingest.
* **A name/lattice disagreement on a full tile is recorded, not fatal.** The VRT
  and the file agree; only the filename is misleading, and 2 039 ragged tiles
  already sit at arbitrary offsets, so names never were the authority.

**Two tiles claiming one window** is a third case, and it needs discrimination
rather than a blanket rule. `nitrogen/0_5`'s `tileSG-017-054_1-1` and `_1-2` are
**byte-identical** 450x450 rasters sharing a tie-point -- `_1-2` is a duplicate
file with a wrong name, so either choice yields the same store. `bdod`'s
`_3-2`/`_3-3` pair, by contrast, differ in 89.5 % of pixels. So:

* `tile_rows` never emits two rows for one window (writing both would make the
  winner depend on which worker finished last). It picks deterministically --
  the tile whose *name* matches the window, else the lexicographically smaller
  path -- and records a `Conflict`.
* `resolve_conflicts` then fetches both and verifies they are interchangeable.
  Byte-identical is recorded and accepted; **differing pixels raise**, because
  choosing between two different rasters for one window is a data-quality
  decision a pipeline must not make silently.

## 4. Encoding

| | native | overview levels |
|---|---|---|
| chunk | `(1, 50, 50)` | `(1, 50, 50)` |
| shard | `(6, 450, 450)` | `(1, 450, 450)` |
| compressor | zstd level 3 | zstd level 3 |
| chunks/shard | 486 (7.8 kB index) | 81 (1.3 kB index) |
| shard raw | 2.43 MB | 405 kB |

A native shard is **exactly one source tile position x all six depths**, so the
ingest unit is "fetch these 6 rasters, write one object", resume granularity is a
single tile, and `validate` can compare a shard byte-for-byte with the source.
50 px divides both 450 and 600, so one chunk grid serves every array and xarray
combines them without rechunking.

Chosen off a measured frontier for the real access pattern -- a ~6x6 px (1.5 km)
field, all six depths. Such a window is inside one chunk 83 % of the time and
always inside one shard, so the read is one 7.8 kB index plus ~7 chunk ranges =
**28.2 kB per property**. Sweeping every chunk size that divides 450:

| chunk px | 15 | 25 | 30 | 45 | 50 | 75 | 150 |
|---|---|---|---|---|---|---|---|
| bytes per field read | 89.7 kB | 37.7 kB | 30.2 kB | 26.6 kB | **28.2 kB** | 45.7 kB | 151 kB |

The optimum is 45; 50 is 6 % off it, inside the noise of the compression
estimates, and 50 also divides 600. Smaller chunks lose to their own 16 B/chunk
index; larger ones lose to read amplification.

**Measured zstd-3 ratio is not flat** (unlike categorical data), which is the
other reason not to go below 50:

| chunk px | 15 | 20 | 25 | 50 | 75 | 100 | 150 | 450 |
|---|---|---|---|---|---|---|---|---|
| sand 0-5 | 1.45 | 1.56 | 1.63 | 1.78 | 1.82 | 1.90 | 1.92 | 1.92 |
| clay 100-200 | 1.71 | 1.89 | 2.01 | 2.17 | 2.22 | 2.27 | 2.27 | 2.24 |
| phh2o (600 px) | 3.75 | 4.73 | 5.33 | 6.84 | 7.05 | 7.17 | 7.18 | 6.41 |

Overview levels use depth-1 **shards** (not just chunks) because they serve map
display, which reads one depth at a time; that also makes the downsample stream
one (property, depth) layer at a time.

**Measured on real data** (5 cells x 6 depths, 194 MB of source pixels):

| | int16 (mapped units) | float32 (decoded) | ratio |
|---|---|---|---|
| native | 108.0 MB | 138.6 MB | 1.28x |
| 2x overview ladder | 39.1 MB | 70.1 MB | 1.79x |
| ladder as share of native | 36 % | **51 %** | |

The overview ladder costs relatively more in float32: averaging produces many
distinct fractional values, where the Int16 path truncated back to a low-cardinality
integer. Projecting to the full product: **~177 GB native + ~90 GB overviews =
~267 GB**.

`config.OVERVIEW_FACTORS = (4, 16, 64, 256)` is the lever if that matters -- a sparse
ladder is 0.20x the overview pixels of the 2x ladder, so ~18 GB instead of ~90 GB and
**~195 GB total**, at the cost of tiling smoothness for map rendering.

Changing `EncodingSpec` rewrites every array, so it needs a fresh store path:
`init-store` refuses outright if an existing array has a different chunk/shard
grid, because shard-aligned writes make a mixed shard grid unsafe.

## 5. Values and units

Stored on disk **exactly as published**: scaled `Int16` in the mapped units. That is
what makes `validate`'s byte-exact tile comparison possible and what keeps the product
a faithful mirror rather than a derivative.

But each array also carries CF **`scale_factor`** (= 1 / conversion_factor) and
**`_FillValue`** (`-32768`), so `xr.open_zarr` decodes to conventional physical units
with `NaN` for no data and **the consumer converts nothing**. `mask_and_scale=False`
returns the raw integers.

| property | long name | decoded units | stored | / |
|---|---|---|---|---|
| `bdod` | bulk density of the fine earth fraction | kg/dm3 | cg/cm3 | 100 |
| `cec` | cation exchange capacity, buffered at pH 7 | cmol(c)/kg | mmol(c)/kg | 10 |
| `cfvo` | volumetric fraction of coarse fragments (> 2 mm) | vol % | cm3/dm3 | 10 |
| `clay` | clay (< 0.002 mm) in the fine earth fraction | mass % | g/kg | 10 |
| `nitrogen` | total nitrogen | g/kg | cg/kg | 100 |
| `ocd` | organic carbon density | kg/m3 | hg/m3 | 10 |
| `ocs` | organic carbon stocks (0-30 cm) | kg/m2 | t/ha | 10 |
| `phh2o` | soil pH in water | pH | pH x 10 | 10 |
| `sand` | sand (> 0.05 mm) in the fine earth fraction | mass % | g/kg | 10 |
| `silt` | silt (0.002-0.05 mm) in the fine earth fraction | mass % | g/kg | 10 |
| `soc` | soil organic carbon in the fine earth fraction | g/kg | dg/kg | 10 |

### Why CF packing here, and not in the sibling repos

`usda_cropland_data` and `usda_gnatsgo` deliberately set **no** `_FillValue` on their
integer arrays, and the playbook states the rule as a flat prohibition. The reason is
specific to their data: CDL's integers are *categorical* (crop-class codes) and
gnatsgo's `mukey` is an *identifier*. Masking a legal code to NaN and upcasting to
float destroys the meaning, and there is no scale to apply.

SoilGrids is the opposite case: the integer is purely a storage encoding of a
continuous measurement, and scaling plus nodata-masking is exactly what CF packing
exists for. Applying the categorical rule here would have forced every consumer to
divide by hand, or forced `float32` on disk -- which would roughly triple the store
(the measured Int16 ratios in section 4 do not survive the switch to floats) and
forfeit byte-exact validation.

Storing the *decoded* units directly is not an option at integer precision: pH 6.5 and
bulk density 1.35 kg/dm3 are not representable in `Int16`.

Note that xarray infers the decoded width from `scale_factor`, which JSON round-trips
as a double, so decoding yields `float64`. Irrelevant for a field-sized read (288 B);
for large windows, `.astype("float32")` or `mask_and_scale=False`.

## 6. Overviews

Levels `2x` ... `256x` as factor-named child groups (parent/child layout, so
native never moves), following zarr-conventions/multiscales with GeoZarr
`proj:`/`spatial:` companions.

Built with `topozarr.engine.downsample_level` -- topozarr's Rust kernel driven
**directly**, not through `create_pyramid`. Three reasons, all load-bearing:

1. **The fill value is passed as an argument.** `create_pyramid` resolves it from
   `encoding["_FillValue"]` -> `attrs["_FillValue"]` -> NaN-if-float -> `None`.
   Our arrays carry no CF `_FillValue` on purpose, so it would resolve to `None`,
   and then: the mean averages `-32768` **as data** (measured: -16207 where the
   correct value is 352, and the error widens one cell per level, so ~64 km of
   corrupted coastline by 256x); the level's declared fill becomes 0, which is a
   legal soil value; and every all-ocean shard gets written instead of omitted.
   `overviews.py` passes `fill_value=-32768` explicitly and `validate` recomputes
   sampled windows against an independent reference.
2. **No fusion path**, so memory is bounded by region size (~1.6 MB) rather than
   by how much RAM happens to be free. `create_pyramid`'s fusion would
   pre-allocate every level as a dense array: 38.1 GB for one property.
3. **The layout is ours.** `Pyramid.write` imposes ordinal level groups (`0/`,
   `1/`) and wants native at `0/<var>`, which would mean duplicating or
   relocating the native data.

Each level is a **chained** stride-2 fill-aware mean of the level above,
truncated toward zero to stay `int16`. Mean-of-means with unequal valid counts is
not identical to a direct mean from native: bounded, but real, and recorded in
each level's `derived_from`. **Native for analysis, overviews for display.**

`downsample_level` matches `xarray.coarsen(boundary="trim")`, so level shapes are
`floor(parent/2)`, not `ceil(native/factor)` -- the grid divides evenly only by 2,
4 and 8 (160 200 = 2^3 * 3^2 * 5^2 * 89; 59 400 = 2^3 * 3^3 * 5^2 * 11):

| factor | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| width | 80 100 | 40 050 | 20 025 | 10 012 | 5 006 | 2 503 | 1 251 | 625 |
| height | 29 700 | 14 850 | 7 425 | 3 712 | 1 856 | 928 | 464 | 232 |

The chain therefore discards at most one coarse pixel per level at the far
right/bottom edge -- by 256x, 200 px (50 km) in x and 8 px (2 km) in y, at
lambda ~ 179.8 deg and ~56 deg S, both open ocean. Deliberate.

## 7. Pipeline

Seven ordered, idempotent phases; see the README. The one thing worth repeating
here: **ISRIC throttles per connection at roughly 40 kB/s** -- one 6 MB VRT takes
~132 s on its own, while eight at once take the same 132 s. Aggregate throughput
(~2.2-2.6 MB/s, flat from 8 to 32 connections) comes *only* from parallelism.
Every phase is therefore built around concurrency, and every phase is resumable,
because the full ingest is ~352 GB of transfer.

Measured on a 5-cell High Plains subset: phase 1 for `sand` (6 VRTs, listings,
HEADs) 22 s, and 24 layers in 64 s; phase 3, 5 cells x 6 depths = 0.19-0.20 GB
per property in 90-115 s (18-23 s/cell, ~2 MB/s); phase 4, all eight pyramid
levels in 15 s.

Raw tiles are never written to disk: they are fetched into a 38.9 MB cell buffer,
placed, and discarded once the cell's shards are written.

Checkpoint commits leave one snapshot each, so a full property at
`COMMIT_EVERY=16` adds ~70 snapshots and the whole product ~800. That is
deliberate -- the history shows exactly how a long backfill progressed -- but it
is why `garbage-collect` exists, and why `icechunk expire_snapshots` is the tool
if the history itself ever needs trimming.

## 8. Identity and drift

SoilGrids filenames carry no release date, and `latest/` is a mutable path.
Identity is therefore:

- `SOURCE_VERSION = "2.0.0"` plus the VRT `<Metadata>` provenance, checked on
  **every** layer VRT (`inspect-source` refuses on any disagreement);

### Per-property model provenance (measured)

`Code_version` (`v2.0.0`) and `WoSIS_version` (`Data stream 7`) are global, but
**`Outputs_version` and `Transformation` are per property**: ISRIC built the
properties in different model runs. Measured, and invariant across the depths of
one property:

| property | Outputs_version | Transformation (as published) | normalised |
|---|---|---|---|
| `bdod` | RUN03 | element present, empty | `none` |
| `cec` | RUN06 | element present, empty | `none` |
| `cfvo` | RUN06 | element **absent** | `none` |
| `clay` | RUN10 | `alr` | `alr` |
| `nitrogen` | RUN05 | element **absent** | `none` |
| `ocd` | RUN03 | element **absent** | `none` |
| `ocs` | RUN03 | element **absent** | `none` |
| `phh2o` | RUN05 | element present, empty | `none` |
| `sand` | RUN10 | `alr` | `alr` |
| `silt` | RUN10 | `alr` | `alr` |
| `soc` | RUN18 | element **absent** | `none` |

Two things follow. **`sand`, `silt` and `clay` share RUN10 and the additive
log-ratio (`alr`) transform** -- they are modelled jointly as a composition,
which is *why* the sand+silt+clay closure check in section 9 holds and is a
meaningful test. And a single global expectation would be wrong: these are
recorded per property in `config.PropertySpec` so the drift check stays strict
without false positives. (It found this: an initial global `RUN10` expectation
correctly rejected `phh2o`, which is RUN05.)

**`Transformation` is compared normalised.** Upstream spells "no transform"
three ways -- the MDI element absent, present but empty, or carrying text -- and
`config.normalise_transformation` collapses all three to `none`. A change
between those spellings is not drift and must not abort a 38-hour ingest; a real
change (`alr` -> `none`) still does. `Outputs_version` is compared strictly,
because it is consistently populated with a real value on every layer.

- these per-property strings, checked on every layer VRT (`Outputs_version`
  strictly, `Transformation` normalised);
- every tile's **ETag and Last-Modified** recorded in
  `work/2.0.0/source_manifest.parquet` (ETag is `inode-mtime-size`, so it changes
  when a file is replaced). `fetch.get` verifies it where recorded, and raises
  `SourceChanged` rather than mixing two snapshots into one store.

## 9. Verification

- `make test` -- synthetic fixtures on a shrunken grid, no network.
- `make inspect-source` -- the structural gate: lattice cross-check for every
  full tile, sampled directory listings against the VRT inventory (a tile on disk
  but absent from a VRT would otherwise be dropped silently), provenance, dtype,
  nodata, pixel size.
- `make validate` -- structural checks, then **byte-exact sampling**: re-fetch
  random source tiles and compare pixel-for-pixel, always including a ragged
  tile; overview windows recomputed against an independent fill-aware mean; and
  the sand+silt+clay ~ 1000 g/kg closure, which only holds if three
  independently-stored arrays agree.

  Sampling is restricted to cells that have actually been materialized: a partial
  ingest would otherwise "fail" on every unfilled cell, where the store is
  correctly holding its fill value.

### What has been verified against the real source

On a 5-cell High Plains subset of `sand`, `silt`, `clay` and `phh2o` (89 checks,
0 failures):

- **byte-exact** agreement for sampled 450 px *and* 600 px tiles;
- all **eight overview levels** matching an independently recomputed fill-aware
  mean;
- **sand + silt + clay closure** over ~200 000 px per window, summing to
  [999, 1001] across four windows -- three separately stored, separately fetched,
  separately placed arrays agreeing to 1 g/kg;
- a point cross-checked against **ISRIC's own REST API**
  (`rest.isric.org/soilgrids/v2.0/properties/query`) at 41.55 N, 102.45 W:
  599/610/610/617/643/681 g/kg, an **exact match on all six depths**, which
  independently validates the whole lon/lat -> IGH -> pixel -> value chain.
- `make release` -- refuses while anything is incomplete, then re-reads every
  array through the immutable tag.
