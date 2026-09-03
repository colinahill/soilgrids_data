# Deferred variables and products

Everything ISRIC publishes that this store does **not** contain, why, and what
each would need. `config.PROPERTIES` carries a `status` field, so promoting a
deferred entry is a config change plus (where noted) new code.

## 1. The other four statistics — `Q0.05`, `Q0.5`, `Q0.95`, `uncertainty`

**Status:** deferred. **Cost:** 244 more layers, ~1.3 TB of transfer.

SoilGrids publishes five statistics per (property, depth). We take `mean` only.
The others are structurally identical — same tiling, same dtype, same nodata — so
this is the cheapest possible extension: no new code, only scope.

`uncertainty` is defined as `(Q0.95 − Q0.05) / Q0.50`, i.e. it is derivable from
the three quantiles, so taking all three makes it redundant (though ISRIC's is
authoritative and cheaper than recomputing).

**What it needs:** a `statistic` axis. Two options, and the choice matters:

- a `statistic` dimension on each array, chunked at 1 — keeps one array per
  property, and unfilled statistics cost nothing (empty chunks). But it changes
  the native shard from `(6, 450, 450)` to `(1, 6, 450, 450)`-ish and so needs a
  fresh store path.
- separate arrays (`sand_q05`, …) — additive to the existing store, no re-chunk,
  but 55 arrays instead of 11.

The additive option is the reason to defer rather than pre-build: `mean` ships
now, and the decision stays open.

**Blocker:** transfer time. At the source's measured ~40 kB/s per connection, the
four extra statistics are roughly a week of wall clock even at full concurrency.

## 2. WRB soil classes — `wrb/`

**Status:** deferred. **Cost:** 32 layers (31 class probabilities + `MostProbable`).

A different product on a **different grid**: EPSG:4326, 172 800 × 67 200, `Byte`,
nodata 255, and ~460 numerically-named tiles (`10.tif`, `100.tif`) rather than the
`tileSG-` lattice.

**What it needs:** a second `GridSpec` and a second tile-naming scheme, i.e. real
work in `grid.py` and `catalog.py` — `config.GRID` is currently a singleton and
`grid.py` assumes the `tileSG` formula. It also needs categorical CF attrs
(`flag_values` / `flag_meanings`) and, for overviews, **`mode` rather than `mean`**
resampling — which does *not* compose, so every pyramid level would have to be
built from native rather than chained.

**Decision needed:** whether class probabilities belong in the same store at all,
given they share no grid with the soil properties. A sibling store
(`v0.1.0-wrb.icechunk`) is probably cleaner.

## 3. `landmask`

**Status:** deferred, and cheap. **Cost:** one 160 300 × 61 319 `Int16` layer.

A single COG (`landmask_SG_052020_COG512.tif`), **Deflate with horizontal
predictor 2** and 512² internal tiles — so `tiff.py` refuses it by design
(uncompressed-only). Its grid is also 160 300 × 61 319, which is *not* the
canonical 160 200 × 59 400: it would need its own extent reconciliation.

**What it needs:** either a decompressing reader path in `tiff.py` (zlib +
predictor), or just reading it with rasterio in a one-off ingest since it is a
single small file. Low effort, low value: the per-property nodata mask already
tells a consumer where there is no soil.

## 4. Aggregated 1 km and 5 km products — `data_aggregated/`

**Status:** deferred, superseded. **Cost:** trivial (199 MB and 9.4 MB per layer).

ISRIC's own coarsened products, in the same IGH projection (39 812 × 14 509 and
7 962 × 2 902). Also Deflate + predictor 2, so `tiff.py` refuses them.

**Largely superseded** by the overview pyramid: `4x` is 1 km and `16x` is 4 km,
built from the same native data. The difference is method — ISRIC's aggregation
may not be a plain fill-aware mean — so if exact agreement with ISRIC's published
1 km product matters, ingest theirs rather than ours.

**Decision needed:** whether anyone wants ISRIC's aggregation specifically. If
not, this stays closed.

## 5. An EPSG:4326 reprojection

**Status:** deferred, deliberately. **Cost:** a second full store.

Interrupted Goode Homolosine is awkward — no EPSG code, interrupted lobes — and a
lat/lon version would be easier for many consumers.

**Why not:** reprojection resamples. This product's contract is that every pixel
is byte-identical to what ISRIC published, which is what makes
`validate --samples` a meaningful check. A reprojected store is a *derived
product* with its own resampling decisions (nearest for categorical closure vs
bilinear for smoothness), its own grid choice, and its own validation story.

**What it needs:** a documented resampling choice, a target grid, and its own
store path. Build it *from* this store, never from ISRIC again.

## 6. Litter layers, `Depth_interval` variants

The VRT metadata carries `Litter_layers: FALSE` and an empty `Depth_interval`
field, and SoilGrids v2.0 publishes only the six standard intervals. Nothing to
defer here — noted so the empty fields are not mistaken for missing ingest.
