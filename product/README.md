# ISRIC SoilGrids 250 m Soil Properties (v2.0)

Global predictions of eleven soil properties at 250 m resolution for six standard
depth intervals, produced by [ISRIC — World Soil Information](https://soilgrids.org).
This product takes the SoilGrids v2.0 GeoTIFF tile tree and reformats it into a cloud-optimized,
version-controlled [Icechunk](https://icechunk.io) Zarr store, where a soil profile
can be sliced by location and depth in one selection, in physical units, with no
mosaicking and no unit conversion.

SoilGrids is designed as a globally consistent, data-driven system that predicts soil
properties using global covariates and globally fitted models. This product carries the
**mean** prediction, the unbiased "expected value" for each cell. ISRIC also publishes
the median and the 5th/95th percentiles; see "Not included" below.

## Contents

One Icechunk repository at `v0.1.0.icechunk/`, on SoilGrids' native grid — no
reprojection, no resampling:

| group | dimensions | contents |
|---|---|---|
| `soil_properties` | `(depth_interval, y, x)` | 10 depth-indexed arrays: particle-size fractions, bulk density, carbon, nitrogen, pH, CEC, coarse fragments |
| `profile_properties` | `(y, x)` | 1 array: `ocs`, organic carbon stocks for a fixed 0–30 cm |
| `{group}/2x` … `{group}/256x` | same as parent | multiscale overviews for map-scale reads |

| | |
|---|---|
| Grid | 160,200 × 59,400 px, 250 m |
| CRS | Interrupted Goode Homolosine — **no EPSG code**, see *Coordinates* |
| Extent | x [−20,037,500, 20,012,500] m, y [−6,249,250, 8,600,750] m |
| Type | `float32`, physical units, `NaN` where there is no data |
| Storage | zarr v3 sharded, Zstd-compressed: `(1, 50, 50)` inner chunks in `(6, 450, 450)` shards |
| Size | ~177 GB native + ~90 GB overviews |

- **Coordinates** are pixel centres in metres. A CF `spatial_ref` variable carries the
  WKT and GeoTransform, so `rioxarray` and GIS tools georeference the arrays directly.
- **Storage layout**: one shard holds all six depths of a 112 km tile, so a
  field-scale query fetches one 7.8 kB shard index plus ~7 inner chunks — about 36 kB
  per property for a whole profile.
- **Units are already converted.** Values are in the conventional units named on each
  array, not ISRIC's packed integers. No scaling required.

### Depth intervals

The `depth_interval` coordinate of `soil_properties` carries the six standard
SoilGrids intervals; auxiliary coordinates `depth_top_cm` and `depth_bottom_cm` carry
the same information numerically.

| `depth_interval` | top (cm) | bottom (cm) |
|---|---:|---:|
| `0_5` | 0 | 5 |
| `5_15` | 5 | 15 |
| `15_30` | 15 | 30 |
| `30_60` | 30 | 60 |
| `60_100` | 60 | 100 |
| `100_200` | 100 | 200 |

`ocs` (organic carbon stocks) is published only for 0–30 cm, which is not one of the
six, so it lives in `profile_properties` as a 2-D array rather than as a seventh
interval that every other property would be empty at.

## Included parameters

Values are stored in the **units you read** column — the conversion from ISRIC's
packed integers has already been applied. The last two columns record how the source
publishes it, so the provenance stays checkable.

### `soil_properties` — by depth interval

| name | description | units you read | source units | ÷ |
|---|---|---|---|---:|
| `bdod` | Bulk density of the fine earth fraction | kg/dm³ | cg/cm³ | 100 |
| `cec` | Cation exchange capacity of the soil (buffered at pH 7) | cmol(c)/kg | mmol(c)/kg | 10 |
| `cfvo` | Volumetric fraction of coarse fragments (> 2 mm) | cm³/100cm³ (vol %) | cm³/dm³ (vol ‰) | 10 |
| `clay` | Proportion of clay particles (< 0.002 mm) in the fine earth fraction | g/100g (%) | g/kg | 10 |
| `nitrogen` | Total nitrogen (N) | g/kg | cg/kg | 100 |
| `ocd` | Organic carbon density | kg/m³ | hg/m³ | 10 |
| `phh2o` | Soil pH in water | pH | pH × 10 | 10 |
| `sand` | Proportion of sand particles (> 0.05 mm) in the fine earth fraction | g/100g (%) | g/kg | 10 |
| `silt` | Proportion of silt particles (≥ 0.002 mm and ≤ 0.05 mm) in the fine earth fraction | g/100g (%) | g/kg | 10 |
| `soc` | Soil organic carbon content in the fine earth fraction | g/kg | dg/kg | 10 |

### `profile_properties` — one value per pixel

| name | description | units you read | source units | ÷ |
|---|---|---|---|---:|
| `ocs` | Organic carbon stocks, 0–30 cm | kg/m² | t/ha | 10 |

`sand`, `silt` and `clay` are modelled jointly as a composition (ISRIC's `RUN10`
outputs, additive log-ratio transform), so the three sum to ~100 % wherever all three
have data. Every array's attributes record its own `source_outputs_version`, because
ISRIC built the properties in different model runs.

Units and conversion factors follow
[ISRIC's SoilGrids layer documentation](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html),
which is authoritative if you need to cross-check them.

## Reading the data

```python
import icechunk, xarray as xr
from pyproj import Transformer

repo = icechunk.Repository.open(
    icechunk.s3_storage(
        bucket="chill", prefix="soilgrids/v0.1.0.icechunk",
        region="us-east-1", endpoint_url="https://data.source.coop", anonymous=True,
    )
)
ds = xr.open_zarr(
    repo.readonly_session(tag="soilgrids-2.0.0").store,
    group="soil_properties", consolidated=False,
)

# SoilGrids uses Interrupted Goode Homolosine, which has NO EPSG code.
# Transform your lon/lat into the store's CRS, then select as usual.
to_igh = Transformer.from_crs("EPSG:4326", ds.spatial_ref.attrs["crs_wkt"], always_xy=True)
x, y = to_igh.transform(-102.45454, 41.55459)      # Nebraska Sandhills

# a whole soil profile at one point, in physical units
print(ds.sand.sel(x=x, y=y, method="nearest").values)
# -> [59.9 61.  61.  61.7 64.3 68.1]  % sand, by depth
print(ds.phh2o.sel(x=x, y=y, method="nearest").values)
# -> [7.4 7.5 7.7 7.9 8.2 8.3]  pH

# a ~1.5 km field window (y descends, so slice high -> low)
half = 750  # metres
field = ds.sand.sel(x=slice(x - half, x + half), y=slice(y + half, y - half))

# organic carbon stocks are 2-D, in their own group
ocs = xr.open_zarr(repo.readonly_session(tag="soilgrids-2.0.0").store,
                   group="profile_properties", consolidated=False).ocs
```

A whole-globe view is one small read from a coarse level:

```python
coarse = xr.open_zarr(repo.readonly_session(tag="soilgrids-2.0.0").store,
                      group="soil_properties/64x", consolidated=False)
coarse.sand.isel(depth_interval=0).plot(vmin=0, vmax=100)   # 2,503 × 928 px
```

### Coordinates

Interrupted Goode Homolosine, `+proj=igh +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m
+no_defs`. Three things to know:

1. **There is no EPSG code.** `crs.to_epsg()` returns `None`. Use the `crs_wkt` attr on
   `spatial_ref`, or the proj4 string in the root attrs. pyproj handles the projection
   exactly — round-trips to nanometre precision.
2. **No lat/lon coordinate arrays are stored.** Longitude is not a function of `x`
   alone, so a 1-D `lon` would be wrong and a 2-D one would be 38 GB. Transform at
   selection time, as above.
3. **The projection is interrupted.** Goode's lobe boundaries run through ocean, so a
   field-scale window is contiguous in practice (a 6×6 px window spans ~0.015° even
   beside the 100°W cut). A window wider than ~100 km may straddle a cut and be
   geographically discontinuous. About 4 % of columns in a mid-latitude row fall
   between lobes and are `NaN`.

## Overviews

Levels `2x` … `256x` are child groups with the same variables, coordinates and
attributes, so any level reads exactly like the native array. They follow the
[zarr-conventions/multiscales](https://github.com/zarr-conventions/multiscales) layout
with GeoZarr `proj:`/`spatial:` companions.

| level | grid (y × x) | pixel |
|---|---|---|
| native | 59,400 × 160,200 | 250 m |
| `2x` | 29,700 × 80,100 | 500 m |
| `8x` | 7,425 × 20,025 | 2 km |
| `32x` | 1,856 × 5,006 | 8 km |
| `256x` | 232 × 625 | 64 km |

Each level is a chained NaN-aware mean of the level above. That means mean-of-means
with unequal valid counts, which is bounded but not identical to a direct mean from
native. **Use the native arrays for analysis and the overviews for display.**

## Versioning

The store path carries the dataset version (`v0.1.0.icechunk/`). A new ISRIC release
gets a fresh path — this dataset has no time dimension, so releases replace rather
than append — and old paths stay readable. Pin a release with the immutable tag:

```python
repo.readonly_session(tag="soilgrids-2.0.0")
```

Tags are never deleted or reused; a corrected re-release gets a `-r2` suffix.

## Provenance & license

Built from `https://files.isric.org/soilgrids/latest/data/` with every source tile's
ETag recorded, so a rebuild detects ISRIC replacing data under the mutable `latest/`
path. Source provenance is carried per array: `Code_version v2.0.0`, WoSIS
`Data stream 7`, quantile regression forests, and a per-property
`source_outputs_version`. The tile manifest and validation reports are published under
`audit/2.0.0/`.

Every value is derived from the source by a single, exactly invertible scaling, and the
pipeline verifies that by re-fetching random source tiles and comparing them
pixel-for-pixel with the store. Pipeline and full data reference:
https://github.com/colinahill/soilgrids_data

Licensed **CC-BY 4.0**, as the source is. Please cite:

```
Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B.,
Ribeiro, E., and Rossiter, D.: SoilGrids 2.0: producing soil information for the
globe with quantified spatial uncertainty, SOIL, 7, 217–240,
https://doi.org/10.5194/soil-7-217-2021, 2021.
```

## Not included

Only the **mean** prediction. SoilGrids also publishes `Q0.05`, `Q0.5` (median) and
`Q0.95` for every layer, plus an `uncertainty` layer — 244 more layers, roughly a week
of transfer at the source's measured throughput. Also excluded: WRB soil-class
probabilities (a different product on an EPSG:4326 grid), the landmask, and ISRIC's
own 1 km/5 km aggregates (largely superseded by the `4x` and `16x` overviews here).
See `docs/future-variables.md` in the pipeline repo for what each would need.
