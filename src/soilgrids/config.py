"""Dataset structure as frozen, reviewable configuration.

Every structural fact about the published store lives here: the canonical grid,
the property inventory, depth intervals, encodings and attribute constants. Any
structural change shows up as a code diff (pattern borrowed from
dynamical-org/reformatters and usda_gnatsgo). Ingest order can never influence
the result because structure is decided once, from this module.

Grid and tiling numbers are MEASURED from the ISRIC SoilGrids v2.0 tile tree
(see docs/data-reference.md), never inferred from documentation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

# ---------------------------------------------------------------------------
# Source and product identity
# ---------------------------------------------------------------------------

# The ISRIC release this DATASET_VERSION was built from. SoilGrids filenames
# carry no release date, so identity comes from the VRT <Metadata> provenance
# (cross-checked by catalog.py) plus every tile's ETag in the source manifest.
SOURCE_VERSION = "2.0.0"
SOURCE_BASE_URL = "https://files.isric.org/soilgrids/latest/data"

# Provenance strings expected in EVERY layer VRT's <Metadata> block. catalog.py
# refuses to run when a layer disagrees: that means ISRIC replaced the data under
# the mutable `latest/` path.
#
# Outputs_version is deliberately NOT here: it is per property (measured, see
# PropertySpec.outputs_version) because ISRIC built the properties in different
# model runs -- sand, silt and clay share RUN10 and the `alr` transform, which is
# why their compositional closure holds, while soc is RUN18 and bdod RUN03. It is
# invariant across the depths of one property (verified).
EXPECTED_VRT_METADATA = {
    "Code_version": "v2.0.0",
    "WoSIS_version": "Data stream 7",
}

# Bumped for every published store. A new ISRIC release bumps minor and builds a
# fresh store at the new v{DATASET_VERSION}.icechunk path (this dataset has no
# time dimension; releases replace, not append). Breaking structural changes
# (re-chunk, re-grid, semantics) bump major. Old store paths remain readable.
DATASET_VERSION = "0.1.0"

# The source tiles are Int16 in scaled "mapped" units with a -32768 sentinel.
# The store holds the DECODED values: float32 in conventional physical units with
# NaN for no data, so a consumer -- xarray or otherwise -- needs no conversion and
# no CF support. The transform is exactly invertible (verified for all 65 535
# Int16 values at both scale factors), so `validate` remains an exact check
# rather than a tolerance.
SOURCE_NODATA = -32768
SOURCE_DTYPE = "int16"
DTYPE = "float32"
FILL_VALUE = float("nan")

DEPTH_DIM = "depth_interval"
DIMS_3D = (DEPTH_DIM, "y", "x")
DIMS_2D = ("y", "x")


# cell geometry, needed by PropertySpec validation before GRID is constructed
GRID_PIXEL_SIZE = 250.0
GRID_CELL_M = 450_000.0


def normalise_transformation(value: str | None) -> str:
    """Normalise the VRT's ``Transformation`` metadata to "none" or a real value.

    ISRIC spells "no transform" three ways, measured across the eleven
    properties: the MDI element is *absent* (cfvo, nitrogen, ocd, soc, ocs),
    *present but empty* (bdod, cec, phh2o), or carries a real value (``alr`` for
    clay, sand and silt -- the compositional trio, modelled jointly, which is why
    their fractions sum to ~100 %).

    Normalising both sides keeps the drift check meaningful (alr -> none would
    still be caught) without aborting a 38-hour ingest because an empty element
    became an absent one.
    """
    if value is None:
        return "none"
    v = value.strip()
    return "none" if v == "" or v.lower() == "none" else v


def is_fill(value) -> bool:
    """Is ``value`` the store's fill? NaN-aware, and tolerant of numpy scalars.

    ``NaN != NaN``, and zarr hands back a ``np.float32`` rather than a Python
    ``float``, so neither ``==`` nor ``isinstance(x, float)`` works here.
    """
    try:
        return bool(value != value)
    except TypeError:
        return False


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Canonical grid
# ---------------------------------------------------------------------------


class GridSpec(FrozenModel):
    """The canonical SoilGrids grid, derived from the tile naming scheme.

    ISRIC publishes no monolithic raster and the per-property VRT bounding boxes
    disagree, so the grid is defined by the tile lattice instead. A tile named
    ``tileSG-{row}-{col}_{r}-{c}.tif`` has its upper-left corner at

        x = x_min + col*CELL_M + (c-1)*tile_px*pixel_size
        y = y_max - row*CELL_M - (r-1)*tile_px*pixel_size

    verified exactly against the GeoTIFF tie-points of all 10 169 full tiles of
    ``sand_0-5cm_mean``. Rows 0-32 and cols 0-88 exist, which fixes the extent.
    """

    # Interrupted Goode Homolosine: no EPSG code exists, so the authority is the
    # WKT read from the source tiles themselves (see catalog.verify_crs).
    proj4: str = "+proj=igh +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    pixel_size: float = GRID_PIXEL_SIZE
    x_min: float = -20_037_500.0  # upper-left corner (edge, not pixel centre)
    y_max: float = 8_600_750.0
    width: int = 160_200  # 356 * 450 = 267 * 600
    height: int = 59_400  # 132 * 450 =  99 * 600
    cell_m: float = GRID_CELL_M  # one tileSG cell: 1800 px
    max_tile_row: int = 32
    max_tile_col: int = 88

    @model_validator(mode="after")
    def _check(self) -> GridSpec:
        # the grid must be a whole number of cells and of both subtile sizes, or
        # shard-aligned writes and the overview ladder both break
        cell_px = round(self.cell_m / self.pixel_size)
        for name, extent, cells in (
            ("width", self.width, self.max_tile_col + 1),
            ("height", self.height, self.max_tile_row + 1),
        ):
            if extent != cells * cell_px:
                raise ValueError(f"{name} {extent} != {cells} cells x {cell_px} px")
        return self

    @property
    def x_max(self) -> float:
        return self.x_min + self.width * self.pixel_size

    @property
    def y_min(self) -> float:
        return self.y_max - self.height * self.pixel_size

    @property
    def geotransform(self) -> str:
        """GDAL GeoTransform string for the spatial_ref attr."""
        return f"{self.x_min} {self.pixel_size} 0.0 {self.y_max} 0.0 {-self.pixel_size}"

    def x_coords(self):
        """Pixel-centre x coordinates, ascending."""
        import numpy as np

        return self.x_min + (np.arange(self.width) + 0.5) * self.pixel_size

    def y_coords(self):
        """Pixel-centre y coordinates, descending (north to south)."""
        import numpy as np

        return self.y_max - (np.arange(self.height) + 0.5) * self.pixel_size


# ---------------------------------------------------------------------------
# Depth intervals
# ---------------------------------------------------------------------------


class DepthInterval(FrozenModel):
    """One entry of the string depth_interval coordinate.

    ``source_token`` is the token used in ISRIC directory and file names
    (``sand_0-5cm_mean``); ``label`` is the snake-case coordinate value.
    """

    label: str
    source_token: str
    top_cm: int
    bottom_cm: int


DEPTH_INTERVALS: list[DepthInterval] = [
    DepthInterval(label="0_5", source_token="0-5cm", top_cm=0, bottom_cm=5),
    DepthInterval(label="5_15", source_token="5-15cm", top_cm=5, bottom_cm=15),
    DepthInterval(label="15_30", source_token="15-30cm", top_cm=15, bottom_cm=30),
    DepthInterval(label="30_60", source_token="30-60cm", top_cm=30, bottom_cm=60),
    DepthInterval(label="60_100", source_token="60-100cm", top_cm=60, bottom_cm=100),
    DepthInterval(label="100_200", source_token="100-200cm", top_cm=100, bottom_cm=200),
]
DEPTH_LABELS: list[str] = [d.label for d in DEPTH_INTERVALS]

# ocs is published for a single, non-standard interval and so lives in its own
# 2-D group rather than as a seventh slot on the depth coordinate.
OCS_INTERVAL = DepthInterval(label="0_30", source_token="0-30cm", top_cm=0, bottom_cm=30)


# ---------------------------------------------------------------------------
# Encoding (zarr v3 sharding, zstd)
# ---------------------------------------------------------------------------


class EncodingSpec(FrozenModel):
    """Sharded zarr v3 encoding.

    A native shard is exactly ONE source tile position x all six depths, so the
    ingest unit is "fetch these 6 rasters, write one object", resume granularity
    is a single tile, and validation can compare a shard byte-for-byte against
    the source. 50 px chunks divide both the 450 px and the 600 px subtile grids,
    so one chunk grid serves every array.

    50/450 was picked off a measured frontier for the real access pattern (a
    ~6x6 px field, all depths): 486 chunks/shard is a 7.8 kB shard index, and the
    read is index + ~7 chunk ranges = 28.2 kB. Smaller chunks lose to their own
    index (15 px -> 89.7 kB) and to a worse zstd ratio; larger ones lose to
    payload (150 px -> 151 kB). Depth is chunked at 1 so depths backfill in
    parallel and a single-depth read costs 1/6.

    Overview levels use depth-1 SHARDS (not just chunks) because they serve map
    display, which reads one depth at a time; that also makes the downsample
    stream one (property, depth) layer at a time.
    """

    chunk_y: int = 50
    chunk_x: int = 50
    shard_y: int = 450
    shard_x: int = 450
    zstd_level: int = 3

    def chunks(self, ndim: int) -> tuple[int, ...]:
        return (1, self.chunk_y, self.chunk_x) if ndim == 3 else (self.chunk_y, self.chunk_x)

    def shards(self, ndim: int, *, depth_extent: int) -> tuple[int, ...]:
        return (depth_extent, self.shard_y, self.shard_x) if ndim == 3 else (self.shard_y, self.shard_x)

    @model_validator(mode="after")
    def _check(self) -> EncodingSpec:
        if self.shard_y % self.chunk_y or self.shard_x % self.chunk_x:
            raise ValueError("zarr v3 requires the shard shape to be a multiple of the chunk shape")
        return self


ENCODING = EncodingSpec()

# Overview ladder. downsample_level matches xarray.coarsen(boundary="trim"), so
# level shapes are floor(parent/2) and the chain discards at most one coarse
# pixel per level at the far right/bottom edge (200 px = 50 km in x and 8 px =
# 2 km in y by 256x, at lambda~179.8 deg and ~56 deg S: open ocean).
# A sparse [4, 16, 64, 256] ladder would cost ~6.7% instead of ~33%, trading
# tiling smoothness.
OVERVIEW_FACTORS: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)
OVERVIEW_RESAMPLING = "mean"


# ---------------------------------------------------------------------------
# Property inventory
# ---------------------------------------------------------------------------

GroupName = Literal["soil_properties", "profile_properties"]


class PropertySpec(FrozenModel):
    """One output array: identity, source tiling, units and provenance.

    ``mapped_units`` / ``conversion_factor`` / ``conventional_units`` are
    recorded as INFORMATIONAL attrs. They are deliberately not CF
    ``scale_factor``: on an integer array that would make xarray mask the fill
    and upcast to float on every read.
    """

    name: str
    group: GroupName
    long_name: str
    mapped_units: str
    conversion_factor: int
    conventional_units: str
    # source subtiling: 4x4 subtiles of 450 px, or 3x3 of 600 px
    tile_px: int
    subtiles_per_cell: int
    rows_per_strip: int  # measured; tiff.py asserts it
    depths: tuple[str, ...]
    # measured from each property's VRT <Metadata>; catalog.py refuses on drift.
    # transformation is NORMALISED (see normalise_transformation): upstream spells
    # "no transform" three different ways.
    outputs_version: str
    transformation: str
    status: Literal["included", "deferred"] = "included"

    @model_validator(mode="after")
    def _check(self) -> PropertySpec:
        if self.tile_px % self.rows_per_strip:
            raise ValueError(
                f"{self.name}: {self.tile_px} rows is not a whole number of {self.rows_per_strip}-row strips"
            )
        if self.group == "soil_properties" and tuple(self.depths) != tuple(DEPTH_LABELS):
            raise ValueError(f"{self.name}: 3-D properties must carry every standard depth")
        if self.group == "profile_properties" and len(self.depths) != 1:
            raise ValueError(f"{self.name}: 2-D properties carry exactly one interval")
        return self

    @property
    def ndim(self) -> int:
        return 3 if self.group == "soil_properties" else 2

    @property
    def dims(self) -> tuple[str, ...]:
        return DIMS_3D if self.ndim == 3 else DIMS_2D

    @property
    def depth_extent(self) -> int:
        return len(self.depths)

    @property
    def array_path(self) -> str:
        return f"{self.group}/{self.name}"

    def layer_dir(self, depth_label: str) -> str:
        """ISRIC directory name for one (property, depth) mean layer."""
        token = _DEPTH_TOKENS[depth_label]
        return f"{self.name}_{token}_mean"

    def layer_url(self, depth_label: str) -> str:
        return f"{SOURCE_BASE_URL}/{self.name}/{self.layer_dir(depth_label)}"


_DEPTH_TOKENS = {d.label: d.source_token for d in DEPTH_INTERVALS} | {OCS_INTERVAL.label: OCS_INTERVAL.source_token}

_STANDARD = tuple(DEPTH_LABELS)

PROPERTIES: dict[str, PropertySpec] = {
    spec.name: spec
    for spec in [
        PropertySpec(
            name="bdod",
            outputs_version="RUN03",
            transformation="none",
            group="soil_properties",
            long_name="bulk density of the fine earth fraction",
            mapped_units="cg/cm3",
            conversion_factor=100,
            conventional_units="kg/dm3",
            tile_px=600,
            subtiles_per_cell=3,
            rows_per_strip=6,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="cec",
            outputs_version="RUN06",
            transformation="none",
            group="soil_properties",
            long_name="cation exchange capacity of the soil, buffered at pH 7",
            mapped_units="mmol(c)/kg",
            conversion_factor=10,
            conventional_units="cmol(c)/kg",
            tile_px=450,
            subtiles_per_cell=4,
            rows_per_strip=9,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="cfvo",
            outputs_version="RUN06",
            transformation="none",
            group="soil_properties",
            long_name="volumetric fraction of coarse fragments (> 2 mm)",
            mapped_units="cm3/dm3",
            conversion_factor=10,
            conventional_units="cm3/100cm3 (vol%)",
            tile_px=450,
            subtiles_per_cell=4,
            rows_per_strip=9,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="clay",
            outputs_version="RUN10",
            transformation="alr",
            group="soil_properties",
            long_name="proportion of clay particles (< 0.002 mm) in the fine earth fraction",
            mapped_units="g/kg",
            conversion_factor=10,
            conventional_units="g/100g (mass%)",
            tile_px=450,
            subtiles_per_cell=4,
            rows_per_strip=9,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="nitrogen",
            outputs_version="RUN05",
            transformation="none",
            group="soil_properties",
            long_name="total nitrogen",
            mapped_units="cg/kg",
            conversion_factor=100,
            conventional_units="g/kg",
            tile_px=450,
            subtiles_per_cell=4,
            rows_per_strip=9,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="ocd",
            outputs_version="RUN03",
            transformation="none",
            group="soil_properties",
            long_name="organic carbon density",
            mapped_units="hg/m3",
            conversion_factor=10,
            conventional_units="kg/m3",
            tile_px=450,
            subtiles_per_cell=4,
            rows_per_strip=9,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="phh2o",
            outputs_version="RUN05",
            transformation="none",
            group="soil_properties",
            long_name="soil pH in water",
            mapped_units="pH x 10",
            conversion_factor=10,
            conventional_units="pH",
            tile_px=600,
            subtiles_per_cell=3,
            rows_per_strip=6,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="sand",
            outputs_version="RUN10",
            transformation="alr",
            group="soil_properties",
            long_name="proportion of sand particles (> 0.05 mm) in the fine earth fraction",
            mapped_units="g/kg",
            conversion_factor=10,
            conventional_units="g/100g (mass%)",
            tile_px=450,
            subtiles_per_cell=4,
            rows_per_strip=9,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="silt",
            outputs_version="RUN10",
            transformation="alr",
            group="soil_properties",
            long_name="proportion of silt particles (>= 0.002 mm and <= 0.05 mm) in the fine earth fraction",
            mapped_units="g/kg",
            conversion_factor=10,
            conventional_units="g/100g (mass%)",
            tile_px=450,
            subtiles_per_cell=4,
            rows_per_strip=9,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="soc",
            outputs_version="RUN18",
            transformation="none",
            group="soil_properties",
            long_name="soil organic carbon content in the fine earth fraction",
            mapped_units="dg/kg",
            conversion_factor=10,
            conventional_units="g/kg",
            tile_px=450,
            subtiles_per_cell=4,
            rows_per_strip=9,
            depths=_STANDARD,
        ),
        PropertySpec(
            name="ocs",
            outputs_version="RUN03",
            transformation="none",
            group="profile_properties",
            long_name="organic carbon stocks",
            mapped_units="t/ha",
            conversion_factor=10,
            conventional_units="kg/m2",
            tile_px=450,
            subtiles_per_cell=4,
            rows_per_strip=9,
            depths=(OCS_INTERVAL.label,),
        ),
    ]
}

GRID = GridSpec()
GROUPS: tuple[str, ...] = ("soil_properties", "profile_properties")


def included_properties() -> list[PropertySpec]:
    return [p for p in PROPERTIES.values() if p.status == "included"]


def layers() -> list[tuple[PropertySpec, str]]:
    """Every (property, depth) mean layer in the product."""
    return [(p, d) for p in included_properties() for d in p.depths]


def validate_consistency(
    grid: GridSpec | None = None,
    properties: dict[str, PropertySpec] | None = None,
    encoding: EncodingSpec | None = None,
) -> None:
    """Cross-check grid x properties x encoding.

    Kept out of the model validators on purpose: GridSpec must not reach into the
    module-level PROPERTIES (an import-order coupling, and it blocks shrinking the
    grid in tests). The CLI calls this at startup instead.
    """
    grid = grid or GRID
    properties = properties if properties is not None else PROPERTIES
    encoding = encoding or ENCODING
    cell_px = round(grid.cell_m / grid.pixel_size)
    for p in properties.values():
        if p.tile_px * p.subtiles_per_cell != cell_px:
            raise ValueError(f"{p.name}: {p.subtiles_per_cell} x {p.tile_px} px != {cell_px} px cell")
        if grid.width % p.tile_px or grid.height % p.tile_px:
            raise ValueError(f"grid {grid.width}x{grid.height} is not a whole number of {p.tile_px} px tiles")
        if p.tile_px % encoding.chunk_y or p.tile_px % encoding.chunk_x:
            raise ValueError(f"{p.name}: {p.tile_px} px tiles are not a whole number of chunks")
    if cell_px % encoding.shard_y or cell_px % encoding.shard_x:
        raise ValueError(f"cell {cell_px} px is not a whole number of {encoding.shard_y} px shards")
    if grid.width % encoding.shard_x or grid.height % encoding.shard_y:
        raise ValueError("grid is not a whole number of shards")


# ---------------------------------------------------------------------------
# Attribute constants
# ---------------------------------------------------------------------------

ROOT_ATTRS: dict[str, object] = {
    "title": "ISRIC SoilGrids v2.0 (250 m) as Icechunk Zarr",
    "summary": (
        "Global predictions of eleven soil properties at 250 m resolution for six standard depth "
        "intervals (plus organic carbon stocks for 0-30 cm), reproduced without resampling or "
        "reprojection from the ISRIC SoilGrids v2.0 GeoTIFF tile tree. Mean predictions only."
    ),
    "source": SOURCE_BASE_URL,
    "source_version": SOURCE_VERSION,
    "dataset_version": DATASET_VERSION,
    "institution": "ISRIC - World Soil Information",
    "license": "CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
    "references": (
        "Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B., Ribeiro, E., "
        "and Rossiter, D.: SoilGrids 2.0: producing soil information for the globe with quantified "
        "spatial uncertainty, SOIL, 7, 217-240, 2021. https://doi.org/10.5194/soil-7-217-2021"
    ),
    "processing_code": "https://github.com/colinahill/soilgrids_data",
    "Conventions": "CF-1.10",
    "comment": (
        "Values are stored exactly as published, as scaled integers in the mapped units recorded on "
        "each array. Divide by the array's conversion_factor for conventional units. The grid is "
        "Interrupted Goode Homolosine, which has no EPSG code: use the crs_wkt on the spatial_ref "
        "variable, and transform your coordinates into it before selecting."
    ),
}
