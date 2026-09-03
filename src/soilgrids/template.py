"""Create the empty store structure: groups, coords, attrs, and arrays.

Only metadata and the small 1-D coordinate arrays are written; data arrays are
created empty (fill-value chunks occupy no storage) and filled later by
materialize / overviews. init is idempotent and ADDITIVE: on an existing store it
creates only what is missing, so a newly included property extends a published
store without touching existing data.

It refuses outright if an existing array was created under a different
EncodingSpec: writes are shard-aligned, so mixing shard grids in one store would
let two workers write the same storage object.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
import zarr
from icechunk import Session
from icechunk.xarray import to_icechunk

from . import config, grid, metadata

log = logging.getLogger(__name__)


class EncodingMismatch(RuntimeError):
    """An existing array has a different chunk/shard grid than config asks for."""


def _group_exists(session: Session, path: str) -> bool:
    try:
        zarr.open_group(session.store, path=path, mode="r")
        return True
    except (KeyError, FileNotFoundError):
        return False


def coords_dataset(
    *, with_depth: bool, shape: tuple[int, int] | None = None, pixel_size: float | None = None
) -> xr.Dataset:
    """Coordinates + CRS for one group or overview level."""
    g = config.GRID
    px = g.pixel_size if pixel_size is None else pixel_size
    h, w = shape if shape is not None else (g.height, g.width)
    ca = metadata.coordinate_attrs()
    coords: dict = {
        "y": ("y", g.y_max - (np.arange(h) + 0.5) * px, ca["y"]),
        "x": ("x", g.x_min + (np.arange(w) + 0.5) * px, ca["x"]),
        "spatial_ref": ((), np.int64(0), metadata.spatial_ref_attrs(px)),
    }
    if with_depth:
        coords.update(metadata.depth_coords())
    return xr.Dataset(coords=coords)


def verify_array_encoding(arr: zarr.Array, spec: config.PropertySpec, *, depth_extent: int) -> None:
    want_chunks = config.ENCODING.chunks(spec.ndim)
    want_shards = config.ENCODING.shards(spec.ndim, depth_extent=depth_extent)
    if tuple(arr.chunks) != want_chunks or tuple(arr.shards or ()) != want_shards:
        raise EncodingMismatch(
            f"{arr.path}: existing chunks/shards {tuple(arr.chunks)}/{tuple(arr.shards or ())} "
            f"!= config {want_chunks}/{want_shards}. Shard-aligned writes make mixing shard "
            f"grids unsafe: bump DATASET_VERSION for a fresh store path, or wipe this one."
        )


def _create_array(
    group: zarr.Group,
    spec: config.PropertySpec,
    shape: tuple[int, ...],
    *,
    depth_extent: int,
    factor: int,
) -> None:
    group.create_array(
        spec.name,
        shape=shape,
        chunks=config.ENCODING.chunks(spec.ndim),
        shards=config.ENCODING.shards(spec.ndim, depth_extent=depth_extent),
        dtype=config.DTYPE,
        fill_value=config.FILL_VALUE,
        compressors=[zarr.codecs.ZstdCodec(level=config.ENCODING.zstd_level)],
        dimension_names=spec.dims,
        attributes=metadata.property_attrs(spec, factor=factor),
    )


def init_store(
    session: Session,
    properties: list[config.PropertySpec] | None = None,
    *,
    factors: tuple[int, ...] | None = None,
) -> list[str]:
    """Create/extend the full structure. No commit. Returns created array paths."""
    config.validate_consistency()
    properties = properties if properties is not None else config.included_properties()
    factors = config.OVERVIEW_FACTORS if factors is None else factors
    g = config.GRID
    shapes = grid.overview_shapes(factors=factors) if factors else {}
    created: list[str] = []

    for group_name in config.GROUPS:
        with_depth = group_name == "soil_properties"
        if not _group_exists(session, group_name):
            to_icechunk(coords_dataset(with_depth=with_depth), session, group=group_name, mode="w")
        gp = zarr.open_group(session.store, path=group_name, mode="r+")
        gp.attrs.update(
            metadata.group_attrs(group_name)
            | metadata.geozarr_attrs((g.height, g.width), g.pixel_size)
            | (metadata.multiscales_attrs(factors) if factors else {})
        )

        # native level
        for spec in (p for p in properties if p.group == group_name):
            if spec.name in gp.array_keys():
                verify_array_encoding(gp[spec.name], spec, depth_extent=spec.depth_extent)
                continue
            shape = (spec.depth_extent, g.height, g.width) if spec.ndim == 3 else (g.height, g.width)
            _create_array(gp, spec, shape, depth_extent=spec.depth_extent, factor=1)
            created.append(f"{group_name}/{spec.name}")

        # overview levels: depth-1 shards, so a level streams one (property,
        # depth) layer at a time and a map tile read touches one depth
        for factor in factors:
            h, w = shapes[factor]
            path = f"{group_name}/{factor}x"
            if not _group_exists(session, path):
                to_icechunk(
                    coords_dataset(with_depth=with_depth, shape=(h, w), pixel_size=g.pixel_size * factor),
                    session,
                    group=path,
                    mode="w",
                )
            lg = zarr.open_group(session.store, path=path, mode="r+")
            lg.attrs.update(
                metadata.geozarr_attrs((h, w), g.pixel_size * factor)
                | {
                    "overview_factor": factor,
                    "derived_from": "." if factor == factors[0] else f"{factors[factors.index(factor) - 1]}x",
                }
            )
            for spec in (p for p in properties if p.group == group_name):
                if spec.name in lg.array_keys():
                    verify_array_encoding(lg[spec.name], spec, depth_extent=1)
                    continue
                shape = (spec.depth_extent, h, w) if spec.ndim == 3 else (h, w)
                _create_array(lg, spec, shape, depth_extent=1, factor=factor)
                created.append(f"{path}/{spec.name}")

    root = zarr.open_group(session.store, mode="a")
    root.attrs.update(config.ROOT_ATTRS | metadata.region_note())
    if created:
        log.info("created %d arrays", len(created))
    return created
