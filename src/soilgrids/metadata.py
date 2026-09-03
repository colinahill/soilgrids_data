"""CF / GeoZarr attribute construction: coordinates, CRS, properties, levels."""

from __future__ import annotations

import numpy as np

from . import config, grid
from .config import DEPTH_INTERVALS, GridSpec, PropertySpec

# The Interrupted Goode Homolosine WKT read from the source tiles is the
# authority: IGH has no EPSG code, so there is nothing else to appeal to.
# catalog/validate compare this against what the tiles actually carry.


def crs():
    from pyproj import CRS

    return CRS.from_proj4(config.GRID.proj4)


def spatial_ref_attrs(pixel_size: float | None = None) -> dict:
    """CF grid-mapping attrs for the scalar spatial_ref variable (rioxarray style)."""
    c = crs()
    attrs = c.to_cf()
    attrs["spatial_ref"] = attrs.get("crs_wkt", c.to_wkt())  # rioxarray compatibility
    g = config.GRID
    px = g.pixel_size if pixel_size is None else pixel_size
    attrs["GeoTransform"] = f"{g.x_min} {px} 0.0 {g.y_max} 0.0 {-px}"
    attrs["comment"] = (
        "Interrupted Goode Homolosine. This CRS has no EPSG code: use crs_wkt (or the "
        "proj4 string in the root attrs) and transform your coordinates into it before "
        "selecting. Longitude is not a function of x alone, so no lon/lat coordinates "
        "are stored."
    )
    return attrs


def coordinate_attrs() -> dict[str, dict]:
    return {
        "x": {
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection (pixel centre)",
            "units": "m",
            "axis": "X",
        },
        "y": {
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection (pixel centre)",
            "units": "m",
            "axis": "Y",
        },
    }


def depth_coords() -> dict[str, tuple[tuple[str, ...], np.ndarray, dict]]:
    """The string depth_interval coordinate plus its auxiliary coordinates."""
    dim = (config.DEPTH_DIM,)
    return {
        config.DEPTH_DIM: (
            dim,
            np.array([d.label for d in DEPTH_INTERVALS]),
            {
                "long_name": "soil depth interval",
                "comment": (
                    "The six standard SoilGrids intervals. Labels are '{top}_{bottom}' in cm; "
                    "see depth_top_cm / depth_bottom_cm. Organic carbon stocks (ocs) are "
                    "published only for 0-30 cm and live in the profile_properties group."
                ),
            },
        ),
        "depth_top_cm": (
            dim,
            np.array([d.top_cm for d in DEPTH_INTERVALS], "int16"),
            {"long_name": "interval top", "units": "cm"},
        ),
        "depth_bottom_cm": (
            dim,
            np.array([d.bottom_cm for d in DEPTH_INTERVALS], "int16"),
            {"long_name": "interval bottom", "units": "cm"},
        ),
    }


def property_attrs(spec: PropertySpec, *, factor: int = 1, extra: dict | None = None) -> dict:
    """Zarr array attrs for one property, built from its spec.

    The array holds decoded float32 in conventional physical units with NaN for
    no data, so there is deliberately **no** CF ``scale_factor``: nothing needs
    unpacking. That matters beyond convenience -- CF decoding is an xarray
    convention, not a zarr one, so a browser client (icechunk-js, zarr-layer)
    reading the multiscale pyramid would render packed integers as if they were
    physical values. Storing decoded units makes every client correct by default.

    ``mapped_units`` / ``conversion_factor`` are kept as provenance: they record
    what ISRIC published and how this array was derived from it.
    """
    attrs: dict = {
        "long_name": spec.long_name,
        "units": spec.conventional_units,
        # No CF _FillValue: the array is float32 and NaN *is* the missing value,
        # so nothing needs masking. Writing it as the string "NaN" also breaks
        # xarray's decode path (it reaches zarr's base64 fill parser).
        "comment": (
            f"Conventional units ({spec.conventional_units}), NaN where there is no data. "
            f"ISRIC publishes this property as Int16 in {spec.mapped_units}; this array is "
            f"that value divided by {spec.conversion_factor}. The conversion is exactly "
            f"invertible, so the store is a faithful mirror of the source."
        ),
        "source_mapped_units": spec.mapped_units,
        "source_conversion_factor": spec.conversion_factor,
        "source_nodata": config.SOURCE_NODATA,
        "statistic": "mean",
        "source_version": config.SOURCE_VERSION,
        # per-property model provenance, measured from the source VRTs: ISRIC
        # built the properties in different runs
        "source_outputs_version": spec.outputs_version,
        "source_transformation": config.normalise_transformation(spec.transformation),
        "source_layers": [spec.layer_dir(d) for d in spec.depths],
        "grid_mapping": "spatial_ref",
        "coordinates": "spatial_ref",
    }
    if spec.ndim == 2:
        interval = config.OCS_INTERVAL
        attrs["depth_interval"] = interval.label
        attrs["depth_top_cm"] = interval.top_cm
        attrs["depth_bottom_cm"] = interval.bottom_cm
    if factor > 1:
        idx = config.OVERVIEW_FACTORS.index(factor)
        attrs["overview_factor"] = factor
        attrs["resampling_method"] = config.OVERVIEW_RESAMPLING
        attrs["derived_from"] = "." if idx == 0 else f"{config.OVERVIEW_FACTORS[idx - 1]}x"
        attrs["comment"] += (
            f" This is a {factor}x coarsened overview, produced by chained stride-2 "
            f"NaN-aware {config.OVERVIEW_RESAMPLING} from the "
            f"{attrs['derived_from'].replace('.', 'native')} level. Mean-of-means with "
            f"unequal valid counts is not identical to a direct mean from native; use the "
            f"native arrays for analysis and the overviews for display."
        )
    if extra:
        attrs.update(extra)
    return attrs


def group_attrs(group: str) -> dict:
    if group == "soil_properties":
        return {
            "long_name": "soil properties on the six standard SoilGrids depth intervals",
            "depth_intervals": config.DEPTH_LABELS,
        }
    return {
        "long_name": "whole-profile soil properties (single fixed interval per variable)",
        "depth_intervals": [config.OCS_INTERVAL.label],
    }


def geozarr_attrs(shape: tuple[int, int], pixel_size: float) -> dict:
    """GeoZarr ``proj:`` / ``spatial:`` attrs for one group or level.

    Built with topozarr's helper so the emitted conventions stay in step with the
    spec it tracks, then corrected: topozarr sets ``proj:code`` to the proj4
    string when the CRS has no authority code, but ``proj:code`` is for authority
    codes only, so it is dropped and ``proj:wkt2`` carries the CRS.
    """
    import xarray as xr
    import xproj  # noqa: F401  (registers the .proj accessor)
    from topozarr.geozarr import create_geozarr_metadata

    g = config.GRID
    h, w = shape
    ds = xr.Dataset(
        {"_": (("y", "x"), np.zeros((min(h, 2), min(w, 2)), "int16"))},
        coords={
            "y": g.y_max - (np.arange(min(h, 2)) + 0.5) * pixel_size,
            "x": g.x_min + (np.arange(min(w, 2)) + 0.5) * pixel_size,
        },
    ).proj.assign_crs(spatial_ref=crs())
    attrs = dict(create_geozarr_metadata(ds, "x", "y", g.proj4))
    if crs().to_epsg() is None:
        attrs.pop("proj:code", None)
    attrs["spatial:shape"] = [h, w]
    attrs["spatial:transform"] = [pixel_size, 0.0, g.x_min, 0.0, -pixel_size, g.y_max]
    attrs["spatial:bbox"] = [g.x_min, g.y_max - h * pixel_size, g.x_min + w * pixel_size, g.y_max]
    return attrs


def multiscales_attrs(factors: tuple[int, ...] | None = None) -> dict:
    """zarr-conventions/multiscales layout: native as ".", then each level.

    Parent/child layout (native at the group root, levels in factor-named
    children) so adding or rebuilding the pyramid never moves native data.
    Ordered fine -> coarse by the layout array, never by name: "16x" sorts
    before "2x".

    ``derived_from`` names the level each one was actually computed from -- the
    previous level, because the ladder is a chained stride-2 reduction. Stating
    "." would claim every level came from native, which would misrepresent the
    (bounded) mean-of-means error.
    """
    factors = factors or config.OVERVIEW_FACTORS
    shapes = grid.overview_shapes(factors=factors)
    g = config.GRID
    layout = [{"asset": ".", "resolution": [g.pixel_size, g.pixel_size]}]
    for i, f in enumerate(factors):
        h, w = shapes[f]
        parent = "." if i == 0 else f"{factors[i - 1]}x"
        layout.append(
            {
                "asset": f"{f}x",
                "derived_from": parent,
                "resampling_method": config.OVERVIEW_RESAMPLING,
                "transform": {
                    "scale": [float(f) if parent == "." else 2.0] * 2,
                    "translation": [0.0, 0.0],
                },
                "resolution": [g.pixel_size * f, g.pixel_size * f],
                "shape": [h, w],
            }
        )
    return {"multiscales": {"layout": layout}}


def region_note(g: GridSpec | None = None) -> dict:
    g = g or config.GRID
    return {
        "grid_width": g.width,
        "grid_height": g.height,
        "grid_pixel_size_m": g.pixel_size,
        "grid_origin_xy": [g.x_min, g.y_max],
        "grid_crs_proj4": g.proj4,
    }
