"""config.py invariants: the store's structure is decided here, once."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soilgrids import config


def test_grid_is_a_whole_number_of_cells_and_of_both_subtilings():
    g = config.GRID
    assert g.width == 356 * 450 == 267 * 600
    assert g.height == 132 * 450 == 99 * 600
    assert g.width == (g.max_tile_col + 1) * 1800
    assert g.height == (g.max_tile_row + 1) * 1800


def test_grid_extent_matches_the_measured_tile_lattice():
    g = config.GRID
    assert (g.x_min, g.y_max) == (-20_037_500.0, 8_600_750.0)
    assert g.x_max == 20_012_500.0
    assert g.y_min == -6_249_250.0
    assert g.geotransform == "-20037500.0 250.0 0.0 8600750.0 0.0 -250.0"


def test_coords_are_pixel_centres():
    g = config.GRID
    x, y = g.x_coords(), g.y_coords()
    assert len(x) == g.width and len(y) == g.height
    assert x[0] == g.x_min + 125.0
    assert y[0] == g.y_max - 125.0
    assert y[1] < y[0], "y must descend north to south"
    assert x[-1] == g.x_max - 125.0


def test_every_property_tiles_a_cell_exactly():
    for p in config.PROPERTIES.values():
        assert p.tile_px * p.subtiles_per_cell == 1800
        assert p.tile_px % p.rows_per_strip == 0, "a full tile must have no partial strip"
        assert config.GRID.width % p.tile_px == 0
        assert config.GRID.height % p.tile_px == 0


def test_chunk_divides_both_subtilings_so_one_grid_serves_every_array():
    enc = config.ENCODING
    for tile_px in {p.tile_px for p in config.PROPERTIES.values()}:
        assert tile_px % enc.chunk_y == 0
        assert tile_px % enc.chunk_x == 0
    assert enc.shard_y % enc.chunk_y == 0
    assert enc.shard_x % enc.chunk_x == 0


def test_shard_index_and_object_size_stay_in_budget():
    enc = config.ENCODING
    chunks_per_shard = 6 * (enc.shard_y // enc.chunk_y) * (enc.shard_x // enc.chunk_x)
    assert chunks_per_shard == 486
    assert chunks_per_shard * 16 < 16 * 1024, "shard index must stay small (playbook: ~4-16 kB)"
    raw = 6 * enc.shard_y * enc.shard_x * 2
    assert raw < 64 * 1024**2, "playbook bounds a shard object at 64 MiB"


def test_layer_inventory_is_61_mean_layers():
    layers = config.layers()
    assert len(layers) == 61
    three_d = [p for p in config.included_properties() if p.ndim == 3]
    two_d = [p for p in config.included_properties() if p.ndim == 2]
    assert len(three_d) == 10 and len(two_d) == 1
    assert sum(len(p.depths) for p in three_d) == 60


def test_source_urls_match_the_isric_layout():
    sand = config.PROPERTIES["sand"]
    assert sand.layer_dir("0_5") == "sand_0-5cm_mean"
    assert sand.layer_dir("100_200") == "phh2o_100-200cm_mean".replace("phh2o", "sand")
    assert sand.layer_url("0_5").endswith("/data/sand/sand_0-5cm_mean")
    ocs = config.PROPERTIES["ocs"]
    assert ocs.layer_dir("0_30") == "ocs_0-30cm_mean"


def test_depths_and_dims():
    assert config.DEPTH_LABELS == ["0_5", "5_15", "15_30", "30_60", "60_100", "100_200"]
    assert config.PROPERTIES["sand"].dims == ("depth_interval", "y", "x")
    assert config.PROPERTIES["ocs"].dims == ("y", "x")
    assert config.PROPERTIES["ocs"].depth_extent == 1
    assert config.PROPERTIES["sand"].depth_extent == 6


def test_units_table_is_complete_and_plausible():
    for p in config.PROPERTIES.values():
        assert p.mapped_units and p.conventional_units and p.long_name
        assert p.conversion_factor in (10, 100)


def test_encoding_rejects_a_shard_that_is_not_a_chunk_multiple():
    with pytest.raises(ValidationError, match="multiple of the chunk shape"):
        config.EncodingSpec(chunk_y=50, chunk_x=50, shard_y=451, shard_x=450)


def test_config_is_self_consistent():
    config.validate_consistency()


def test_consistency_check_rejects_a_subtiling_that_does_not_fill_a_cell():
    bad = {"sand": config.PropertySpec.model_validate(config.PROPERTIES["sand"].model_dump() | {"tile_px": 405})}
    with pytest.raises(ValueError, match="!= 1800 px cell"):
        config.validate_consistency(properties=bad)


def test_consistency_check_rejects_a_chunk_that_does_not_divide_the_tile():
    with pytest.raises(ValueError, match="not a whole number of chunks"):
        config.validate_consistency(encoding=config.EncodingSpec(chunk_y=64, chunk_x=64, shard_y=448, shard_x=448))


def test_property_rejects_a_partial_strip_on_a_full_tile():
    with pytest.raises(ValidationError, match="whole number of"):
        config.PropertySpec.model_validate(config.PROPERTIES["sand"].model_dump() | {"rows_per_strip": 7})


def test_overview_ladder_is_powers_of_two():
    assert config.OVERVIEW_FACTORS == (2, 4, 8, 16, 32, 64, 128, 256)
    assert config.OVERVIEW_RESAMPLING == "mean"


def test_store_holds_decoded_physical_units_with_nan_fill():
    """The consumer-facing contract: physical units, NaN, nothing to convert."""
    from soilgrids import metadata

    assert config.DTYPE == "float32"
    assert config.is_fill(config.FILL_VALUE)
    assert config.SOURCE_DTYPE == "int16" and config.SOURCE_NODATA == -32768
    for spec in config.included_properties():
        a = metadata.property_attrs(spec)
        assert a["units"] == spec.conventional_units, "units name what you read"
        # no CF packing attrs: there is nothing left to unpack
        assert "scale_factor" not in a
        assert "_FillValue" not in a
        # but the source encoding stays recorded as provenance
        assert a["source_mapped_units"] == spec.mapped_units
        assert a["source_conversion_factor"] == spec.conversion_factor
        assert a["source_nodata"] == config.SOURCE_NODATA


def test_is_fill_is_nan_aware_and_tolerates_numpy_scalars():
    import numpy as np

    assert config.is_fill(float("nan"))
    assert config.is_fill(np.float32("nan"))
    assert not config.is_fill(0.0)
    assert not config.is_fill(-32768)
    assert not config.is_fill(None)


def test_measured_conversion_factors():
    """Settled against real data; two widely-circulated tables get these wrong."""
    f = {p.name: p.conversion_factor for p in config.included_properties()}
    # cfvo is cm3/dm3 (per-mille) -> vol %, so 10, not 100
    assert f["cfvo"] == 10
    # ocd is hg/m3 -> kg/m3, so 10 (kg/dm3 would be 44 tonnes per litre)
    assert f["ocd"] == 10
    assert f["nitrogen"] == f["bdod"] == 100
    assert f["sand"] == f["silt"] == f["clay"] == f["phh2o"] == f["soc"] == 10
