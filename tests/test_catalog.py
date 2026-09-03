"""catalog.py: the VRT is the inventory, the tile lattice is the cross-check."""

from __future__ import annotations

import re

import pytest

from soilgrids import catalog, config, grid
from soilgrids.fetch import parse_listing

# A minimal but structurally faithful VRT: same element shapes as ISRIC's, and
# the same VRT-frame-to-canonical offset (sand starts 351 px east, 959 px south).
VRT_X0, VRT_Y0 = -19_949_750.0, 8_361_000.0
DX, DY = 351, 959


def _vrt(
    sources: str,
    *,
    dtype: str = "Int16",
    nodata: str = "-32768",
    outputs: str = "RUN10",
    transform: str = "alr",
    code: str = "v2.0.0",
    wosis: str = "Data stream 7",
    x0: float = VRT_X0,
    y0: float = VRT_Y0,
) -> str:
    return f"""<VRTDataset rasterXSize="159246" rasterYSize="58034">
  <SRS>PROJCS["Interrupted_Goode_Homolosine"]</SRS>
  <GeoTransform>{x0}, 250.0, 0.0, {y0}, 0.0, -250.0</GeoTransform>
  <Metadata>
    <MDI key="Code_version">{code}</MDI>
    <MDI key="Outputs_version">{outputs}</MDI>
    <MDI key="Transformation">{transform}</MDI>
    <MDI key="WoSIS_version">{wosis}</MDI>
  </Metadata>
  <VRTRasterBand dataType="{dtype}" band="1">
    <NoDataValue>{nodata}</NoDataValue>
{sources}
  </VRTRasterBand>
</VRTDataset>"""


def _source(cell: str, sub: str, w: int, h: int, dx: float, dy: float, block_y: int = 9) -> str:
    return f"""    <ComplexSource>
      <SourceFilename relativeToVRT="1">./sand_0-5cm_mean/{cell}/{cell}_{sub}.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{w}" RasterYSize="{h}" DataType="Int16" BlockXSize="{w}" BlockYSize="{block_y}" />
      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}" />
      <DstRect xOff="{dx}" yOff="{dy}" xSize="{w}" ySize="{h}" />
      <NODATA>-32768</NODATA>
    </ComplexSource>"""


def _full_source(row: int, col: int, r: int, c: int, tile_px: int = 450) -> str:
    """A source placed exactly where the tile lattice says it belongs."""
    ref = grid.TileRef(row, col, r, c, tile_px)
    x, y = ref.pixel_offset
    cell = ref.cell
    return _source(cell, f"{r}-{c}", tile_px, tile_px, x - DX, y - DY)


SAND = config.PROPERTIES["sand"]


def _inv(xml: str, spec=SAND, depth: str = "0_5") -> catalog.LayerInventory:
    return catalog.inventory_layer(None, spec, depth, xml=xml)


def test_parses_header_provenance_and_sources():
    header, sources = catalog.parse_vrt(_vrt(_full_source(18, 27, 1, 1)))
    assert header["dtype"] == "Int16"
    assert header["nodata"] == -32768.0
    assert header["metadata"]["Outputs_version"] == "RUN10"
    assert len(sources) == 1
    assert sources[0].relpath == "tileSG-018-027/tileSG-018-027_1-1.tif"
    assert (sources[0].width, sources[0].height) == (450, 450)


def test_vrt_frame_offset_matches_the_measured_value():
    header, _ = catalog.parse_vrt(_vrt(_full_source(18, 27, 1, 1)))
    assert catalog.vrt_to_canonical(header) == (DX, DY)


def test_full_tiles_are_cross_checked_against_the_lattice():
    xml = _vrt("\n".join(_full_source(18, 27, r, c) for r, c in grid.subtiles(450)))
    rows = catalog.tile_rows(_inv(xml), SAND)
    assert len(rows) == 16
    assert all(r["is_full"] for r in rows)
    for row in rows:
        ref = grid.parse_tile_name(row["relpath"].split("/")[1], 450)
        assert (row["x_off"], row["y_off"]) == ref.pixel_offset


def test_a_misplaced_full_tile_is_recorded_not_fatal():
    """Measured upstream: bdod's tileSG-015-023_3-3 sits one subtile west of its name.

    The VRT agrees with the file's own tie-point, so it is the NAME that is
    misleading, and names were never the authority for placement. Record it and
    keep the tie-point placement.
    """
    ref = grid.TileRef(18, 27, 3, 3, 450)
    x, y = ref.pixel_offset
    xml = _vrt(_source(ref.cell, "3-3", 450, 450, x - DX - 450, y - DY))
    inv = _inv(xml)
    rows = catalog.tile_rows(inv, SAND)
    assert rows[0]["x_off"] == x - 450, "placement follows the VRT/tie-point, not the name"
    assert any("its name implies" in a for a in inv.anomalies)


def test_two_tiles_claiming_one_window_are_recorded_and_resolved():
    """Measured: nitrogen/0_5 has a byte-identical duplicate with a wrong name."""
    ref = grid.TileRef(17, 54, 1, 1, 450)
    x, y = ref.pixel_offset
    xml = _vrt(
        _source(ref.cell, "1-1", 450, 450, x - DX, y - DY) + "\n" + _source(ref.cell, "1-2", 450, 450, x - DX, y - DY)
    )
    inv = _inv(xml)
    rows = catalog.tile_rows(inv, SAND)
    assert len(rows) == 1, "exactly one row per window; writing both would race"
    assert len(inv.conflicts) == 1
    c = inv.conflicts[0]
    assert c.keep.endswith("_1-1.tif"), "the tile whose name matches the window wins"
    assert c.drop.endswith("_1-2.tif")
    assert rows[0]["relpath"] == c.keep


def test_the_kept_tile_is_the_same_whichever_order_the_vrt_lists_them():
    """The choice must not depend on VRT ordering, or re-runs would disagree."""
    ref = grid.TileRef(17, 54, 1, 1, 450)
    x, y = ref.pixel_offset
    a = _source(ref.cell, "1-1", 450, 450, x - DX, y - DY)
    b = _source(ref.cell, "1-2", 450, 450, x - DX, y - DY)
    keeps = []
    for xml in (_vrt(a + "\n" + b), _vrt(b + "\n" + a)):
        inv = _inv(xml)
        rows = catalog.tile_rows(inv, SAND)
        keeps.append((rows[0]["relpath"], inv.conflicts[0].keep))
    assert keeps[0] == keeps[1]
    assert keeps[0][0].endswith("_1-1.tif")


def test_a_tile_outside_the_canonical_grid_is_refused():
    # ragged, so it skips the lattice check and reaches the containment check
    xml = _vrt(_source("tileSG-000-000", "1-1", 100, 100, -DX - 10, -DY))
    with pytest.raises(catalog.SourceError, match="does not fit the canonical grid"):
        catalog.tile_rows(_inv(xml), SAND)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"dtype": "Float32"}, "dataType"),
        ({"nodata": "-9999"}, "NoDataValue"),
        ({"code": "v2.1.0"}, "Code_version"),
        ({"wosis": "Data stream 8"}, "WoSIS_version"),
        ({"outputs": "RUN18"}, "Outputs_version"),  # sand is RUN10; soc is RUN18
        ({"transform": ""}, "Transformation"),  # sand is alr; losing it is real drift
    ],
)
def test_source_drift_is_refused(kwargs, message):
    """`latest/` is mutable: any provenance or encoding change must stop the run."""
    xml = _vrt(_full_source(18, 27, 1, 1), **kwargs)
    with pytest.raises(catalog.SourceError, match=message):
        _inv(xml)


@pytest.mark.parametrize("spelling", ["", "   ", "None", "none"])
def test_absent_or_empty_transformation_all_mean_the_same_thing(spelling):
    """Upstream spells "no transform" three ways; a change between them is not drift.

    Measured: the MDI element is absent for cfvo/nitrogen/ocd/soc/ocs, present
    but empty for bdod/cec/phh2o, and carries "alr" for clay/sand/silt.
    """
    cfvo = config.PROPERTIES["cfvo"]
    assert cfvo.transformation == "none"
    ref = grid.TileRef(18, 27, 1, 1, cfvo.tile_px)
    x, y = ref.pixel_offset
    src = _source(ref.cell, "1-1", cfvo.tile_px, cfvo.tile_px, x - DX, y - DY)
    xml = _vrt(src, outputs=cfvo.outputs_version, transform=spelling)
    inv = catalog.inventory_layer(None, cfvo, "0_5", xml=xml)  # must not raise
    assert catalog.tile_rows(inv, cfvo)[0]["is_full"]


def test_a_missing_transformation_element_is_also_accepted():
    """cfvo's VRT has no Transformation element at all -- the case that aborted."""
    cfvo = config.PROPERTIES["cfvo"]
    ref = grid.TileRef(18, 27, 1, 1, cfvo.tile_px)
    x, y = ref.pixel_offset
    xml = _vrt(
        _source(ref.cell, "1-1", cfvo.tile_px, cfvo.tile_px, x - DX, y - DY),
        outputs=cfvo.outputs_version,
    )
    xml = re.sub(r'\s*<MDI key="Transformation">[^<]*</MDI>', "", xml)
    assert "Transformation" not in xml
    catalog.inventory_layer(None, cfvo, "0_5", xml=xml)  # must not raise


def test_per_property_provenance_is_accepted_for_the_right_property():
    """phh2o is RUN05 with no transform: a global RUN10 expectation would reject it."""
    phh2o = config.PROPERTIES["phh2o"]
    ref = grid.TileRef(18, 27, 1, 1, 600)
    x, y = ref.pixel_offset
    xml = _vrt(_source(ref.cell, "1-1", 600, 600, x - DX, y - DY, block_y=6), outputs="RUN05", transform="")
    inv = catalog.inventory_layer(None, phh2o, "0_5", xml=xml)
    rows = catalog.tile_rows(inv, phh2o)
    assert rows[0]["is_full"] and rows[0]["rows_per_strip"] == 6


def test_resampled_sources_are_flagged_not_fatal():
    """The real sand VRT contains a 1x2 tile stretched to 62x124."""
    ref = grid.TileRef(10, 49, 1, 1, 450)
    x, y = ref.pixel_offset
    xml = _vrt(f"""    <ComplexSource>
      <SourceFilename relativeToVRT="1">./sand_0-5cm_mean/{ref.cell}/{ref.cell}_1-1.tif</SourceFilename>
      <SourceProperties RasterXSize="1" RasterYSize="2" DataType="Int16" BlockXSize="1" BlockYSize="2" />
      <SrcRect xOff="0" yOff="0" xSize="1" ySize="2" />
      <DstRect xOff="{x - DX}" yOff="{y - DY}" xSize="62" ySize="124" />
    </ComplexSource>""")
    inv = _inv(xml)
    assert any("resamples" in a for a in inv.anomalies)
    assert inv.sources[0].resampled


def test_listing_crosscheck_spots_a_tile_missing_from_the_vrt():
    xml = _vrt(_full_source(18, 27, 1, 1))
    rows = catalog.tile_rows(_inv(xml), SAND)
    html = """<tr><td>tileSG-018-027_1-1.tif</td><td>TIF-File</td><td>406,513 Bytes</td><td>x</td></tr>
              <tr><td>tileSG-018-027_1-2.tif</td><td>TIF-File</td><td>406,513 Bytes</td><td>x</td></tr>"""
    disk_only, vrt_only = catalog.check_listing(rows, "tileSG-018-027", html)
    assert disk_only == ["tileSG-018-027_1-2.tif"] and vrt_only == []
    assert rows[0]["size"] == 406513, "real sizes are recorded from the listing"


def test_listing_crosscheck_spots_a_vrt_tile_missing_from_disk():
    xml = _vrt(_full_source(18, 27, 1, 1))
    rows = catalog.tile_rows(_inv(xml), SAND)
    disk_only, vrt_only = catalog.check_listing(
        rows,
        "tileSG-018-027",
        "<tr><td>checksum.sha256.txt</td><td>TXT-File</td><td>10 Bytes</td><td>x</td></tr>",
    )
    assert disk_only == [] and vrt_only == ["tileSG-018-027_1-1.tif"]


def test_listing_parser_handles_directories_and_sizes():
    html = """<tr class="directory"><td><a>tileSG-000-019</a></td><td>Directory</td><td>-</td><td>x</td></tr>
              <tr><td>a.tif</td><td>TIF-File</td><td>1,234,567 Bytes</td><td>x</td></tr>"""
    assert parse_listing(html) == {"tileSG-000-019": None, "a.tif": 1234567}


def test_empty_vrt_is_refused():
    with pytest.raises(catalog.SourceError, match="lists no sources"):
        catalog.parse_vrt(_vrt(""))


def test_manifest_schema_is_declared_not_inferred():
    """size/etag are populated only for sampled rows, so inference sees Nulls."""
    schema = catalog._manifest_schema()
    assert set(schema) >= {"property", "depth", "x_off", "y_off", "is_full", "size", "etag"}
    xml = _vrt("\n".join(_full_source(18, 27, r, c) for r, c in grid.subtiles(450)))
    rows = catalog.tile_rows(_inv(xml), SAND)
    rows[0]["size"] = 406513  # one populated row among Nulls: the case that broke
    import polars as pl

    df = pl.DataFrame(rows, schema=schema)
    assert df.height == 16
    assert df["size"].null_count() == 15


def test_identical_duplicates_resolve_and_differing_ones_raise():
    ref = grid.TileRef(17, 54, 1, 1, 450)
    keep = f"{ref.cell}/{ref.cell}_1-1.tif"
    drop = f"{ref.cell}/{ref.cell}_1-2.tif"
    conflict = [catalog.Conflict(window=ref.pixel_offset, keep=keep, drop=drop)]
    same = _tile_bytes(17, 54, 1, 1, 450)

    notes = catalog.resolve_conflicts(_HeaderSource({keep: same, drop: same}), SAND, "0_5", conflict)
    assert len(notes) == 1 and "byte-identical duplicate" in notes[0]

    other = _tile_bytes(17, 54, 1, 1, 450, fill=999)
    with pytest.raises(catalog.SourceError, match="pixels DIFFER"):
        catalog.resolve_conflicts(_HeaderSource({keep: same, drop: other}), SAND, "0_5", conflict)


class _HeaderSource:
    """Serves synthetic tile headers, standing in for the WebDAV tree."""

    workers = 1

    def __init__(self, tiles):
        self.tiles = tiles

    def _key(self, url):
        return url.rsplit("/", 2)[-2] + "/" + url.rsplit("/", 1)[-1]

    def get_range(self, url, start, length):
        return self.tiles[self._key(url)][start : start + length]

    def get(self, url, **kw):
        return self.tiles[self._key(url)]

    def map(self, fn, items, **kw):
        return [fn(i) for i in items]


def _tile_bytes(row, col, r, c, tile_px, size=None, offset=(0, 0), fill=300):
    import numpy as np

    from .conftest import build_tile

    g = config.GRID
    ref = grid.TileRef(row, col, r, c, tile_px)
    x, y = ref.pixel_offset
    x += offset[0]
    y += offset[1]
    h, w = size or (tile_px, tile_px)
    return build_tile(
        np.full((h, w), fill, "int16"),
        (g.x_min + x * g.pixel_size, g.y_max - y * g.pixel_size),
    )


def test_repair_recovers_a_tile_the_vrt_omitted():
    """Measured: 8 such tiles per bdod layer hold ~26,000 km2 nothing else covers."""
    xml = _vrt(_full_source(15, 23, 2, 1))
    rows = catalog.tile_rows(_inv(xml), SAND)
    missing = "tileSG-015-023/tileSG-015-023_3-1.tif"
    src = _HeaderSource({missing: _tile_bytes(15, 23, 3, 1, 450)})
    extra, notes = catalog.repair_from_disk(src, SAND, "0_5", rows, {"tileSG-015-023": ["tileSG-015-023_3-1.tif"]})
    assert len(extra) == 1
    assert (extra[0]["x_off"], extra[0]["y_off"]) == grid.TileRef(15, 23, 3, 1, 450).pixel_offset
    assert extra[0]["is_full"] and extra[0]["property"] == "sand"
    assert any("recovered" in n for n in notes)


def test_repair_skips_an_off_lattice_tile():
    """One such tile exists in bdod/0_5; it cannot be placed at all."""
    xml = _vrt(_full_source(15, 23, 2, 1))
    rows = catalog.tile_rows(_inv(xml), SAND)
    name = "tileSG-015-023/tileSG-015-023_3-1.tif"
    src = _HeaderSource({name: _tile_bytes(15, 23, 3, 1, 450, size=(2, 1), offset=(0.5, 0.5))})
    extra, notes = catalog.repair_from_disk(src, SAND, "0_5", rows, {"tileSG-015-023": ["tileSG-015-023_3-1.tif"]})
    assert extra == []
    assert any("off the 250 m lattice" in n for n in notes)


def test_repair_keeps_the_vrt_tile_when_a_recovered_one_duplicates_it():
    """Deterministic by construction: the VRT always wins, so re-runs agree."""
    xml = _vrt(_full_source(15, 23, 3, 3))
    rows = catalog.tile_rows(_inv(xml), SAND)
    dup = "tileSG-015-023/tileSG-015-023_3-2.tif"
    src = _HeaderSource({dup: _tile_bytes(15, 23, 3, 3, 450)})
    extra, notes = catalog.repair_from_disk(src, SAND, "0_5", rows, {"tileSG-015-023": ["tileSG-015-023_3-2.tif"]})
    assert extra == []
    assert any("duplicates the window" in n and "keeping the VRT" in n for n in notes)
