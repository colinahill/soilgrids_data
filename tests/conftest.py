"""Shared fixtures: synthetic SoilGrids tiles and a shrunken canonical grid.

Tests never touch the network. The grid is shrunk (playbook: keep the
multi-chunk / multi-shard code paths hot at toy scale) by scaling the cell and
tile sizes together, so the real 4x4-of-450 and 3x3-of-600 subtilings both stay
exercised.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from soilgrids import config

NODATA = config.SOURCE_NODATA  # the source sentinel, for building tiles
FILL = config.FILL_VALUE  # what the store holds where there is no data


def build_tile(
    values: np.ndarray,
    tiepoint: tuple[float, float],
    *,
    rows_per_strip: int = 9,
    pixel_size: float = 250.0,
    nodata: int | None = NODATA,
    byteorder: str = "<",
    compression: int = 1,
    header_pad: int = 0,
) -> bytes:
    """Build a striped, uncompressed, single-band Int16 GeoTIFF like ISRIC's.

    Layout mirrors the real files: IFD first, then the strips back to back, so
    the pixel data is one contiguous range at the end of the file.
    """
    values = np.ascontiguousarray(values, dtype=byteorder + "i2")
    h, w = values.shape
    n_strips = -(-h // rows_per_strip)
    counts = [min(rows_per_strip, h - i * rows_per_strip) * w * 2 for i in range(n_strips)]

    nodata_bytes = (str(nodata) + "\0").encode() if nodata is not None else b""
    entries: list[tuple[int, int, int, bytes]] = []  # (tag, type, count, inline-or-external)
    external = bytearray()

    def add(tag: int, typ: int, vals) -> None:
        if typ == 2:
            payload = vals if isinstance(vals, bytes) else (str(vals) + "\0").encode()
            entries.append((tag, 2, len(payload), payload))
            return
        fmt = {1: "B", 3: "H", 4: "I", 12: "d"}[typ]
        payload = struct.pack(byteorder + fmt * len(vals), *vals)
        entries.append((tag, typ, len(vals), payload))

    add(256, 3, [w])
    add(257, 3, [h])
    add(258, 3, [16])
    add(259, 3, [compression])
    add(262, 3, [1])
    add(273, 4, [0] * n_strips)  # patched once the layout is known
    add(277, 3, [1])
    add(278, 3, [rows_per_strip])
    add(279, 4, counts)
    add(284, 3, [1])
    add(339, 3, [2])
    add(33550, 12, [pixel_size, pixel_size, 0.0])
    add(33922, 12, [0.0, 0.0, 0.0, tiepoint[0], tiepoint[1], 0.0])
    add(34737, 2, b"Interrupted_Goode_Homolosine|GCS unnamed ellipse|\0")
    if nodata_bytes:
        add(42113, 2, nodata_bytes)
    entries.sort(key=lambda e: e[0])

    ifd_off = 8
    n = len(entries)
    ifd_size = 2 + n * 12 + 4
    ext_off = ifd_off + ifd_size
    layout: list[tuple[int, int, int, bytes, int | None]] = []
    for tag, typ, count, payload in entries:
        if len(payload) <= 4:
            layout.append((tag, typ, count, payload, None))
        else:
            layout.append((tag, typ, count, payload, ext_off + len(external)))
            external += payload
            if len(external) % 2:
                external += b"\0"

    data_off = ext_off + len(external) + header_pad
    strip_offsets = []
    o = data_off
    for c in counts:
        strip_offsets.append(o)
        o += c

    out = bytearray()
    out += (b"II" if byteorder == "<" else b"MM") + struct.pack(byteorder + "HI", 42, ifd_off)
    out += struct.pack(byteorder + "H", n)
    for tag, typ, count, payload, ptr in layout:
        out += struct.pack(byteorder + "HHI", tag, typ, count)
        if ptr is None:
            out += payload.ljust(4, b"\0")
        else:
            out += struct.pack(byteorder + "I", ptr)
    out += struct.pack(byteorder + "I", 0)
    out += external
    out += b"\0" * header_pad
    assert len(out) == data_off, (len(out), data_off)
    out += values.tobytes()

    # patch StripOffsets now that we know where the data starts
    for i, (tag, _typ, _count, _payload, ptr) in enumerate(layout):
        if tag == 273:
            packed = struct.pack(byteorder + "I" * n_strips, *strip_offsets)
            if ptr is None:
                pos = ifd_off + 2 + i * 12 + 8
                out[pos : pos + 4] = packed.ljust(4, b"\0")[:4]
            else:
                out[ptr : ptr + len(packed)] = packed
    return bytes(out)


@pytest.fixture
def small_grid(monkeypatch):
    """A 2x2-cell grid whose cell is 90 px, with 45 px and 30 px subtilings.

    Scaled from the real geometry by 20x: cell 1800 -> 90 px, tiles 450 -> 45 px
    (4x4) and 600 -> 30 px (3x3), shard 450 -> 45, chunk 50 -> 5. Still exercises
    multi-chunk shards, multi-shard cells, both subtilings, and ragged tiles.
    """
    px = config.GRID.pixel_size
    grid = config.GridSpec(
        pixel_size=px,
        x_min=config.GRID.x_min,
        y_max=config.GRID.y_max,
        width=180,
        height=180,
        cell_m=90 * px,
        max_tile_row=1,
        max_tile_col=1,
    )
    enc = config.EncodingSpec(chunk_y=5, chunk_x=5, shard_y=45, shard_x=45)
    props = {
        "sand": config.PROPERTIES["sand"].model_copy(
            update={"tile_px": 45, "subtiles_per_cell": 2, "rows_per_strip": 9}
        ),
        "phh2o": config.PROPERTIES["phh2o"].model_copy(
            update={"tile_px": 30, "subtiles_per_cell": 3, "rows_per_strip": 6}
        ),
        "ocs": config.PROPERTIES["ocs"].model_copy(update={"tile_px": 45, "subtiles_per_cell": 2, "rows_per_strip": 9}),
    }
    monkeypatch.setattr(config, "GRID", grid)
    monkeypatch.setattr(config, "ENCODING", enc)
    monkeypatch.setattr(config, "PROPERTIES", props)
    return grid
