"""A minimal, strict TIFF reader for SoilGrids tiles.

SoilGrids 250 m tiles are uncompressed, little-endian, single-band Int16 striped
GeoTIFFs whose strips are byte-contiguous. That is simple enough to parse
directly, which buys three things over going through GDAL:

* the tile's own tie-point, shape, dtype and nodata are checked against the
  manifest at write time -- the FILE is the ground truth, not the VRT;
* no GDAL/vsicurl in the hot loop, so retries, keep-alive and concurrency stay
  under our control;
* ragged (coastline-clipped) tiles are handled by the same code path, since we
  only ever ask a tile where it starts and how big it is.

Anything unexpected raises. Silently mis-parsing a soil property is worse than
failing a run.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# TIFF tags we read; everything else is ignored.
IMAGE_WIDTH = 256
IMAGE_LENGTH = 257
BITS_PER_SAMPLE = 258
COMPRESSION = 259
STRIP_OFFSETS = 273
SAMPLES_PER_PIXEL = 277
ROWS_PER_STRIP = 278
STRIP_BYTE_COUNTS = 279
PLANAR_CONFIG = 284
PREDICTOR = 317
TILE_WIDTH = 322
SAMPLE_FORMAT = 339
MODEL_PIXEL_SCALE = 33550
MODEL_TIEPOINT = 33922
GEO_ASCII_PARAMS = 34737
GDAL_NODATA = 42113

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d", 16: "Q", 17: "q", 18: "Q"}

COMPRESSION_NONE = 1
SAMPLE_FORMAT_INT = 2


class TiffError(ValueError):
    """The tile is not the shape of file this pipeline accepts."""


@dataclass(frozen=True, slots=True)
class TiffTile:
    """Parsed header of one source tile."""

    width: int
    height: int
    rows_per_strip: int
    strip_offsets: tuple[int, ...]
    strip_byte_counts: tuple[int, ...]
    tiepoint: tuple[float, float]  # upper-left (x, y) in projected metres
    pixel_size: tuple[float, float]
    nodata: int | None
    crs_wkt: str | None
    dtype: np.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def data_nbytes(self) -> int:
        return self.width * self.height * self.dtype.itemsize

    @property
    def contiguous(self) -> bool:
        """Are the strips one unbroken run, so the whole tile is a single range?"""
        return all(
            self.strip_offsets[i] + self.strip_byte_counts[i] == self.strip_offsets[i + 1]
            for i in range(len(self.strip_offsets) - 1)
        )

    @property
    def data_range(self) -> tuple[int, int]:
        """(offset, length) of the pixel data, valid only when contiguous."""
        if not self.contiguous:
            raise TiffError("strips are not contiguous; there is no single data range")
        return self.strip_offsets[0], self.data_nbytes


def parse_header(buf: bytes) -> TiffTile:
    """Parse the first IFD of a (Big)TIFF from a buffer holding at least the header."""
    if len(buf) < 16:
        raise TiffError(f"buffer too short for a TIFF header ({len(buf)} bytes)")
    if buf[:2] == b"II":
        bo = "<"
    elif buf[:2] == b"MM":
        bo = ">"
    else:
        raise TiffError(f"not a TIFF: byte-order mark {buf[:2]!r}")
    magic = struct.unpack(bo + "H", buf[2:4])[0]
    if magic == 42:
        big, off_fmt, count_fmt, entry_size = False, "I", "H", 12
        ifd_off = struct.unpack(bo + "I", buf[4:8])[0]
    elif magic == 43:
        big, off_fmt, count_fmt, entry_size = True, "Q", "Q", 20
        if struct.unpack(bo + "H", buf[4:6])[0] != 8:
            raise TiffError("BigTIFF with a non-8-byte offset size")
        ifd_off = struct.unpack(bo + "Q", buf[8:16])[0]
    else:
        raise TiffError(f"not a TIFF: magic {magic}")

    n_size = 8 if big else 2
    if ifd_off + n_size > len(buf):
        raise TiffError(f"IFD at {ifd_off} is beyond the {len(buf)}-byte buffer")
    n_entries = struct.unpack(bo + count_fmt, buf[ifd_off : ifd_off + n_size])[0]
    base = ifd_off + n_size
    if base + n_entries * entry_size > len(buf):
        raise TiffError(f"IFD of {n_entries} entries is beyond the {len(buf)}-byte buffer")

    tags: dict[int, object] = {}
    for i in range(n_entries):
        e = base + i * entry_size
        tag, typ = struct.unpack(bo + "HH", buf[e : e + 4])
        count = struct.unpack(bo + off_fmt, buf[e + 4 : e + 4 + (8 if big else 4)])[0]
        value_off = e + 4 + (8 if big else 4)
        inline = 8 if big else 4
        size = _TYPE_SIZE.get(typ, 1) * count
        if size <= inline:
            raw = buf[value_off : value_off + size]
        else:
            ptr = struct.unpack(bo + off_fmt, buf[value_off : value_off + inline])[0]
            if ptr + size > len(buf):
                continue  # value lives past the buffer; only matters for tags we skip
            raw = buf[ptr : ptr + size]
        if typ == 2:
            tags[tag] = raw.split(b"\0")[0].decode("latin-1")
        elif typ in _TYPE_FMT:
            tags[tag] = struct.unpack(bo + _TYPE_FMT[typ] * count, raw[: _TYPE_SIZE[typ] * count])
        else:
            tags[tag] = raw

    def one(tag: int, what: str) -> int:
        v = tags.get(tag)
        if v is None:
            raise TiffError(f"missing {what} (tag {tag})")
        return int(v[0])  # type: ignore[index]

    if TILE_WIDTH in tags:
        raise TiffError("tiled TIFF; SoilGrids 250 m tiles are striped")
    if (comp := one(COMPRESSION, "Compression")) != COMPRESSION_NONE:
        raise TiffError(f"Compression={comp}; expected {COMPRESSION_NONE} (uncompressed)")
    if PREDICTOR in tags and one(PREDICTOR, "Predictor") != 1:
        raise TiffError(f"Predictor={one(PREDICTOR, 'Predictor')}; not representable as raw bytes")
    if (spp := one(SAMPLES_PER_PIXEL, "SamplesPerPixel")) != 1:
        raise TiffError(f"SamplesPerPixel={spp}; expected a single band")
    if (pc := one(PLANAR_CONFIG, "PlanarConfiguration")) != 1:
        raise TiffError(f"PlanarConfiguration={pc}; expected chunky")
    bits = one(BITS_PER_SAMPLE, "BitsPerSample")
    fmt = one(SAMPLE_FORMAT, "SampleFormat")
    if (bits, fmt) != (16, SAMPLE_FORMAT_INT):
        raise TiffError(f"BitsPerSample={bits}, SampleFormat={fmt}; expected 16-bit signed integer")
    dtype = np.dtype("<i2" if bo == "<" else ">i2")

    width, height = one(IMAGE_WIDTH, "ImageWidth"), one(IMAGE_LENGTH, "ImageLength")
    rps = one(ROWS_PER_STRIP, "RowsPerStrip")
    offsets = tuple(int(v) for v in tags[STRIP_OFFSETS])  # type: ignore[union-attr]
    counts = tuple(int(v) for v in tags[STRIP_BYTE_COUNTS])  # type: ignore[union-attr]
    expected_strips = -(-height // rps)
    if not (len(offsets) == len(counts) == expected_strips):
        raise TiffError(
            f"{len(offsets)} strip offsets / {len(counts)} byte counts, expected {expected_strips} "
            f"for {height} rows at {rps} rows per strip"
        )
    for i, c in enumerate(counts):
        rows = min(rps, height - i * rps)
        if c != rows * width * dtype.itemsize:
            raise TiffError(f"strip {i} is {c} bytes; expected {rows} rows x {width} px x {dtype.itemsize}")

    tp = tags.get(MODEL_TIEPOINT)
    if tp is None or len(tp) < 6:  # type: ignore[arg-type]
        raise TiffError("missing or short ModelTiepointTag (33922)")
    ps = tags.get(MODEL_PIXEL_SCALE)
    if ps is None or len(ps) < 2:  # type: ignore[arg-type]
        raise TiffError("missing or short ModelPixelScaleTag (33550)")
    nodata_raw = tags.get(GDAL_NODATA)
    nodata = int(float(nodata_raw)) if isinstance(nodata_raw, str) and nodata_raw else None

    return TiffTile(
        width=width,
        height=height,
        rows_per_strip=rps,
        strip_offsets=offsets,
        strip_byte_counts=counts,
        tiepoint=(float(tp[3]), float(tp[4])),  # type: ignore[index]
        pixel_size=(float(ps[0]), float(ps[1])),  # type: ignore[index]
        nodata=nodata,
        crs_wkt=tags.get(GEO_ASCII_PARAMS) if isinstance(tags.get(GEO_ASCII_PARAMS), str) else None,
        dtype=dtype,
    )


def decode(buf: bytes, header: TiffTile | None = None) -> tuple[TiffTile, np.ndarray]:
    """Parse a whole-tile buffer and return (header, (height, width) int16 array)."""
    hdr = header or parse_header(buf)
    end = hdr.strip_offsets[-1] + hdr.strip_byte_counts[-1]
    if end > len(buf):
        raise TiffError(f"pixel data ends at {end} but the buffer is {len(buf)} bytes")
    if hdr.contiguous:
        off, length = hdr.data_range
        body = buf[off : off + length]
    else:
        body = b"".join(buf[o : o + c] for o, c in zip(hdr.strip_offsets, hdr.strip_byte_counts, strict=True))
    if len(body) != hdr.data_nbytes:
        raise TiffError(f"assembled {len(body)} pixel bytes, expected {hdr.data_nbytes}")
    return hdr, np.frombuffer(body, dtype=hdr.dtype).reshape(hdr.shape)
