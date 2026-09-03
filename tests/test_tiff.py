"""tiff.py must accept exactly the shape of file ISRIC publishes, and no other."""

from __future__ import annotations

import numpy as np
import pytest

from soilgrids import tiff

from .conftest import NODATA, build_tile


def _values(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-100, 900, size=(h, w), dtype="int16")


@pytest.mark.parametrize(
    "h,w,rps,pad",
    [
        (450, 450, 9, 0),  # the 450 px family, exactly as measured
        (600, 600, 6, 0),  # bdod / phh2o
        (403, 450, 9, 0),  # ragged: partial trailing strip
        (179, 401, 10, 0),  # ragged, both axes clipped
        (450, 450, 9, 397),  # ocs-style shorter header, and a padded one
        (1, 2, 2, 0),  # the degenerate 1x2 source the sand VRT contains
    ],
)
def test_roundtrip_real_shapes(h, w, rps, pad):
    want = _values(h, w)
    buf = build_tile(want, (-7_887_500.0, 500_750.0), rows_per_strip=rps, header_pad=pad)
    hdr, got = tiff.decode(buf)
    assert hdr.shape == (h, w)
    assert hdr.rows_per_strip == rps
    assert hdr.nodata == NODATA
    assert hdr.tiepoint == (-7_887_500.0, 500_750.0)
    assert hdr.contiguous
    assert hdr.data_range == (len(buf) - h * w * 2, h * w * 2)
    np.testing.assert_array_equal(got, want)


def test_nodata_survives_roundtrip():
    want = _values(450, 450)
    want[:20, :] = NODATA
    buf = build_tile(want, (0.0, 0.0))
    _, got = tiff.decode(buf)
    np.testing.assert_array_equal(got, want)
    assert (got == NODATA).sum() == 20 * 450


def test_big_endian_is_read_correctly():
    want = _values(90, 90)
    buf = build_tile(want, (0.0, 0.0), rows_per_strip=9, byteorder=">")
    hdr, got = tiff.decode(buf)
    assert hdr.dtype.byteorder == ">"
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda b: b"XX" + b[2:], "byte-order mark"),
        (lambda b: b[:2] + bytes([99, 0]) + b[4:], "magic"),
        (lambda b: b[:20], "beyond"),  # header ok, IFD entries past the buffer
        (lambda b: b[:8], "too short"),
    ],
)
def test_malformed_headers_raise(mutate, message):
    buf = build_tile(_values(90, 90), (0.0, 0.0))
    with pytest.raises(tiff.TiffError, match=message):
        tiff.parse_header(mutate(buf))


def test_compressed_tile_is_refused():
    # a Deflate tile (as used by landmask and the 1 km/5 km aggregates) must not
    # be silently read as raw bytes
    buf = build_tile(_values(90, 90), (0.0, 0.0), compression=8)
    with pytest.raises(tiff.TiffError, match="Compression=8"):
        tiff.parse_header(buf)


def test_truncated_pixel_data_raises():
    buf = build_tile(_values(450, 450), (0.0, 0.0))
    with pytest.raises(tiff.TiffError, match="buffer is"):
        tiff.decode(buf[:-1000])


def test_strip_byte_count_mismatch_raises():
    import struct

    buf = bytearray(build_tile(_values(450, 450), (0.0, 0.0)))
    hdr = tiff.parse_header(bytes(buf))
    # corrupt the first strip byte count; the header check must catch it
    ptr = None
    n = struct.unpack("<H", buf[8:10])[0]
    for i in range(n):
        e = 10 + i * 12
        tag = struct.unpack("<H", buf[e : e + 2])[0]
        if tag == 279:
            ptr = struct.unpack("<I", buf[e + 8 : e + 12])[0]
    assert ptr is not None and len(hdr.strip_byte_counts) > 1
    buf[ptr : ptr + 4] = struct.pack("<I", 1234)
    with pytest.raises(tiff.TiffError, match="strip 0 is 1234 bytes"):
        tiff.parse_header(bytes(buf))
