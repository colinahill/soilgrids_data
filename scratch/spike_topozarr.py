"""Spike: drive topozarr's kernel directly, with our own layout and an explicit fill.

Checks (a) fill-aware mean, (b) what the fill-value trap would have done,
(c) trim semantics for level shapes, (d) geozarr metadata with a CRS that has no
EPSG code, (e) that native keeps our encoding while levels get depth-1 shards.
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import icechunk
import numpy as np
import zarr
from pyproj import CRS
from topozarr.engine import downsample_level

IGH = CRS.from_proj4("+proj=igh +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs")
NODATA = -32768
OUT = Path("scratch/out/spike.icechunk")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.parent.mkdir(parents=True, exist_ok=True)

repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(OUT)))
session = repo.writable_session("main")
root = zarr.open_group(session.store, path="soil_properties", mode="a")

D, H, W = 6, 900, 1800
native = root.create_array(
    "sand", shape=(D, H, W), chunks=(1, 50, 50), shards=(D, 450, 450),
    dtype="int16", fill_value=NODATA,
    compressors=[zarr.codecs.ZstdCodec(level=3)],
    dimension_names=("depth_interval", "y", "x"),
)

# synthetic land: a smooth field, with the left third and a diagonal set to nodata
yy, xx = np.mgrid[0:H, 0:W]
base = (300 + 100 * np.sin(xx / 90) + 50 * np.cos(yy / 70)).astype("int16")
land = (xx > W // 3) & (np.abs(yy - xx * 0.3) > 40)
data = np.where(land, base, NODATA).astype("int16")
native[:] = np.broadcast_to(data, (D, H, W))
print(f"native written: {native.shape} chunks={native.chunks} shards={native.shards} "
      f"fill={native.fill_value}  land={100 * land.mean():.1f}%")

# --- levels, our layout: factor-named child groups, depth-1 shards -------------
def make_level(factor: int, h: int, w: int) -> zarr.Array:
    g = zarr.open_group(session.store, path=f"soil_properties/{factor}x", mode="a")
    return g.create_array(
        "sand", shape=(D, h, w), chunks=(1, 50, 50), shards=(1, 450, 450),
        dtype="int16", fill_value=NODATA,
        compressors=[zarr.codecs.ZstdCodec(level=3)],
        dimension_names=("depth_interval", "y", "x"),
    )

# trim semantics: floor, not ceil
l2 = make_level(2, H // 2, W // 2)
with ThreadPoolExecutor(4) as ex:
    for f in downsample_level(native, l2, stride=(1, 2, 2), method="mean",
                              fill_value=NODATA, executor=ex):
        f.result()

# --- (a) is the mean fill-aware? ----------------------------------------------
got = l2[0]
blocks = data.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(H // 2, W // 2, 4)
valid = blocks != NODATA
nvalid = valid.sum(axis=-1)
with np.errstate(invalid="ignore"):
    ref = np.where(nvalid > 0,
                   np.trunc(np.where(valid, blocks, 0).sum(axis=-1) / np.maximum(nvalid, 1)),
                   NODATA).astype("int16")
mixed = (nvalid > 0) & (nvalid < 4)
print(f"\n(a) fill-aware mean: exact match {np.array_equal(got, ref)}  "
      f"| mixed land/nodata cells checked: {mixed.sum()}  "
      f"| max abs diff {np.abs(got.astype(int) - ref.astype(int)).max()}")
print(f"    all-nodata cells read back as fill: {np.all(got[nvalid == 0] == NODATA)}")

# --- (b) what the trap would have produced ------------------------------------
from topozarr.engine import block_reduce  # noqa: E402
# pick a window that actually straddles the nodata boundary
ys, xs = np.where(mixed)
y0, x0 = (ys[len(ys) // 2] // 25) * 50, (xs[len(xs) // 2] // 25) * 50
blk = np.ascontiguousarray(data[None, y0:y0 + 100, x0:x0 + 100])
good = block_reduce(blk, (1, 2, 2), "mean", NODATA, True)
trap = block_reduce(blk, (1, 2, 2), "mean", None, True)
m = mixed[y0 // 2:y0 // 2 + 50, x0 // 2:x0 // 2 + 50]
print(f"    window at y={y0} x={x0}, mixed cells in it: {m.sum()}")
print(f"\n(b) fill_value=None on mixed cells: min {trap[0][m].min()} (correct: {good[0][m].min()}) "
      f"-> {'CORRUPTED as predicted' if trap[0][m].min() < -1000 else 'no difference'}")

# --- (c) trim: level shapes must be floor ------------------------------------
print(f"\n(c) trim: 160200/16 = {160200 / 16}, floor {160200 // 16}; "
      f"59400/16 = {59400 / 16}, floor {59400 // 16}")
sh = [(160200, 59400)]
for _ in range(8):
    w, h = sh[-1]
    sh.append((w // 2, h // 2))
print("    stride-2 chain on the real grid:",
      " ".join(f"{i}:{w}x{h}" for i, (w, h) in enumerate(sh)))
print(f"    edge lost at 256x: {160200 - sh[8][0] * 256} px x = {(160200 - sh[8][0] * 256) * 250 / 1000:.0f} km, "
      f"{59400 - sh[8][1] * 256} px y = {(59400 - sh[8][1] * 256) * 250 / 1000:.0f} km")

# --- (d) geozarr metadata with no EPSG code ----------------------------------
import xarray as xr  # noqa: E402
import xproj  # noqa: F401,E402
from topozarr.geozarr import create_geozarr_metadata  # noqa: E402
ds = xr.Dataset(
    {"sand": (("y", "x"), data[:10, :10])},
    coords={"y": np.arange(10.0), "x": np.arange(10.0)},
).proj.assign_crs(spatial_ref=IGH)
try:
    md = create_geozarr_metadata(ds, x_dim="x", y_dim="y")
    keys = sorted(md)
    print(f"\n(d) create_geozarr_metadata OK; keys: {keys}")
    for k in keys:
        v = md[k]
        print(f"    {k} = {str(v)[:160]}")
except Exception as e:
    print(f"\n(d) create_geozarr_metadata FAILED: {type(e).__name__}: {e}")

snap = session.commit("spike")
print(f"\ncommitted {snap}")
