#!/usr/bin/env python3
"""
Convert all .npy and .tif files to left-edge labeling (pixel-as-area with
upper-left origin), writing standardized GeoTIFF outputs.

Usage examples:
  # Convert all .npy under a root directory to GeoTIFF with left-edge origin
  python convert_left_edge.py --root /path/to/data --grid 0.1 --lat-max 60 --lon-min 0 --mode npy_to_tif

  # Fix existing GeoTIFFs assumed to be center-labeled → left-edge by shifting origin
  python convert_left_edge.py --root /path/to/data --grid 0.1 --assume-center --mode fix_tif

  # Auto: process both .npy and .tif in one pass
  python convert_left_edge.py --root /path/to/data --grid 0.1 --lat-max 60 --lon-min 0 --assume-center --mode auto
"""

import argparse
import os
import sys
from typing import Optional, Tuple, Iterable

import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _try_import_rasterio():
    try:
        import rasterio  # type: ignore
        from rasterio.transform import from_origin  # type: ignore
        return rasterio, from_origin
    except Exception as e:
        raise RuntimeError(
            "Cần cài rasterio để làm việc với GeoTIFF: pip install rasterio"
        ) from e


def find_files(root: str, patterns: Tuple[str, ...]) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            lower = fn.lower()
            if any(lower.endswith(pat) for pat in patterns):
                yield os.path.join(dirpath, fn)


def npy_to_tif(npy_path: str, out_path: str, grid_deg: float, lat_max: float, lon_min: float, dtype: Optional[str] = None, compress: str = "deflate") -> None:
    rasterio, from_origin = _try_import_rasterio()
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"Chỉ hỗ trợ NPY 2D. File: {npy_path} có ndim={arr.ndim}")
    if dtype:
        arr = arr.astype(dtype)

    rows, cols = arr.shape
    transform = from_origin(lon_min, lat_max, grid_deg, grid_deg)
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": str(arr.dtype),
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": compress,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


def fix_tif_left_edge(tif_path: str, out_path: str, grid_deg: float, assume_center: bool, force_origin: Optional[Tuple[float, float]] = None, compress: str = "deflate") -> None:
    rasterio, from_origin = _try_import_rasterio()
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        src_profile = src.profile.copy()
        transform = src.transform

    # Determine new transform for left-edge labeling
    if force_origin is not None:
        lon_min, lat_max = force_origin
        new_transform = from_origin(lon_min, lat_max, grid_deg, grid_deg)
    elif assume_center:
        # Shift origin from center-labeled to left-edge: (-grid/2 lon, +grid/2 lat)
        a, b, c, d, e, f = transform.to_gdal()  # type: ignore[attr-defined]
        # GDAL affine: [c, a, b, f, d, e], but rasterio.Affine stores (a, b, c, d, e, f)
        # We want to move origin by (-grid/2, +grid/2)
        from affine import Affine  # lazy import to avoid hard dep if not used
        new_transform = Affine(transform.a, transform.b, transform.c - grid_deg / 2.0,
                               transform.d, transform.e, transform.f + grid_deg / 2.0)
    else:
        # Keep existing transform (assume already left-edge). But we still enforce pixel size.
        from affine import Affine
        new_transform = Affine(grid_deg, transform.b, transform.c,
                               transform.d, -grid_deg if transform.e < 0 else grid_deg, transform.f)

    rows, cols = arr.shape
    profile = src_profile
    profile.update({
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": str(arr.dtype),
        "transform": new_transform,
        "crs": "EPSG:4326",
        "compress": compress,
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .npy/.tif sang quy ước nhãn mép trái (left-edge)")
    parser.add_argument("--root", required=True, help="Thư mục gốc để quét file")
    parser.add_argument("--mode", choices=["npy_to_tif", "fix_tif", "auto", "drop_lon"], default="auto", help="Chế độ xử lý")
    parser.add_argument("--grid", type=float, required=True, help="Kích thước lưới (độ), ví dụ 0.1")
    parser.add_argument("--lat-max", type=float, default=None, help="lat_max cho NPY → TIF")
    parser.add_argument("--lon-min", type=float, default=None, help="lon_min cho NPY → TIF")
    parser.add_argument("--lon-cut", type=float, default=None, help="Kinh độ mép trái cột cần cắt bỏ (left-edge)")
    parser.add_argument("--assume-center", action="store_true", help="TIF đầu vào đang center-labeled, cần dịch sang left-edge")
    parser.add_argument("--force-origin", action="store_true", help="Buộc dùng (lon_min, lat_max) cho TIF đầu vào")
    parser.add_argument("--out-dir", default=None, help="Thư mục lưu kết quả (mặc định ghi cạnh file)")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè nếu file đích tồn tại")
    parser.add_argument("--dtype", default=None, help="Ép kiểu khi ghi (ví dụ float32, int16)")
    parser.add_argument("--in-place", action="store_true", help="Ghi đè lên file gốc khi drop_lon")
    args = parser.parse_args()

    root = args.root
    mode = args.mode
    grid = float(args.grid)
    lat_max = args.lat_max
    lon_min = args.lon_min
    out_dir = args.out_dir
    overwrite = bool(args.overwrite)
    dtype = args.dtype

    if mode in ("npy_to_tif", "auto"):
        if lat_max is None or lon_min is None:
            print("[ERR] Cần --lat-max và --lon-min để chuyển NPY → TIF.")
            return 1
    if mode == "drop_lon" and args.lon_cut is None:
        print("[ERR] drop_lon yêu cầu --lon-cut (kinh độ mép trái cần cắt)")
        return 1

    npy_paths = []
    tif_paths = []
    if mode in ("npy_to_tif", "auto"):
        npy_paths = list(find_files(root, (".npy",)))
    if mode in ("fix_tif", "auto"):
        tif_paths = list(find_files(root, (".tif", ".tiff")))
    if mode == "drop_lon":
        npy_paths = list(find_files(root, (".npy",)))
        tif_paths = list(find_files(root, (".tif", ".tiff")))

    # Process NPY files
    for p in tqdm(npy_paths, desc="NPY", unit="file"):
        rows_cols = None
        try:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 2:
                print(f"[SKIP] Không phải NPY 2D: {p}")
                continue
            rows_cols = arr.shape
        except Exception as e:
            print(f"[WARN] Không thể đọc NPY {p}: {e}")
            continue
        if mode in ("npy_to_tif", "auto"):
            out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".tif") if out_dir else os.path.splitext(p)[0] + ".tif"
            if os.path.exists(out_path) and not overwrite:
                print(f"[SKIP] Đã tồn tại: {out_path}")
                continue
            try:
                npy_to_tif(p, out_path, grid, float(lat_max), float(lon_min), dtype=dtype)
                print(f"[OK] NPY→TIF ({rows_cols[0]}x{rows_cols[1]}): {out_path}")
            except Exception as e:
                print(f"[ERR] NPY→TIF thất bại {p}: {e}")
        elif mode == "drop_lon":
            # compute column index from lon_cut (left-edge labeling)
            lon_cut = float(args.lon_cut)
            if lon_min is None:
                print(f"[ERR] Cần --lon-min để cắt NPY theo nhãn mép trái: {p}")
                continue
            col_idx = int(np.floor((lon_cut - float(lon_min)) / grid))
            if col_idx < 0 or col_idx >= rows_cols[1]:
                print(f"[SKIP] lon_cut={lon_cut} ngoài phạm vi cột 0..{rows_cols[1]-1} ở {p}")
                continue
            try:
                arr2 = np.load(p)
                arr2 = np.delete(arr2, col_idx, axis=1)
                if args.in_place:
                    out_path = p
                else:
                    suffix = f"_drop{lon_cut:g}"
                    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + f"{suffix}.npy") if out_dir else os.path.splitext(p)[0] + f"{suffix}.npy"
                if os.path.exists(out_path) and not overwrite:
                    print(f"[SKIP] Đã tồn tại: {out_path}")
                    continue
                np.save(out_path, arr2)
                print(f"[OK] NPY drop lon={lon_cut} (col={col_idx}) → {out_path}")
            except Exception as e:
                print(f"[ERR] NPY drop_lon thất bại {p}: {e}")

    # Process TIF files
    for p in tqdm(tif_paths, desc="TIF", unit="file"):
        if mode in ("fix_tif", "auto"):
            out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + "_leftedge.tif") if out_dir else os.path.splitext(p)[0] + "_leftedge.tif"
            if os.path.exists(out_path) and not overwrite:
                print(f"[SKIP] Đã tồn tại: {out_path}")
                continue
            try:
                force_origin = (float(lon_min), float(lat_max)) if args.force_origin and (lat_max is not None and lon_min is not None) else None
                fix_tif_left_edge(p, out_path, grid, assume_center=bool(args.assume_center), force_origin=force_origin)
                print(f"[OK] Fix TIF→left-edge: {out_path}")
            except Exception as e:
                print(f"[ERR] Fix TIF thất bại {p}: {e}")
        elif mode == "drop_lon":
            try:
                import rasterio  # type: ignore
                from rasterio.transform import Affine as RAffine  # type: ignore
            except Exception as e:
                print("[ERR] Cần rasterio để cắt TIFF. pip install rasterio")
                return 1
            lon_cut = float(args.lon_cut)
            try:
                with rasterio.open(p) as src:
                    arr = src.read(1)
                    transform = src.transform
                    profile = src.profile.copy()
                    rows, cols = arr.shape
                    # derive grid and lon_min from transform
                    px_w = float(transform.a)
                    px_h = float(-transform.e) if transform.e < 0 else float(transform.e)
                    grid_x = px_w
                    lon_min_tif = float(transform.c)
                    col_idx = int(np.floor((lon_cut - lon_min_tif) / grid_x))
                    if col_idx < 0 or col_idx >= cols:
                        print(f"[SKIP] lon_cut={lon_cut} ngoài phạm vi cột 0..{cols-1} ở {p}")
                        continue
                    arr2 = np.delete(arr, col_idx, axis=1)
                    # adjust origin only if we removed first column
                    if col_idx == 0:
                        from affine import Affine
                        new_transform = Affine(transform.a, transform.b, transform.c + grid_x,
                                               transform.d, transform.e, transform.f)
                    else:
                        new_transform = transform
                if args.in_place:
                    out_path = p
                else:
                    suffix = f"_drop{lon_cut:g}"
                    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + f"{suffix}.tif") if out_dir else os.path.splitext(p)[0] + f"{suffix}.tif"
                if os.path.exists(out_path) and not overwrite:
                    print(f"[SKIP] Đã tồn tại: {out_path}")
                    continue
                profile.update({
                    "width": arr2.shape[1],
                    "transform": new_transform,
                })
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(arr2, 1)
                print(f"[OK] TIF drop lon={lon_cut} (col={col_idx}) → {out_path}")
            except Exception as e:
                print(f"[ERR] TIF drop_lon thất bại {p}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


