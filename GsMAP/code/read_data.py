#!/usr/bin/env python3
"""
Read GSMaP binary .dat or .dat.gz, infer grid shape, and show stats.

Examples (PowerShell):
  # Đọc và in thống kê (tự suy đoán kích thước)
  python read_data.py --input .\data\gsmap_nrt.20240901.0.1d.daily.00Z-23Z.dat.gz

  # Chỉ định kích thước thủ công nếu cần (hàng, cột)
  python read_data.py --input .\data\file.dat --rows 1800 --cols 3600

  # Xuất GeoTIFF (nếu đã cài rasterio)
  python read_data.py --input .\data\file.dat.gz --export-tiff .\data\file.tif --grid 0.1
"""

import argparse
import gzip
import os
import shutil
import sys
from typing import Optional, Tuple, List

import numpy as np


COMMON_GRIDS = [
    # (rows, cols, grid_deg, lat_max, lon_min)
    (1800, 3600, 0.1, 90.0, -180.0),     # 0.1 degree global (full globe)
    (480,  1440, 0.25, 90.0, -180.0),    # 0.25 degree global (full globe)
    (1200, 3600, 0.1, 60.0, 0.0),        # 0.1 degree, 60S..60N, 0..360 lon (GSMaP NRT thường gặp)
    (600,  1440, 0.25, 60.0, 0.0),       # 0.25 degree, 60S..60N, 0..360 lon
]


def gunzip_if_needed(path: str) -> str:
    if not path.lower().endswith(".gz"):
        return path
    out_path = path[:-3]
    if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(path):
        return out_path
    with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def infer_shape_from_size(file_size: int) -> Optional[Tuple[int, int, float, float, float]]:
    # Assume float32 values without headers
    for rows, cols, grd, lat_max, lon_min in COMMON_GRIDS:
        expected = rows * cols * 4
        if expected == file_size:
            return rows, cols, grd, lat_max, lon_min
    return None


def load_binary_grid(path: str, rows: int, cols: int, dtype: str = "float32") -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    if data.size != rows * cols:
        raise ValueError(f"Kích thước dữ liệu không khớp: {data.size} != {rows}*{cols}")
    return data.reshape((rows, cols))


def export_geotiff(tif_path: str, grid: np.ndarray, grid_deg: float, lat_max: float, lon_min: float) -> None:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except Exception as e:
        raise RuntimeError("Cần cài đặt rasterio để xuất GeoTIFF: pip install rasterio") from e

    rows, cols = grid.shape
    # Dùng góc trên trái (lon_min, lat_max)
    transform = from_origin(lon_min, lat_max, grid_deg, grid_deg)
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": str(grid.dtype),
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(grid, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Giải nén và đọc GSMaP .dat/.dat.gz")
    parser.add_argument("--input", required=True, help="Đường dẫn tới .dat hoặc .dat.gz")
    parser.add_argument("--rows", type=int, default=None, help="Số hàng (nếu biết)")
    parser.add_argument("--cols", type=int, default=None, help="Số cột (nếu biết)")
    parser.add_argument("--dtype", default="float32", help="Kiểu dữ liệu (mặc định float32)")
    parser.add_argument("--grid", type=float, default=None, help="Kích thước lưới (độ), ví dụ 0.1 hoặc 0.25 để export GeoTIFF")
    parser.add_argument("--lat-max", type=float, default=None, help="Vĩ độ trên cùng (mặc định suy đoán: 90 hoặc 60)")
    parser.add_argument("--lon-min", type=float, default=None, help="Kinh độ trái (mặc định suy đoán: -180 hoặc 0)")
    parser.add_argument("--export-tiff", default=None, help="Xuất GeoTIFF ra đường dẫn này")
    parser.add_argument("--info", action="store_true", help="Chỉ in thông tin file và các cấu hình lưới khả dĩ rồi thoát")
    # Trích xuất box quanh một điểm
    parser.add_argument("--point-lat", type=float, default=None, help="Vĩ độ tâm box cần trích (deg)")
    parser.add_argument("--point-lon", type=float, default=None, help="Kinh độ tâm box cần trích (deg)")
    parser.add_argument("--half-size-deg", type=float, default=None, help="Nửa cạnh box (độ). Ví dụ 0.5 → box 1.0°")
    parser.add_argument("--half-size-px", type=int, default=None, help="Nửa cạnh box (số pixel). Ví dụ 5 → box 11x11")
    parser.add_argument("--out-csv", default=None, help="Lưu box ra CSV")
    parser.add_argument("--out-npy", default=None, help="Lưu box ra NumPy .npy")
    parser.add_argument("--out-tiff", default=None, help="Lưu box ra GeoTIFF")
    # Cắt theo bbox
    parser.add_argument("--bbox-lat-min", type=float, default=None, help="Vĩ độ tối thiểu của bbox")
    parser.add_argument("--bbox-lat-max", type=float, default=None, help="Vĩ độ tối đa của bbox")
    parser.add_argument("--bbox-lon-min", type=float, default=None, help="Kinh độ tối thiểu của bbox")
    parser.add_argument("--bbox-lon-max", type=float, default=None, help="Kinh độ tối đa của bbox")
    parser.add_argument("--bbox-out-csv", default=None, help="Lưu bbox ra CSV")
    parser.add_argument("--bbox-out-npy", default=None, help="Lưu bbox ra NumPy .npy")
    parser.add_argument("--bbox-out-tiff", default=None, help="Lưu bbox ra GeoTIFF")

    args = parser.parse_args()

    src = args.input
    if not os.path.exists(src):
        print(f"[ERR] Không tìm thấy file: {src}")
        return 1

    # Nhánh đọc GeoTIFF trực tiếp
    if src.lower().endswith((".tif", ".tiff")):
        try:
            import rasterio
        except Exception as e:
            print("[ERR] Cần cài rasterio để đọc GeoTIFF: conda install -c conda-forge rasterio -y")
            return 1
        try:
            with rasterio.open(src) as ds:
                arr = ds.read(1)
                rows, cols = arr.shape
                bounds = ds.bounds
                crs = ds.crs
                transform = ds.transform
                finite = np.isfinite(arr)
                valid_count = int(finite.sum())
                total = arr.size
                vmin = float(np.nanmin(np.where(finite, arr, np.nan))) if valid_count > 0 else float("nan")
                vmax = float(np.nanmax(np.where(finite, arr, np.nan))) if valid_count > 0 else float("nan")
                vmean = float(np.nanmean(np.where(finite, arr, np.nan))) if valid_count > 0 else float("nan")
                print(f"[INFO] GeoTIFF size: {rows} x {cols}")
                print(f"[INFO] CRS: {crs}")
                print(f"[INFO] Bounds: left={bounds.left}, bottom={bounds.bottom}, right={bounds.right}, top={bounds.top}")
                print(f"[INFO] Transform: {transform}")
                print(f"[INFO] Hợp lệ: {valid_count}/{total} | Min: {vmin:.4f} | Max: {vmax:.4f} | Mean: {vmean:.4f}")

                # Xuất nếu yêu cầu
                if args.out_npy:
                    np.save(args.out_npy, arr)
                    print(f"[OK] Lưu NPY: {args.out_npy}")
                if args.out_csv:
                    np.savetxt(args.out_csv, arr, delimiter=",", fmt="%.6f")
                    print(f"[OK] Lưu CSV: {args.out_csv}")
            return 0
        except Exception as e:
            print(f"[ERR] Lỗi đọc GeoTIFF: {e}")
            return 1

    dat_path = gunzip_if_needed(src)
    size = os.path.getsize(dat_path)

    if args.info:
        print(f"[INFO] File: {dat_path}")
        print(f"[INFO] Kích thước (bytes): {size}")
        matches = []
        for r, c, g, la, lo in COMMON_GRIDS:
            if r * c * 4 == size:
                matches.append((r, c, g, la, lo))
        if matches:
            print("[INFO] Khả dĩ (rows, cols, grid_deg, lat_max, lon_min):")
            for r, c, g, la, lo in matches:
                print(f"  - {r} x {c}, {g}°, lat_max={la}, lon_min={lo}")
            # Gợi ý lệnh đọc nhanh cho cấu hình đầu tiên
            r, c, g, la, lo = matches[0]
            print("[HINT] Lệnh đọc gợi ý:")
            print(f"  python read_data.py --input {dat_path} --rows {r} --cols {c}")
            print("[HINT] Xuất GeoTIFF:")
            print(f"  python read_data.py --input {dat_path} --rows {r} --cols {c} --grid {g} --lat-max {la} --lon-min {lo} --export-tiff {os.path.splitext(dat_path)[0]}.tif")
        else:
            print("[WARN] Không khớp mẫu lưới mặc định. Hãy xem README/DataFormatDescription hoặc cung cấp --rows/--cols.")
        return 0

    rows = args.rows
    cols = args.cols
    grid_deg: Optional[float] = args.grid
    lat_max: Optional[float] = args.lat_max
    lon_min: Optional[float] = args.lon_min

    if rows is None or cols is None:
        inferred = infer_shape_from_size(size)
        if inferred is None:
            print(f"[ERR] Không suy đoán được kích thước từ file size={size}. Hãy cung cấp --rows và --cols.")
            return 1
        rows, cols, inferred_grid, inferred_lat_max, inferred_lon_min = inferred
        if grid_deg is None:
            grid_deg = inferred_grid
        if lat_max is None:
            lat_max = inferred_lat_max
        if lon_min is None:
            lon_min = inferred_lon_min
        print(f"[INFO] Suy đoán kích thước: rows={rows}, cols={cols}, grid≈{inferred_grid}°, lat_max={lat_max}, lon_min={lon_min}")

    try:
        arr = load_binary_grid(dat_path, rows, cols, dtype=args.dtype)
    except Exception as e:
        print(f"[ERR] Lỗi khi đọc dữ liệu: {e}")
        return 1

    # Thống kê cơ bản
    finite = np.isfinite(arr)
    valid_count = int(finite.sum())
    total = arr.size
    vmin = float(np.nanmin(np.where(finite, arr, np.nan))) if valid_count > 0 else float("nan")
    vmax = float(np.nanmax(np.where(finite, arr, np.nan))) if valid_count > 0 else float("nan")
    vmean = float(np.nanmean(np.where(finite, arr, np.nan))) if valid_count > 0 else float("nan")

    print(f"[INFO] Kích thước: {rows} x {cols} | Tổng ô: {total}")
    print(f"[INFO] Hợp lệ: {valid_count} | Min: {vmin:.4f} | Max: {vmax:.4f} | Mean: {vmean:.4f}")

    # Trích xuất box quanh một điểm nếu được yêu cầu
    if args.point_lat is not None and args.point_lon is not None:
        if grid_deg is None or lat_max is None or lon_min is None:
            print("[ERR] Cần biết --grid, --lat-max, --lon-min (hoặc để script suy đoán) để lập chỉ số.")
            return 1
        half_px: Optional[int] = args.half_size_px
        if half_px is None:
            if args.half_size_deg is None:
                print("[ERR] Cần --half-size-deg hoặc --half-size-px để xác định kích thước box.")
                return 1
            half_px = max(1, int(round(args.half_size_deg / grid_deg)))

        # Chuẩn hóa kinh độ đầu vào theo lon_min (0..360 hay -180..180)
        lon = float(args.point_lon)
        if lon_min >= 0 and lon < 0:
            lon = (lon + 360.0) % 360.0
        if lon_min < 0 and lon > 180.0:
            lon = ((lon + 180.0) % 360.0) - 180.0

        lat = float(args.point_lat)
        row_center = int(round((lat_max - lat) / grid_deg))
        col_center = int(round((lon - lon_min) / grid_deg))

        r0 = max(0, row_center - half_px)
        r1 = min(rows, row_center + half_px + 1)
        c0 = max(0, col_center - half_px)
        c1 = min(cols, col_center + half_px + 1)

        box = arr[r0:r1, c0:c1]
        print(f"[INFO] Box shape: {box.shape} | rows[{r0}:{r1}) cols[{c0}:{c1})")
        if box.size == 0:
            print("[WARN] Box rỗng do tọa độ ở sát/ngoài biên.")

        # Lưu tùy chọn
        if args.out_npy:
            try:
                np.save(args.out_npy, box)
                print(f"[OK] Lưu NPY: {args.out_npy}")
            except Exception as e:
                print(f"[ERR] Lưu NPY thất bại: {e}")
        if args.out_csv:
            try:
                np.savetxt(args.out_csv, box, delimiter=",", fmt="%.6f")
                print(f"[OK] Lưu CSV: {args.out_csv}")
            except Exception as e:
                print(f"[ERR] Lưu CSV thất bại: {e}")
        if args.out_tiff:
            try:
                # Tính lại góc trái-trên của box
                lat_max_box = lat_max - r0 * grid_deg
                lon_min_box = lon_min + c0 * grid_deg
                export_geotiff(args.out_tiff, box, grid_deg, lat_max_box, lon_min_box)
                print(f"[OK] Lưu GeoTIFF: {args.out_tiff}")
            except Exception as e:
                print(f"[ERR] Lưu GeoTIFF thất bại: {e}")

    if args.export_tiff:
        if grid_deg is None:
            print("[WARN] Không biết grid degree để georeference. Dùng --grid 0.1 hoặc 0.25.")
        # Mặc định hợp lý nếu thiếu thông tin
        if lat_max is None:
            lat_max = 90.0
        if lon_min is None:
            lon_min = -180.0
        try:
            export_geotiff(args.export_tiff, arr, grid_deg if grid_deg is not None else 0.1, lat_max, lon_min)
            print(f"[OK] Đã xuất GeoTIFF: {args.export_tiff}")
        except Exception as e:
            print(f"[ERR] Xuất GeoTIFF thất bại: {e}")

    # Cắt theo bbox nếu được yêu cầu
    if (args.bbox_lat_min is not None and args.bbox_lat_max is not None and
        args.bbox_lon_min is not None and args.bbox_lon_max is not None):
        if grid_deg is None or lat_max is None or lon_min is None:
            print("[ERR] Cần biết --grid, --lat-max, --lon-min (hoặc để script suy đoán) để cắt bbox.")
            return 1

        lat_min_req = float(args.bbox_lat_min)
        lat_max_req = float(args.bbox_lat_max)
        lon_min_req = float(args.bbox_lon_min)
        lon_max_req = float(args.bbox_lon_max)
        print(f"[INFO] BBOX: lat_min_req={lat_min_req}, lat_max_req={lat_max_req}, lon_min_req={lon_min_req}, lon_max_req={lon_max_req}")
        # Chuẩn hóa thứ tự
        if lat_min_req > lat_max_req:
            lat_min_req, lat_max_req = lat_max_req, lat_min_req
        print(f"[INFO] BBOX: lat_min_req={lat_min_req}, lat_max_req={lat_max_req}")
        # Chuyển lon theo hệ của dữ liệu
        def norm_lon_to_data(lon: float) -> float:
            if lon_min >= 0 and lon < 0:
                return (lon + 360.0) % 360.0
            if lon_min < 0 and lon > 180.0:
                return ((lon + 180.0) % 360.0) - 180.0
            return lon
        print(f"[INFO] BBOX: lon_min_req={lon_min_req}, lon_max_req={lon_max_req}")
        lon_min_n = norm_lon_to_data(lon_min_req)
        lon_max_n = norm_lon_to_data(lon_max_req)
        print(f"[INFO] BBOX: lon_min_n={lon_min_n}, lon_max_n={lon_max_n}")
        # Tính chỉ số hàng
        top = int(np.floor((lat_max - lat_max_req) / grid_deg))
        bottom = int(np.ceil((lat_max - lat_min_req) / grid_deg))
        r0 = max(0, top)
        r1 = min(rows, bottom)
        print(f"[INFO] BBOX: r0={r0}, r1={r1}")
        # Tính chỉ số cột, xử lý qua đường đổi ngày nếu lon_max_n < lon_min_n
        def col_of(lon: float) -> int:
            return int(round((lon - lon_min) / grid_deg))
        print(f"[INFO] BBOX: lon_max_n={lon_max_n}, lon_min_n={lon_min_n}")
        if lon_max_n >= lon_min_n:
            c0 = max(0, int(np.floor((lon_min_n - lon_min) / grid_deg)))
            c1 = min(cols, int(np.ceil((lon_max_n - lon_min) / grid_deg)) + 1)
            bbox = arr[r0:r1, c0:c1]
            lon_min_bbox = lon_min + c0 * grid_deg
        else:
            # wrap-around: [lon_min_n .. 360(or 180)] U [base .. lon_max_n]
            c0a = max(0, int(np.floor((lon_min_n - lon_min) / grid_deg)))
            c1a = cols
            c0b = 0
            c1b = min(cols, int(np.ceil((lon_max_n - lon_min) / grid_deg)) + 1)
            part_a = arr[r0:r1, c0a:c1a]
            part_b = arr[r0:r1, c0b:c1b]
            bbox = np.concatenate([part_a, part_b], axis=1)
            lon_min_bbox = lon_min + c0a * grid_deg
        print(f"[INFO] BBOX: lon_min_bbox={lon_min_bbox}")          
        print(f"[INFO] BBOX rows[{r0}:{r1}) cols[{c0}:{c1}] -> shape {bbox.shape}")

        if args.bbox_out_npy:
            try:
                np.save(args.bbox_out_npy, bbox)
                print(f"[OK] Lưu NPY: {args.bbox_out_npy}")
            except Exception as e:
                print(f"[ERR] Lưu NPY thất bại: {e}")
        if args.bbox_out_csv:
            try:
                np.savetxt(args.bbox_out_csv, bbox, delimiter=",", fmt="%.6f")
                print(f"[OK] Lưu CSV: {args.bbox_out_csv}")
            except Exception as e:
                print(f"[ERR] Lưu CSV thất bại: {e}")
        if args.bbox_out_tiff:
            try:
                lat_max_bbox = lat_max - r0 * grid_deg
                export_geotiff(args.bbox_out_tiff, bbox, grid_deg, lat_max_bbox, lon_min_bbox)
                print(f"[OK] Lưu GeoTIFF: {args.bbox_out_tiff}")
            except Exception as e:
                print(f"[ERR] Lưu GeoTIFF thất bại: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


