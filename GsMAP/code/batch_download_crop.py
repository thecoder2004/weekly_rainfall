#!/usr/bin/env python3
"""
Batch downloader + cropper for GSMaP data.

Modes supported:
- v6_daily01 (default):
  Remote: /realtime_ver/v6/daily0.1/00Z-23Z/YYYYMM/gsmap_nrt.YYYYMMDD.0.1d.daily.00Z-23Z.dat.gz
  Grid default: rows=1200 cols=3600 grid=0.1 lat_max=60 lon_min=0

- v8_hourly_mvk:
  Remote: /standard/v8/hourly/YYYY/MM/DD/gsmap_mvk.YYYYMMDD.HH00.v8.0000.0.dat.gz
  Grid default: rows=1800 cols=3600 grid=0.1 lat_max=90 lon_min=-180

After each download, crop a bbox to NumPy array (.npy) and GeoTIFF via read_data.py, then delete the .dat file.

Examples (PowerShell):
  # Tải 2004..2023 daily (v6) cho bbox VN và lưu .\out
  python batch_download_crop.py --mode v6_daily01 --year-start 2004 --year-end 2023 \
    --bbox-lat-min 8 --bbox-lat-max 24.5 --bbox-lon-min 102 --bbox-lon-max 110 \
    --out-root .\out

  # Tải 1 ngày hourly (v8) theo giờ
  python batch_download_crop.py --mode v8_hourly_mvk --date-start 2024-10-22 --date-end 2024-10-22 \
    --bbox-lat-min 8 --bbox-lat-max 24.5 --bbox-lon-min 102 --bbox-lon-max 110 \
    --out-root .\out

Notes:
- Requires get_data.py and read_data.py in the same directory.
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
from typing import Optional, List, Tuple

try:
    from tqdm import tqdm as _tqdm
    def progress(iterable, total: Optional[int] = None, desc: Optional[str] = None):
        return _tqdm(iterable, total=total, desc=desc)
except Exception:
    def progress(iterable, total: Optional[int] = None, desc: Optional[str] = None):
        return iterable

def run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout, end="")
    return proc.returncode

def log_error(log_file: str, message: str):
    """Ghi lỗi vào file log"""
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"[WARN] Cannot write to log file: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch download + crop GSMaP data")
    parser.add_argument("--mode", choices=["v6_daily01", "v8_hourly_mvk"], default="v8_hourly_mvk")
    # Date range for daily mode
    parser.add_argument("--year-start", type=int, default=2004)
    parser.add_argument("--year-end", type=int, default=2023)
    # Date range for hourly mode
    parser.add_argument("--date-start", type=str, default='2004-01-01', help="YYYY-MM-DD (for hourly mode)")
    parser.add_argument("--date-end", type=str, default='2023-12-31', help="YYYY-MM-DD (for hourly mode)")
    parser.add_argument("--out-npy", default="../npy", help="Thư mục chứa file NPY")
    parser.add_argument("--out-tiff", default="../tiff", help="Tên thư mục chứa file TIFF")
    parser.add_argument("--remote-root", default=None, help="Gốc remote. Tự đặt theo mode nếu bỏ trống")
    # Grid defaults (will be overridden by mode if not provided)
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--cols", type=int, default=3600)
    parser.add_argument("--grid", type=float, default=0.1)
    parser.add_argument("--lat-max", type=float, default=60)
    parser.add_argument("--lon-min", type=float, default=0)
    # BBox to crop (set defaults as requested)
    parser.add_argument("--bbox-lat-min", type=float, default=8.0)
    parser.add_argument("--bbox-lat-max", type=float, default=24.0)
    parser.add_argument("--bbox-lon-min", type=float, default=102.0)
    parser.add_argument("--bbox-lon-max", type=float, default=117.0)
    # Control
    parser.add_argument("--skip-existing", action="store_true", help="Bỏ qua nếu CSV/TIFF đã tồn tại")
    parser.add_argument("--retries", type=int, default=2, help="Số lần retry tải 1 file khi lỗi")
    parser.add_argument("--raw-dir", default="../raw", help="Thư mục tạm lưu file tải về")
    parser.add_argument("--log-file", default="../log.txt", help="File log để ghi lại lỗi")

    args = parser.parse_args()

    os.makedirs(args.out_npy, exist_ok=True)
    os.makedirs(args.out_tiff, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)
    
    # Khởi tạo log file
    log_error(args.log_file, f"=== Bắt đầu batch download - Mode: {args.mode} ===")
    if args.mode == "v8_hourly_mvk":
        log_error(args.log_file, f"Khoảng thời gian: {args.date_start} đến {args.date_end}")
    else:
        log_error(args.log_file, f"Khoảng thời gian: {args.year_start} đến {args.year_end}")

    def date_iter(y0: int, y1: int):
        cur = dt.date(y0, 1, 1)
        end = dt.date(y1, 12, 31)
        one = dt.timedelta(days=1)
        while cur <= end:
            yield cur
            cur += one

    def hour_iter(d0: dt.date, d1: dt.date):
        cur = dt.datetime(d0.year, d0.month, d0.day, 0, 0)
        end = dt.datetime(d1.year, d1.month, d1.day, 23, 0)
        one = dt.timedelta(hours=1)
        while cur <= end:
            yield cur
            cur += one

    py = sys.executable
    base_dir = os.path.dirname(os.path.abspath(__file__))
    get_data_py = os.path.join(base_dir, "get_data.py")
    read_data_py = os.path.join(base_dir, "read_data.py")

    # Configure mode defaults
    mode = args.mode
    if mode == "v6_daily01":
        remote_root = args.remote_root or "/realtime_ver/v6/daily0.1/00Z-23Z"
        rows = args.rows if args.rows is not None else 1200
        cols = args.cols if args.cols is not None else 3600
        grid = args.grid if args.grid is not None else 0.1
        lat_max = args.lat_max if args.lat_max is not None else 60.0
        lon_min = args.lon_min if args.lon_min is not None else 0.0
        # Loop by day
        # total days inclusive
        start_date = dt.date(args.year_start, 1, 1)
        end_date = dt.date(args.year_end, 12, 31)
        num_days = (end_date - start_date).days + 1
        for day in progress(date_iter(args.year_start, args.year_end), total=num_days, desc="v6 daily"):
            ymd = day.strftime("%Y%m%d")
            ym = day.strftime("%Y%m")
            filename = f"gsmap_nrt.{ymd}.0.1d.daily.00Z-23Z.dat.gz"
            remote_file = f"{remote_root}/{ym}/{filename}"

            out_npy = os.path.join(args.out_npy, f"{ymd}.npy")
            out_tif = os.path.join(args.out_tiff, f"{ymd}.tif")
            if args.skip_existing and os.path.exists(out_npy) and os.path.exists(out_tif):
                print(f"[SKIP] {ymd} outputs existed")
                continue

            raw_path = os.path.join(args.raw_dir, filename)
            for attempt in range(args.retries + 1):
                print(f"[INFO] Download {remote_file} -> {raw_path} (try {attempt+1})")
                code = run([py, get_data_py, "--remote-file", remote_file, "--out", args.raw_dir, "--out-file", filename])
                if code == 0:
                    break
            else:
                error_msg = f"Download failed after {args.retries+1} tries: {remote_file}"
                print(f"[FAIL] {error_msg}")
                log_error(args.log_file, error_msg)
                continue

            cmd = [
                py, read_data_py,
                "--input", raw_path,
                "--rows", str(rows),
                "--cols", str(cols),
                "--grid", str(grid),
                "--lat-max", str(lat_max),
                "--lon-min", str(lon_min),
                "--bbox-lat-min", str(args.bbox_lat_min),
                "--bbox-lat-max", str(args.bbox_lat_max),
                "--bbox-lon-min", str(args.bbox_lon_min),
                "--bbox-lon-max", str(args.bbox_lon_max),
                "--bbox-out-npy", out_npy,
                "--bbox-out-tiff", out_tif,
            ]
            print(f"[INFO] Crop {ymd} -> {out_npy}, {out_tif}")
            code = run(cmd)
            if code != 0:
                error_msg = f"Crop failed for {ymd}"
                print(f"[FAIL] {error_msg}")
                log_error(args.log_file, error_msg)
                continue

            # Xóa file .dat đã giải nén
            dat_path = raw_path[:-3] if raw_path.endswith('.gz') else None
            if dat_path and os.path.exists(dat_path):
                try:
                    os.remove(dat_path)
                    print(f"[INFO] Deleted {dat_path}")
                except Exception as e:
                    print(f"[WARN] Cannot delete {dat_path}: {e}")
            
            # Xóa file .dat.gz gốc
            if os.path.exists(raw_path):
                try:
                    os.remove(raw_path)
                    print(f"[INFO] Deleted {raw_path}")
                except Exception as e:
                    print(f"[WARN] Cannot delete {raw_path}: {e}")

    elif mode == "v8_hourly_mvk":
        try:
            d0 = dt.datetime.strptime(args.date_start, "%Y-%m-%d").date()
            d1 = dt.datetime.strptime(args.date_end, "%Y-%m-%d").date()
        except Exception as e:
            print(f"[ERR] Sai định dạng ngày: {e}")
            return 1

        remote_root = args.remote_root or "/standard/v8/hourly"
        rows = args.rows if args.rows is not None else 1200
        cols = args.cols if args.cols is not None else 3600
        grid = args.grid if args.grid is not None else 0.1
        lat_max = args.lat_max if args.lat_max is not None else 60.0
        lon_min = args.lon_min if args.lon_min is not None else 0.0

        # total hours inclusive (from 00:00 of d0 to 23:00 of d1)
        total_hours = int(((dt.datetime(d1.year, d1.month, d1.day, 23) - dt.datetime(d0.year, d0.month, d0.day, 0)).total_seconds() // 3600) + 1)
        for ts in progress(hour_iter(d0, d1), total=total_hours, desc="v8 hourly"):
            ymd = ts.strftime("%Y%m%d")
            ym = ts.strftime("%Y%m")
            y = ts.strftime("%Y")
            m = ts.strftime("%m")
            d = ts.strftime("%d")
            hh = ts.strftime("%H")
            out_npy = os.path.join(args.out_npy, f"{ymd}_{hh}00.npy")
            out_tif = os.path.join(args.out_tiff, f"{ymd}_{hh}00.tif")
            if args.skip_existing and os.path.exists(out_npy) and os.path.exists(out_tif):
                print(f"[SKIP] {ymd} {hh}00 outputs existed")
                continue

            # Thử tải với R=[0,1] và J=[0,1]
            download_success = False
            for R in [0, 1]:
                for J in [0, 1]:
                    filename = f"gsmap_mvk.{ymd}.{hh}00.v8.{R}000.{J}.dat.gz"
                    remote_file = f"{remote_root}/{y}/{m}/{d}/{filename}"
                    raw_path = os.path.join(args.raw_dir, filename)
                    
                    print(f"[INFO] Download {remote_file} -> {raw_path} (R={R}, J={J})")
                    code = run([py, get_data_py, "--remote-file", remote_file, "--out", args.raw_dir, "--out-file", filename])
                    if code == 0:
                        download_success = True
                        break
                    else:
                        print(f"[WARN] Download failed with R={R}, J={J}")
                        log_error(args.log_file, f"Download failed: {remote_file} (R={R}, J={J})")
                
                if download_success:
                    break
            
            if not download_success:
                error_msg = f"Download failed with all combinations for {ymd} {hh}00"
                print(f"[FAIL] {error_msg}")
                log_error(args.log_file, error_msg)
                continue

            cmd = [
                py, read_data_py,
                "--input", raw_path,
                "--rows", str(rows),
                "--cols", str(cols),
                "--grid", str(grid),
                "--lat-max", str(lat_max),
                "--lon-min", str(lon_min),
                "--bbox-lat-min", str(args.bbox_lat_min),
                "--bbox-lat-max", str(args.bbox_lat_max),
                "--bbox-lon-min", str(args.bbox_lon_min),
                "--bbox-lon-max", str(args.bbox_lon_max),
                "--bbox-out-npy", out_npy,
                "--bbox-out-tiff", out_tif,
            ]
            print(f"[INFO] Crop {ymd} {hh}00 -> {out_npy}, {out_tif}")
            code = run(cmd)
            if code != 0:
                error_msg = f"Crop failed for {ymd} {hh}00"
                print(f"[FAIL] {error_msg}")
                log_error(args.log_file, error_msg)
                continue

            # Xóa file .dat đã giải nén
            dat_path = raw_path[:-3] if raw_path.endswith('.gz') else None
            if dat_path and os.path.exists(dat_path):
                try:
                    os.remove(dat_path)
                    print(f"[INFO] Deleted {dat_path}")
                except Exception as e:
                    print(f"[WARN] Cannot delete {dat_path}: {e}")
            
            # Xóa file .dat.gz gốc
            if os.path.exists(raw_path):
                try:
                    os.remove(raw_path)
                    print(f"[INFO] Deleted {raw_path}")
                except Exception as e:
                    print(f"[WARN] Cannot delete {raw_path}: {e}")

    print("[DONE] Batch completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


