#!/usr/bin/env python3
"""
Script để kiểm tra file completeness trong folder npy/tiff của GSMaP data.

Hỗ trợ 2 mode:
- v6_daily01: Kiểm tra file daily (YYYYMMDD.npy)
- v8_hourly_mvk: Kiểm tra file hourly (YYYYMMDD_HH00.npy)

Usage:
  python check_file_completeness.py --mode v8_hourly_mvk --date-start 2004-01-01 --date-end 2004-01-31 --npy-dir /path/to/npy
  python check_file_completeness.py --mode v6_daily01 --year-start 2004 --year-end 2004 --npy-dir /path/to/npy
"""

import argparse
import datetime as dt
import os
import sys
from typing import List, Set, Tuple, Optional
from collections import defaultdict

def date_iter(y0: int, y1: int):
    """Iterator cho các ngày từ y0 đến y1"""
    cur = dt.date(y0, 1, 1)
    end = dt.date(y1, 12, 31)
    one = dt.timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one

def hour_iter(d0: dt.date, d1: dt.date):
    """Iterator cho các giờ từ d0 00:00 đến d1 23:00"""
    cur = dt.datetime(d0.year, d0.month, d0.day, 0, 0)
    end = dt.datetime(d1.year, d1.month, d1.day, 23, 0)
    one = dt.timedelta(hours=1)
    while cur <= end:
        yield cur
        cur += one

def get_expected_files_daily(year_start: int, year_end: int) -> Set[str]:
    """Tạo danh sách file cần thiết cho mode daily"""
    expected = set()
    for day in date_iter(year_start, year_end):
        ymd = day.strftime("%Y%m%d")
        expected.add(f"{ymd}.npy")
    return expected

def get_expected_files_hourly(date_start: str, date_end: str) -> Set[str]:
    """Tạo danh sách file cần thiết cho mode hourly"""
    try:
        d0 = dt.datetime.strptime(date_start, "%Y-%m-%d").date()
        d1 = dt.datetime.strptime(date_end, "%Y-%m-%d").date()
    except Exception as e:
        print(f"[ERR] Sai định dạng ngày: {e}")
        return set()
    
    expected = set()
    for ts in hour_iter(d0, d1):
        ymd = ts.strftime("%Y%m%d")
        hh = ts.strftime("%H")
        expected.add(f"{ymd}_{hh}00.npy")
    return expected

def scan_existing_files(npy_dir: str, tiff_dir: Optional[str] = None) -> Tuple[Set[str], Set[str]]:
    """Quét file có sẵn trong folder npy và tiff"""
    npy_files = set()
    tiff_files = set()
    
    # Scan npy files
    if os.path.exists(npy_dir):
        for filename in os.listdir(npy_dir):
            if filename.endswith('.npy'):
                npy_files.add(filename)
    
    # Scan tiff files nếu có
    if tiff_dir and os.path.exists(tiff_dir):
        for filename in os.listdir(tiff_dir):
            if filename.endswith('.tif') or filename.endswith('.tiff'):
                tiff_files.add(filename)
    
    return npy_files, tiff_files

def analyze_missing_files(expected: Set[str], existing: Set[str]) -> Tuple[List[str], List[str], int, int]:
    """Phân tích file missing và extra"""
    missing = sorted(expected - existing)
    extra = sorted(existing - expected)
    total_expected = len(expected)
    total_existing = len(existing)
    
    return missing, extra, total_expected, total_existing

def print_report(mode: str, missing: List[str], extra: List[str], 
                total_expected: int, total_existing: int, 
                npy_dir: str, tiff_dir: Optional[str] = None):
    """In báo cáo chi tiết"""
    print("=" * 80)
    print(f"BÁO CÁO KIỂM TRA FILE COMPLETENESS - MODE: {mode}")
    print("=" * 80)
    print(f"Folder NPY: {npy_dir}")
    if tiff_dir:
        print(f"Folder TIFF: {tiff_dir}")
    print()
    
    print(f"📊 TỔNG QUAN:")
    print(f"   - Tổng file cần thiết: {total_expected:,}")
    print(f"   - Tổng file có sẵn: {total_existing:,}")
    print(f"   - File thiếu: {len(missing):,}")
    print(f"   - File thừa: {len(extra):,}")
    print(f"   - Tỷ lệ hoàn thành: {((total_existing - len(extra)) / total_expected * 100):.1f}%")
    print()
    
    if missing:
        print(f"❌ FILE THIẾU ({len(missing)} files):")
        if len(missing) <= 20:
            for f in missing:
                print(f"   - {f}")
        else:
            for f in missing[:10]:
                print(f"   - {f}")
            print(f"   ... và {len(missing) - 10} files khác")
        print()
    
    if extra:
        print(f"⚠️  FILE THỪA ({len(extra)} files):")
        if len(extra) <= 20:
            for f in extra:
                print(f"   - {f}")
        else:
            for f in extra[:10]:
                print(f"   - {f}")
            print(f"   ... và {len(extra) - 10} files khác")
        print()
    
    # Phân tích theo tháng (chỉ cho hourly mode)
    if mode == "v8_hourly_mvk" and missing:
        print("📅 PHÂN TÍCH FILE THIẾU THEO THÁNG:")
        monthly_missing = defaultdict(int)
        for f in missing:
            if f.endswith('.npy'):
                date_part = f.split('_')[0]
                if len(date_part) == 8:
                    year_month = date_part[:6]
                    monthly_missing[year_month] += 1
        
        for ym in sorted(monthly_missing.keys()):
            year = ym[:4]
            month = ym[4:6]
            count = monthly_missing[ym]
            print(f"   - {year}-{month}: {count} files thiếu")
        print()

def save_missing_list(missing: List[str], output_file: str):
    """Lưu danh sách file thiếu vào file"""
    try:
        with open(output_file, 'w') as f:
            for filename in missing:
                f.write(f"{filename}\n")
        print(f"💾 Đã lưu danh sách file thiếu vào: {output_file}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra file completeness cho GSMaP data")
    parser.add_argument("--mode", choices=["v6_daily01", "v8_hourly_mvk"], default="v8_hourly_mvk",
                       help="Mode kiểm tra")
    
    # Tham số cho daily mode
    parser.add_argument("--year-start", type=int, default=2004, help="Năm bắt đầu (daily mode)")
    parser.add_argument("--year-end", type=int, default=2004, help="Năm kết thúc (daily mode)")
    
    # Tham số cho hourly mode
    parser.add_argument("--date-start", type=str, default='2004-01-01', 
                       help="Ngày bắt đầu YYYY-MM-DD (hourly mode)")
    parser.add_argument("--date-end", type=str, default='2004-01-31', 
                       help="Ngày kết thúc YYYY-MM-DD (hourly mode)")
    
    # Tham số folder
    parser.add_argument("--npy-dir", default="/mnt/disk3/longnd/env_data/GSMaP/npy",
                       help="Thư mục chứa file NPY")
    parser.add_argument("--tiff-dir", default=None,
                       help="Thư mục chứa file TIFF (optional)")
    
    # Tham số output
    parser.add_argument("--output-missing", type=str, default=None,
                       help="File để lưu danh sách file thiếu")
    parser.add_argument("--quiet", action="store_true", help="Chỉ hiển thị tổng quan")
    
    args = parser.parse_args()
    
    # Tạo danh sách file cần thiết
    if args.mode == "v6_daily01":
        expected = get_expected_files_daily(args.year_start, args.year_end)
    elif args.mode == "v8_hourly_mvk":
        expected = get_expected_files_hourly(args.date_start, args.date_end)
    else:
        print(f"[ERR] Mode không hỗ trợ: {args.mode}")
        return 1
    
    if not expected:
        print("[ERR] Không có file nào cần kiểm tra")
        return 1
    
    # Quét file có sẵn
    npy_files, tiff_files = scan_existing_files(args.npy_dir, args.tiff_dir)
    
    # Phân tích
    missing, extra, total_expected, total_existing = analyze_missing_files(expected, npy_files)
    
    # In báo cáo
    if not args.quiet:
        print_report(args.mode, missing, extra, total_expected, total_existing, 
                    args.npy_dir, args.tiff_dir)
    else:
        print(f"Expected: {total_expected}, Existing: {total_existing}, Missing: {len(missing)}, Extra: {len(extra)}")
    
    # Lưu danh sách file thiếu nếu được yêu cầu
    if args.output_missing and missing:
        save_missing_list(missing, args.output_missing)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
