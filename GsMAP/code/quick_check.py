#!/usr/bin/env python3
"""
Script đơn giản để check nhanh file completeness trong folder npy
"""

import os
import datetime as dt
from collections import defaultdict

def quick_check_npy_folder(npy_dir: str, date_start: str = "2004-01-01", date_end: str = "2004-01-31"):
    """Check nhanh file trong folder npy"""
    
    print(f"🔍 Kiểm tra folder: {npy_dir}")
    print(f"📅 Khoảng thời gian: {date_start} đến {date_end}")
    print("-" * 60)
    
    # Tạo danh sách file cần thiết
    d0 = dt.datetime.strptime(date_start, "%Y-%m-%d").date()
    d1 = dt.datetime.strptime(date_end, "%Y-%m-%d").date()
    
    expected_files = set()
    cur = dt.datetime(d0.year, d0.month, d0.day, 0, 0)
    end = dt.datetime(d1.year, d1.month, d1.day, 23, 0)
    one = dt.timedelta(hours=1)
    
    while cur <= end:
        ymd = cur.strftime("%Y%m%d")
        hh = cur.strftime("%H")
        expected_files.add(f"{ymd}_{hh}00.npy")
        cur += one
    
    # Quét file có sẵn
    existing_files = set()
    if os.path.exists(npy_dir):
        for filename in os.listdir(npy_dir):
            if filename.endswith('.npy'):
                existing_files.add(filename)
    
    # Phân tích
    missing = expected_files - existing_files
    extra = existing_files - expected_files
    
    print(f"📊 KẾT QUẢ:")
    print(f"   - Tổng file cần thiết: {len(expected_files):,}")
    print(f"   - Tổng file có sẵn: {len(existing_files):,}")
    print(f"   - File thiếu: {len(missing):,}")
    print(f"   - File thừa: {len(extra):,}")
    print(f"   - Tỷ lệ hoàn thành: {((len(existing_files) - len(extra)) / len(expected_files) * 100):.1f}%")
    
    if missing:
        print(f"\n❌ MỘT SỐ FILE THIẾU:")
        missing_list = sorted(list(missing))
        for i, f in enumerate(missing_list[:10]):
            print(f"   - {f}")
        if len(missing_list) > 10:
            print(f"   ... và {len(missing_list) - 10} files khác")
    
    # Phân tích theo ngày
    if missing:
        print(f"\n📅 PHÂN TÍCH THEO NGÀY:")
        daily_missing = defaultdict(int)
        for f in missing:
            if f.endswith('.npy') and '_' in f:
                date_part = f.split('_')[0]
                if len(date_part) == 8:
                    daily_missing[date_part] += 1
        
        for date in sorted(daily_missing.keys())[:10]:
            count = daily_missing[date]
            print(f"   - {date}: {count} files thiếu")
        if len(daily_missing) > 10:
            print(f"   ... và {len(daily_missing) - 10} ngày khác")

if __name__ == "__main__":
    npy_dir = "/mnt/disk3/longnd/env_data/GSMaP/npy"
    quick_check_npy_folder(npy_dir, "2004-01-01", "2004-01-31")
