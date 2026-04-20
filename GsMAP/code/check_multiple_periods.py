#!/usr/bin/env python3
"""
Script để check file completeness cho nhiều khoảng thời gian khác nhau
"""

import os
import datetime as dt
from collections import defaultdict

def check_period(npy_dir: str, date_start: str, date_end: str, period_name: str = ""):
    """Check file completeness cho một khoảng thời gian"""
    
    print(f"\n{'='*60}")
    print(f"📅 KIỂM TRA: {period_name}")
    print(f"📅 Khoảng thời gian: {date_start} đến {date_end}")
    print(f"{'='*60}")
    
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
    
    if len(expected_files) > 0:
        completion_rate = ((len(existing_files) - len(extra)) / len(expected_files) * 100)
        print(f"   - Tỷ lệ hoàn thành: {completion_rate:.1f}%")
    
    # Hiển thị một số file thiếu
    if missing:
        print(f"\n❌ MỘT SỐ FILE THIẾU:")
        missing_list = sorted(list(missing))
        for i, f in enumerate(missing_list[:5]):
            print(f"   - {f}")
        if len(missing_list) > 5:
            print(f"   ... và {len(missing_list) - 5} files khác")
    
    # Phân tích theo tháng
    if missing:
        print(f"\n📅 PHÂN TÍCH THEO THÁNG:")
        monthly_missing = defaultdict(int)
        for f in missing:
            if f.endswith('.npy') and '_' in f:
                date_part = f.split('_')[0]
                if len(date_part) == 8:
                    year_month = date_part[:6]
                    monthly_missing[year_month] += 1
        
        for ym in sorted(monthly_missing.keys()):
            year = ym[:4]
            month = ym[4:6]
            count = monthly_missing[ym]
            print(f"   - {year}-{month}: {count} files thiếu")
    
    return {
        'expected': len(expected_files),
        'existing': len(existing_files),
        'missing': len(missing),
        'extra': len(extra),
        'completion_rate': completion_rate if len(expected_files) > 0 else 0
    }

def main():
    npy_dir = "/mnt/disk3/longnd/env_data/GSMaP/npy"
    
    # Định nghĩa các khoảng thời gian cần check
    periods = [
        ("2004-01-01", "2004-01-31", "Tháng 1/2004"),
        ("2004-02-01", "2004-02-29", "Tháng 2/2004"),
        ("2004-03-01", "2004-03-31", "Tháng 3/2004"),
        ("2004-04-01", "2004-04-30", "Tháng 4/2004"),
        ("2004-05-01", "2004-05-31", "Tháng 5/2004"),
        ("2004-06-01", "2004-06-30", "Tháng 6/2004"),
        ("2004-07-01", "2004-07-31", "Tháng 7/2004"),
        ("2004-08-01", "2004-08-31", "Tháng 8/2004"),
        ("2004-09-01", "2004-09-30", "Tháng 9/2004"),
        ("2004-10-01", "2004-10-31", "Tháng 10/2004"),
        ("2004-11-01", "2004-11-30", "Tháng 11/2004"),
        ("2004-12-01", "2004-12-31", "Tháng 12/2004"),
    ]
    
    results = []
    
    for date_start, date_end, period_name in periods:
        result = check_period(npy_dir, date_start, date_end, period_name)
        result['period'] = period_name
        results.append(result)
    
    # Tổng kết
    print(f"\n{'='*80}")
    print("📊 TỔNG KẾT TẤT CẢ CÁC THÁNG")
    print(f"{'='*80}")
    print(f"{'Tháng':<15} {'Cần thiết':<10} {'Có sẵn':<10} {'Thiếu':<10} {'Thừa':<10} {'Hoàn thành':<12}")
    print("-" * 80)
    
    total_expected = 0
    total_existing = 0
    total_missing = 0
    total_extra = 0
    
    for result in results:
        print(f"{result['period']:<15} {result['expected']:<10,} {result['existing']:<10,} "
              f"{result['missing']:<10,} {result['extra']:<10,} {result['completion_rate']:<11.1f}%")
        
        total_expected += result['expected']
        total_existing += result['existing']
        total_missing += result['missing']
        total_extra += result['extra']
    
    print("-" * 80)
    print(f"{'TỔNG CỘNG':<15} {total_expected:<10,} {total_existing:<10,} "
          f"{total_missing:<10,} {total_extra:<10,} "
          f"{((total_existing - total_extra) / total_expected * 100):<11.1f}%")

if __name__ == "__main__":
    main()
