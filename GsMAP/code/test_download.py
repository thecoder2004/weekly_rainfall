#!/usr/bin/env python3
"""
Script test để kiểm tra download với R=[0,1] và J=[0,1]
"""

import subprocess
import sys
import os

def test_download():
    """Test download với một vài file mẫu"""
    
    # Tham số test
    mode = "v8_hourly_mvk"
    date_start = "2004-01-01"
    date_end = "2004-01-01"  # Chỉ test 1 ngày
    npy_dir = "/mnt/disk3/longnd/env_data/GSMaP/npy"
    tiff_dir = "/mnt/disk3/longnd/env_data/GSMaP/tiff"
    raw_dir = "/mnt/disk3/longnd/env_data/GSMaP/raw"
    log_file = "test_log.txt"
    
    # Tạo thư mục nếu chưa có
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(tiff_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    # Xóa log file cũ nếu có
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Chạy script
    cmd = [
        sys.executable, "batch_download_crop.py",
        "--mode", mode,
        "--date-start", date_start,
        "--date-end", date_end,
        "--out-npy", npy_dir,
        "--out-tiff", tiff_dir,
        "--raw-dir", raw_dir,
        "--log-file", log_file,
        "--skip-existing"
    ]
    
    print("🚀 Bắt đầu test download...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        print(f"\n✅ Script hoàn thành với exit code: {result.returncode}")
        
        # Kiểm tra log file
        if os.path.exists(log_file):
            print(f"\n📄 Nội dung log file ({log_file}):")
            print("-" * 40)
            with open(log_file, 'r', encoding='utf-8') as f:
                print(f.read())
        else:
            print("⚠️  Không tìm thấy log file")
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy script: {e}")

if __name__ == "__main__":
    test_download()
