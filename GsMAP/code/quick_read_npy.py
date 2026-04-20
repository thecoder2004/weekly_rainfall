#!/usr/bin/env python3
"""
Script đơn giản để đọc nhanh file NPY
"""

import numpy as np
import sys
import os

def quick_read_npy(file_path: str):
    """Đọc nhanh file NPY và hiển thị thông tin cơ bản"""
    
    if not os.path.exists(file_path):
        print(f"❌ File không tồn tại: {file_path}")
        return
    
    try:
        print(f"📁 Đọc file: {file_path}")
        data = np.load(file_path)
        
        print(f"✅ Thành công!")
        print(f"   - Shape: {data.shape}")
        print(f"   - Type: {data.dtype}")
        print(f"   - Size: {data.size:,} elements")
        print(f"   - Memory: {data.nbytes / 1024 / 1024:.2f} MB")
        
        if data.size > 0:
            print(f"   - Min: {np.min(data):.6f}")
            print(f"   - Max: {np.max(data):.6f}")
            print(f"   - Mean: {np.mean(data):.6f}")
            
            # Kiểm tra NaN và Inf
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            
            if nan_count > 0:
                print(f"   - NaN: {nan_count:,} ({nan_count/data.size*100:.1f}%)")
            if inf_count > 0:
                print(f"   - Inf: {inf_count:,} ({inf_count/data.size*100:.1f}%)")
            
            # Hiển thị một vài giá trị mẫu
            print(f"\n👀 Mẫu dữ liệu (5 phần tử đầu):")
            flat_data = data.flatten()
            for i in range(min(5, len(flat_data))):
                value = flat_data[i]
                if np.isnan(value):
                    print(f"   [{i}]: NaN")
                elif np.isinf(value):
                    print(f"   [{i}]: {'+Inf' if value > 0 else '-Inf'}")
                else:
                    print(f"   [{i}]: {value:.6f}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_read_npy.py <file.npy>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    quick_read_npy(file_path)
