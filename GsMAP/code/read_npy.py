#!/usr/bin/env python3
"""
Script để đọc và hiển thị thông tin file NPY của GSMaP data.

Usage:
  python read_npy.py /path/to/file.npy
  python read_npy.py /path/to/file.npy --show-data
  python read_npy.py /path/to/file.npy --save-plot plot.png
"""

import argparse
import numpy as np
import os
import sys
from typing import Optional

def read_npy_file(file_path: str) -> Optional[np.ndarray]:
    """Đọc file NPY và trả về numpy array"""
    try:
        if not os.path.exists(file_path):
            print(f"❌ File không tồn tại: {file_path}")
            return None
        
        if not file_path.endswith('.npy'):
            print(f"⚠️  File không phải định dạng .npy: {file_path}")
        
        print(f"📁 Đang đọc file: {file_path}")
        data = np.load(file_path)
        print(f"✅ Đọc thành công!")
        return data
    
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return None

def analyze_data(data: np.ndarray, filename: str):
    """Phân tích và hiển thị thông tin về data"""
    print(f"\n📊 THÔNG TIN DATA:")
    print(f"   - Tên file: {os.path.basename(filename)}")
    print(f"   - Shape: {data.shape}")
    print(f"   - Data type: {data.dtype}")
    print(f"   - Size: {data.size:,} elements")
    print(f"   - Memory usage: {data.nbytes / 1024 / 1024:.2f} MB")
    
    # Thống kê cơ bản
    if data.size > 0:
        print(f"\n📈 THỐNG KÊ:")
        print(f"   - Min value: {np.min(data):.6f}")
        print(f"   - Max value: {np.max(data):.6f}")
        print(f"   - Mean: {np.mean(data):.6f}")
        print(f"   - Std: {np.std(data):.6f}")
        
        # Đếm giá trị đặc biệt
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        zero_count = (data == 0).sum()
        negative_count = (data < 0).sum()
        
        print(f"\n🔍 GIÁ TRỊ ĐẶC BIỆT:")
        print(f"   - NaN values: {nan_count:,} ({nan_count/data.size*100:.2f}%)")
        print(f"   - Inf values: {inf_count:,} ({inf_count/data.size*100:.2f}%)")
        print(f"   - Zero values: {zero_count:,} ({zero_count/data.size*100:.2f}%)")
        print(f"   - Negative values: {negative_count:,} ({negative_count/data.size*100:.2f}%)")
        
        # Phân tích theo hàng/cột nếu là 2D
        if len(data.shape) == 2:
            print(f"\n📐 PHÂN TÍCH 2D:")
            print(f"   - Rows: {data.shape[0]}")
            print(f"   - Cols: {data.shape[1]}")
            
            # Tìm hàng/cột có nhiều NaN nhất
            row_nan_counts = np.isnan(data).sum(axis=1)
            col_nan_counts = np.isnan(data).sum(axis=0)
            
            max_nan_row = np.argmax(row_nan_counts)
            max_nan_col = np.argmax(col_nan_counts)
            
            print(f"   - Row có nhiều NaN nhất: {max_nan_row} ({row_nan_counts[max_nan_row]} NaN)")
            print(f"   - Col có nhiều NaN nhất: {max_nan_col} ({col_nan_counts[max_nan_col]} NaN)")

def show_sample_data(data: np.ndarray, sample_size: int = 10):
    """Hiển thị một phần data mẫu"""
    print(f"\n👀 DỮ LIỆU MẪU (first {sample_size} elements):")
    
    if data.size == 0:
        print("   - Data rỗng")
        return
    
    # Flatten data để hiển thị
    flat_data = data.flatten()
    sample_data = flat_data[:sample_size]
    
    for i, value in enumerate(sample_data):
        if np.isnan(value):
            print(f"   [{i:2d}]: NaN")
        elif np.isinf(value):
            print(f"   [{i:2d}]: {'+Inf' if value > 0 else '-Inf'}")
        else:
            print(f"   [{i:2d}]: {value:.6f}")

def create_plot(data: np.ndarray, output_file: str):
    """Tạo plot và lưu vào file"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        
        print(f"\n🎨 Tạo plot và lưu vào: {output_file}")
        
        # Tạo figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'GSMaP Data Analysis - {os.path.basename(output_file)}', fontsize=14)
        
        # Plot 1: Heatmap của data
        ax1 = axes[0, 0]
        if len(data.shape) == 2:
            im1 = ax1.imshow(data, cmap='viridis', aspect='auto')
            ax1.set_title('Data Heatmap')
            ax1.set_xlabel('Column')
            ax1.set_ylabel('Row')
            plt.colorbar(im1, ax=ax1)
        else:
            ax1.plot(data.flatten()[:1000])  # Plot first 1000 points
            ax1.set_title('Data Line Plot (first 1000 points)')
            ax1.set_xlabel('Index')
            ax1.set_ylabel('Value')
        
        # Plot 2: Histogram
        ax2 = axes[0, 1]
        valid_data = data[~np.isnan(data) & ~np.isinf(data)]
        if len(valid_data) > 0:
            ax2.hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
            ax2.set_title('Value Distribution')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
        else:
            ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Value Distribution (No valid data)')
        
        # Plot 3: NaN mask
        ax3 = axes[1, 0]
        if len(data.shape) == 2:
            nan_mask = np.isnan(data)
            ax3.imshow(nan_mask, cmap='Reds', aspect='auto')
            ax3.set_title('NaN Mask (Red = NaN)')
            ax3.set_xlabel('Column')
            ax3.set_ylabel('Row')
        else:
            nan_mask = np.isnan(data.flatten())
            ax3.plot(nan_mask[:1000])
            ax3.set_title('NaN Mask (first 1000 points)')
            ax3.set_xlabel('Index')
            ax3.set_ylabel('Is NaN')
        
        # Plot 4: Statistics by row/column
        ax4 = axes[1, 1]
        if len(data.shape) == 2:
            row_means = np.nanmean(data, axis=1)
            ax4.plot(row_means)
            ax4.set_title('Mean by Row')
            ax4.set_xlabel('Row')
            ax4.set_ylabel('Mean Value')
        else:
            # Rolling mean for 1D data
            window = min(100, len(data) // 10)
            if window > 1:
                rolling_mean = np.convolve(data.flatten(), np.ones(window)/window, mode='valid')
                ax4.plot(rolling_mean)
                ax4.set_title(f'Rolling Mean (window={window})')
                ax4.set_xlabel('Index')
                ax4.set_ylabel('Mean Value')
            else:
                ax4.plot(data.flatten()[:1000])
                ax4.set_title('Data (first 1000 points)')
                ax4.set_xlabel('Index')
                ax4.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Đã lưu plot thành công!")
        
    except ImportError:
        print("❌ Cần cài đặt matplotlib để tạo plot:")
        print("   pip install matplotlib")
    except Exception as e:
        print(f"❌ Lỗi khi tạo plot: {e}")

def main():
    parser = argparse.ArgumentParser(description="Đọc và phân tích file NPY")
    parser.add_argument("file_path", help="Đường dẫn đến file .npy")
    parser.add_argument("--show-data", action="store_true", 
                       help="Hiển thị dữ liệu mẫu")
    parser.add_argument("--save-plot", type=str, default=None,
                       help="Lưu plot vào file (ví dụ: plot.png)")
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Số phần tử mẫu để hiển thị (default: 10)")
    
    args = parser.parse_args()
    
    # Đọc file
    data = read_npy_file(args.file_path)
    if data is None:
        return 1
    
    # Phân tích data
    analyze_data(data, args.file_path)
    
    # Hiển thị dữ liệu mẫu nếu được yêu cầu
    if args.show_data:
        show_sample_data(data, args.sample_size)
    
    # Tạo plot nếu được yêu cầu
    if args.save_plot:
        create_plot(data, args.save_plot)
    
    print(f"\n✅ Hoàn thành phân tích file: {args.file_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
