# Thay đổi trong batch_download_crop.py

## 🔄 Các thay đổi chính

### 1. **Thử tải với R=[0,1] và J=[0,1]**
- **Trước**: Chỉ thử với J=[0,1]
- **Sau**: Thử với tất cả combinations của R=[0,1] và J=[0,1]
- **Lý do**: Tăng khả năng tải thành công file

### 2. **Thêm logging lỗi**
- **Thêm tham số**: `--log-file` (mặc định: `log.txt`)
- **Function mới**: `log_error()` để ghi lỗi vào file
- **Log các lỗi**:
  - Download failed với từng combination R,J
  - Download failed sau khi thử tất cả combinations
  - Crop failed

### 3. **Cải thiện thông báo**
- Hiển thị R và J trong thông báo download
- Ghi timestamp vào log file
- Thông báo khởi tạo log file

## 📝 Cách sử dụng

### Cú pháp cơ bản:
```bash
python batch_download_crop.py --mode v8_hourly_mvk \
  --date-start 2004-01-01 --date-end 2004-01-31 \
  --log-file my_log.txt
```

### Tham số mới:
- `--log-file`: File để ghi log lỗi (mặc định: `log.txt`)

## 🔍 Thứ tự thử download

Với mode `v8_hourly_mvk`, script sẽ thử download theo thứ tự:

1. **R=0, J=0**: `gsmap_mvk.YYYYMMDD.HH00.v8.0000.0.dat.gz`
2. **R=0, J=1**: `gsmap_mvk.YYYYMMDD.HH00.v8.0000.1.dat.gz`
3. **R=1, J=0**: `gsmap_mvk.YYYYMMDD.HH00.v8.1000.0.dat.gz`
4. **R=1, J=1**: `gsmap_mvk.YYYYMMDD.HH00.v8.1000.1.dat.gz`

Nếu thành công ở bất kỳ bước nào, script sẽ dừng và tiếp tục với file tiếp theo.

## 📊 Log file format

```
[2024-01-15 10:30:15] === Bắt đầu batch download - Mode: v8_hourly_mvk ===
[2024-01-15 10:30:15] Khoảng thời gian: 2004-01-01 đến 2004-01-31
[2024-01-15 10:30:20] Download failed: /standard/v8/hourly/2004/01/01/gsmap_mvk.20040101.0000.v8.0000.0.dat.gz (R=0, J=0)
[2024-01-15 10:30:25] Download failed: /standard/v8/hourly/2004/01/01/gsmap_mvk.20040101.0000.v8.0000.1.dat.gz (R=0, J=1)
[2024-01-15 10:30:30] Download failed with all combinations for 20040101 0000
```

## 🧪 Test script

Sử dụng `test_download.py` để test:
```bash
python test_download.py
```

Script này sẽ:
- Test download 1 ngày (2004-01-01)
- Hiển thị log file
- Kiểm tra kết quả
