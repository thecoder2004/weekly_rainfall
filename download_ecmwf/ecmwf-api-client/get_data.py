from ecmwfapi import ECMWFDataServer
import datetime
import os

server = ECMWFDataServer()

BASE_DIR = "/mnt/disk3/longnd/env_data/S2S_0.125/raw_data/Step6h"
step_6h_params = ["122"] #  "165", "166", "121", , , "122"
# 121 : max temperature per 6 hours
# 122 : min temperature per 6 hours
step_24h_params = ["143", "146", "147", "175", "176", "177", "179", "174008"]

start_date = datetime.date(2024, 1, 1)
end_date = datetime.date(2024, 12, 31)

dates = []
current_date = start_date
day_steps = [3, 4]  # Trước 11-11
step_index = 0

# Tạo danh sách dates
while current_date <= end_date:
    dates.append(current_date.strftime("%Y-%m-%d"))
    # Từ 11-11 trở đi dùng bước 2 ngày
    if current_date >= datetime.date(2024, 11, 11):
        current_date += datetime.timedelta(days=2)
    else:
        current_date += datetime.timedelta(days=day_steps[step_index])
        step_index = 1 - step_index

years = list(range(2004, 2024))

def log_error(message):
    with open("error.log", "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {message}\n")

def download_data(param, step_interval):
    param_dir = os.path.join(BASE_DIR, param)
    os.makedirs(param_dir, exist_ok=True)

    for date in dates:
        # Xử lý đặc biệt cho ngày 29-2
        if date == "2024-02-29":
            hdate_list = [f"{year}-02-28" for year in years]
        else:
            hdate_list = [f"{year}-{date[5:]}" for year in years]
        
        hdate_str = "/".join(hdate_list)
        target_file = os.path.join(param_dir, f"{date[-5:]}.nc")

        if os.path.exists(target_file):
            print(f"File {target_file} đã tồn tại, bỏ qua.")
            continue

        print(f"Downloading param {param} for {date} ...")
        # 1104/6/4=46
        if param in ["121", "122"]:
            step_values = "/".join(str(i) for i in range(6, 1104 + 1, step_interval))
        else:
            step_values = "/".join(str(i) for i in range(0, 1104 + 1, step_interval))

        try:
            server.retrieve(
                {
                    "class": "s2",
                    "dataset": "s2s",
                    "date": date,
                    "expver": "prod",
                    "hdate": hdate_str,
                    "levtype": "sfc",
                    "model": "glob",
                    "origin": "ecmf",
                    "param": param,
                    "area": "42.5/90/0/126",
                    "grid": "0.125/0.125",
                    "format": "netcdf",
                    "step": step_values,
                    "stream": "enfh",
                    "time": "00:00:00",
                    "type": "cf",
                    "target": target_file,
                }
            )
            print(f"Saved {target_file}\n")
        except Exception as e:
            error_message = f"Lỗi tải param {param} cho ngày {date}: {e}"
            print(error_message)
            log_error(error_message)
            continue

def download_data_individual(param, date, step_interval):
    param_dir = os.path.join(BASE_DIR, param)
    os.makedirs(param_dir, exist_ok=True)

    numbers = list(range(1, 11))
    for number in numbers:
        for year in years:
            # Xử lý đặc biệt cho ngày 29-2
            if date == "2024-02-29":
                hdate = f"{year}-02-28"
            else:
                hdate = f"{year}-{date[5:]}"
                
            target_file = os.path.join(param_dir, f"{date}_number{number}_hdate{year}.nc")

            if os.path.exists(target_file):
                print(f"File {target_file} đã tồn tại, bỏ qua.")
                continue

            print(f"Downloading param {param}, number {number}, hdate {hdate} for {date} ...")
            if param in ["121", "122"]:
                step_values = "/".join(str(i) for i in range(6, 1104 + 1, step_interval))
            else:
                step_values = "/".join(str(i) for i in range(0, 1104 + 1, step_interval))

            try:
                server.retrieve(
                    {
                        "class": "s2",
                        "dataset": "s2s",
                        "date": date,
                        "expver": "prod",
                        "hdate": hdate,
                        "levtype": "sfc",
                        "model": "glob",
                        "number": str(number),
                        "origin": "ecmf",
                        "param": param,
                        "area": "25/100/8/115",
                        "format": "netcdf",
                        "step": step_values,
                        "stream": "enfh",
                        "time": "00:00:00",
                        "type": "pf",
                        "target": target_file,
                    }
                )
                print(f"Saved {target_file}\n")
            except Exception as e:
                error_message = f"Lỗi tải param {param}, number {number}, hdate {hdate} cho ngày {date}: {e}"
                print(error_message)
                log_error(error_message)
                continue

for param in step_6h_params:
    download_data(param, 6)

# for param in step_24h_params:
#     download_data(param, 24)