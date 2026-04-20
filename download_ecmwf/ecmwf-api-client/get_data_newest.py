from ecmwfapi import ECMWFDataServer
import os
server = ECMWFDataServer()

BASE_DIR = "/mnt/disk3/longnd/env_data/S2S_0.125_newest/raw_data"
step_6h_params = ["121", "122", "165", "166", "228"] #  "165", "166", "121", , , "122"
step_24h_params = ["143", "146", "147", "175", "176", "177", "179", "174008"]
date = "2025-05-14"

for param in step_6h_params:
    step_interval = 6
    if param in ["121", "122"]:
            step_values = "/".join(str(i) for i in range(6, 1104 + 1, step_interval))
    else:
            step_values = "/".join(str(i) for i in range(0, 1104 + 1, step_interval))

    param_dir = os.path.join(BASE_DIR, 'Step6h', param)
    os.makedirs(param_dir, exist_ok=True)
    target_file = os.path.join(param_dir, f"{date[-5:]}.nc")

    if os.path.exists(target_file):
        print(f"File {target_file} đã tồn tại, bỏ qua.")
        continue
    
    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date,
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": "ecmf",
        "param": param,
        "step": step_values,
        "stream": "enfo",
        "time": "00:00:00",
        "type": "cf",
        "target": target_file,
        "area": "25/100/8/115",
        "grid": "0.125/0.125",
        "format": "netcdf",
    })

for param in step_24h_params:
    step_interval = 24
    step_values = "/".join(str(i) for i in range(0, 1104 + 1, step_interval))

    param_dir = os.path.join(BASE_DIR, 'Step24h', param)
    os.makedirs(param_dir, exist_ok=True)
    target_file = os.path.join(param_dir, f"{date[-5:]}.nc")

    if os.path.exists(target_file):
        print(f"File {target_file} đã tồn tại, bỏ qua.")
        continue

    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date,
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": "ecmf",
        "param": param,
        "step": step_values,
        "stream": "enfo",
        "time": "00:00:00",
        "type": "cf",
        "target": target_file,
        "area": "25/100/8/115",
        "grid": "0.125/0.125",
        "format": "netcdf",
    })