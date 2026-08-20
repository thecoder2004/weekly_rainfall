# 🌧️ Weekly Rainfall Forecasting

This project focuses on **weekly rainfall prediction** using satellite-based datasets such as ECMWF and GSMaP.  
It integrates data preprocessing pipelines with deep learning models including **Vision Transformer (ViT)**.


## 📁 Project Structure
```
.
├── config/                 # Experiment configuration
├── download_ecmwf/         # ECMWF data downloading scripts
├── GsMap/                  # GSMap data processing pipeline
├── src/
│   ├── model/              # Model architectures (VIT)
│   └── utils/              # Training, evaluation, and utilities
├── Final_Data_*.csv        # Processed dataset
└── main.py                 # Entry point
└── main.sh                 # Run project
```
## 🚀 Getting Started
# 1. Installation

Make sure you have Python 3.11 installed.

Then, install the required dependencies:

```
pip install -r requirements.txt
```
## 

# 2. Configuration
Before running the project, you need to configure the dataset paths in `main.sh`.

Open `main.sh` and replace the following placeholders with the actual paths on your machine:

```bash
<YOUR_DATA_IDX_DIR>
<YOUR_GAUGE_DATA_PATH>
<YOUR_NPYARR_DIR>
<YOUR_PROCESSED_ECMWF_DIR>
<YOUR_GSMAP_DATA_PATH>
```
# 3. Run the Project

After installing all dependencies, run the project using:
```
bash main.sh
```