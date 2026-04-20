# 🌧️ Weekly Rainfall Forecasting

This project focuses on **weekly rainfall prediction** using satellite-based datasets such as ECMWF and GSMaP.  
It integrates data preprocessing pipelines with deep learning models including **Vision Transformer (ViT)**.

---

## 📁 Project Structure
.
├── config/                 # Experiment configuration
├── download_ecmwf/         # ECMWF data downloading scripts
├── GsMap/                  # GSMap data processing pipeline
├── src/
│   ├── model/              # Model architectures (VIT)
│   └── utils/              # Training, evaluation, and utilities
├── Final_Data_*.csv        # Processed dataset
└── main.py                 # Entry point