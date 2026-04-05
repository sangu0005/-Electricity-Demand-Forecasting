# ⚡ Electricity Demand Forecasting

A machine learning project that forecasts electricity demand using historical consumption data, weather features, and time-series engineering — built with **XGBoost** and trained on hourly electricity records.

---

## 📌 Problem Statement

Accurate electricity demand forecasting is critical for grid operators, energy providers, and policymakers. Over- or under-estimating demand leads to either wasteful overgeneration or damaging power shortages. This project builds a data-driven forecasting pipeline that predicts future electricity demand (in MW) from historical patterns and environmental signals.

---

## 📊 Dataset

| Field | Description |
|---|---|
| `Timestamp` | Hourly datetime index |
| `Temperature` | Ambient temperature (°C) |
| `Humidity` | Relative humidity (%) |
| `Demand` | Electricity demand in MW *(target)* |
| `hour`, `month`, `year` | Extracted time components |
| `dayofweek`, `dayofyear` | Calendar features |

> **Note:** Dataset is stored at `Data/Electricity_Demand_Dataset.csv`.

---

## 🔧 Project Pipeline

```
Raw Data → Cleaning → Feature Engineering → EDA → Model Training → Evaluation → Export
```

### 1. Data Cleaning
- Dropped rows where **all** values were null
- Applied **forward fill** on time-based columns (`hour`, `month`, `year`, etc.)
- Applied **backward fill** on `Temperature` and `Humidity`
- Used **time-based interpolation** on `Demand` to preserve temporal continuity

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `quarter` | Calendar quarter (1–4) |
| `weekofyear` | ISO week number |
| `is_weekend` | Binary flag (1 = Saturday/Sunday) |
| `Demand_lag_24hr` | Demand from 24 hours ago |
| `Demand_lag_168hr` | Demand from 7 days ago (same hour, last week) |
| `Demand_rolling_mean_24hr` | 24-hour rolling average demand |
| `Demand_rolling_std_24hr` | 24-hour rolling standard deviation |

### 3. Exploratory Data Analysis
- Demand trend over time
- Demand distribution by hour and month
- Demand vs. Temperature correlation
- Full feature correlation heatmap

### 4. Model — XGBoost Regressor
- **Train set:** All data up to `2023-12-31`
- **Test set:** Data from `2024-01-01` onwards
- Time-based split (no shuffling) to prevent data leakage

```python
XGBRegressor(
    n_estimators       = 1000,
    early_stopping_rounds = 50,
    learning_rate      = 0.01,
    objective          = 'reg:squarederror',
    random_state       = 42
)
```

---

## 📈 Results

| Metric | Score |
|---|---|
| **RMSE** | *175.07965552482665* |
| **MAE** | *123.32653722857253* |
| **R² Score** | *0.9846467164984454* |

Actual vs. predicted demand is visualized across the 2024 test window.

---

## 🗂️ Project Structure

```
Electricity-Demand-Forecasting/
│
├── Data/
│   └── Electricity_Demand_Dataset.csv
│
├── Electricity_FC.ipynb          # Main notebook
├── Electricity_xgb_prediction_model.pkl  # Saved XGBoost model
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn joblib
```

### Run the Notebook

```bash
git clone https://github.com/your-username/electricity-demand-forecasting.git
cd electricity-demand-forecasting
jupyter notebook Electricity_FC.ipynb
```

### Load the Saved Model

```python
import joblib

model = joblib.load('Electricity_xgb_prediction_model.pkl')
predictions = model.predict(X_features)
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-lightblue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Metrics-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-teal)

---

## 🙋 Author

**Sangamesh Hallur**  
B.E. Data Science — Sri Siddhartha Institute of Technology, Bengaluru  
[LinkedIn](https://linkedin.com/in/your-profile) • [GitHub](https://github.com/your-sangu0005)

---
