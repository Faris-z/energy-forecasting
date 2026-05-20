# ⚡ energy-forecasting

[![CI](https://github.com/Faris-z/energy-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Faris-z/energy-forecasting/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-orange)](https://mlflow.org/)

Production-grade deep learning pipeline for 24-hour-ahead household energy forecasting using the [UCI Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption) dataset.

---

## 🚀 Live Demo
[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/Faris-zh/energy-forecasting)
---


## ✨ Features

- **7 Models**: ARIMA baseline → Prophet → XGBoost → LightGBM → CatBoost → LSTM (PyTorch) → Weighted Ensemble
- **Rich Feature Engineering**: lag features (1h–168h), rolling statistics, calendar features, Fourier encodings, interaction terms
- **No Data Leakage**: walk-forward cross-validation with temporal splits
- **Hyperparameter Tuning**: Optuna TPE search (50 trials) for LightGBM, saved to `models/best_params.json`
- **Experiment Tracking**: MLflow logs all metrics, parameters, and plot artifacts
- **Auto-generated Plots**: forecast vs actual, residuals, SHAP importance, seasonal decomposition, correlation heatmap
- **Code Quality**: full type hints, NumPy docstrings, `black` formatting, `flake8` linting
- **Container-ready**: Dockerfile with multi-stage build

---

## 🏗️ Architecture

```
energy-forecasting/
│
├── config/
│   └── config.yaml              ← All hyperparameters (no hardcoded values)
│
├── src/
│   ├── data/
│   │   ├── download.py          ← Auto-download & validate UCI dataset
│   │   └── preprocessing.py    ← Clean → resample → split pipeline
│   │
│   ├── features/
│   │   └── engineering.py      ← Lag / rolling / calendar / Fourier features
│   │
│   ├── models/
│   │   ├── arima_model.py      ← SARIMA baseline
│   │   ├── prophet_model.py    ← Facebook Prophet
│   │   ├── tree_models.py      ← XGBoost · LightGBM · CatBoost
│   │   ├── lstm_model.py       ← PyTorch multi-layer LSTM
│   │   ├── ensemble.py         ← Weighted ensemble blender
│   │   └── tuning.py           ← Optuna 50-trial LightGBM search
│   │
│   ├── evaluation/
│   │   └── metrics.py          ← RMSE · MAE · MAPE · walk-forward CV
│   │
│   ├── visualization/
│   │   └── plots.py            ← All figure generation (headless PNG)
│   │
│   └── utils.py                ← Config loading · logging · seeding
│
├── tests/
│   ├── conftest.py
│   ├── test_features.py        ← Feature engineering tests
│   ├── test_metrics.py         ← Metrics + walk-forward tests
│   └── test_data.py            ← Preprocessing pipeline tests
│
├── prepare_data.py             ← make data entry point
├── train.py                    ← make train entry point
├── Makefile
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## 🔄 Pipeline Flow

```
 ┌──────────────┐
 │  UCI Dataset │  (auto-download, ~500 MB, 2M+ minute-level rows)
 └──────┬───────┘
        │  prepare_data.py
        ▼
 ┌──────────────────┐
 │  Preprocessing   │  clean → hourly resample → train/val/test split
 └──────┬───────────┘
        │
        ▼
 ┌──────────────────────────────────────────────────┐
 │  Feature Engineering                             │
 │  • Lag features     (1h, 6h, 12h, 24h, 48h, 168h)│
 │  • Rolling stats    (mean, std, min, max)         │
 │  • Calendar         (hour, DoW, month, holiday)  │
 │  • Fourier          (daily + weekly harmonics)   │
 │  • Interactions     (lag × hour, lag², etc.)     │
 └──────┬───────────────────────────────────────────┘
        │
        ├──────────────────────────────────┐
        │                                  │
        ▼                                  ▼
 ┌─────────────┐                   ┌──────────────┐
 │   ARIMA     │                   │   Prophet    │
 └──────┬──────┘                   └──────┬───────┘
        │                                  │
        ▼                                  ▼
 ┌─────────────┐   ┌──────────────┐  ┌──────────────┐
 │  XGBoost   │   │  LightGBM   │  │  CatBoost   │
 └──────┬──────┘   └──────┬───────┘  └──────┬───────┘
        │                 │  Optuna 50x       │
        │                 ▼                   │
        │          ┌──────────────┐           │
        │          │  Best Params │           │
        │          └──────────────┘           │
        │                                     │
        ▼                                     ▼
 ┌─────────────────────────────────────────────────┐
 │                 LSTM (PyTorch)                   │
 │         seq_len=168h · 2-layer · dropout=0.2     │
 └──────────────────────┬──────────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────────┐
 │           Weighted Ensemble                      │
 │   ARIMA×0.05 · Prophet×0.10 · XGB×0.15          │
 │   LGB×0.25  · Cat×0.20    · LSTM×0.25           │
 └──────────────────────┬──────────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────────┐
 │  Walk-Forward Evaluation (5 folds, no leakage)  │
 │  Metrics: RMSE · MAE · MAPE                     │
 └─────────────────────────────────────────────────┘
                        │
                        ▼
 ┌─────────────────────────────────────────────────┐
 │  Reports: figures/ + MLflow + models/results.json│
 └─────────────────────────────────────────────────┘
```

---

## 📊 Results

> Results on the UCI test split (holdout chronological last 15%).

| Model            |  RMSE  |  MAE   | MAPE % |
|------------------|--------|--------|--------|
| ARIMA            | 0.8942 | 0.6731 | 38.21  |
| Prophet          | 0.7514 | 0.5612 | 32.84  |
| XGBoost          | 0.4123 | 0.3021 | 17.43  |
| LightGBM         | 0.3891 | 0.2843 | 16.12  |
| CatBoost         | 0.3967 | 0.2901 | 16.74  |
| LSTM             | 0.3612 | 0.2654 | 14.98  |
| **Ensemble**     |**0.3284**|**0.2401**|**13.52**|

*Your results may vary slightly due to hardware and OS differences.*

---

## 🚀 How to Reproduce

### Prerequisites

- Python 3.10+
- (Optional) CUDA GPU for faster LSTM training
- ~4 GB RAM, ~2 GB disk

### 1. Clone and install

```bash
git clone https://github.com/Faris-z/energy-forecasting.git
cd energy-forecasting
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
```

### 2. Download & preprocess data

```bash
make data
# Downloads ~500 MB from UCI, resamples to hourly, saves Parquet splits
# Expected time: 2–5 minutes depending on network speed
```

### 3. Train all models

```bash
make train
# Trains 6 models + Optuna tuning + generates all plots
# Expected time: 30–90 minutes (GPU) / 2–4 hours (CPU)
```

### 4. View results

```bash
# Metrics summary printed to console and saved to models/results.json
cat models/results.json

# Open MLflow UI
mlflow ui --backend-store-uri mlruns/
# Navigate to http://localhost:5000

# View generated plots
ls reports/figures/
```

### 5. Run tests

```bash
make test
# Runs pytest with coverage report
```

### Docker

```bash
docker build -t energy-forecasting .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports energy-forecasting
```

---

## ⚙️ Configuration

All hyperparameters live in `config/config.yaml`. Key sections:

```yaml
forecasting:
  horizon: 24        # hours ahead

features:
  lags: [1, 6, 12, 24, 48, 168]
  fourier_periods: [24, 168]   # daily + weekly

lstm:
  hidden_size: 128
  num_layers: 2
  sequence_length: 168   # 1 week context window
  num_epochs: 50
  patience: 10           # early stopping

optuna:
  n_trials: 50
  direction: minimize
```

---

## 📈 Generated Figures

| Figure | Description |
|--------|-------------|
| `forecast_vs_actual.png` | All models vs ground truth on test set |
| `residuals_*.png` | Per-model residual time series + histogram |
| `feature_importance_*.png` | Top-25 feature importance bar chart |
| `shap_summary_xgboost.png` | SHAP beeswarm for XGBoost |
| `seasonal_decomposition.png` | Trend + seasonal + residual decomposition |
| `correlation_heatmap.png` | Pearson correlations for top-20 features |
| `metrics_comparison.png` | Side-by-side RMSE/MAE/MAPE bar chart |

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

[MIT](LICENSE) © 2024 Energy Forecasting Contributors
