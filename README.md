# energy-forecasting

[![CI](https://github.com/Faris-z/energy-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Faris-z/energy-forecasting/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hourly household energy forecasting pipeline on the UCI Individual Household Electric Power Consumption dataset, comparing gradient-boosted trees, an LSTM, an ensemble, and classical baselines.

> **Status:** the pipeline runs end to end, but the evaluation setup has known issues (see [Limitations](#limitations--next-steps)). No verified results are published yet.

## Problem

Predict household electricity use (`Global_active_power`, kW) at an hourly resolution from its own history and calendar patterns, and compare classical, gradient-boosting and deep learning approaches under the same split.

## Approach

**Data** (`src/data/`)
- [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption): 2M+ minute-level readings from one household, downloaded automatically.
- Cleaned, resampled to hourly means, and split chronologically into train / validation / test (70% / 15% / 15%).

**Features** (`src/features/engineering.py`, 51 engineered features)
- Lags of the target: 1, 6, 12, 24, 48, 168 hours (6)
- Rolling mean / std / min / max over 6, 12, 24, 48, 168 hours, computed on the shifted series (20)
- Calendar: hour, day of week, month, quarter, day of year, week of year, weekend flag, holiday flag (8)
- Fourier terms for daily and weekly periods, 3 harmonics each, sin and cos (12)
- Interactions such as `lag24 x hour` and `lag168 x dayofweek` (5)

Features come only from the target's own history and the timestamp. The other raw measurements (`Global_intensity`, `Voltage`, `Global_reactive_power`, sub-meterings) are recorded in the same hour as the target, so `build_features` drops them to avoid leakage.

**Models** (`src/models/`)
- Baselines: SARIMA and Prophet
- Gradient boosting: XGBoost, LightGBM, CatBoost (validation split passed as the eval set)
- LSTM in PyTorch: 2 layers, hidden size 128, 168-hour input window, early stopping
- Weighted ensemble of all six, with weights fixed in `config/config.yaml` (not learned)

**Tuning and tracking**
- Optuna (TPE, 50 trials) tunes LightGBM, scoring each trial by walk-forward RMSE on train + validation data.
- MLflow logs parameters, per-model metrics, the results file and the generated figures.

## Results

**TODO: no verified results yet.** The pipeline has not been re-run since the same-hour leakage was removed, the repository does not contain a saved results file or MLflow run, and the evaluation setup still needs the fixes listed below before any numbers are meaningful.

What `train.py` currently evaluates:
- **Holdout:** every model is scored on the chronological last 15% of the hourly series.
- **Walk-forward:** LightGBM only, 5 expanding-window folds (`TimeSeriesSplit`) with 168-hour test windows, taken from the full series.
- **Metrics:** RMSE, MAE, MAPE. Results are written to `models/results.json` and logged to MLflow.

## Tech Stack

Python, pandas, NumPy, scikit-learn, statsmodels, Prophet, XGBoost, LightGBM, CatBoost, PyTorch, Optuna, MLflow, SHAP, matplotlib / seaborn, pytest, GitHub Actions (flake8, black, isort, pytest), Docker.

## How to Run

Requires Python 3.10+.

```bash
git clone https://github.com/Faris-z/energy-forecasting.git
cd energy-forecasting
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
make data    # download the UCI dataset, resample to hourly, save the splits
make train   # train all models, run Optuna, write models/results.json and figures
make test    # pytest with coverage
```

Runs on CPU; the LSTM uses a CUDA GPU automatically if one is available.

Results and figures:

```bash
cat models/results.json
mlflow ui --backend-store-uri mlruns/   # then open http://localhost:5000
ls reports/figures/
```

Docker:

```bash
docker build -t energy-forecasting .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports energy-forecasting
```

All hyperparameters live in `config/config.yaml`.

Live demo: **TODO.** The linked Hugging Face Space is currently in a runtime-error state and runs on synthetic data, not this pipeline.

## Limitations / Next Steps

- **Horizon:** the tree models and the LSTM predict one step (1 hour) ahead from actual lagged values, while ARIMA and Prophet forecast the whole test period from the end of training, so the comparison is not like for like. Next: build a true multi-step target (for example shift the target by the horizon and drop lags shorter than it) and evaluate every model on it.
- **Single holdout:** the per-model metrics come from one chronological 15% split. The walk-forward evaluation covers LightGBM only, and its folds fall inside that same holdout period. Next: run walk-forward evaluation for all models on a fixed protocol.
- **No saved results:** commit `models/results.json` (or an MLflow export) and the key figures from a real run once the fixes above are in.
- **Holiday flag:** uses US holidays, while the dataset comes from a household in France. Next: use French holidays.
- **Ensemble:** weights are hand-set in the config. Next: fit them on the validation split.

## License

[MIT](LICENSE)
