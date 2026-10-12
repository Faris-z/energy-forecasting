# energy-forecasting

[![CI](https://github.com/Faris-z/energy-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Faris-z/energy-forecasting/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hourly household energy forecasting pipeline on the UCI Individual Household Electric Power Consumption dataset, comparing gradient-boosted trees, an LSTM, an ensemble, and classical baselines.

> **Status:** the pipeline runs end to end with a consistent 24-hour-ahead setup for every model. No verified results are published yet (see [Results](#results)).

## Problem

Predict household electricity use (`Global_active_power`, kW) for each hour, 24 hours in advance, from the series' own history and calendar patterns, and compare classical, gradient-boosting and deep learning approaches under the same split and horizon.

## Approach

**Data** (`src/data/`)
- [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption): 2M+ minute-level readings from one household, downloaded automatically.
- Cleaned, resampled to hourly means, and split chronologically into train / validation / test (70% / 15% / 15%).

**Forecast horizon:** 24 hours (`forecasting.horizon` in the config). Every feature for hour *t* uses only data up to hour *t − 24*, so each row is a direct 24-hour-ahead forecast. Lags shorter than the horizon are skipped.

**Features** (`src/features/engineering.py`, 52 engineered features)
- Lags of the target: the same hour 1 to 7 days earlier (24, 48, ..., 168 hours) (7)
- Rolling mean / std / min / max over 6, 12, 24, 48, 168 hours, with each window ending at *t − 24* (20)
- Calendar: hour, day of week, month, quarter, day of year, week of year, weekend flag, holiday flag (8)
- Fourier terms for daily and weekly periods, 3 harmonics each, sin and cos (12)
- Interactions such as `lag24 x hour` and `lag168 x dayofweek` (5)

Features come only from the target's own history and the timestamp. The other raw measurements (`Global_intensity`, `Voltage`, `Global_reactive_power`, sub-meterings) are recorded in the same hour as the target, so `build_features` drops them to avoid leakage. Features are built once on the full series over a gap-free hourly index, so lags count hours (not rows) across missing stretches, and the first validation and test rows take their lags from the period just before them.

**Models** (`src/models/`), all scored on the same test hours at the same 24-hour horizon
- Seasonal naive reference: the value at the same hour one day earlier (reported only, not in the ensemble)
- SARIMA: parameters fit once on the training split; each test hour is forecast 24 steps ahead from a rolling origin 24 hours earlier, updating the state with newly observed hours (no refitting)
- Prophet: fit once on the training split. Its forecast depends only on the timestamp, so it never uses data after training
- Gradient boosting: XGBoost, LightGBM, CatBoost (validation split passed as the eval set)
- LSTM in PyTorch: 2 layers, hidden size 128, 168-hour input window, early stopping. The first test hours take their input window from the end of the validation split
- Weighted ensemble of the six models above, with weights fixed in `config/config.yaml` (not learned)

**Tuning and tracking**
- Optuna (TPE, 50 trials) tunes LightGBM, scoring each trial by walk-forward RMSE on train + validation data.
- MLflow logs parameters, per-model metrics, the results file and the generated figures.

## Results

**TODO: no verified results yet.** The pipeline has not been run on the real dataset since the leakage and horizon fixes, and the repository does not contain a saved results file or MLflow run.

What `train.py` evaluates:
- **Holdout:** every model forecasts the same hours of the chronological last 15% of the series, 24 hours ahead (hours whose lags fall in a data gap are skipped for all models).
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

- **Single horizon:** models are evaluated at exactly 24 hours ahead. Next: report error by horizon (1 to 24 hours) to show how accuracy decays.
- **Prophet is not refit:** it is trained once on the training split, so it is at a disadvantage against models that see recent data. Next: refit it on a rolling schedule.
- **Single holdout:** the per-model metrics come from one chronological 15% split. The walk-forward evaluation covers LightGBM only, and its folds fall inside that same holdout period. Next: run walk-forward evaluation for all models on a fixed protocol.
- **No saved results:** commit `models/results.json` (or an MLflow export) and the key figures from a real run once the fixes above are in.
- **Holiday flag:** uses US holidays, while the dataset comes from a household in France. Next: use French holidays.
- **Ensemble:** weights are hand-set in the config. Next: fit them on the validation split.

## License

[MIT](LICENSE)
