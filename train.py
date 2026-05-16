"""Main training script for the energy-forecasting project.

Orchestrates the full pipeline:
  1. Load processed data
  2. Feature engineering
  3. Train all models (ARIMA, Prophet, XGBoost, LightGBM, CatBoost, LSTM)
  4. Optuna hyperparameter tuning for LightGBM
  5. Weighted ensemble
  6. Walk-forward evaluation
  7. Generate all visualizations
  8. Log everything to MLflow

Run: python train.py  or  make train
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.preprocessing import load_processed_data
from src.evaluation.metrics import compute_all_metrics, walk_forward_split
from src.features.engineering import build_features, get_feature_columns
from src.models.arima_model import ARIMAForecaster
from src.models.ensemble import WeightedEnsemble
from src.models.lstm_model import LSTMForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.tree_models import CatBoostForecaster, LightGBMForecaster, XGBoostForecaster
from src.models.tuning import run_optuna_tuning
from src.utils import ensure_dirs, load_config, load_json, save_json, set_seed, setup_logging
from src.visualization.plots import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_forecast_vs_actual,
    plot_metrics_comparison,
    plot_residuals,
    plot_seasonal_decomposition,
    plot_shap_summary,
)

logger = logging.getLogger(__name__)


class _NullContext:
    """No-op context manager used when MLflow is unavailable."""

    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


def train_arima(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: Dict[str, Any],
) -> np.ndarray:
    """Fit ARIMA on training series and forecast the test horizon.

    Parameters
    ----------
    train : pd.DataFrame
        Training data with target column.
    test : pd.DataFrame
        Test data.
    config : Dict[str, Any]
        Project configuration.

    Returns
    -------
    np.ndarray
        Forecast array of length len(test).
    """
    target = config["data"]["target_col"]
    model = ARIMAForecaster.from_config(config)
    model.fit(train[target])
    horizon = len(test)
    preds = model.predict(horizon=horizon)
    preds = np.clip(preds, 0, None)
    logger.info("ARIMA forecast done (horizon=%d).", horizon)
    return preds


def train_prophet(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: Dict[str, Any],
) -> np.ndarray:
    """Fit Prophet on training series and forecast the test horizon.

    Parameters
    ----------
    train : pd.DataFrame
        Training data with target column.
    test : pd.DataFrame
        Test data.
    config : Dict[str, Any]
        Project configuration.

    Returns
    -------
    np.ndarray
        Forecast array of length len(test).
    """
    target = config["data"]["target_col"]
    model = ProphetForecaster.from_config(config)
    model.fit(train[target])
    preds = model.predict(horizon=len(test))
    preds = np.clip(preds, 0, None)
    logger.info("Prophet forecast done.")
    return preds


def train_tree_model(
    model_cls: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    config: Dict[str, Any],
    model_name: str,
) -> tuple:
    """Train a gradient-boosted tree model and generate test predictions.

    Parameters
    ----------
    model_cls : type
        One of XGBoostForecaster, LightGBMForecaster, CatBoostForecaster.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation targets.
    X_test : np.ndarray
        Test features.
    config : Dict[str, Any]
        Project configuration.
    model_name : str
        Human-readable name for logging.

    Returns
    -------
    np.ndarray
        Test predictions.
    """
    model = model_cls.from_config(config)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)
    logger.info("%s predictions done.", model_name)
    return model, preds


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Train LSTM and generate test predictions.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation targets.
    X_test : np.ndarray
        Test features.
    config : Dict[str, Any]
        Project configuration.

    Returns
    -------
    np.ndarray
        Test predictions of length len(X_test).
    """
    model = LSTMForecaster.from_config(config)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    preds = preds[:len(X_test)]
    preds = np.clip(preds, 0, None)
    model.save(str(Path(config["paths"]["models_dir"]) / "lstm"))
    logger.info("LSTM predictions done.")
    return preds


def run_training(config: Dict[str, Any]) -> None:
    """Run the full training pipeline end-to-end.

    Parameters
    ----------
    config : Dict[str, Any]
        Project configuration dictionary.
    """
    setup_logging(
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
        fmt=config["logging"]["format"],
    )
    set_seed(config["project"]["random_seed"])
    ensure_dirs(config)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    try:
        import mlflow

        mlflow.set_tracking_uri(config["project"]["mlflow_tracking_uri"])
        mlflow.set_experiment(config["project"]["mlflow_experiment"])
        mlflow_enabled = True
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment tracking.")
        mlflow_enabled = False

    target = config["data"]["target_col"]

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading processed data ...")
    train_raw, val_raw, test_raw = load_processed_data(config)

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Building features ...")
    train_feat = build_features(train_raw, config, drop_na=True)
    val_feat = build_features(val_raw, config, drop_na=True)
    test_feat = build_features(test_raw, config, drop_na=True)

    feature_cols = get_feature_columns(train_feat, target)
    logger.info("Feature count: %d", len(feature_cols))

    X_train = train_feat[feature_cols].values
    y_train = train_feat[target].values
    X_val = val_feat[feature_cols].values
    y_val = val_feat[target].values
    X_test = test_feat[feature_cols].values
    y_test = test_feat[target].values

    # ── Exploratory plots ─────────────────────────────────────────────────────
    logger.info("Generating exploratory visualizations ...")
    plot_seasonal_decomposition(train_raw[target], config)
    plot_correlation_heatmap(train_feat, config)

    all_predictions: Dict[str, np.ndarray] = {}
    all_results: Dict[str, Dict[str, float]] = {}
    run_context = mlflow.start_run(run_name="full_pipeline") if mlflow_enabled else _NullContext()

    with run_context:
        if mlflow_enabled:
            mlflow.log_params({
                "n_features": len(feature_cols),
                "train_size": len(train_feat),
                "val_size": len(val_feat),
                "test_size": len(test_feat),
                "horizon": config["forecasting"]["horizon"],
            })

        # ── ARIMA ─────────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Training ARIMA ...")
        t0 = time.time()
        try:
            arima_preds = train_arima(train_raw, test_raw, config)
            arima_preds = arima_preds[:len(y_test)]
            all_predictions["arima"] = arima_preds
            arima_metrics = compute_all_metrics(y_test[:len(arima_preds)], arima_preds)
            all_results["arima"] = arima_metrics
            logger.info("ARIMA — %s  (%.1fs)", arima_metrics, time.time() - t0)
            if mlflow_enabled:
                mlflow.log_metrics({f"arima_{k}": v for k, v in arima_metrics.items()})
            plot_residuals(y_test[:len(arima_preds)], arima_preds, "ARIMA", test_feat.index[:len(arima_preds)], config)
        except Exception as exc:
            logger.error("ARIMA failed: %s — using naive forecast.", exc)
            all_predictions["arima"] = np.full(len(y_test), y_train[-1])
            all_results["arima"] = compute_all_metrics(y_test, all_predictions["arima"])

        # ── Prophet ───────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Training Prophet ...")
        t0 = time.time()
        try:
            prophet_preds = train_prophet(train_raw, test_raw, config)
            prophet_preds = prophet_preds[:len(y_test)]
            all_predictions["prophet"] = prophet_preds
            prophet_metrics = compute_all_metrics(y_test[:len(prophet_preds)], prophet_preds)
            all_results["prophet"] = prophet_metrics
            logger.info("Prophet — %s  (%.1fs)", prophet_metrics, time.time() - t0)
            if mlflow_enabled:
                mlflow.log_metrics({f"prophet_{k}": v for k, v in prophet_metrics.items()})
            plot_residuals(y_test[:len(prophet_preds)], prophet_preds, "Prophet", test_feat.index[:len(prophet_preds)], config)
        except Exception as exc:
            logger.error("Prophet failed: %s — using naive forecast.", exc)
            all_predictions["prophet"] = np.full(len(y_test), y_train[-1])
            all_results["prophet"] = compute_all_metrics(y_test, all_predictions["prophet"])

        # ── XGBoost ───────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Training XGBoost ...")
        t0 = time.time()
        xgb_model, xgb_preds = train_tree_model(
            XGBoostForecaster, X_train, y_train, X_val, y_val, X_test, config, "XGBoost"
        )
        all_predictions["xgboost"] = xgb_preds
        xgb_metrics = compute_all_metrics(y_test, xgb_preds)
        all_results["xgboost"] = xgb_metrics
        logger.info("XGBoost — %s  (%.1fs)", xgb_metrics, time.time() - t0)
        if mlflow_enabled:
            mlflow.log_metrics({f"xgboost_{k}": v for k, v in xgb_metrics.items()})
        plot_residuals(y_test, xgb_preds, "XGBoost", test_feat.index, config)
        plot_feature_importance(feature_cols, xgb_model.get_feature_importance(), "XGBoost", config)
        plot_shap_summary(xgb_model._model, X_test, feature_cols, "XGBoost", config)

        # ── LightGBM ──────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Training LightGBM ...")
        t0 = time.time()
        lgb_model, lgb_preds = train_tree_model(
            LightGBMForecaster, X_train, y_train, X_val, y_val, X_test, config, "LightGBM"
        )
        all_predictions["lightgbm"] = lgb_preds
        lgb_metrics = compute_all_metrics(y_test, lgb_preds)
        all_results["lightgbm"] = lgb_metrics
        logger.info("LightGBM — %s  (%.1fs)", lgb_metrics, time.time() - t0)
        if mlflow_enabled:
            mlflow.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
        plot_residuals(y_test, lgb_preds, "LightGBM", test_feat.index, config)
        plot_feature_importance(feature_cols, lgb_model.get_feature_importance(), "LightGBM", config)

        # ── Optuna tuning for LightGBM ────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Running Optuna hyperparameter tuning for LightGBM ...")
        full_feat = build_features(
            pd.concat([train_raw, val_raw]), config, drop_na=True
        )
        try:
            best_lgb_params = run_optuna_tuning(full_feat, config)
            if mlflow_enabled:
                mlflow.log_params({f"optuna_lgb_{k}": v for k, v in best_lgb_params.items()})
        except Exception as exc:
            logger.error("Optuna tuning failed: %s", exc)
            best_lgb_params = config["lightgbm"]

        logger.info("Retraining LightGBM with Optuna best params ...")
        tuned_config = {**config, "lightgbm": best_lgb_params}
        lgb_model, lgb_preds = train_tree_model(
            LightGBMForecaster, X_train, y_train, X_val, y_val, X_test, tuned_config, "LightGBM-tuned"
        )
        all_predictions["lightgbm"] = lgb_preds
        lgb_metrics = compute_all_metrics(y_test, lgb_preds)
        all_results["lightgbm"] = lgb_metrics
        logger.info("LightGBM (tuned) — %s", lgb_metrics)
        if mlflow_enabled:
            mlflow.log_metrics({f"lightgbm_tuned_{k}": v for k, v in lgb_metrics.items()})

        # ── CatBoost ──────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Training CatBoost ...")
        t0 = time.time()
        cat_model, cat_preds = train_tree_model(
            CatBoostForecaster, X_train, y_train, X_val, y_val, X_test, config, "CatBoost"
        )
        all_predictions["catboost"] = cat_preds
        cat_metrics = compute_all_metrics(y_test, cat_preds)
        all_results["catboost"] = cat_metrics
        logger.info("CatBoost — %s  (%.1fs)", cat_metrics, time.time() - t0)
        if mlflow_enabled:
            mlflow.log_metrics({f"catboost_{k}": v for k, v in cat_metrics.items()})
        plot_residuals(y_test, cat_preds, "CatBoost", test_feat.index, config)
        plot_feature_importance(feature_cols, cat_model.get_feature_importance(), "CatBoost", config)

        # ── LSTM ──────────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Training LSTM ...")
        t0 = time.time()
        lstm_preds = train_lstm(X_train, y_train, X_val, y_val, X_test, config)
        all_predictions["lstm"] = lstm_preds
        lstm_metrics = compute_all_metrics(y_test, lstm_preds[:len(y_test)])
        all_results["lstm"] = lstm_metrics
        logger.info("LSTM — %s  (%.1fs)", lstm_metrics, time.time() - t0)
        if mlflow_enabled:
            mlflow.log_metrics({f"lstm_{k}": v for k, v in lstm_metrics.items()})
        plot_residuals(y_test, lstm_preds[:len(y_test)], "LSTM", test_feat.index, config)

        # ── Ensemble ──────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Computing weighted ensemble ...")
        ensemble = WeightedEnsemble.from_config(config)
        aligned_preds = {k: v[:len(y_test)] for k, v in all_predictions.items()}
        ens_preds = ensemble.predict(aligned_preds)
        all_predictions["ensemble"] = ens_preds
        ens_metrics = compute_all_metrics(y_test, ens_preds)
        all_results["ensemble"] = ens_metrics
        logger.info("Ensemble — %s", ens_metrics)
        if mlflow_enabled:
            mlflow.log_metrics({f"ensemble_{k}": v for k, v in ens_metrics.items()})
        plot_residuals(y_test, ens_preds, "Ensemble", test_feat.index, config)

        # ── Final plots ───────────────────────────────────────────────────────
        logger.info("Generating final comparison plots ...")
        aligned_all = {k: v[:len(y_test)] for k, v in all_predictions.items()}
        plot_forecast_vs_actual(
            y_test, aligned_all, test_feat.index, config
        )
        plot_metrics_comparison(all_results, config)

        # ── Save results ──────────────────────────────────────────────────────
        # ── Walk-forward evaluation ───────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Running walk-forward evaluation ...")
        full_feat = build_features(
            pd.concat([train_raw, val_raw, test_raw]), config, drop_na=True
        )
        wf_splits = walk_forward_split(full_feat, config)
        wf_results: Dict[str, list] = {}
        for fold_idx, (wf_train, wf_test) in enumerate(wf_splits):
            wf_X_train = wf_train[feature_cols].values
            wf_y_train = wf_train[target].values
            wf_X_test = wf_test[feature_cols].values
            wf_y_test = wf_test[target].values
            _, wf_preds = train_tree_model(
                LightGBMForecaster, wf_X_train, wf_y_train,
                wf_X_train, wf_y_train, wf_X_test, tuned_config, f"LightGBM-wf-{fold_idx}"
            )
            fold_metrics = compute_all_metrics(wf_y_test, wf_preds[:len(wf_y_test)])
            for k, v in fold_metrics.items():
                wf_results.setdefault(k, []).append(v)
        wf_summary = {k: float(np.mean(v)) for k, v in wf_results.items()}
        logger.info("Walk-forward mean metrics — %s", wf_summary)
        all_results["walk_forward"] = wf_summary
        if mlflow_enabled:
            mlflow.log_metrics({f"wf_{k}": v for k, v in wf_summary.items()})

        results_path = Path(config["paths"]["models_dir"]) / "results.json"
        save_json(all_results, str(results_path))
        logger.info("Results saved to %s", results_path)

        if mlflow_enabled:
            mlflow.log_artifact(str(results_path))
            mlflow.log_artifacts(config["paths"]["figures_dir"], artifact_path="figures")

    # ── Print final leaderboard ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINAL RESULTS (Test Set)")
    logger.info("%-15s  %10s  %10s  %10s", "Model", "RMSE", "MAE", "MAPE%")
    logger.info("-" * 50)
    for model_name, metrics in sorted(all_results.items(), key=lambda x: x[1]["rmse"]):
        logger.info(
            "%-15s  %10.4f  %10.4f  %10.2f",
            model_name,
            metrics["rmse"],
            metrics["mae"],
            metrics["mape"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    run_training(cfg)
