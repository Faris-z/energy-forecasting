"""Evaluation metrics and walk-forward validation framework.

Provides RMSE, MAE, and MAPE implementations, plus a walk-forward
(time-series cross-validation) splitter that prevents data leakage.
"""

import logging
from typing import Dict, Any, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE score (lower is better).
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAE score (lower is better).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Compute Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.
    epsilon : float
        Small constant added to denominator to avoid division by zero.

    Returns
    -------
    float
        MAPE in percentage points (lower is better).
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE, MAE, and MAPE in a single call.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys ``'rmse'``, ``'mae'``, ``'mape'``.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def walk_forward_split(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Generate walk-forward (expanding window) train/test splits.

    Each fold expands the training set by one test window, ensuring
    no future data is used in training (no data leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Full feature-engineered DataFrame sorted chronologically.
    config : Dict[str, Any]
        Project configuration dictionary. Uses:
        ``evaluation.walk_forward.n_splits``,
        ``evaluation.walk_forward.test_size``.

    Yields
    ------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(train_fold, test_fold)`` DataFrames for each split.
    """
    wf_cfg = config["evaluation"]["walk_forward"]
    n_splits = wf_cfg["n_splits"]
    test_size = wf_cfg["test_size"]

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_fold = df.iloc[train_idx]
        test_fold = df.iloc[test_idx]
        logger.debug(
            "Fold %d — Train: %d rows [%s → %s]  Test: %d rows [%s → %s]",
            fold_idx + 1,
            len(train_fold),
            train_fold.index.min().date(),
            train_fold.index.max().date(),
            len(test_fold),
            test_fold.index.min().date(),
            test_fold.index.max().date(),
        )
        yield train_fold, test_fold


def evaluate_model(
    predict_fn: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Dict[str, float]:
    """Evaluate a model on a single train/test split.

    Parameters
    ----------
    predict_fn : callable
        A function or object with a ``predict(X)`` method.
    train : pd.DataFrame
        Training data (features + target).
    test : pd.DataFrame
        Test data (features + target).
    target_col : str
        Name of the target column.
    feature_cols : List[str]
        List of feature column names.

    Returns
    -------
    Dict[str, float]
        Metric dictionary with 'rmse', 'mae', 'mape'.
    """
    X_test = test[feature_cols].values
    y_true = test[target_col].values
    y_pred = predict_fn(X_test)
    return compute_all_metrics(y_true, y_pred)


def cross_validate_model(
    model_class: Any,
    model_params: Dict[str, Any],
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Run walk-forward cross-validation and return averaged metrics.

    Parameters
    ----------
    model_class : Any
        Model class with ``fit(X, y)`` and ``predict(X)`` interface.
    model_params : Dict[str, Any]
        Keyword arguments passed to ``model_class``.
    df : pd.DataFrame
        Full feature-engineered DataFrame.
    target_col : str
        Name of the target column.
    feature_cols : List[str]
        Feature column names.
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    Dict[str, float]
        Mean metrics across all folds: 'rmse', 'mae', 'mape'.
    """
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_fold, test_fold) in enumerate(
        walk_forward_split(df, config)
    ):
        X_train = train_fold[feature_cols].values
        y_train = train_fold[target_col].values

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model.predict, train_fold, test_fold, target_col, feature_cols
        )
        fold_metrics.append(metrics)
        logger.info(
            "Fold %d — RMSE: %.4f  MAE: %.4f  MAPE: %.2f%%",
            fold_idx + 1,
            metrics["rmse"],
            metrics["mae"],
            metrics["mape"],
        )

    mean_metrics = {
        k: float(np.mean([m[k] for m in fold_metrics]))
        for k in ["rmse", "mae", "mape"]
    }
    logger.info(
        "CV Mean — RMSE: %.4f  MAE: %.4f  MAPE: %.2f%%",
        mean_metrics["rmse"],
        mean_metrics["mae"],
        mean_metrics["mape"],
    )
    return mean_metrics
