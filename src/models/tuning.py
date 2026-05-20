"""Optuna hyperparameter optimisation for LightGBM.

Runs a 50-trial Bayesian search over LightGBM hyperparameters using
walk-forward cross-validation as the objective function. Best parameters
are saved to ``models/best_params.json``.
"""

import logging
from typing import Any, Dict

import numpy as np

from src.evaluation.metrics import rmse, walk_forward_split
from src.features.engineering import get_feature_columns
from src.utils import save_json

logger = logging.getLogger(__name__)


def _lgb_objective(
    trial: Any,
    df: Any,
    target_col: str,
    feature_cols: list,
    config: Dict[str, Any],
) -> float:
    """Optuna objective function: walk-forward RMSE for LightGBM.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial.
    df : pd.DataFrame
        Feature-engineered DataFrame.
    target_col : str
        Target column name.
    feature_cols : list
        Feature column names.
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    float
        Mean RMSE across walk-forward folds.
    """
    import lightgbm as lgb

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
    }

    fold_rmses = []

    for train_fold, test_fold in walk_forward_split(df, config):
        X_train = train_fold[feature_cols].values
        y_train = train_fold[target_col].values
        X_test = test_fold[feature_cols].values
        y_test = test_fold[target_col].values

        callbacks = [lgb.log_evaluation(period=-1)]
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=callbacks,
        )

        y_pred = model.predict(X_test)
        fold_rmses.append(rmse(y_test, y_pred))

    mean_rmse = float(np.mean(fold_rmses))
    return mean_rmse


def run_optuna_tuning(
    df: Any,
    config: Dict[str, Any],
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter search for LightGBM.

    Saves the best parameters to the path specified in config and
    returns them as a dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with all splits.
    config : Dict[str, Any]
        Project configuration dictionary.
    n_trials : int
        Number of Optuna trials to run (overrides config value if given).

    Returns
    -------
    Dict[str, Any]
        Best LightGBM hyperparameters found.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required: pip install optuna") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    target_col = config["data"]["target_col"]
    feature_cols = get_feature_columns(df, target_col)
    opt_cfg = config["optuna"]
    n_trials = n_trials or opt_cfg["n_trials"]

    logger.info(
        "Starting Optuna LightGBM tuning: %d trials, direction=%s",
        n_trials,
        opt_cfg["direction"],
    )

    study = optuna.create_study(
        study_name=opt_cfg["study_name"],
        direction=opt_cfg["direction"],
        sampler=optuna.samplers.TPESampler(seed=config["project"]["random_seed"]),
    )

    study.optimize(
        lambda trial: _lgb_objective(trial, df, target_col, feature_cols, config),
        n_trials=n_trials,
        timeout=opt_cfg.get("timeout"),
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_value = study.best_value

    logger.info(
        "Optuna complete. Best RMSE: %.4f\nBest params: %s",
        best_value,
        best_params,
    )

    output: Dict[str, Any] = {
        "best_params": best_params,
        "best_rmse": best_value,
        "n_trials": n_trials,
        "study_name": opt_cfg["study_name"],
    }

    params_path = opt_cfg["params_output"]
    save_json(output, params_path)
    logger.info("Best params saved to %s", params_path)

    return best_params
