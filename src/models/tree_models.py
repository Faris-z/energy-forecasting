"""Gradient-boosted tree models for time series regression.

Wraps XGBoost, LightGBM, and CatBoost with consistent interfaces,
early stopping support, and project config integration.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """XGBoost regression model with early stopping.

    Parameters
    ----------
    **params
        XGBoost hyperparameters passed to ``xgb.XGBRegressor``.
    """

    def __init__(self, **params: Any) -> None:
        self.params = params
        self._model: Optional[Any] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostForecaster":
        """Fit XGBoost regressor with optional early stopping.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target values.
        X_val : Optional[np.ndarray]
            Validation features for early stopping.
        y_val : Optional[np.ndarray]
            Validation targets for early stopping.

        Returns
        -------
        XGBoostForecaster
            Fitted instance (self).
        """
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required: pip install xgboost") from exc

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        logger.info(
            "Fitting XGBoost on %d samples, %d features ...",
            X_train.shape[0],
            X_train.shape[1],
        )
        self._model = xgb.XGBRegressor(**self.params)
        self._model.fit(X_train, y_train, **fit_kwargs)
        logger.info("XGBoost fitting complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importances (gain-based).

        Returns
        -------
        np.ndarray
            Feature importance scores.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.feature_importances_

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "XGBoostForecaster":
        """Instantiate from project configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Project configuration dictionary.

        Returns
        -------
        XGBoostForecaster
            New instance.
        """
        return cls(**config["xgboost"])


class LightGBMForecaster:
    """LightGBM regression model with early stopping.

    Parameters
    ----------
    **params
        LightGBM hyperparameters passed to ``lgb.LGBMRegressor``.
    """

    def __init__(self, **params: Any) -> None:
        self.params = params
        self._model: Optional[Any] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LightGBMForecaster":
        """Fit LightGBM regressor.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target values.
        X_val : Optional[np.ndarray]
            Validation features for early stopping.
        y_val : Optional[np.ndarray]
            Validation targets for early stopping.
        feature_names : Optional[List[str]]
            Column names for feature importance logging.

        Returns
        -------
        LightGBMForecaster
            Fitted instance (self).
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm is required: pip install lightgbm") from exc

        logger.info(
            "Fitting LightGBM on %d samples, %d features ...",
            X_train.shape[0],
            X_train.shape[1],
        )

        callbacks = [lgb.log_evaluation(period=-1)]

        fit_kwargs: Dict[str, Any] = {"callbacks": callbacks}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
        self._model = lgb.LGBMRegressor(**params)
        self._model.fit(X_train, y_train, **fit_kwargs)
        logger.info("LightGBM fitting complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importances (gain-based).

        Returns
        -------
        np.ndarray
            Feature importance scores.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.feature_importances_

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LightGBMForecaster":
        """Instantiate from project configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Project configuration dictionary.

        Returns
        -------
        LightGBMForecaster
            New instance.
        """
        return cls(**config["lightgbm"])


class CatBoostForecaster:
    """CatBoost regression model with early stopping.

    Parameters
    ----------
    **params
        CatBoost hyperparameters passed to ``CatBoostRegressor``.
    """

    def __init__(self, **params: Any) -> None:
        self.params = params
        self._model: Optional[Any] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CatBoostForecaster":
        """Fit CatBoost regressor.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target values.
        X_val : Optional[np.ndarray]
            Validation features for early stopping.
        y_val : Optional[np.ndarray]
            Validation targets for early stopping.

        Returns
        -------
        CatBoostForecaster
            Fitted instance (self).
        """
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError as exc:
            raise ImportError("catboost is required: pip install catboost") from exc

        logger.info(
            "Fitting CatBoost on %d samples, %d features ...",
            X_train.shape[0],
            X_train.shape[1],
        )

        params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
        self._model = CatBoostRegressor(**params)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val)

        self._model.fit(X_train, y_train, eval_set=eval_set, use_best_model=(eval_set is not None))
        logger.info("CatBoost fitting complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importances.

        Returns
        -------
        np.ndarray
            Feature importance scores.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.get_feature_importance()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CatBoostForecaster":
        """Instantiate from project configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Project configuration dictionary.

        Returns
        -------
        CatBoostForecaster
            New instance.
        """
        return cls(**config["catboost"])
