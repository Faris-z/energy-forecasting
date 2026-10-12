"""SARIMA baseline model for horizon-hours-ahead forecasting.

Wraps ``statsmodels`` SARIMA with a consistent sklearn-style interface
so it can be used interchangeably with the other models in the pipeline.
"""

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """Seasonal ARIMA wrapper for multi-step-ahead forecasting.

    Fits a SARIMA model on the target time series. :meth:`predict_rolling`
    produces a horizon-step-ahead forecast for every target hour from a
    rolling forecast origin.

    Parameters
    ----------
    p : int
        AR order.
    d : int
        Differencing order.
    q : int
        MA order.
    seasonal_p : int
        Seasonal AR order.
    seasonal_d : int
        Seasonal differencing order.
    seasonal_q : int
        Seasonal MA order.
    seasonal_period : int
        Number of periods per season (24 for hourly data).
    """

    def __init__(
        self,
        p: int = 2,
        d: int = 1,
        q: int = 2,
        seasonal_p: int = 1,
        seasonal_d: int = 1,
        seasonal_q: int = 1,
        seasonal_period: int = 24,
    ) -> None:
        self.p = p
        self.d = d
        self.q = q
        self.seasonal_p = seasonal_p
        self.seasonal_d = seasonal_d
        self.seasonal_q = seasonal_q
        self.seasonal_period = seasonal_period
        self._model_fit: Optional[Any] = None
        self._train_series: Optional[pd.Series] = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """Fit the SARIMA model on a time series.

        Parameters
        ----------
        series : pd.Series
            Hourly target time series with a DatetimeIndex.

        Returns
        -------
        ARIMAForecaster
            Fitted instance (self).
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for ARIMA: pip install statsmodels"
            ) from exc

        logger.info(
            "Fitting SARIMA(%d,%d,%d)×(%d,%d,%d)[%d] on %d observations ...",
            self.p,
            self.d,
            self.q,
            self.seasonal_p,
            self.seasonal_d,
            self.seasonal_q,
            self.seasonal_period,
            len(series),
        )

        # Gap-free hourly index: missing hours become NaN, which the Kalman
        # filter skips, so the seasonal period stays 24 *hours*.
        series = series.asfreq("1h")
        self._train_series = series.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                order=(self.p, self.d, self.q),
                seasonal_order=(
                    self.seasonal_p,
                    self.seasonal_d,
                    self.seasonal_q,
                    self.seasonal_period,
                ),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._model_fit = model.fit(disp=False)

        logger.info("ARIMA fitting complete.")
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        """Generate out-of-sample forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Array of length ``horizon`` with point forecasts.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._model_fit is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        forecast = self._model_fit.forecast(steps=horizon)
        return np.array(forecast)

    def predict_rolling(
        self,
        history: pd.Series,
        target_index: pd.DatetimeIndex,
        horizon: int,
    ) -> np.ndarray:
        """Forecast each target hour from ``horizon`` hours earlier.

        For every timestamp ``t`` in ``target_index``, the forecast is the
        ``horizon``-step-ahead prediction made at origin ``t - horizon``,
        using observations up to that origin only. Parameters stay fixed from
        :meth:`fit`; the state is updated with new observations through
        ``extend`` (no refitting).

        Parameters
        ----------
        history : pd.Series
            Observed series after the training period (e.g. validation +
            test), hourly with a DatetimeIndex. Gaps are allowed.
        target_index : pd.DatetimeIndex
            Timestamps to forecast, sorted ascending.
        horizon : int
            Forecast horizon in hours.

        Returns
        -------
        np.ndarray
            One forecast per timestamp in ``target_index``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If the first origin falls inside the training period.
        """
        if self._model_fit is None or self._train_series is None:
            raise RuntimeError("Model must be fitted before calling predict_rolling().")

        step = pd.Timedelta(hours=1)
        train_end = self._train_series.index.max()
        first_origin = target_index.min() - horizon * step
        last_origin = target_index.max() - horizon * step
        if first_origin < train_end:
            raise ValueError("First forecast origin must not be before the end of training.")

        observed = history.reindex(pd.date_range(train_end + step, last_origin, freq="1h"))
        by_origin: Dict[pd.Timestamp, float] = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self._model_fit
            warmup = observed.loc[:first_origin]
            if len(warmup):
                results = results.extend(warmup)
            origin = first_origin
            while True:
                by_origin[origin] = float(np.asarray(results.forecast(steps=horizon))[-1])
                if origin >= last_origin:
                    break
                origin = origin + step
                results = results.extend(observed.loc[[origin]])

        logger.info("ARIMA rolling forecast done (%d origins).", len(by_origin))
        return np.array([by_origin[t - horizon * step] for t in target_index])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ARIMAForecaster":
        """Instantiate ARIMAForecaster from project config.

        Parameters
        ----------
        config : Dict[str, Any]
            Project configuration dictionary.

        Returns
        -------
        ARIMAForecaster
            New instance with hyperparameters from config.
        """
        cfg = config["arima"]
        return cls(
            p=cfg["p"],
            d=cfg["d"],
            q=cfg["q"],
            seasonal_p=cfg["seasonal_p"],
            seasonal_d=cfg["seasonal_d"],
            seasonal_q=cfg["seasonal_q"],
            seasonal_period=cfg["seasonal_period"],
        )
