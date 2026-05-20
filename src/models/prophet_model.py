"""Facebook Prophet model wrapper for 24-hour-ahead energy forecasting.

Provides a consistent interface for fitting Prophet and generating
multi-step-ahead forecasts compatible with the ensemble pipeline.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProphetForecaster:
    """Prophet wrapper with sklearn-compatible interface.

    Parameters
    ----------
    changepoint_prior_scale : float
        Flexibility of the trend changepoints.
    seasonality_prior_scale : float
        Strength of seasonality components.
    holidays_prior_scale : float
        Strength of holiday effects.
    seasonality_mode : str
        'additive' or 'multiplicative'.
    yearly_seasonality : bool
        Whether to model yearly seasonality.
    weekly_seasonality : bool
        Whether to model weekly seasonality.
    daily_seasonality : bool
        Whether to model daily seasonality.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self._model: Optional[Any] = None
        self._last_date: Optional[pd.Timestamp] = None

    def fit(self, series: pd.Series) -> "ProphetForecaster":
        """Fit Prophet on a time series.

        Parameters
        ----------
        series : pd.Series
            Hourly target series with a DatetimeIndex named 'datetime'.

        Returns
        -------
        ProphetForecaster
            Fitted instance (self).
        """
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError("prophet is required: pip install prophet") from exc

        logger.info("Fitting Prophet on %d observations ...", len(series))

        prophet_df = pd.DataFrame(
            {
                "ds": series.index,
                "y": series.values,
            }
        )

        self._model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(prophet_df)

        self._last_date = series.index.max()
        logger.info("Prophet fitting complete.")
        return self

    def predict(self, horizon: int = 24) -> np.ndarray:
        """Generate out-of-sample forecasts.

        Parameters
        ----------
        horizon : int
            Number of hourly steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Array of length ``horizon`` with point forecasts.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._model is None or self._last_date is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        future_dates = pd.date_range(
            start=self._last_date + pd.Timedelta(hours=1),
            periods=horizon,
            freq="1h",
        )
        future_df = pd.DataFrame({"ds": future_dates})
        forecast = self._model.predict(future_df)
        predictions = forecast["yhat"].values
        predictions = np.clip(predictions, 0, None)
        return predictions

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ProphetForecaster":
        """Instantiate ProphetForecaster from project config.

        Parameters
        ----------
        config : Dict[str, Any]
            Project configuration dictionary.

        Returns
        -------
        ProphetForecaster
            New instance with hyperparameters from config.
        """
        cfg = config["prophet"]
        return cls(
            changepoint_prior_scale=cfg["changepoint_prior_scale"],
            seasonality_prior_scale=cfg["seasonality_prior_scale"],
            holidays_prior_scale=cfg["holidays_prior_scale"],
            seasonality_mode=cfg["seasonality_mode"],
            yearly_seasonality=cfg["yearly_seasonality"],
            weekly_seasonality=cfg["weekly_seasonality"],
            daily_seasonality=cfg["daily_seasonality"],
        )
