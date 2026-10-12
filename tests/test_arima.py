"""Unit tests for the SARIMA baseline's rolling-origin forecasts."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from src.models.arima_model import ARIMAForecaster  # noqa: E402

HORIZON = 6


@pytest.fixture()
def series() -> pd.Series:
    """Return a 300-hour series with a daily cycle and noise."""
    idx = pd.date_range("2022-01-01", periods=300, freq="1h")
    rng = np.random.default_rng(0)
    values = 2.0 + np.sin(np.arange(300) * 2 * np.pi / 24) + rng.normal(0, 0.1, 300)
    return pd.Series(values, index=idx, name="Global_active_power")


@pytest.fixture()
def fitted(series: pd.Series) -> ARIMAForecaster:
    """Return an AR(1) model fitted on the first 200 hours."""
    model = ARIMAForecaster(p=1, d=0, q=0, seasonal_p=0, seasonal_d=0, seasonal_q=0)
    return model.fit(series.iloc[:200])


class TestPredictRolling:
    """Tests for ARIMAForecaster.predict_rolling."""

    def test_one_forecast_per_target(self, fitted: ARIMAForecaster, series: pd.Series) -> None:
        """Returns exactly one value per requested timestamp."""
        target_index = series.index[250:300]
        preds = fitted.predict_rolling(series.iloc[200:], target_index, HORIZON)
        assert preds.shape == (50,)
        assert np.isfinite(preds).all()

    def test_matches_refiltering_up_to_origin(
        self, fitted: ARIMAForecaster, series: pd.Series
    ) -> None:
        """Each forecast equals an h-step forecast from the data up to its origin."""
        target_index = series.index[250:260]
        preds = fitted.predict_rolling(series.iloc[200:], target_index, HORIZON)
        for t, pred in zip(target_index, preds):
            origin = t - pd.Timedelta(hours=HORIZON)
            refit = fitted._model_fit.apply(series.loc[:origin].asfreq("1h"))
            expected = np.asarray(refit.forecast(steps=HORIZON))[-1]
            assert pred == pytest.approx(expected, rel=1e-6)

    def test_ignores_data_after_origin(self, fitted: ARIMAForecaster, series: pd.Series) -> None:
        """Values after a forecast's origin must not change that forecast."""
        target_index = series.index[[250]]
        origin = target_index[0] - pd.Timedelta(hours=HORIZON)
        changed = series.copy()
        changed[changed.index > origin] += 100.0
        original = fitted.predict_rolling(series.iloc[200:], target_index, HORIZON)
        perturbed = fitted.predict_rolling(changed.iloc[200:], target_index, HORIZON)
        np.testing.assert_allclose(original, perturbed)

    def test_origin_inside_training_raises(
        self, fitted: ARIMAForecaster, series: pd.Series
    ) -> None:
        """Forecast origins inside the training period would reuse fitted data."""
        with pytest.raises(ValueError):
            fitted.predict_rolling(series.iloc[200:], series.index[201:210], HORIZON)
