"""Unit tests for ProphetForecaster.predict_at (uses a stub, no Prophet install needed)."""

import numpy as np
import pandas as pd
import pytest

from src.models.prophet_model import ProphetForecaster


class _StubProphet:
    """Mimics Prophet.predict: yhat depends only on the timestamp."""

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        hours = future["ds"].dt.hour.to_numpy()
        return pd.DataFrame({"ds": future["ds"], "yhat": hours - 2.0})


class TestPredictAt:
    """Tests for ProphetForecaster.predict_at."""

    def test_one_forecast_per_timestamp_clipped(self) -> None:
        """Returns one value per timestamp, in order, clipped at zero."""
        model = ProphetForecaster()
        model._model = _StubProphet()
        timestamps = pd.date_range("2010-01-01", periods=5, freq="1h")
        np.testing.assert_allclose(model.predict_at(timestamps), [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_unfitted_raises(self) -> None:
        """Calling before fit is an error."""
        with pytest.raises(RuntimeError):
            ProphetForecaster().predict_at(pd.date_range("2010-01-01", periods=2, freq="1h"))
