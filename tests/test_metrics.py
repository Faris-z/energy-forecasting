"""Unit tests for evaluation metrics.

Verifies mathematical correctness of RMSE, MAE, MAPE and the
walk-forward splitter's no-leakage guarantee.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    mae,
    mape,
    rmse,
    walk_forward_split,
)


class TestRMSE:
    """Tests for the rmse function."""

    def test_perfect_prediction(self) -> None:
        """RMSE should be 0 for identical arrays."""
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """RMSE of [0,0,0] vs [1,1,1] should be 1.0."""
        assert rmse(np.zeros(3), np.ones(3)) == pytest.approx(1.0)

    def test_symmetric(self) -> None:
        """RMSE(a, b) == RMSE(b, a)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5, 2.5])
        assert rmse(a, b) == pytest.approx(rmse(b, a))

    def test_returns_float(self) -> None:
        """Return type must be Python float."""
        result = rmse(np.array([1.0]), np.array([2.0]))
        assert isinstance(result, float)

    def test_single_element(self) -> None:
        """RMSE of scalar arrays should equal absolute difference."""
        assert rmse(np.array([3.0]), np.array([5.0])) == pytest.approx(2.0)


class TestMAE:
    """Tests for the mae function."""

    def test_perfect_prediction(self) -> None:
        """MAE should be 0 for identical arrays."""
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """MAE of [1,2,3] vs [2,3,4] should be 1.0."""
        assert mae(np.array([1, 2, 3]), np.array([2, 3, 4])) == pytest.approx(1.0)

    def test_non_negative(self) -> None:
        """MAE must always be non-negative."""
        a = np.random.default_rng(0).uniform(-5, 5, size=100)
        b = np.random.default_rng(1).uniform(-5, 5, size=100)
        assert mae(a, b) >= 0.0


class TestMAPE:
    """Tests for the mape function."""

    def test_perfect_prediction(self) -> None:
        """MAPE should be approximately 0 for identical arrays."""
        y = np.array([1.0, 2.0, 3.0])
        assert mape(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self) -> None:
        """MAPE must always be non-negative."""
        a = np.abs(np.random.default_rng(0).uniform(0.1, 5, size=100))
        b = np.abs(np.random.default_rng(1).uniform(0.1, 5, size=100))
        assert mape(a, b) >= 0.0

    def test_returns_percentage(self) -> None:
        """MAPE of 50% over-prediction on unit actuals should be ~50."""
        y_true = np.ones(10)
        y_pred = np.ones(10) * 1.5
        result = mape(y_true, y_pred)
        assert result == pytest.approx(50.0, rel=0.01)

    def test_zero_actuals_handled(self) -> None:
        """Zero actual values should not raise ZeroDivisionError."""
        y_true = np.zeros(5)
        y_pred = np.ones(5)
        result = mape(y_true, y_pred)
        assert np.isfinite(result)


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_keys_present(self) -> None:
        """Result dict must have 'rmse', 'mae', 'mape'."""
        result = compute_all_metrics(np.ones(5), np.zeros(5))
        assert set(result.keys()) == {"rmse", "mae", "mape"}

    def test_all_values_finite(self) -> None:
        """All metric values must be finite floats."""
        result = compute_all_metrics(np.ones(10), np.random.rand(10))
        for v in result.values():
            assert np.isfinite(v)


class TestWalkForwardSplit:
    """Tests for walk_forward_split."""

    @pytest.fixture()
    def big_df(self) -> pd.DataFrame:
        """Return a sufficiently large DataFrame to test splitting."""
        idx = pd.date_range("2018-01-01", periods=10_000, freq="1h")
        rng = np.random.default_rng(99)
        return pd.DataFrame({"x": rng.random(10_000), "y": rng.random(10_000)}, index=idx)

    @pytest.fixture()
    def wf_config(self) -> dict:
        """Return a minimal config for walk-forward splitting."""
        return {
            "evaluation": {
                "walk_forward": {
                    "n_splits": 3,
                    "test_size": 168,
                    "gap": 0,
                }
            }
        }

    def test_correct_number_of_folds(
        self, big_df: pd.DataFrame, wf_config: dict
    ) -> None:
        """Generator must yield exactly n_splits folds."""
        folds = list(walk_forward_split(big_df, wf_config))
        assert len(folds) == wf_config["evaluation"]["walk_forward"]["n_splits"]

    def test_no_temporal_leakage(
        self, big_df: pd.DataFrame, wf_config: dict
    ) -> None:
        """Train end must always be before test start in every fold."""
        for train_fold, test_fold in walk_forward_split(big_df, wf_config):
            assert train_fold.index.max() < test_fold.index.min()

    def test_test_size_respected(
        self, big_df: pd.DataFrame, wf_config: dict
    ) -> None:
        """Each test fold must have exactly test_size rows."""
        expected = wf_config["evaluation"]["walk_forward"]["test_size"]
        for _, test_fold in walk_forward_split(big_df, wf_config):
            assert len(test_fold) == expected

    def test_train_grows_across_folds(
        self, big_df: pd.DataFrame, wf_config: dict
    ) -> None:
        """Expanding-window: each train fold should be larger than the last."""
        sizes = [len(tr) for tr, _ in walk_forward_split(big_df, wf_config)]
        assert sizes == sorted(sizes)
