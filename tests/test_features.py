"""Unit tests for the feature engineering module.

Tests cover individual feature generators, the full pipeline,
and edge-case behaviour (empty DataFrames, missing columns, etc.).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_calendar_features,
    add_fourier_features,
    add_interaction_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    get_feature_columns,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Return a minimal hourly DataFrame for testing."""
    idx = pd.date_range("2022-01-01", periods=300, freq="1h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"Global_active_power": rng.uniform(0.2, 4.0, size=300)},
        index=idx,
    )


@pytest.fixture()
def config() -> dict:
    """Return a minimal configuration dictionary for testing."""
    return {
        "data": {"target_col": "Global_active_power"},
        "features": {
            "lags": [1, 6, 12, 24],
            "rolling_windows": [6, 24],
            "rolling_stats": ["mean", "std", "min", "max"],
            "fourier_periods": [24],
            "fourier_terms": 2,
            "calendar_features": [
                "hour",
                "dayofweek",
                "month",
                "is_weekend",
            ],
        },
    }


class TestLagFeatures:
    """Tests for add_lag_features."""

    def test_lag_columns_created(self, sample_df: pd.DataFrame) -> None:
        """Expected lag columns should be present after transformation."""
        result = add_lag_features(sample_df, "Global_active_power", [1, 6, 24])
        assert "Global_active_power_lag_1h" in result.columns
        assert "Global_active_power_lag_6h" in result.columns
        assert "Global_active_power_lag_24h" in result.columns

    def test_lag_shifts_correctly(self, sample_df: pd.DataFrame) -> None:
        """Lag-1 at row i should equal the original value at row i-1."""
        result = add_lag_features(sample_df, "Global_active_power", [1])
        original = sample_df["Global_active_power"].values
        lagged = result["Global_active_power_lag_1h"].values
        assert np.isnan(lagged[0])
        assert lagged[1] == pytest.approx(original[0])

    def test_no_lags_unchanged(self, sample_df: pd.DataFrame) -> None:
        """Passing an empty lag list should leave the DataFrame unchanged."""
        result = add_lag_features(sample_df, "Global_active_power", [])
        assert list(result.columns) == list(sample_df.columns)

    def test_original_not_modified(self, sample_df: pd.DataFrame) -> None:
        """Input DataFrame must not be mutated."""
        original_cols = list(sample_df.columns)
        add_lag_features(sample_df, "Global_active_power", [1, 24])
        assert list(sample_df.columns) == original_cols


class TestRollingFeatures:
    """Tests for add_rolling_features."""

    def test_rolling_columns_created(self, sample_df: pd.DataFrame) -> None:
        """Expected rolling columns should exist."""
        result = add_rolling_features(
            sample_df, "Global_active_power", [6], ["mean", "std"]
        )
        assert "Global_active_power_roll_6h_mean" in result.columns
        assert "Global_active_power_roll_6h_std" in result.columns

    def test_no_future_leakage(self, sample_df: pd.DataFrame) -> None:
        """Rolling mean at time t should only use data up to t-1."""
        result = add_rolling_features(
            sample_df, "Global_active_power", [3], ["mean"]
        )
        col = result["Global_active_power_roll_3h_mean"]
        orig = sample_df["Global_active_power"]
        for i in range(3, 10):
            expected = orig.iloc[i - 3: i].mean()
            assert col.iloc[i] == pytest.approx(expected, rel=1e-5)

    def test_unknown_stat_skipped(self, sample_df: pd.DataFrame) -> None:
        """Unknown stat names should be skipped without raising."""
        result = add_rolling_features(
            sample_df, "Global_active_power", [6], ["nonexistent"]
        )
        assert result.shape == sample_df.shape


class TestCalendarFeatures:
    """Tests for add_calendar_features."""

    def test_hour_range(self, sample_df: pd.DataFrame) -> None:
        """Hour values must be in [0, 23]."""
        result = add_calendar_features(sample_df, ["hour"])
        assert result["hour"].between(0, 23).all()

    def test_dayofweek_range(self, sample_df: pd.DataFrame) -> None:
        """Day-of-week values must be in [0, 6]."""
        result = add_calendar_features(sample_df, ["dayofweek"])
        assert result["dayofweek"].between(0, 6).all()

    def test_is_weekend_binary(self, sample_df: pd.DataFrame) -> None:
        """is_weekend must be 0 or 1 only."""
        result = add_calendar_features(sample_df, ["is_weekend"])
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_raises_on_non_datetime_index(self) -> None:
        """Should raise ValueError for integer-indexed DataFrames."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            add_calendar_features(df)


class TestFourierFeatures:
    """Tests for add_fourier_features."""

    def test_fourier_columns_created(self, sample_df: pd.DataFrame) -> None:
        """Sin and cos columns must be created for each (period, term)."""
        result = add_fourier_features(sample_df, periods=[24], n_terms=2)
        assert "fourier_sin_24h_k1" in result.columns
        assert "fourier_cos_24h_k1" in result.columns
        assert "fourier_sin_24h_k2" in result.columns
        assert "fourier_cos_24h_k2" in result.columns

    def test_fourier_values_in_range(self, sample_df: pd.DataFrame) -> None:
        """Sin/cos values must be in [-1, 1]."""
        result = add_fourier_features(sample_df, periods=[24], n_terms=1)
        assert result["fourier_sin_24h_k1"].between(-1.0, 1.0).all()
        assert result["fourier_cos_24h_k1"].between(-1.0, 1.0).all()

    def test_multiple_periods(self, sample_df: pd.DataFrame) -> None:
        """Multiple periods should each produce their own feature set."""
        result = add_fourier_features(sample_df, periods=[24, 168], n_terms=1)
        assert "fourier_sin_24h_k1" in result.columns
        assert "fourier_sin_168h_k1" in result.columns


class TestInteractionFeatures:
    """Tests for add_interaction_features."""

    def test_interactions_added_when_dependencies_present(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Interaction columns require lag and calendar columns to be present."""
        df = add_lag_features(sample_df, "Global_active_power", [24, 168])
        df = add_calendar_features(df, ["hour", "dayofweek", "is_weekend", "month"])
        result = add_interaction_features(df, "Global_active_power")
        assert "lag24_x_hour" in result.columns
        assert "lag168_x_dow" in result.columns

    def test_no_interactions_without_dependencies(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Should not fail even if required columns are absent."""
        result = add_interaction_features(sample_df, "Global_active_power")
        assert isinstance(result, pd.DataFrame)


class TestBuildFeatures:
    """Tests for the full build_features pipeline."""

    def test_output_has_more_columns(
        self, sample_df: pd.DataFrame, config: dict
    ) -> None:
        """Feature engineering should significantly increase column count."""
        result = build_features(sample_df, config, drop_na=True)
        assert result.shape[1] > sample_df.shape[1]

    def test_no_nans_when_drop_na(
        self, sample_df: pd.DataFrame, config: dict
    ) -> None:
        """With drop_na=True, result should have zero NaN values."""
        result = build_features(sample_df, config, drop_na=True)
        assert result.isnull().sum().sum() == 0

    def test_target_column_preserved(
        self, sample_df: pd.DataFrame, config: dict
    ) -> None:
        """The target column must still exist after feature engineering."""
        result = build_features(sample_df, config, drop_na=True)
        assert "Global_active_power" in result.columns


class TestGetFeatureColumns:
    """Tests for get_feature_columns."""

    def test_excludes_target(self, sample_df: pd.DataFrame, config: dict) -> None:
        """Target column should not appear in the feature list."""
        df = build_features(sample_df, config, drop_na=True)
        cols = get_feature_columns(df, "Global_active_power")
        assert "Global_active_power" not in cols

    def test_returns_list_of_strings(
        self, sample_df: pd.DataFrame, config: dict
    ) -> None:
        """Should return a plain Python list of strings."""
        df = build_features(sample_df, config, drop_na=True)
        cols = get_feature_columns(df, "Global_active_power")
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)
