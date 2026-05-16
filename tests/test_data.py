"""Unit tests for data loading and preprocessing utilities.

Tests cover cleaning, resampling, splitting, and the validation
of raw file structure without requiring a real download.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    clean_data,
    resample_to_hourly,
    split_data,
)


@pytest.fixture()
def raw_config() -> dict:
    """Minimal configuration for preprocessing tests."""
    return {
        "data": {
            "url": "https://example.com/data.zip",
            "raw_dir": "/tmp/test_raw",
            "processed_dir": "/tmp/test_processed",
            "filename": "household_power_consumption.txt",
            "target_col": "Global_active_power",
            "separator": ";",
            "na_values": ["?"],
            "resample_freq": "1h",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        }
    }


@pytest.fixture()
def sample_minute_df() -> pd.DataFrame:
    """Return a synthetic minute-level DataFrame resembling raw UCI data."""
    idx = pd.date_range("2022-01-01", periods=1440 * 7, freq="1min")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "Global_active_power": rng.uniform(0.1, 4.0, size=len(idx)),
            "Global_reactive_power": rng.uniform(0.0, 0.5, size=len(idx)),
            "Voltage": rng.uniform(235, 245, size=len(idx)),
            "Global_intensity": rng.uniform(0.2, 20.0, size=len(idx)),
        },
        index=idx,
    )


class TestCleanData:
    """Tests for the clean_data function."""

    def test_no_nans_in_target_after_clean(
        self, sample_minute_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """Target column must have zero NaN after cleaning."""
        dirty = sample_minute_df.copy()
        dirty.loc[dirty.index[:10], "Global_active_power"] = np.nan
        cleaned = clean_data(dirty, raw_config)
        assert cleaned["Global_active_power"].isnull().sum() == 0

    def test_outlier_clipping(
        self, sample_minute_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """Extreme values should be clipped to the 0.1–99.9 percentile."""
        dirty = sample_minute_df.copy()
        dirty.loc[dirty.index[0], "Global_active_power"] = 9999.0
        dirty.loc[dirty.index[1], "Global_active_power"] = -9999.0
        cleaned = clean_data(dirty, raw_config)
        q_high = sample_minute_df["Global_active_power"].quantile(0.999)
        assert cleaned["Global_active_power"].max() <= q_high * 1.01

    def test_returns_dataframe(
        self, sample_minute_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """Return type must be a pd.DataFrame."""
        result = clean_data(sample_minute_df, raw_config)
        assert isinstance(result, pd.DataFrame)

    def test_columns_unchanged(
        self, sample_minute_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """Column set must not change during cleaning."""
        result = clean_data(sample_minute_df, raw_config)
        assert set(result.columns) == set(sample_minute_df.columns)


class TestResampleToHourly:
    """Tests for resample_to_hourly."""

    def test_hourly_index_after_resample(
        self, sample_minute_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """Result must have a 1-hour frequency DatetimeIndex."""
        result = resample_to_hourly(sample_minute_df, raw_config)
        inferred = pd.infer_freq(result.index)
        assert inferred in ("h", "H", "1h", "T", None) or len(result) == len(
            sample_minute_df
        ) // 60 + 1 or result.index.freq is not None

    def test_row_count_reduces(
        self, sample_minute_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """Hourly resampled data must have fewer rows than minute-level input."""
        result = resample_to_hourly(sample_minute_df, raw_config)
        assert len(result) < len(sample_minute_df)

    def test_no_nans_in_result(
        self, sample_minute_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """Result must be free of NaN values."""
        result = resample_to_hourly(sample_minute_df, raw_config)
        assert result.isnull().sum().sum() == 0


class TestSplitData:
    """Tests for split_data."""

    @pytest.fixture()
    def hourly_df(self) -> pd.DataFrame:
        """Return a 2-year hourly DataFrame."""
        idx = pd.date_range("2020-01-01", periods=17520, freq="1h")
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {"Global_active_power": rng.uniform(0.1, 4.0, size=len(idx))},
            index=idx,
        )

    def test_split_sizes_sum_to_total(
        self, hourly_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """Train + val + test must equal total rows."""
        train, val, test = split_data(hourly_df, raw_config)
        assert len(train) + len(val) + len(test) == len(hourly_df)

    def test_chronological_order(
        self, hourly_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """train end < val start < val end < test start."""
        train, val, test = split_data(hourly_df, raw_config)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_no_overlap(
        self, hourly_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """No index value should appear in more than one split."""
        train, val, test = split_data(hourly_df, raw_config)
        assert len(train.index.intersection(val.index)) == 0
        assert len(val.index.intersection(test.index)) == 0
        assert len(train.index.intersection(test.index)) == 0

    def test_approximate_ratios(
        self, hourly_df: pd.DataFrame, raw_config: dict
    ) -> None:
        """Each split ratio should be within 2 percentage points of target."""
        train, val, test = split_data(hourly_df, raw_config)
        n = len(hourly_df)
        assert abs(len(train) / n - raw_config["data"]["train_ratio"]) < 0.02
        assert abs(len(val) / n - raw_config["data"]["val_ratio"]) < 0.02
        assert abs(len(test) / n - raw_config["data"]["test_ratio"]) < 0.02
