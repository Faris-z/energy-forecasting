"""Shared pytest fixtures and configuration.

Fixtures defined here are automatically available to all test modules
without needing explicit imports.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def global_config() -> dict:
    """Return the full project configuration loaded from YAML.

    Uses the real config file so tests stay in sync with production settings.
    """
    from src.utils import load_config

    return load_config("config/config.yaml")


@pytest.fixture()
def small_hourly_series() -> pd.Series:
    """Return a 30-day hourly time series for lightweight tests."""
    idx = pd.date_range("2022-06-01", periods=720, freq="1h")
    rng = np.random.default_rng(0)
    return pd.Series(rng.uniform(0.5, 3.5, size=720), index=idx, name="Global_active_power")


@pytest.fixture()
def small_feature_df(small_hourly_series: pd.Series) -> pd.DataFrame:
    """Return a small feature-engineered DataFrame for fast tests."""
    from src.features.engineering import build_features

    config = {
        "data": {"target_col": "Global_active_power"},
        "features": {
            "lags": [1, 24],
            "rolling_windows": [6],
            "rolling_stats": ["mean", "std"],
            "fourier_periods": [24],
            "fourier_terms": 1,
            "calendar_features": ["hour", "dayofweek", "is_weekend"],
        },
    }
    df = small_hourly_series.to_frame()
    return build_features(df, config, drop_na=True)
