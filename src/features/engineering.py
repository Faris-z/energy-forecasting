"""Feature engineering for time series forecasting.

Creates lag features, rolling statistics, calendar features,
Fourier encodings, and interaction features from the hourly
power consumption time series.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int],
    min_lag: int = 1,
) -> pd.DataFrame:
    """Add lagged values of the target column as new features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a datetime index.
    target_col : str
        Name of the target column to lag.
    lags : List[int]
        List of lag offsets in hours (e.g., [1, 6, 12, 24]).
    min_lag : int
        Smallest lag allowed. Lags below it are skipped, because they would
        not be known at forecast time (set this to the forecast horizon).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``lag_{n}h`` columns.
    """
    df = df.copy()
    for lag in lags:
        if lag < min_lag:
            logger.warning("Skipping lag %dh: shorter than the %dh horizon.", lag, min_lag)
            continue
        col_name = f"{target_col}_lag_{lag}h"
        df[col_name] = df[target_col].shift(lag)
        logger.debug("Added lag feature: %s", col_name)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int],
    stats: List[str],
    shift: int = 1,
) -> pd.DataFrame:
    """Add rolling window statistics for the target column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a datetime index.
    target_col : str
        Name of the target column to compute rolling stats on.
    windows : List[int]
        Rolling window sizes in hours.
    stats : List[str]
        Statistics to compute per window. Supported: 'mean', 'std',
        'min', 'max'.
    shift : int
        Hours to shift the series before rolling, so each window ends this
        many hours before the row it describes (set this to the forecast
        horizon).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling statistic columns.
    """
    df = df.copy()
    stat_fn_map = {
        "mean": lambda s, w: s.rolling(w, min_periods=1).mean(),
        "std": lambda s, w: s.rolling(w, min_periods=2).std(),
        "min": lambda s, w: s.rolling(w, min_periods=1).min(),
        "max": lambda s, w: s.rolling(w, min_periods=1).max(),
    }

    for window in windows:
        series = df[target_col].shift(shift)  # avoid leakage
        for stat in stats:
            if stat not in stat_fn_map:
                logger.warning("Unknown rolling stat '%s' — skipping.", stat)
                continue
            col_name = f"{target_col}_roll_{window}h_{stat}"
            df[col_name] = stat_fn_map[stat](series, window)
            logger.debug("Added rolling feature: %s", col_name)
    return df


def add_calendar_features(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add calendar-based temporal features from the datetime index.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a ``DatetimeIndex``.
    feature_names : Optional[List[str]]
        Subset of calendar features to compute. If None, all are added.
        Supported: 'hour', 'dayofweek', 'month', 'quarter',
        'dayofyear', 'weekofyear', 'is_weekend', 'is_holiday'.

    Returns
    -------
    pd.DataFrame
        DataFrame with new calendar feature columns.

    Raises
    ------
    ValueError
        If the DataFrame index is not a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    df = df.copy()
    idx = df.index

    all_features: Dict[str, Any] = {
        "hour": idx.hour,
        "dayofweek": idx.dayofweek,
        "month": idx.month,
        "quarter": idx.quarter,
        "dayofyear": idx.dayofyear,
        "weekofyear": idx.isocalendar().week.astype(int).values,
        "is_weekend": (idx.dayofweek >= 5).astype(int),
        "is_holiday": _compute_us_holidays(idx),
    }

    if feature_names is None:
        feature_names = list(all_features.keys())

    for name in feature_names:
        if name in all_features:
            df[name] = all_features[name]
            logger.debug("Added calendar feature: %s", name)
        else:
            logger.warning("Unknown calendar feature '%s' — skipping.", name)

    return df


def _compute_us_holidays(idx: pd.DatetimeIndex) -> np.ndarray:
    """Compute a binary holiday indicator for US federal holidays.

    Parameters
    ----------
    idx : pd.DatetimeIndex
        Datetime index to evaluate.

    Returns
    -------
    np.ndarray
        Integer array (0 or 1) indicating holiday status.
    """
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=idx.min(), end=idx.max())
        return idx.normalize().isin(holidays).astype(int).values
    except Exception:
        logger.debug("Could not compute US holidays; using zeros.")
        return np.zeros(len(idx), dtype=int)


def add_fourier_features(
    df: pd.DataFrame,
    periods: List[int],
    n_terms: int,
) -> pd.DataFrame:
    """Add Fourier series encodings to capture periodic patterns.

    For each (period, term) pair, adds both sine and cosine components,
    enabling linear models to represent non-linear cyclic behaviour.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a ``DatetimeIndex``.
    periods : List[int]
        Cycle lengths in hours (e.g., 24 for daily, 168 for weekly).
    n_terms : int
        Number of harmonic terms per period.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional Fourier encoding columns.
    """
    df = df.copy()
    t = np.arange(len(df))

    for period in periods:
        for k in range(1, n_terms + 1):
            sin_col = f"fourier_sin_{period}h_k{k}"
            cos_col = f"fourier_cos_{period}h_k{k}"
            angle = 2 * np.pi * k * t / period
            df[sin_col] = np.sin(angle)
            df[cos_col] = np.cos(angle)
            logger.debug("Added Fourier features: %s, %s", sin_col, cos_col)

    return df


def add_interaction_features(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Add selected interaction features between calendar and lag columns.

    Creates power-law and multiplicative interactions that capture
    non-linear relationships (e.g., load × hour-of-day).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    target_col : str
        Name of the target column (for lag reference names).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional interaction features.
    """
    df = df.copy()
    lag_24 = f"{target_col}_lag_24h"
    lag_168 = f"{target_col}_lag_168h"

    if lag_24 in df.columns and "hour" in df.columns:
        df["lag24_x_hour"] = df[lag_24] * df["hour"]
        logger.debug("Added interaction: lag24_x_hour")

    if lag_168 in df.columns and "dayofweek" in df.columns:
        df["lag168_x_dow"] = df[lag_168] * df["dayofweek"]
        logger.debug("Added interaction: lag168_x_dow")

    if "hour" in df.columns and "is_weekend" in df.columns:
        df["hour_x_weekend"] = df["hour"] * df["is_weekend"]
        logger.debug("Added interaction: hour_x_weekend")

    if lag_24 in df.columns:
        df["lag24_squared"] = df[lag_24] ** 2
        logger.debug("Added interaction: lag24_squared")

    if "month" in df.columns and "hour" in df.columns:
        df["month_x_hour"] = df["month"] * df["hour"]
        logger.debug("Added interaction: month_x_hour")

    return df


def build_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    drop_na: bool = True,
) -> pd.DataFrame:
    """Run the complete feature engineering pipeline.

    Applies all feature transformations in the correct order to avoid
    data leakage. Only the target column is kept from the input: the other
    raw measurements (e.g. ``Global_intensity``, ``Voltage``) are recorded
    in the same hour as the target and would leak it. Optionally drops rows
    with NaN (caused by initial lags).

    Every target-derived feature for hour ``t`` uses only values from hour
    ``t - horizon`` or earlier (``config['forecasting']['horizon']``, default
    1), so each row is a direct ``horizon``-hours-ahead forecast. Features are
    computed on a gap-free hourly index, so lags and windows are measured in
    hours even when the input is missing some hours.

    Parameters
    ----------
    df : pd.DataFrame
        Input hourly DataFrame with at least the target column. Any other
        columns are dropped.
    config : Dict[str, Any]
        Project configuration dictionary.
    drop_na : bool
        If True, drop rows with any NaN values after feature creation.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame ready for model input.
    """
    target = config["data"]["target_col"]
    feat_cfg = config["features"]
    horizon = config.get("forecasting", {}).get("horizon", 1)

    logger.info("Starting feature engineering (horizon=%dh) ...", horizon)

    same_hour_cols = [c for c in df.columns if c != target]
    if same_hour_cols:
        logger.info("Dropping same-hour input columns to avoid leakage: %s", same_hour_cols)
    original_index = df.index
    full_index = pd.date_range(original_index.min(), original_index.max(), freq="1h")
    df = df[[target]].reindex(full_index)

    df = add_lag_features(df, target, feat_cfg["lags"], min_lag=horizon)
    df = add_rolling_features(
        df, target, feat_cfg["rolling_windows"], feat_cfg["rolling_stats"], shift=horizon
    )
    df = add_calendar_features(df, feat_cfg.get("calendar_features"))
    df = add_fourier_features(
        df,
        periods=feat_cfg["fourier_periods"],
        n_terms=feat_cfg["fourier_terms"],
    )
    df = add_interaction_features(df, target)
    df = df.loc[original_index]

    if drop_na:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        logger.info("Dropped %d rows with NaN after feature engineering.", dropped)

    logger.info(
        "Feature engineering complete: %d rows × %d columns",
        df.shape[0],
        df.shape[1],
    )
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Return the list of feature column names (excludes the target).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    target_col : str
        Name of the target column to exclude.

    Returns
    -------
    List[str]
        Sorted list of feature column names.
    """
    return [c for c in df.columns if c != target_col]
