"""Data loading, cleaning, and preprocessing pipeline.

Transforms the raw UCI Household Electric Power Consumption text file
into a clean, hourly-resampled Pandas DataFrame ready for feature
engineering and model training.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(raw_path: Path, config: Dict[str, Any]) -> pd.DataFrame:
    """Load the raw UCI household power consumption file.

    Parameters
    ----------
    raw_path : Path
        Path to the raw ``.txt`` data file.
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with a parsed ``datetime`` index.
    """
    logger.info("Loading raw data from %s", raw_path)

    df = pd.read_csv(
        raw_path,
        sep=config["data"]["separator"],
        na_values=config["data"]["na_values"],
        low_memory=False,
    )

    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime")
    df = df.sort_index()

    numeric_cols = df.columns.tolist()
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    logger.info(
        "Loaded %d rows x %d columns spanning %s -> %s",
        len(df),
        df.shape[1],
        df.index.min(),
        df.index.max(),
    )
    return df


def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Handle missing values and remove obvious outliers.

    Strategy:
    - Interpolate short gaps (≤ 6 consecutive missing minutes) linearly.
    - Forward-fill remaining gaps up to 60 minutes.
    - Drop rows where the target is still NaN.
    - Clip extreme values at the 0.1 / 99.9 percentile.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with a datetime index.
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    target = config["data"]["target_col"]
    logger.info("Missing values before cleaning:\n%s", df.isnull().sum())

    df = df.interpolate(method="linear", limit=6)
    df = df.ffill(limit=60)

    before = len(df)
    df = df.dropna(subset=[target])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN in target column.", dropped)

    q_low = df[target].quantile(0.001)
    q_high = df[target].quantile(0.999)
    clipped = ((df[target] < q_low) | (df[target] > q_high)).sum()
    df[target] = df[target].clip(lower=q_low, upper=q_high)
    logger.info("Clipped %d extreme outliers in target column.", clipped)

    logger.info("Missing values after cleaning:\n%s", df.isnull().sum())
    return df


def resample_to_hourly(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Resample the minute-level data to hourly frequency.

    Aggregates all numeric columns by mean and forward-fills any
    remaining gaps after resampling.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned minute-level DataFrame.
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Hourly-resampled DataFrame.
    """
    freq = config["data"]["resample_freq"]
    logger.info("Resampling to %s frequency ...", freq)

    df_hourly = df.resample(freq).mean()
    df_hourly = df_hourly.ffill(limit=24)
    df_hourly = df_hourly.dropna()

    logger.info("Resampled shape: %d rows × %d columns", df_hourly.shape[0], df_hourly.shape[1])
    return df_hourly


def split_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the time series into train, validation, and test sets.

    Uses chronological splitting to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Full hourly DataFrame.
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` DataFrames.
    """
    n = len(df)
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info(
        "Split sizes — Train: %d  Val: %d  Test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def preprocess_pipeline(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full preprocessing pipeline end-to-end.

    Loads raw data, cleans, resamples, and splits into train/val/test.
    Saves processed splits to disk.

    Parameters
    ----------
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` DataFrames with hourly resolution.
    """
    from src.data.download import download_dataset, validate_raw_file

    raw_path = download_dataset(config)
    validate_raw_file(raw_path, config)

    df = load_raw_data(raw_path, config)
    df = clean_data(df, config)
    df = resample_to_hourly(df, config)

    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = split_data(df, config)

    df.to_parquet(processed_dir / "full.parquet")
    train.to_parquet(processed_dir / "train.parquet")
    val.to_parquet(processed_dir / "val.parquet")
    test.to_parquet(processed_dir / "test.parquet")

    logger.info("Saved processed datasets to %s", processed_dir)
    return train, val, test


def load_processed_data(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved processed splits from Parquet files.

    Parameters
    ----------
    config : Dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` DataFrames.

    Raises
    ------
    FileNotFoundError
        If processed files are missing. Run ``make data`` first.
    """
    processed_dir = Path(config["data"]["processed_dir"])
    for split in ["train", "val", "test"]:
        p = processed_dir / f"{split}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Processed file not found: {p}. Run `make data` first.")

    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")

    logger.info(
        "Loaded splits — Train: %d  Val: %d  Test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test
