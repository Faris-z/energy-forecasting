"""Standalone data preparation script.

Downloads the UCI Household Electric Power Consumption dataset,
cleans, resamples to hourly frequency, and saves train/val/test
Parquet splits to ``data/processed/``.

Run: python prepare_data.py  or  make data
"""

import logging
import sys
from pathlib import Path

from src.data.preprocessing import preprocess_pipeline
from src.utils import ensure_dirs, load_config, setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Entry point for the data preparation pipeline.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    config = load_config("config/config.yaml")

    setup_logging(
        level=config["logging"]["level"],
        log_file=config["logging"]["log_file"],
        fmt=config["logging"]["format"],
    )
    ensure_dirs(config)

    logger.info("=" * 60)
    logger.info("Energy Forecasting — Data Preparation")
    logger.info("=" * 60)

    try:
        train, val, test = preprocess_pipeline(config)
        logger.info("Data preparation complete.")
        logger.info(
            "Train: %d rows  Val: %d rows  Test: %d rows",
            len(train),
            len(val),
            len(test),
        )
        logger.info("Date range: %s -> %s", train.index.min(), test.index.max())
        return 0
    except Exception as exc:
        logger.exception("Data preparation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
