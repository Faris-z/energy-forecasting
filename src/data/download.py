"""Dataset download and extraction utilities.

Downloads the UCI Household Electric Power Consumption dataset,
extracts it, and validates the raw file before preprocessing.
"""

import logging
import zipfile
from pathlib import Path
from typing import Dict, Any
from urllib.request import urlretrieve
from urllib.error import URLError

logger = logging.getLogger(__name__)


def download_dataset(config: Dict[str, Any]) -> Path:
    """Download and extract the UCI power consumption dataset.

    Skips download if the raw file already exists locally.

    Parameters
    ----------
    config : Dict[str, Any]
        Project configuration dictionary. Uses keys:
        ``data.url``, ``data.raw_dir``, ``data.filename``.

    Returns
    -------
    Path
        Path to the extracted raw data file.

    Raises
    ------
    URLError
        If the download fails due to network issues.
    RuntimeError
        If the extracted file cannot be found in the archive.
    """
    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_file = raw_dir / config["data"]["filename"]

    if target_file.exists():
        logger.info("Raw dataset already exists at %s — skipping download.", target_file)
        return target_file

    url = config["data"]["url"]
    zip_path = raw_dir / "household_power_consumption.zip"

    logger.info("Downloading dataset from %s ...", url)
    try:
        urlretrieve(url, zip_path)
    except URLError as exc:
        raise URLError(f"Failed to download dataset from {url}: {exc}") from exc

    logger.info("Download complete. Extracting archive ...")
    _extract_zip(zip_path, raw_dir)

    if not target_file.exists():
        raise RuntimeError(
            f"Expected file '{config['data']['filename']}' not found after extraction."
        )

    zip_path.unlink()
    logger.info("Dataset ready at %s", target_file)
    return target_file


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP archive to the given directory.

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP file.
    extract_to : Path
        Directory where contents will be extracted.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    logger.debug("Extracted %s to %s", zip_path.name, extract_to)


def validate_raw_file(raw_path: Path, config: Dict[str, Any]) -> None:
    """Perform basic sanity checks on the raw CSV/TXT file.

    Parameters
    ----------
    raw_path : Path
        Path to the raw data file.
    config : Dict[str, Any]
        Project configuration dictionary.

    Raises
    ------
    ValueError
        If the file appears malformed or too small.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    size_mb = raw_path.stat().st_size / (1024**2)
    logger.info("Raw file size: %.1f MB", size_mb)

    if size_mb < 1:
        raise ValueError(
            f"Raw file appears too small ({size_mb:.2f} MB). " "Download may have been incomplete."
        )

    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
        first_row = f.readline().strip()

    logger.debug("Header: %s", header)
    logger.debug("First row: %s", first_row)

    expected_cols = ["Date", "Time", "Global_active_power"]
    for col in expected_cols:
        if col not in header:
            raise ValueError(f"Expected column '{col}' not found in header: {header}")

    logger.info("Raw file validation passed.")
