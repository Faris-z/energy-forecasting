"""Utility functions for the energy-forecasting project.

Provides configuration loading, logging setup, reproducibility
helpers, and shared I/O utilities used across the project.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file and return as nested dictionary.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML file cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: Optional[str] = None,
) -> None:
    """Configure root logger with console and optional file handlers.

    Parameters
    ----------
    level : str
        Logging level string, e.g. 'INFO', 'DEBUG', 'WARNING'.
    log_file : Optional[str]
        If provided, also write logs to this file path.
    fmt : Optional[str]
        Log format string. Defaults to a timestamped format.
    """
    if fmt is None:
        fmt = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"

    handlers: list = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries.

    Parameters
    ----------
    seed : int
        Integer seed value for Python random, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dirs(config: Dict[str, Any]) -> None:
    """Create all project output directories if they do not exist.

    Parameters
    ----------
    config : Dict[str, Any]
        Project configuration dictionary containing path settings.
    """
    dirs = [
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["paths"]["models_dir"],
        config["paths"]["reports_dir"],
        config["paths"]["figures_dir"],
        config["paths"]["logs_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Serialize a dictionary to a JSON file.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary to serialize.
    path : str
        Destination file path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return as dictionary.

    Parameters
    ----------
    path : str
        Source file path.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(p, "r") as f:
        return json.load(f)
