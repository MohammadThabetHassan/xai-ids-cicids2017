"""
Centralized logging configuration for the XAI-IDS pipeline.

Provides structured logging to both console and file with timestamps,
log levels, and module identification.
"""

import logging
import os
from pathlib import Path


def setup_logger(
    name: str = "xai_ids",
    log_dir: str = "outputs/logs",
    level: int = logging.INFO,
    log_file: str = "pipeline.log",
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically module name).
    log_dir : str
        Directory to store log files.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    log_file : str
        Name of the log file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve an existing logger by name, or create one with defaults.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
