import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "neuroguard",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with specified parameters.

    Args:
        name (str): Name of the logger
        log_file (Optional[str]): Path to log file (if None, only console logging)
        log_level (int): Logging level (default: logging.INFO)
        format_string (Optional[str]): Format string for log messages

    Returns:
        logging.Logger: Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(format_string)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "neuroguard") -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name (str): Name of the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
