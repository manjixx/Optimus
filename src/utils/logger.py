import logging
import sys
from typing import Optional

def setup_logger(level: int = logging.INFO) -> None:
    """Set up the root logger with console output."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler and formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a named logger. If level is provided, set its level."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger