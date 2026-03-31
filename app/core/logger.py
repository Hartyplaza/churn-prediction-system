from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import PROJECT_ROOT


def setup_logger(name: str = "churn_prediction_system") -> logging.Logger:
    log_dir = PROJECT_ROOT / "artifacts" / "logs"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "system.log"
        file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Streamlit/cloud environments may not always allow file writes.
        pass

    logger.propagate = False
    return logger
