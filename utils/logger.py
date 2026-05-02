"""Logging utility"""

import logging
import sys
import os
from datetime import datetime


def setup_logger(name: str = "JunctionDetector", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper()))

    # Fix Unicode trên Windows: force UTF-8 cho stdout
    if sys.platform == "win32":
        import io
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    else:
        stream = sys.stdout

    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(logging.DEBUG)

    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # File log luôn dùng UTF-8
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger