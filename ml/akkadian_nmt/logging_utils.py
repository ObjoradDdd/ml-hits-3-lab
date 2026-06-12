"""Singleton logger for the whole pipeline.

Spec requirement: everything (data loading, training, decoding, errors) is
logged to ./data/log_file.log, viewable inside the container.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER: logging.Logger | None = None

LOG_DIR = Path("./data")
LOG_FILE = LOG_DIR / "log_file.log"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str = "akkadian_nmt") -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger("akkadian_nmt")
        root.setLevel(logging.INFO)
        formatter = logging.Formatter(_FORMAT)

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        _LOGGER = root
    if name == "akkadian_nmt":
        return _LOGGER
    return _LOGGER.getChild(name.removeprefix("akkadian_nmt."))
