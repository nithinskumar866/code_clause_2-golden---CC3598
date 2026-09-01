"""One configured logger for the whole service."""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _build() -> logging.Logger:
    log = logging.getLogger("company_intel")
    if log.handlers:  # uvicorn reloads import this module twice
        return log
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


logger = _build()
