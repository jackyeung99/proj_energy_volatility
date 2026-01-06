from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional


def setup_logging(
    level: str | int = "INFO",
    *,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure stdout logging (best for Docker/ECS -> CloudWatch).
    Idempotent: safe to call multiple times.
    """
    logger = logging.getLogger(name) if name else logging.getLogger()
    if logger.handlers:
        return logger  # already configured

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    datefmt = os.getenv("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    logger.addHandler(handler)
    logger.propagate = False

    # quieter third-party libs (optional)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3fs").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logger


class log_step:
    """
    Context manager to mark pipeline steps with timing + exceptions.
    Usage:
        with log_step(logger, "ingest"):
            ...
    """
    def __init__(self, logger: logging.Logger, step: str):
        self.logger = logger
        self.step = step
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info("STEP_START %s", self.step)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc is None:
            self.logger.info("STEP_OK %s (%.2fs)", self.step, dt)
            return False
        self.logger.exception("STEP_FAIL %s (%.2fs)", self.step, dt)
        return False  # re-raise
