"""Console and File Logger."""

from __future__ import annotations

from loguru import logger


def setup_logger(console: bool = True, file: str | None = None, **extra):
    import sys

    time = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    level = "<level>{level:<7}</level>"
    message = "<level>{message}</level>"

    formatter = f"{time} {level} - {message}"
    logger.remove()

    if console:
        logger.add(sys.stdout, format=formatter, enqueue=True)
    if file is not None:
        logger.add(file, format=formatter, enqueue=True)

    logger.configure(extra=extra)

    return logger
