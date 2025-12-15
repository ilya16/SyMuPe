""" Console and File Logger. """
from __future__ import annotations

from loguru import logger


def setup_logger(console: bool = True, file: str | None = None, **extra):
    import sys

    time = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    level = "<level>{level:<7}</level>"
    process = "<level>{extra[process]}</level>"
    node_info = "<level>{extra[node_info]}</level>"
    module = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
    message = "<level>{message}</level>"

    # formatter = f"{time} {level} {module} - {message_level:6s} {message}"
    # formatter = f"{time} {level} - {message_level:6s} {message}"
    # formatter = f"{time} {level} {module} - {message}"
    formatter = f"{time} {level} - {message}"
    logger.remove()

    if console:
        logger.add(sys.stdout, format=formatter, enqueue=True)
    if file is not None:
        logger.add(file, format=formatter, enqueue=True)

    logger.configure(extra=extra)

    return logger
