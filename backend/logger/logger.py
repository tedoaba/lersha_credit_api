"""Logger module for the Lersha Credit Scoring backend.

Provides a production-ready rotating file + console logger.
Usage: from backend.logger.logger import get_logger
       logger = get_logger(__name__)
"""
import logging
import logging.handlers

from backend.config.config import config


def get_logger(
    name: str = __name__,
    log_file: str = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Return a production-ready logger instance.

    Args:
        name: Logger name, usually ``__name__`` from the calling module.
        log_file: Path to the log file. Defaults to ``config.log_file``.
        level: Logging level (e.g. ``logging.INFO``).
        max_bytes: Max size of a log file before rotation.
        backup_count: Number of backup log files to keep.

    Returns:
        logging.Logger: Configured logger instance with console + rotating file handlers.
    """
    if log_file is None:
        log_file = config.log_file

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — added only once
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Rotating file handler — added only once
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        fh = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
