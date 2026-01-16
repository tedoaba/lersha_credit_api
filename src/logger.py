import logging
import logging.handlers

from config.config import config


def get_logger(
    name: str = __name__,
    log_file: str = config.log_file,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Returns a production-ready logger instance.
    Args:
        name (str): Logger name, usually __name__ from the calling module.
        log_file (str): Path to the log file.
        level (int): Logging level, e.g., logging.INFO.
        max_bytes (int): Max size of a log file before rotation.
        backup_count (int): Number of backup files to keep.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler with rotation
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        fh = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
