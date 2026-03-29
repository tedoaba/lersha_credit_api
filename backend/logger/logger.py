"""Logger module for the Lersha Credit Scoring backend.

Provides a structured JSON logger for production use.
In containerised deployments log lines are captured by the container runtime
from stdout; no file rotation is needed.

Usage::

    from backend.logger.logger import get_logger
    logger = get_logger(__name__)
"""

import logging

from pythonjsonlogger.json import JsonFormatter


def get_logger(
    name: str = __name__,
    log_file: str | None = None,  # kept for API compatibility — unused in container mode
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # kept for API compatibility — unused
    backup_count: int = 5,  # kept for API compatibility — unused
) -> logging.Logger:
    """Return a production-ready JSON logger instance.

    All log output is written to stdout as newline-delimited JSON.  Every log
    line includes ``asctime``, ``levelname``, ``name``, and ``message`` fields
    so that log-aggregation systems (Datadog, CloudWatch, Loki) can parse
    individual fields without custom parsing rules.

    The ``log_file``, ``max_bytes``, and ``backup_count`` parameters are
    preserved for backward compatibility but are ignored — containers must not
    write logs to the container filesystem.

    Args:
        name: Logger name, usually ``__name__`` from the calling module.
        log_file: Ignored (kept for API compatibility).
        level: Logging level (e.g. ``logging.INFO``).
        max_bytes: Ignored (kept for API compatibility).
        backup_count: Ignored (kept for API compatibility).

    Returns:
        logging.Logger: Configured logger with a single JSON stdout handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Add handler only once (guards against repeated get_logger() calls)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
