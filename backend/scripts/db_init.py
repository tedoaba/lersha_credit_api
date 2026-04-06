"""Database data-loading script (CSV → PostgreSQL).

This script is data-only. All schema/DDL is managed exclusively by Alembic
migrations (``uv run alembic upgrade head``). Run this script AFTER migrations
have been applied to populate the tables with initial CSV data.

Usage::

    uv run python backend/scripts/db_init.py
"""

from backend.config.config import config
from backend.logger.logger import get_logger
from backend.services.db_utils import load_data_to_database

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting CSV data load into PostgreSQL...")

    # Load raw farmer data from CSV if both path and table are configured
    if config.csv_general and config.farmer_data_all:
        try:
            load_data_to_database(config.csv_general, config.farmer_data_all)
            logger.info("CSV data loaded successfully into '%s'", config.farmer_data_all)
        except Exception as exc:
            logger.warning("CSV preload skipped: %s", exc)
    else:
        logger.info("No CSV data path configured — skipping data load")

    logger.info("Data loading complete")
