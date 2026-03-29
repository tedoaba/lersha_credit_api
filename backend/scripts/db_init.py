"""Database initialisation script.

Run once to create all required tables.

Usage:
    uv run python backend/scripts/db_init.py
"""
from sqlalchemy import create_engine, text

from backend.config.config import config
from backend.logger.logger import get_logger
from backend.services.db_utils import load_data_to_database

logger = get_logger(__name__)


def create_tables() -> None:
    """Create all application tables if they do not already exist."""
    engine = create_engine(config.db_uri)

    statements = [
        # Farmer raw data table (loaded from CSV)
        f"""
        CREATE TABLE IF NOT EXISTS {config.farmer_data_all or 'farmer_data_all'} (
            farmer_uid VARCHAR(100) PRIMARY KEY
        );
        """,
        # Candidate result table
        f"""
        CREATE TABLE IF NOT EXISTS {config.candidate_result} (
            id SERIAL PRIMARY KEY,
            farmer_uid VARCHAR(100) NOT NULL,
            first_name VARCHAR(100),
            middle_name VARCHAR(100),
            last_name VARCHAR(100),
            predicted_class_name VARCHAR(100) NOT NULL,
            top_feature_contributions JSONB NOT NULL,
            rag_explanation TEXT NOT NULL,
            model_name TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL
        );
        """,
        # Inference jobs table (async job tracking)
        """
        CREATE TABLE IF NOT EXISTS inference_jobs (
            job_id VARCHAR(36) PRIMARY KEY,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            result JSONB,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        );
        """,
    ]

    with engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt.strip()))
        conn.commit()

    logger.info("All tables created (or already exist)")


if __name__ == "__main__":
    logger.info("Initialising database schema...")
    create_tables()

    # Optionally load CSV data if CSV path is set
    if config.testing_csv_path and config.farmer_data_all:
        try:
            load_data_to_database(config.testing_csv_path, config.farmer_data_all)
        except Exception as exc:
            logger.warning("CSV preload skipped: %s", exc)

    logger.info("Database initialisation complete")
