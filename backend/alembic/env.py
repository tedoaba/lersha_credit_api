"""Alembic environment configuration for the Lersha Credit Scoring backend.

Connects Alembic to the project's SQLAlchemy engine and ORM metadata so that
``alembic revision --autogenerate`` can derive migrations from the ORM models
without any hard-coded SQL.

Configuration:
  - ``sqlalchemy.url`` is injected at runtime from ``config.db_uri`` to avoid
    storing credentials in ``alembic.ini``.
  - ``target_metadata`` points at ``Base.metadata`` from ``backend.services.db_model``
    which contains all ORM table definitions.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Ensure the repo root is on the Python path ────────────────────────────────
# backend/alembic/env.py → backend/ → repo root (two levels up from this file)
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Load .env file before importing app config (which validates required vars)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_repo_root / ".env", override=False)

# ── Project imports ───────────────────────────────────────────────────────────
from backend.config.config import config as app_config  # noqa: E402
from backend.services.db_model import Base  # noqa: E402

# ── Alembic Config object (access to alembic.ini values) ─────────────────────
alembic_config = context.config

# Inject DB URI from the application config singleton — never from alembic.ini
alembic_config.set_main_option("sqlalchemy.url", app_config.db_uri)

# ── Logging setup (if a logging section exists in alembic.ini) ────────────────
if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

# ── Target metadata — points Alembic at all ORM tables ───────────────────────
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no live DB connection required).

    Generates SQL scripts that can be reviewed and applied manually.
    """
    url = alembic_config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connects to a live database).

    Uses the application's connection pool settings for consistency.
    """
    connectable = engine_from_config(
        alembic_config.get_section(alembic_config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # single-use connection appropriate for migrations
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
