"""add_audit_cache_fields

Revision ID: 004
Revises: 003
Create Date: 2026-04-02

Adds two columns to the ``rag_audit_log`` table to support the RagService
cache-hit tracking and prompt-version audit trail:

  - ``cache_hit``      BOOLEAN NOT NULL DEFAULT FALSE
      True when the explanation was served from the Redis cache without
      calling the Gemini API.

  - ``prompt_version`` VARCHAR(20) NULL
      The active prompt version (e.g. 'v1') at the time of generation.
      NULL for pure retrieval events (no explanation generated).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    """Add cache_hit and prompt_version columns to rag_audit_log.

    Both columns are additive — existing rows receive the default
    value (FALSE / NULL) without disrupting live data.
    """
    op.add_column(
        "rag_audit_log",
        sa.Column(
            "cache_hit",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.add_column(
        "rag_audit_log",
        sa.Column("prompt_version", sa.String(length=20), nullable=True),
    )


def downgrade() -> None:
    """Remove cache_hit and prompt_version columns from rag_audit_log."""
    op.drop_column("rag_audit_log", "prompt_version")
    op.drop_column("rag_audit_log", "cache_hit")
