"""add_inference_jobs

Revision ID: 002
Revises: 001
Create Date: 2026-03-29

Creates the inference_jobs table for async job tracking (Celery-based inference).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    """Create the inference_jobs table."""
    op.create_table(
        "inference_jobs",
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("job_id"),
    )


def downgrade() -> None:
    """Drop the inference_jobs table."""
    op.drop_table("inference_jobs")
