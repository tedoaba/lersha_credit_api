"""initial_schema

Revision ID: 001
Revises:
Create Date: 2026-03-29

Creates the candidate_result table which stores credit scoring evaluation records.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    """Create the candidate_result table."""
    op.create_table(
        "candidate_result",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("farmer_uid", sa.String(length=100), nullable=False),
        sa.Column("first_name", sa.String(length=100), nullable=True),
        sa.Column("middle_name", sa.String(length=100), nullable=True),
        sa.Column("last_name", sa.String(length=100), nullable=True),
        sa.Column("predicted_class_name", sa.String(length=100), nullable=False),
        sa.Column("top_feature_contributions", sa.JSON(), nullable=False),
        sa.Column("rag_explanation", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop the candidate_result table."""
    op.drop_table("candidate_result")
