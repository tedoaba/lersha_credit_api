"""add_probability_columns

Revision ID: 007
Revises: 006
Create Date: 2026-04-06

Adds class_probabilities (JSONB) and confidence_score (Float) columns
to candidate_result for per-class probability tracking.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("candidate_result", sa.Column("class_probabilities", JSONB, nullable=True))
    op.add_column("candidate_result", sa.Column("confidence_score", sa.Float, nullable=True))


def downgrade() -> None:
    op.drop_column("candidate_result", "confidence_score")
    op.drop_column("candidate_result", "class_probabilities")
