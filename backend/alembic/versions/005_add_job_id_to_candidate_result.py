"""add_job_id_to_candidate_result

Revision ID: 005
Revises: 004
Create Date: 2026-04-06

Adds job_id column to candidate_result so results can be filtered
by the inference job that created them.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("candidate_result", sa.Column("job_id", sa.String(36), nullable=True))
    op.create_index("ix_candidate_result_job_id", "candidate_result", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_candidate_result_job_id", table_name="candidate_result")
    op.drop_column("candidate_result", "job_id")
