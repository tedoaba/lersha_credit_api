"""add_performance_indexes

Revision ID: 006
Revises: 005
Create Date: 2026-04-06

Adds indexes on frequently queried columns in candidate_result
to improve JOIN, filter, and sort performance at scale.
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_candidate_result_farmer_uid", "candidate_result", ["farmer_uid"])
    op.create_index("ix_candidate_result_timestamp", "candidate_result", ["timestamp"])
    op.create_index("ix_candidate_result_model_name", "candidate_result", ["model_name"])


def downgrade() -> None:
    op.drop_index("ix_candidate_result_model_name", table_name="candidate_result")
    op.drop_index("ix_candidate_result_timestamp", table_name="candidate_result")
    op.drop_index("ix_candidate_result_farmer_uid", table_name="candidate_result")
