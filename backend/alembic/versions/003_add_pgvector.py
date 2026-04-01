"""add_pgvector_rag_tables

Revision ID: 003
Revises: 002
Create Date: 2026-04-01

Enables the pgvector extension and creates two new tables:
  - rag_documents: semantic knowledge store with VECTOR(384) embedding column,
    an IVFFlat ANN index for cosine-distance retrieval, and a category index.
  - rag_audit_log: immutable record of every RAG retrieval event for
    compliance auditing and performance monitoring.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    """Enable pgvector extension and create RAG document + audit tables.

    Execution order is significant:
      1. Extension must exist before VECTOR columns can be created.
      2. Tables must exist before indexes can reference them.
      3. The IVFFlat index is created last (after the embedding column exists).

    NOTE: IVFFlat requires at least `lists` (100) rows to be effective.
    Run backend/scripts/populate_pgvector.py after migration before
    benchmarking retrieval latency.
    """
    # 1. Enable pgvector extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Create rag_documents table
    op.create_table(
        "rag_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "doc_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        # VECTOR(384) — requires pgvector extension enabled above.
        # Using server_default=None here; the column type is expressed as raw DDL
        # because SQLAlchemy's alembic autogenerate does not natively render Vector.
        sa.Column(
            "embedding",
            sa.Text(),  # placeholder type — overridden by ALTER below
            nullable=False,
        ),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default="{}"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=True, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=True, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_id"),
    )

    # Alter embedding column to the correct VECTOR(384) type now that the extension is active
    op.execute("ALTER TABLE rag_documents ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)")

    # 3. Create rag_audit_log table
    op.create_table(
        "rag_audit_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("retrieved_ids", postgresql.ARRAY(sa.Integer()), nullable=True),
        sa.Column("prediction", sa.String(length=100), nullable=True),
        sa.Column("model_name", sa.String(length=100), nullable=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("generated_text", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=True, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("id"),
    )

    # 4. IVFFlat ANN index for cosine-distance retrieval (requires pgvector >= 0.3.0)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding "
        "ON rag_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # 5. B-tree index for category equality filter (pre-filters before ANN scan)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_rag_documents_category ON rag_documents (category)"
    )


def downgrade() -> None:
    """Remove RAG indexes, tables, and optionally the vector extension.

    NOTE: The vector extension is NOT dropped here because other tables or
    pg_catalog objects may depend on it. Drop it manually if needed:
        DROP EXTENSION IF EXISTS vector CASCADE;
    """
    # Reverse order: indexes first, then tables, extension last (skipped)
    op.execute("DROP INDEX IF EXISTS idx_rag_documents_category")
    op.execute("DROP INDEX IF EXISTS idx_rag_documents_embedding")
    op.drop_table("rag_audit_log")
    op.drop_table("rag_documents")
    # Extension intentionally not dropped — see NOTE above.
