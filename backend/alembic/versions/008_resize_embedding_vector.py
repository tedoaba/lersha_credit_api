"""resize_embedding_vector_384_to_1024

Revision ID: 008
Revises: 007
Create Date: 2026-04-06

Resizes the rag_documents.embedding column from VECTOR(384) to VECTOR(1024)
to accommodate the Ollama mxbai-embed-large model output dimensions.
Drops and recreates the IVFFlat ANN index on the resized column.
Truncates existing rows since embeddings must be regenerated with the new model.
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # 1. Drop the existing IVFFlat index (references vector(384))
    op.drop_index("idx_rag_documents_embedding", table_name="rag_documents")

    # 2. Truncate existing rows — embeddings from the old model are incompatible
    op.execute("TRUNCATE TABLE rag_documents")

    # 3. Resize the embedding column from vector(384) to vector(1024)
    op.execute("ALTER TABLE rag_documents ALTER COLUMN embedding TYPE vector(1024) USING embedding::vector(1024)")

    # 4. Recreate the IVFFlat index for cosine-distance retrieval
    op.execute(
        "CREATE INDEX idx_rag_documents_embedding "
        "ON rag_documents USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )


def downgrade() -> None:
    op.drop_index("idx_rag_documents_embedding", table_name="rag_documents")
    op.execute("TRUNCATE TABLE rag_documents")
    op.execute("ALTER TABLE rag_documents ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)")
    op.execute(
        "CREATE INDEX idx_rag_documents_embedding "
        "ON rag_documents USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )
