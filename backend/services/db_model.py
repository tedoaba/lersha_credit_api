"""SQLAlchemy ORM models for the Lersha Credit Scoring backend.

Extended in 006-migrate-chroma-pgvector (2026-04-01):
  - RagDocumentDB: semantic knowledge document store (replaces ChromaDB).
  - RagAuditLogDB: immutable retrieval audit trail for compliance.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON as SA_JSON
from sqlalchemy import TIMESTAMP, Boolean, Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base

from backend.config.config import config

Base = declarative_base()


class CreditScoringRecordDB(Base):
    """ORM model for the candidate_result table."""

    __tablename__ = config.candidate_result

    id = Column(Integer, primary_key=True, autoincrement=True)

    farmer_uid = Column(String(100), nullable=False)
    first_name = Column(String(100), nullable=True)
    middle_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)

    predicted_class_name = Column(String(100), nullable=False)
    top_feature_contributions = Column(SA_JSON, nullable=False)
    rag_explanation = Column(Text, nullable=False)
    model_name = Column(Text, nullable=False)

    job_id = Column(String(36), nullable=True, index=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)


class InferenceJobDB(Base):
    """ORM model for the inference_jobs table (async job tracking)."""

    __tablename__ = "inference_jobs"

    job_id = Column(String(36), primary_key=True)  # UUID stored as string
    status = Column(String(20), nullable=False, default="pending")
    result = Column(SA_JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)


class RagDocumentDB(Base):
    """ORM model for the rag_documents table.

    Stores semantic knowledge documents (feature definitions, policy rules)
    with a 384-dimensional pgvector embedding for cosine-distance retrieval.
    Managed by Alembic migration 003_add_pgvector.

    Attributes:
        id: Auto-incrementing surrogate primary key.
        doc_id: Stable UUID identifier for upsert deduplication.
        category: Document classification (e.g. 'feature_definition', 'policy_rule').
        title: Human-readable document title.
        content: Full document text used for RAG context assembly.
        embedding: 384-dimensional sentence-transformer embedding vector.
        doc_metadata: Extensible JSONB bag for arbitrary document attributes.
        created_at: UTC timestamp of first ingestion.
        updated_at: UTC timestamp of last update (set on upsert).
    """

    __tablename__ = "rag_documents"

    id: Column = Column(Integer, primary_key=True, autoincrement=True)
    doc_id: Column = Column(PG_UUID(as_uuid=True), unique=True, nullable=False)
    category: Column = Column(String(100), nullable=False)
    title: Column = Column(String(255), nullable=False)
    content: Column = Column(Text, nullable=False)
    embedding: Column = Column(Vector(384), nullable=False)
    doc_metadata: Column = Column("metadata", JSONB, nullable=True, default={})
    created_at: Column = Column(TIMESTAMP(timezone=True), nullable=True)
    updated_at: Column = Column(TIMESTAMP(timezone=True), nullable=True)


class RagAuditLogDB(Base):
    """ORM model for the rag_audit_log table.

    Records every RAG retrieval event for compliance auditing and
    performance diagnostics. Written by RagService after each retrieval
    or explanation generation (cache hit or miss). Never updated — append-only.

    Attributes:
        id: Auto-incrementing surrogate primary key.
        query_text: The raw query string used for embedding and retrieval.
        retrieved_ids: Array of rag_documents.id values returned (may be empty).
        prediction: ML model prediction label (e.g. 'Eligible').
        model_name: Name of the ML model that produced the prediction.
        job_id: UUID linking this audit record to an inference_jobs row.
        generated_text: Gemini-generated explanation text (null on retrieval-only events).
        latency_ms: End-to-end latency in milliseconds.
        cache_hit: True when the explanation was served from Redis cache.
        prompt_version: Active prompt version used (e.g. 'v1'). Null on retrieval-only events.
        created_at: UTC timestamp of the event.
    """

    __tablename__ = "rag_audit_log"

    id: Column = Column(Integer, primary_key=True, autoincrement=True)
    query_text: Column = Column(Text, nullable=False)
    retrieved_ids: Column = Column(ARRAY(Integer), nullable=True)
    prediction: Column = Column(String(100), nullable=True)
    model_name: Column = Column(String(100), nullable=True)
    job_id: Column = Column(PG_UUID(as_uuid=True), nullable=True)
    generated_text: Column = Column(Text, nullable=True)
    latency_ms: Column = Column(Integer, nullable=True)
    cache_hit: Column = Column(Boolean, nullable=False, default=False)
    prompt_version: Column = Column(String(20), nullable=True)
    created_at: Column = Column(TIMESTAMP(timezone=True), nullable=True)
