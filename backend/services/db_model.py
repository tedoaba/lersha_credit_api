"""SQLAlchemy ORM models for the Lersha Credit Scoring backend."""
from sqlalchemy import JSON, TIMESTAMP, Column, Integer, String, Text
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
    top_feature_contributions = Column(JSON, nullable=False)
    rag_explanation = Column(Text, nullable=False)
    model_name = Column(Text, nullable=False)

    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)


class InferenceJobDB(Base):
    """ORM model for the inference_jobs table (async job tracking)."""

    __tablename__ = "inference_jobs"

    job_id = Column(String(36), primary_key=True)  # UUID stored as string
    status = Column(String(20), nullable=False, default="pending")
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
