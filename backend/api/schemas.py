"""Pydantic I/O schemas for the Lersha Credit Scoring API v1.

All API request and response models are defined here. Field names use
``result_xgboost`` and ``result_random_forest`` exclusively — the legacy
fields ``result_18``, ``result_44``, and ``result_featured`` must never appear.

Explain endpoint models (added in feature 007-rag-service-hardening):
  - ExplainRequest:  POST /v1/explain request body.
  - ExplainResponse: POST /v1/explain response body.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FeatureContribution(BaseModel):
    """A single feature's SHAP contribution."""

    feature: str
    value: float


class PredictRequest(BaseModel):
    """Request body for POST /v1/predict."""

    source: Literal["Single Value", "Batch Prediction"]
    farmer_uid: str | None = None
    number_of_rows: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_source_fields(self) -> PredictRequest:
        """Enforce cross-field requirements based on ``source``."""
        if self.source == "Single Value" and not self.farmer_uid:
            raise ValueError("farmer_uid is required for Single Value prediction")
        if self.source == "Batch Prediction" and not self.number_of_rows:
            raise ValueError("number_of_rows is required for Batch Prediction")
        return self


class JobAcceptedResponse(BaseModel):
    """Response body for 202 Accepted on POST /v1/predict."""

    job_id: str
    status: Literal["accepted"] = "accepted"


class EvaluationRecord(BaseModel):
    """A single farmer's inference result."""

    predicted_class_name: str
    top_feature_contributions: list[FeatureContribution]
    rag_explanation: str
    model_name: str


class ModelResult(BaseModel):
    """Inference results for a single model across all processed rows."""

    status: str
    records_processed: int
    evaluations: list[EvaluationRecord]


class JobStatusResponse(BaseModel):
    """Response body for GET /v1/predict/{job_id}."""

    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    result: dict[str, Any] | None = None  # keys: result_xgboost, result_random_forest
    error: str | None = None


class ResultsRecord(BaseModel):
    """A single row from the candidate_result table."""

    farmer_uid: str
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None
    predicted_class_name: str
    top_feature_contributions: list[dict]
    rag_explanation: str
    model_name: str
    timestamp: datetime | None = None


class ResultsResponse(BaseModel):
    """Response body for GET /v1/results."""

    total: int
    records: list[ResultsRecord]


# ── Explain endpoint ───────────────────────────────────────────────────────────


class ExplainRequest(BaseModel):
    """Request body for POST /v1/explain.

    Attributes:
        job_id: UUID string of a completed inference job whose records
            hold the prediction outcome and SHAP values.
        record_index: Zero-based index of the farmer record within the job's
            evaluation list.  Must be non-negative.
        model_name: Name of the ML model used for scoring
            (e.g. ``"xgboost"``, ``"random_forest"``).
    """

    job_id: str = Field(..., description="UUID of the completed inference job")
    record_index: int = Field(..., ge=0, description="0-based index of the farmer record in the job")
    model_name: str = Field(..., min_length=1, description="ML model name, e.g. 'xgboost'")

    @field_validator("job_id")
    @classmethod
    def validate_job_id_format(cls, v: str) -> str:
        """Validate that job_id is a non-empty string (UUID format recommended)."""
        if not v or not v.strip():
            raise ValueError("job_id must be a non-empty string")
        return v.strip()


class ExplainResponse(BaseModel):
    """Response body for POST /v1/explain.

    Attributes:
        farmer_uid: Unique farmer identifier from the prediction record.
        prediction: Predicted class label (e.g. ``"Eligible"``).
        explanation: AI-generated natural-language explanation (2–3 sentences).
        retrieved_doc_ids: Ordered list of rag_documents.id values that formed
            the retrieval context for this explanation.
        cache_hit: ``True`` when the explanation was served from the Redis cache
            without invoking the AI generation service.
        prompt_version: Version of the prompt template used (e.g. ``"v1"``).
        latency_ms: End-to-end latency from request receipt to response in ms.
    """

    farmer_uid: str
    prediction: str
    explanation: str
    retrieved_doc_ids: list[int]
    cache_hit: bool
    prompt_version: str
    latency_ms: int
