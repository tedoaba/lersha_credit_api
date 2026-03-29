"""Pydantic I/O schemas for the Lersha Credit Scoring API v1.

All API request and response models are defined here. Field names use
``result_xgboost`` and ``result_random_forest`` exclusively — the legacy
fields ``result_18``, ``result_44``, and ``result_featured`` must never appear.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
