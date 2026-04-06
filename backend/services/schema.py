"""Pydantic schemas for the Lersha Credit Scoring backend.

These are the validation-layer models that sit between external data
(API input / database rows) and the application logic.
"""

from datetime import datetime

from pydantic import BaseModel


class FeatureContribution(BaseModel):
    """A single feature's SHAP contribution to a prediction."""

    feature: str
    value: float


class CreditScoringRecord(BaseModel):
    """Validated credit scoring evaluation record before ORM write."""

    farmer_uid: str
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None

    predicted_class_name: str
    top_feature_contributions: list[FeatureContribution]
    rag_explanation: str
    model_name: str
    class_probabilities: dict[str, float] | None = None
    confidence_score: float | None = None
    job_id: str | None = None

    timestamp: datetime
