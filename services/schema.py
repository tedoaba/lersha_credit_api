from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class FeatureContribution(BaseModel):
    feature: str
    value: float

class CreditScoringRecord(BaseModel):
    farmer_uid: str
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None

    predicted_class_name: str
    top_feature_contributions: List[FeatureContribution]
    rag_explanation: str
    model_name: str

    timestamp: datetime
