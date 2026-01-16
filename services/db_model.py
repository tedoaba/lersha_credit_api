from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base

from config.config import config

Base = declarative_base()

class CreditScoringRecordDB(Base):
    __tablename__ = config.candidate_result

    id = Column(Integer, primary_key=True, autoincrement=True)

    farmer_uid = Column(String(100), nullable=False)
    first_name = Column(String(100), nullable=True)
    middle_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)

    predicted_class_name = Column(String(100), nullable=False)
    top_feature_contributions = Column(JSON, nullable=False)
    rag_explanation = Column(String, nullable=False)
    model_name = Column(String, nullable=False)

    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
