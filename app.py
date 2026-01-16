
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from infer import infer
from services.db_utils import fetch_rows, get_data_from_database
from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Credit Scoring Model API")


class UserData(BaseModel):
    source: str
    farmer_uid: Optional[str] = None
    number_of_rows: Optional[int] = None

@app.get("/")
def home():
    return {"message": "Credit Scoring Model API is running"}

@app.post("/predict")
async def submit_item(item: UserData):
    result_18, result_44, result_featured = infer(item.source, item.farmer_uid, item.number_of_rows)
    return {
            "result_18": result_18,
            "result_44": result_44,
            "result_featured": result_featured
        }