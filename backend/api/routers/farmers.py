"""Farmers router — GET /v1/farmers/search."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import require_api_key
from backend.api.schemas import FarmerSearchResponse
from backend.logger.logger import get_logger
from backend.services import db_utils

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/search", response_model=FarmerSearchResponse)
async def search_farmers(
    q: str = Query(..., min_length=1, description="Search term"),
    limit: int = Query(default=10, ge=1, le=50),
) -> FarmerSearchResponse:
    """Search farmers by name or UID for autocomplete."""
    results = db_utils.search_farmers(query=q, limit=limit)
    logger.info("Farmer search q='%s' returned %d results", q, len(results))
    return FarmerSearchResponse(results=results)
