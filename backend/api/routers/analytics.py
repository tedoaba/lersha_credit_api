"""Analytics router — GET /v1/analytics/summary."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import require_api_key
from backend.api.schemas import AnalyticsSummaryResponse
from backend.logger.logger import get_logger
from backend.services import db_utils

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary() -> AnalyticsSummaryResponse:
    """Aggregated dashboard analytics: counts by decision, gender breakdown, recent activity."""
    summary = db_utils.get_analytics_summary()
    logger.info(
        "Analytics summary: total=%d, decisions=%s",
        summary["total"],
        summary["by_decision"],
    )
    return AnalyticsSummaryResponse(**summary)
