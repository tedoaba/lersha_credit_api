"""Results router — GET /v1/results."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import require_api_key
from backend.api.schemas import ResultsResponse
from backend.logger.logger import get_logger
from backend.services import db_utils

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/", response_model=ResultsResponse)
async def get_results(
    limit: int = Query(default=500, ge=1, le=1000),
    model_name: str | None = Query(default=None),
) -> ResultsResponse:
    """Retrieve paginated evaluation results from the candidate_result table.

    Args:
        limit: Maximum number of records to return (1–1000, default 500).
        model_name: Optional filter by model name (e.g. ``"xgboost"``).

    Returns:
        ResultsResponse: Total count and list of evaluation records.
    """
    records = db_utils.get_all_results(limit=limit, model_name=model_name)
    logger.info("Returning %d results (limit=%d, model=%s)", len(records), limit, model_name)
    return ResultsResponse(total=len(records), records=records)
