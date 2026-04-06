"""Results router — GET /v1/results."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import require_api_key
from backend.api.schemas import PaginatedResultsResponse, ResultsResponse
from backend.logger.logger import get_logger
from backend.services import db_utils

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/", response_model=PaginatedResultsResponse | ResultsResponse)
async def get_results(
    limit: int = Query(default=500, ge=1, le=1000),
    model_name: str | None = Query(default=None),
    page: int | None = Query(default=None, ge=1),
    per_page: int | None = Query(default=None, ge=1, le=100),
    search: str | None = Query(default=None),
    decision: str | None = Query(default=None),
    gender: str | None = Query(default=None),
) -> PaginatedResultsResponse | ResultsResponse:
    """Retrieve evaluation results from the candidate_result table.

    Supports two modes:
    - Legacy: ``limit`` only (no ``page`` param) returns flat ResultsResponse.
    - Paginated: provide ``page`` to get PaginatedResultsResponse with filters.
    """
    if page is not None:
        result = db_utils.get_results_paginated(
            page=page,
            per_page=per_page or 20,
            search=search,
            decision=decision,
            gender=gender,
            model_name=model_name,
        )
        logger.info(
            "Returning page %d (%d results, total=%d)",
            result["page"],
            len(result["records"]),
            result["total"],
        )
        return PaginatedResultsResponse(**result)

    records = db_utils.get_all_results(limit=limit, model_name=model_name)
    logger.info("Returning %d results (limit=%d, model=%s)", len(records), limit, model_name)
    return ResultsResponse(total=len(records), records=records)
