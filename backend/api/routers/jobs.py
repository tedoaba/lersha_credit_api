"""Jobs router — GET /v1/jobs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import require_api_key
from backend.api.schemas import JobsListResponse
from backend.logger.logger import get_logger
from backend.services import db_utils

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/", response_model=JobsListResponse)
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
) -> JobsListResponse:
    """List recent inference jobs ordered by creation time (newest first)."""
    jobs = db_utils.get_recent_jobs(limit=limit)
    logger.info("Returning %d jobs", len(jobs))
    return JobsListResponse(jobs=jobs)
