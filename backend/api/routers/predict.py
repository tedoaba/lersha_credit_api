"""Prediction router — POST /v1/predict and GET /v1/predict/{job_id}.

Response field names are ``result_xgboost`` and ``result_random_forest``.
Legacy names (result_18, result_44, result_featured) must not appear.

Rate limiting: 10 requests per minute per IP (enforced by slowapi).
Async inference: jobs are dispatched to the Celery worker via Redis.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from backend.api.dependencies import limiter, require_api_key
from backend.api.schemas import JobAcceptedResponse, JobStatusResponse, PredictRequest
from backend.logger.logger import get_logger
from backend.services import db_utils
from backend.worker import run_inference_task

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_model=JobAcceptedResponse)
@limiter.limit("10/minute")
async def submit_prediction(
    request: Request,
    item: Annotated[PredictRequest, Body()],
) -> JobAcceptedResponse:
    """Submit an inference job for asynchronous processing via Celery.

    Creates a job record in the ``inference_jobs`` table with status ``pending``
    and dispatches the inference task to the Celery worker queue.

    Rate limited: 10 requests per minute per IP address.

    Args:
        request: FastAPI request object (required by slowapi rate limiter).
        item: Validated prediction request (source, farmer_uid or number_of_rows).

    Returns:
        JobAcceptedResponse: Job ID and ``"accepted"`` status.
    """
    job_id = str(uuid.uuid4())
    db_utils.create_job(job_id)
    run_inference_task.delay(job_id, item.dict())
    logger.info("Inference job '%s' accepted (source=%s)", job_id, item.source)
    return JobAcceptedResponse(job_id=job_id)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_prediction_status(job_id: str) -> JobStatusResponse:
    """Poll the status of an inference job.

    Returns HTTP 202 if the job is still pending/processing,
    HTTP 200 if completed or failed, HTTP 404 if the job_id is unknown.

    Args:
        job_id: UUID string returned by POST /v1/predict.

    Returns:
        JobStatusResponse: Current job status and result (if complete).
    """
    job = db_utils.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )
