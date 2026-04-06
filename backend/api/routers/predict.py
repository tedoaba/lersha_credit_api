"""Prediction router — POST /v1/predict and GET /v1/predict/{job_id}.

Response field names are ``result_xgboost`` and ``result_random_forest``.
Legacy names (result_18, result_44, result_featured) must not appear.

Rate limiting: 10 requests per minute per IP (enforced by slowapi).
Async inference:
  - Production (Redis + Celery): dispatched via run_inference_task.delay()
  - Local dev (CELERY_TASK_ALWAYS_EAGER=true): dispatched via FastAPI
    BackgroundTasks so the endpoint always returns 202 immediately.
"""

import os
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from backend.api.dependencies import limiter, require_api_key
from backend.api.schemas import JobAcceptedResponse, JobStatusResponse, PredictRequest
from backend.logger.logger import get_logger
from backend.services import db_utils
from backend.worker import run_inference_task

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])

# When CELERY_TASK_ALWAYS_EAGER=true Celery executes tasks synchronously,
# which would block the HTTP request for the full inference duration.
# In that case we use FastAPI BackgroundTasks to run inference in a thread
# AFTER the 202 response has been sent — same UX as production Celery.
_EAGER_MODE = os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() in ("1", "true", "yes")


def _run_inference_background(job_id: str, payload: dict) -> None:
    """Trampoline that calls the Celery task function body directly.

    Used as a FastAPI BackgroundTask in local dev (eager mode).
    The task's ``.run()`` method bypasses Celery dispatch and executes
    the function synchronously in the background thread.
    """
    run_inference_task.run(job_id, payload)


@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_model=JobAcceptedResponse)
@limiter.limit("10/minute")
async def submit_prediction(
    request: Request,
    item: Annotated[PredictRequest, Body()],
    background_tasks: BackgroundTasks,
) -> JobAcceptedResponse:
    """Submit an inference job for asynchronous processing.

    Creates a job record in the ``inference_jobs`` table with status ``pending``
    and dispatches the inference task — via Celery (production) or FastAPI
    BackgroundTasks (local dev eager mode).  Always returns 202 immediately.

    Rate limited: 10 requests per minute per IP address.

    Args:
        request: FastAPI request object (required by slowapi rate limiter).
        item: Validated prediction request (source, farmer_uid or number_of_rows).
        background_tasks: FastAPI background task registry (used in eager mode).

    Returns:
        JobAcceptedResponse: Job ID and ``"accepted"`` status.
    """
    job_id = str(uuid.uuid4())
    db_utils.create_job(job_id)

    if _EAGER_MODE:
        # Dev mode: run inference in a background thread after 202 is returned.
        # This avoids blocking the HTTP request while inference runs (~30–120 s).
        background_tasks.add_task(_run_inference_background, job_id, item.dict())
        logger.info("Inference job '%s' queued via BackgroundTasks (eager/dev mode, source=%s)", job_id, item.source)
    else:
        # Production: dispatch to Celery worker via Redis broker.
        run_inference_task.delay(job_id, item.dict())
        logger.info("Inference job '%s' dispatched to Celery (source=%s)", job_id, item.source)

    return JobAcceptedResponse(job_id=job_id)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_prediction_status(job_id: str) -> JSONResponse:
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

    body = JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )
    http_status = status.HTTP_202_ACCEPTED if job["status"] in ("pending", "processing") else status.HTTP_200_OK
    return JSONResponse(content=body.model_dump(mode="json"), status_code=http_status)
