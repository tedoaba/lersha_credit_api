"""Prediction router — POST /v1/predict and GET /v1/predict/{job_id}.

Response field names are ``result_xgboost`` and ``result_random_forest``.
Legacy names (result_18, result_44, result_featured) must not appear.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from backend.api.dependencies import require_api_key
from backend.api.schemas import JobAcceptedResponse, JobStatusResponse, PredictRequest
from backend.config.config import config
from backend.core.pipeline import match_inputs, run_inferences
from backend.logger.logger import get_logger
from backend.services import db_utils

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_model=JobAcceptedResponse)
async def submit_prediction(
    item: PredictRequest,
    background_tasks: BackgroundTasks,
) -> JobAcceptedResponse:
    """Submit an inference job for asynchronous processing.

    Creates a job record in the ``inference_jobs`` table with status ``pending``
    and adds the heavy ML pipeline as a background task.

    Args:
        item: Validated prediction request (source, farmer_uid or number_of_rows).
        background_tasks: FastAPI BackgroundTasks scheduler.

    Returns:
        JobAcceptedResponse: Job ID and ``"accepted"`` status.
    """
    job_id = str(uuid.uuid4())
    db_utils.create_job(job_id)
    background_tasks.add_task(_run_prediction_background, job_id, item)
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


# ── Background task ─────────────────────────────────────────────────────────

async def _run_prediction_background(job_id: str, item: PredictRequest) -> None:
    """Execute the full ML pipeline in the background.

    Runs inference for both active models (xgboost, random_forest), assembles
    the result under ``result_xgboost`` and ``result_random_forest`` keys,
    and persists the outcome to the ``inference_jobs`` table.

    On any exception the job is marked ``failed`` and the error message saved.

    Args:
        job_id: UUID of the inference job to update.
        item: The original prediction request.
    """
    try:
        original_data, selected_data = match_inputs(
            source=item.source,
            filters=item.farmer_uid,
            number_of_rows=item.number_of_rows,
        )

        active_models: list[str] = config.hyperparams.get("models", {}).get("active", ["xgboost", "random_forest"])
        result: dict = {}

        for model_name in active_models:
            model_result = run_inferences(
                model_name=model_name,
                original_data=original_data,
                selected_data=selected_data,
                feature_column=config.feature_column_36,
                target_column=config.target_column_36,
            )
            result[f"result_{model_name}"] = model_result

        db_utils.update_job_result(job_id, result)
        logger.info("Job '%s' completed successfully", job_id)

    except Exception as exc:
        logger.error("Job '%s' failed: %s", job_id, exc, exc_info=True)
        db_utils.update_job_error(job_id, str(exc))
