"""Celery application and inference task for the Lersha Credit Scoring backend.

This module defines the Celery app and the ``run_inference_task`` task that
executes the full ML pipeline asynchronously, decoupling the HTTP response
from inference computation.

Start the worker::

    uv run celery -A backend.worker worker --loglevel=info

Production (4 concurrent processes)::

    uv run celery -A backend.worker worker --loglevel=info --concurrency=4

Required environment variables:
  - REDIS_URL  : Celery broker and result backend (default: redis://redis:6379/0)
  - DB_URI     : PostgreSQL connection string
  - API_KEY    : Application API key (validated at config import)
  - GEMINI_API_KEY : Gemini API key
  - GEMINI_MODEL   : Gemini model identifier
"""

from __future__ import annotations

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

from backend.config.config import config  # noqa: E402
from backend.logger.logger import get_logger  # noqa: E402
from backend.services import db_utils  # noqa: E402

logger = get_logger(__name__)

# ── Celery application ────────────────────────────────────────────────────────

celery_app = Celery(
    "lersha",
    broker=config.redis_url,
    backend=config.redis_url,
)

# CELERY_TASK_ALWAYS_EAGER is checked in the predict router, not here.
# Celery always runs in normal (non-eager) mode so it can be used with Redis in prod.

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


# ── Inference task ────────────────────────────────────────────────────────────


@celery_app.task(name="run_inference_task")
def run_inference_task(job_id: str, payload: dict) -> None:
    """Execute the full ML pipeline as a Celery background task.

    Updates the job row in ``inference_jobs`` to reflect each stage of
    execution (processing → completed|failed).  On success, the result dict
    is persisted.  On any exception, the error message is stored and the job
    is marked ``failed`` — workers continue processing subsequent tasks.

    Imports are deferred into the function body so that the worker module can
    be imported by FastAPI without loading the full ML stack (which requires
    model files and GPU/CPU environment).

    Args:
        job_id: UUID string of the inference job to process (from create_job()).
        payload: Serialised ``PredictRequest`` dict with keys:
                 ``source``, ``farmer_uid`` (optional), ``number_of_rows`` (optional).
    """
    # Deferred imports — ML stack only loaded inside the actual Celery worker process
    from backend.core.pipeline import match_inputs, run_inferences  # noqa: PLC0415

    logger.info("Worker picked up job '%s' (source=%s)", job_id, payload.get("source"))

    try:
        db_utils.update_job_status(job_id, "processing")

        original_data, selected_data = match_inputs(
            source=payload["source"],
            filters=payload.get("farmer_uid"),
            number_of_rows=payload.get("number_of_rows"),
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

    except Exception as exc:  # noqa: BLE001
        logger.error("Job '%s' failed: %s", job_id, exc, exc_info=True)
        db_utils.update_job_error(job_id, str(exc))
