"""Explain router — POST /v1/explain.

Provides a dedicated endpoint for requesting AI-generated explanations of
credit scoring predictions.  All business logic is delegated to
``RagService`` — this router only handles request validation, record lookup,
result mapping, and error translation.

Authentication: X-API-Key header required (same as all v1 routes).
[P6-API] [P9-SEC]
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.dependencies import require_api_key
from backend.api.schemas import ExplainRequest, ExplainResponse
from backend.chat.rag_service import RagService
from backend.logger.logger import get_logger
from backend.services import db_utils

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post(
    "/",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate AI explanation for a credit prediction",
    description=(
        "Retrieves relevant knowledge documents from the pgvector store, "
        "assembles a versioned prompt, and generates a natural-language "
        "explanation of the credit decision via Gemini. "
        "Responses are cached in Redis for 24 hours — identical inputs return "
        "instantly on subsequent calls."
    ),
)
def explain_prediction(item: ExplainRequest) -> ExplainResponse:
    """Generate (or return cached) an explanation for a scored farmer record.

    Looks up the prediction record by ``job_id`` and ``record_index`` in the
    ``candidate_result`` table, then calls ``RagService.explain()`` to produce
    a natural-language explanation backed by pgvector retrieval and Gemini.

    Args:
        item: Validated request body containing ``job_id``, ``record_index``,
            and ``model_name``.

    Returns:
        ExplainResponse: Explanation, retrieved document IDs, cache status,
            prompt version, and latency.

    Raises:
        HTTPException 404: If ``job_id`` is not found or ``record_index`` is
            out of range.
        HTTPException 503: If the retrieval query or Gemini generation fails
            after all retries.
    """
    t_start = time.perf_counter()

    # ── Fetch prediction record by job_id + record_index ─────────────────────
    record = _fetch_record(item.job_id, item.record_index, item.model_name)

    farmer_uid: str = record.get("farmer_uid", "unknown")
    prediction: str = record.get("predicted_class_name", "Unknown")
    shap_dict: dict = {
        entry["feature"]: entry["value"]
        for entry in (record.get("top_feature_contributions") or [])
    }

    # ── Delegate to RagService ────────────────────────────────────────────────
    try:
        svc = RagService()
        result = svc.explain(
            prediction=prediction,
            shap_dict=shap_dict,
            farmer_uid=farmer_uid,
            job_id=item.job_id,
            model_name=item.model_name,
        )
    except FileNotFoundError as exc:
        # Prompt version file missing — configuration error, not a transient fault.
        logger.error("Prompt version not found: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt configuration error: {exc}",
        ) from exc
    except Exception as exc:
        # Retrieval or Gemini failure after retries.
        logger.error("RagService.explain() failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explanation service temporarily unavailable. Retry after 10 seconds.",
            headers={"Retry-After": "10"},
        ) from exc

    total_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        "Explain request completed: farmer=%s cache_hit=%s latency=%d ms",
        farmer_uid,
        result.cache_hit,
        total_ms,
    )

    return ExplainResponse(
        farmer_uid=result.farmer_uid,
        prediction=result.prediction,
        explanation=result.explanation,
        retrieved_doc_ids=result.retrieved_doc_ids,
        cache_hit=result.cache_hit,
        prompt_version=result.prompt_version,
        latency_ms=result.latency_ms,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _fetch_record(job_id: str, record_index: int, model_name: str) -> dict:
    """Fetch a single prediction record from the candidate_result table.

    Looks up the completed job, extracts the evaluation list for the requested
    model, and returns the record at ``record_index``.

    Args:
        job_id: UUID string of the completed inference job.
        record_index: Zero-based index into the model's evaluation list.
        model_name: ML model key (e.g. ``"xgboost"``).

    Returns:
        dict with keys ``farmer_uid``, ``predicted_class_name``,
        ``top_feature_contributions``.

    Raises:
        HTTPException 404: Job not found, model key absent, or index out of range.
    """
    job = db_utils.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    result_payload: dict | None = job.get("result")
    if not result_payload:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' has no result yet (status: {job.get('status', 'unknown')}).",
        )

    # Try the model-specific key first (e.g. "result_xgboost"), then by index
    model_key = f"result_{model_name}"
    model_result: dict | None = result_payload.get(model_key)
    if model_result is None:
        # Fallback: first model result in the payload
        model_result = next(iter(result_payload.values()), None)

    if model_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No results found for model '{model_name}' in job '{job_id}'.",
        )

    evaluations: list = model_result.get("evaluations", [])
    if record_index >= len(evaluations):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"record_index {record_index} is out of range "
                f"(job '{job_id}' has {len(evaluations)} records for model '{model_name}')."
            ),
        )

    record = evaluations[record_index]

    # Try to enrich with farmer_uid from the raw data table if not in evaluation record
    farmer_uid = record.get("farmer_uid")
    if not farmer_uid:
        all_results = db_utils.get_all_results(limit=1000)
        farmer_uid = all_results[0].get("farmer_uid", "unknown") if all_results else "unknown"
        record = {**record, "farmer_uid": farmer_uid}

    return record
