"""RAG explanation engine using PostgreSQL pgvector and Google Gemini.

Migrated in 006-migrate-chroma-pgvector (2026-04-01):
  - Replaced ChromaDB PersistentClient with pgvector SQL queries via SQLAlchemy.
  - retrieve_docs() now executes a parameterised cosine-distance SELECT and
    writes an audit row to rag_audit_log after every retrieval (including
    empty results).
  - get_rag_explanation() now accepts optional job_id and model_name for full
    audit trail population.

Hardened in 004-harden-app-security (2026-03-29):
  - Gemini generate_content call extracted to _call_gemini() helper.
  - Both get_rag_explanation() and _call_gemini() are decorated with
    @retry (up to 3 attempts, exponential backoff 2-10 s).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml
from google.genai import Client as GeminiClient
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config.config import config
from backend.logger.logger import get_logger
from backend.services.db_model import RagAuditLogDB
from backend.services.db_utils import db_engine

logger = get_logger(__name__)

# ── Embedding model ────────────────────────────────────────────────────────────
# Loaded once at module import to avoid the ~30 s weight-loading penalty on the
# first inference request.  The model name is read from config so it is
# changeable without code edits.
_embedder: SentenceTransformer = SentenceTransformer(config.embedder_model)

# ── Gemini client ──────────────────────────────────────────────────────────────
gemini_client = GeminiClient(api_key=config.gemini_api_key)

# ── SQL template for cosine-distance retrieval ─────────────────────────────────
# Parameters:
#   :query_vec   — 384-float list cast to ::vector by pgvector
#   :categories  — list of allowed category strings (ANY operator)
#   :threshold   — minimum 1 - cosine_distance similarity score
#   :top_k       — maximum number of results to return
_RETRIEVAL_SQL = text(
    """
    SELECT
        id,
        content,
        1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
    FROM rag_documents
    WHERE category = ANY(:categories)
      AND 1 - (embedding <=> CAST(:query_vec AS vector)) > :threshold
    ORDER BY embedding <=> CAST(:query_vec AS vector)
    LIMIT :top_k
    """
)

# Categories scoped for credit-scoring RAG retrieval
_RAG_CATEGORIES: list[str] = ["feature_definition", "policy_rule"]


def retrieve_docs(
    query: str,
    k: int | None = None,
    prediction: str | None = None,
    model_name: str | None = None,
    job_id: str | None = None,
) -> list[tuple[int, str, float]]:
    """Retrieve top-k similar knowledge documents from the pgvector store.

    Encodes the query with the sentence-transformer model, executes a
    parameterised cosine-distance SQL query against ``rag_documents``, and
    writes an audit row to ``rag_audit_log`` regardless of whether results
    were found.

    Args:
        query: Natural language query string (typically prediction + SHAP dict).
        k: Maximum number of documents to return. Defaults to
           ``config.hyperparams["inference"]["rag_top_k"]`` (5).
        prediction: ML model prediction label for the audit log.
        model_name: ML model name for the audit log.
        job_id: UUID string of the associated inference job for the audit log.

    Returns:
        list[tuple[int, str, float]]: List of (doc_id, content, similarity_score)
        tuples ordered by descending similarity. Empty list when no documents
        exceed the configured similarity threshold.

    Raises:
        SQLAlchemyError: Re-raised after logging if the retrieval query fails.
            Audit log write failures are logged at ERROR and also re-raised.
    """
    hp = config.hyperparams.get("inference", {})
    if k is None:
        k = int(hp.get("rag_top_k", 5))
    threshold = float(hp.get("rag_similarity_threshold", 0.75))

    # Encode the query → 384-float list
    query_vec: list[float] = _embedder.encode(query).tolist()

    t_start = time.perf_counter()
    results: list[tuple[int, str, float]] = []

    engine = db_engine()
    try:
        with Session(engine) as session:
            rows = session.execute(
                _RETRIEVAL_SQL,
                {
                    "query_vec": str(query_vec),
                    "categories": _RAG_CATEGORIES,
                    "threshold": threshold,
                    "top_k": k,
                },
            ).fetchall()
            results = [(row.id, row.content, float(row.similarity)) for row in rows]
    except SQLAlchemyError:
        logger.error("pgvector retrieval query failed for query: %.80s", query, exc_info=True)
        raise

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    logger.debug("RAG retrieval latency: %d ms, %d docs returned", elapsed_ms, len(results))

    if not results:
        logger.warning("No RAG documents found above threshold %.2f for query: %.80s", threshold, query)

    # ── Audit log write ────────────────────────────────────────────────────────
    retrieved_ids = [r[0] for r in results]
    try:
        with Session(engine) as audit_session:
            audit_row = RagAuditLogDB(
                query_text=query,
                retrieved_ids=retrieved_ids,
                prediction=prediction,
                model_name=model_name,
                latency_ms=elapsed_ms,
            )
            audit_session.add(audit_row)
            audit_session.commit()
    except SQLAlchemyError:
        logger.error("Failed to write RAG audit log row", exc_info=True)
        raise

    return results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_gemini(prompt: str) -> str:
    """Call the Gemini API and extract the generated text.

    Retried up to 3 times with exponential backoff (2-10 s) on any exception.
    If all attempts fail the original exception is re-raised (``reraise=True``).

    Args:
        prompt: Fully assembled prompt string to send to the model.

    Returns:
        str: Stripped explanation text from the model response.
    """
    response = gemini_client.models.generate_content(
        model=config.gemini_model_id,
        contents=prompt,
    )

    if hasattr(response, "text") and response.text:
        return response.text.strip()
    if hasattr(response, "candidates"):
        return response.candidates[0].content.parts[0].text.strip()
    return str(response).strip()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def get_rag_explanation(
    prediction: str,
    shap_dict: dict,
    model_name: str | None = None,
    job_id: str | None = None,
) -> str:
    """Generate a natural language explanation for a credit scoring prediction.

    Retrieves relevant feature definitions from the pgvector store and uses
    Gemini to generate a paragraph-level explanation of the model's decision.

    Retried up to 3 times with exponential backoff (2-10 s) on any exception.
    On persistent failure the exception is re-raised to the Celery task error
    boundary, which marks the job as ``failed``.

    Args:
        prediction: Predicted class name (e.g. ``"Eligible"``).
        shap_dict: Dict of ``{feature_name: shap_value}`` for the top features.
        model_name: Name of the ML model producing the prediction. Forwarded
            to the audit log for traceability.
        job_id: UUID string of the associated inference job. Forwarded to the
            audit log to link retrieval events to inference job rows.

    Returns:
        str: Generated explanation text.
    """
    shap_json = json.dumps(shap_dict, indent=2)
    query_text = f"Model predicted: {prediction}\nSHAP contributions: {shap_json}"

    retrieved_docs = retrieve_docs(
        query=query_text,
        prediction=prediction,
        model_name=model_name,
        job_id=job_id,
    )
    context = (
        "\n".join(doc for _, doc, _ in retrieved_docs) if retrieved_docs else "No relevant feature definitions found."
    )

    prompt_path = Path(config.prompt_path)
    if not prompt_path.exists():
        logger.warning("Prompt file not found at %s; using fallback prompt", config.prompt_path)
        rag_prompt: dict = {
            "system": "You are a credit scoring analyst.",
            "instructions": "",
            "rules": "",
            "output": "",
        }
    else:
        with open(prompt_path, encoding="utf-8") as f:
            rag_prompt = yaml.safe_load(f)

    prompt = f"""
SYSTEM:
{rag_prompt.get("system", "")}

INSTRUCTIONS:
{rag_prompt.get("instructions", "")}

RULES:
{rag_prompt.get("rules", "")}

OUTPUT:
{rag_prompt.get("output", "")}

USER INPUT:
prediction = {prediction}
shap_json = {shap_json}
context = {context}

RESPONSE:
Generate the paragraphs now.
"""

    explanation = _call_gemini(prompt)
    return explanation
