"""RAG explanation engine using ChromaDB (persistent) and Google Gemini.

Fixed in monorepo refactor (2026-03-29):
  - Replaced ephemeral ``chromadb.Client()`` with ``chromadb.PersistentClient``
    so the vector store survives service restarts.
  - Re-enabled at the pipeline layer (previously commented out).

Hardened in 004-harden-app-security (2026-03-29):
  - Gemini generate_content call extracted to ``_call_gemini()`` helper.
  - Both ``get_rag_explanation()`` and ``_call_gemini()`` are decorated with
    ``@retry`` (up to 3 attempts, exponential backoff 2–10 s) so transient
    network or quota errors are handled automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import yaml
from chromadb.utils import embedding_functions
from google.genai import Client as GeminiClient
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config.config import config
from backend.logger.logger import get_logger

logger = get_logger(__name__)

# ── Embedding function ──────────────────────────────────────────────────────
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.embedder_model)

# ── Gemini client ─────────────────────────────────────────────────────────
gemini_client = GeminiClient(api_key=config.gemini_api_key)

# ── ChromaDB PersistentClient ─────────────────────────────────────────────
# FIXED: was chromadb.Client() (ephemeral). Now uses PersistentClient so
# the collection is retained across restarts.
_chroma_client = chromadb.PersistentClient(path=str(config.chroma_db_path))
collection = _chroma_client.get_or_create_collection(
    name="credit_features",
    embedding_function=embedder,
)


def retrieve_docs(query: str, k: int | None = None) -> list[str]:
    """Retrieve top-k similar feature definition documents from ChromaDB.

    Args:
        query: Natural language query string (typically the prediction + SHAP dict).
        k: Number of documents to retrieve. Defaults to ``config.hyperparams``
           inference.rag_top_k (5).

    Returns:
        list[str]: Retrieved document strings, or empty list if none found.
    """
    if k is None:
        k = config.hyperparams.get("inference", {}).get("rag_top_k", 5)

    results = collection.query(query_texts=[query], n_results=k)
    documents = results.get("documents", [[]])[0]

    if not documents:
        logger.warning("No RAG documents found for query: %.80s...", query)
        return []

    return [str(doc) for doc in documents]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_gemini(prompt: str) -> str:
    """Call the Gemini API and extract the generated text.

    Retried up to 3 times with exponential backoff (2–10 s) on any exception.
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
def get_rag_explanation(prediction: str, shap_dict: dict) -> str:
    """Generate a natural language explanation for a credit scoring prediction.

    Retrieves relevant feature definitions from ChromaDB and uses Gemini
    to generate a paragraph-level explanation of the model's decision.

    Retried up to 3 times with exponential backoff (2–10 s) on any exception.
    On persistent failure the exception is re-raised to the Celery task error
    boundary, which marks the job as ``failed``.

    Args:
        prediction: Predicted class name (e.g. ``"Eligible"``).
        shap_dict: Dict of ``{feature_name: shap_value}`` for the top features.

    Returns:
        str: Generated explanation text.
    """
    shap_json = json.dumps(shap_dict, indent=2)
    query_text = f"Model predicted: {prediction}\nSHAP contributions: {shap_json}"

    retrieved_docs = retrieve_docs(query_text)
    context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant feature definitions found."

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

    return _call_gemini(prompt)
