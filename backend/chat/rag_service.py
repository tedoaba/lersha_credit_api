"""RagService — hardened RAG explanation service for the Lersha Credit API.

This module replaces the module-level functions in ``rag_engine.py`` for new
callers.  The legacy ``get_rag_explanation()`` function in ``rag_engine.py``
is preserved unchanged so existing pipeline code does not break.

Key capabilities
----------------
* ``RagService.retrieve()`` — pgvector cosine-distance retrieval with audit write.
* ``RagService.explain()``  — versioned-prompt + Redis-cached Gemini generation
  with audit write on every call (cache hit or miss).

Cache key
---------
``SHA-256(canonical_json({prediction, shap (sorted, 6 dp), version}))``
TTL: 86 400 s (24 hours).  On Redis failure the service degrades gracefully:
explanation is generated normally and a WARNING is logged; no 5xx is raised.

Prompt versioning
-----------------
Active prompt file: ``config.prompt_dir / f"{config.prompt_version}.yaml"``.
Switch version via the ``PROMPT_VERSION`` env var — no code change or
container rebuild required.

Implemented for feature 007-rag-service-hardening (2026-04-02).
[P1-MODULAR] [P5-CONFIG] [P6-API] [P9-SEC]
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import redis
import yaml  # type: ignore[import-untyped]
from google.genai import Client as GeminiClient
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config.config import config
from backend.logger.logger import get_logger
from backend.services.db_model import RagAuditLogDB
from backend.services.db_utils import db_engine

logger = get_logger(__name__)

# ── SQL template (shared with rag_engine.py for consistency) ──────────────────
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

_RAG_CATEGORIES: list[str] = ["feature_definition", "policy_rule"]

_CACHE_TTL_SECONDS: int = 86_400  # 24 hours


# ── Value objects ─────────────────────────────────────────────────────────────


@dataclass
class RetrievedDoc:
    """A single knowledge document returned by the pgvector retrieval query.

    Attributes:
        doc_id: The ``rag_documents.id`` integer primary key.
        content: Full document text used for prompt context assembly.
        similarity: Cosine similarity score in the range [0, 1].
    """

    doc_id: int
    content: str
    similarity: float


@dataclass
class ExplainResult:
    """The result of a RagService.explain() call.

    Attributes:
        farmer_uid: Unique farmer identifier.
        prediction: ML model prediction label (e.g. ``"Eligible"``).
        explanation: AI-generated natural-language explanation.
        retrieved_doc_ids: Ordered list of ``rag_documents.id`` values used.
        cache_hit: ``True`` when served from Redis without calling Gemini.
        prompt_version: Active prompt version key (e.g. ``"v1"``).
        latency_ms: Wall-clock latency from method entry to return in ms.
    """

    farmer_uid: str
    prediction: str
    explanation: str
    retrieved_doc_ids: list[int] = field(default_factory=list)
    cache_hit: bool = False
    prompt_version: str = "v1"
    latency_ms: int = 0


# ── Service ───────────────────────────────────────────────────────────────────


class RagService:
    """Retrieval-Augmented Generation service for credit-scoring explanations.

    Encapsulates all RAG retrieval, caching, prompt assembly, Gemini generation,
    and audit logging logic.  The HTTP transport layer never contains business
    logic — the explain router delegates entirely to this class.

    Dependency injection is supported for testability: pass a pre-built
    ``engine`` and ``redis_client`` to bypass real connections in unit tests.

    Args:
        engine: Optional SQLAlchemy engine.  Defaults to ``db_engine()``.
        redis_client: Optional Redis client.  Defaults to
            ``redis.Redis.from_url(config.redis_url)``.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        redis_client: redis.Redis | None = None,  # type: ignore[type-arg]
    ) -> None:
        """Initialise the service, lazily loading heavy dependencies."""
        self._config = config
        self._engine: Engine = engine or db_engine()
        self._redis: redis.Redis = redis_client or redis.Redis.from_url(  # type: ignore[type-arg]
            self._config.redis_url,
            decode_responses=False,
        )
        # Load embedder once per service instance — avoids ~30 s startup penalty
        # on the first request.  The model name is read from config.
        self._embedder = SentenceTransformer(self._config.embedder_model)
        self._gemini = GeminiClient(api_key=self._config.gemini_api_key)
        logger.debug(
            "RagService initialised (prompt_version=%s, embedder=%s)",
            self._config.prompt_version,
            self._config.embedder_model,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        prediction: str | None = None,
        model_name: str | None = None,
        job_id: str | None = None,
    ) -> list[RetrievedDoc]:
        """Retrieve top-k similar knowledge documents from pgvector.

        Encodes the query, executes a parameterised cosine-distance SQL query,
        and writes an audit row to ``rag_audit_log`` regardless of the result
        count.  The audit write failure is logged at ERROR but does not
        propagate — the retrieval result is always returned to the caller.

        Args:
            query: Natural-language query string.
            k: Maximum documents to return. Defaults to ``hyperparams.rag_top_k``.
            prediction: ML prediction label for the audit row.
            model_name: ML model name for the audit row.
            job_id: Inference job UUID string for the audit row.

        Returns:
            Ordered list of :class:`RetrievedDoc` instances (highest similarity
            first).  Empty list when no documents exceed the similarity threshold.

        Raises:
            SQLAlchemyError: Re-raised after logging if the retrieval query fails.
        """
        hp = self._config.hyperparams.get("inference", {})
        top_k = int(k or hp.get("rag_top_k", 5))
        threshold = float(hp.get("rag_similarity_threshold", 0.75))

        query_vec: list[float] = self._embedder.encode(query).tolist()

        t_start = time.perf_counter()
        docs: list[RetrievedDoc] = []

        try:
            with Session(self._engine) as session:
                rows = session.execute(
                    _RETRIEVAL_SQL,
                    {
                        "query_vec": str(query_vec),
                        "categories": _RAG_CATEGORIES,
                        "threshold": threshold,
                        "top_k": top_k,
                    },
                ).fetchall()
                docs = [RetrievedDoc(doc_id=row.id, content=row.content, similarity=float(row.similarity)) for row in rows]
        except SQLAlchemyError:
            logger.error("pgvector retrieval failed for query: %.80s", query, exc_info=True)
            raise

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

        if not docs:
            logger.warning("No RAG docs above threshold %.2f for query: %.80s", threshold, query)
        else:
            logger.debug("Retrieved %d docs in %d ms", len(docs), elapsed_ms)

        self._write_audit(
            query_text=query,
            retrieved_ids=[d.doc_id for d in docs],
            prediction=prediction,
            model_name=model_name,
            job_id=job_id,
            generated_text=None,
            latency_ms=elapsed_ms,
            cache_hit=False,
            prompt_version=None,
        )
        return docs

    def explain(
        self,
        prediction: str,
        shap_dict: dict,
        farmer_uid: str,
        job_id: str | None = None,
        model_name: str | None = None,
    ) -> ExplainResult:
        """Generate (or return cached) a natural-language credit explanation.

        Flow on **cache miss**: retrieve docs → assemble versioned prompt →
        call Gemini → store in Redis → write audit row (cache_hit=False).

        Flow on **cache hit**: return cached text → write audit row
        (cache_hit=True). Gemini is NOT called.

        Redis failures are caught and logged at WARNING level; the method
        proceeds as a cache miss so the caller always receives a result.

        Args:
            prediction: Predicted class label (e.g. ``"Eligible"``).
            shap_dict: Feature-importance dict ``{feature_name: shap_value}``.
            farmer_uid: Unique farmer identifier (used in prompt and audit).
            job_id: Inference job UUID string for the audit row.
            model_name: ML model name for the audit row.

        Returns:
            :class:`ExplainResult` with explanation text and metadata.
        """
        t_start = time.perf_counter()
        prompt_version = self._config.prompt_version
        cache_key = self._build_cache_key(prediction, shap_dict, prompt_version)

        # ── Cache lookup ──────────────────────────────────────────────────────
        cached_text: str | None = None
        try:
            raw = self._redis.get(cache_key)
            if raw is not None:
                cached_text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                logger.debug("Cache HIT for key %s…", cache_key[-12:])
        except redis.RedisError:
            logger.warning("Redis GET failed — proceeding without cache", exc_info=True)

        if cached_text is not None:
            elapsed_ms = int((time.perf_counter() - t_start) * 1000)
            self._write_audit(
                query_text=self._build_query(prediction, shap_dict),
                retrieved_ids=[],
                prediction=prediction,
                model_name=model_name,
                job_id=job_id,
                generated_text=cached_text,
                latency_ms=elapsed_ms,
                cache_hit=True,
                prompt_version=prompt_version,
            )
            return ExplainResult(
                farmer_uid=farmer_uid,
                prediction=prediction,
                explanation=cached_text,
                retrieved_doc_ids=[],
                cache_hit=True,
                prompt_version=prompt_version,
                latency_ms=elapsed_ms,
            )

        # ── Cache miss: retrieve → generate ───────────────────────────────────
        query = self._build_query(prediction, shap_dict)
        docs = self.retrieve(query, prediction=prediction, model_name=model_name, job_id=job_id)

        context = (
            "\n\n".join(f"[{i + 1}] {d.content}" for i, d in enumerate(docs))
            if docs
            else "No relevant knowledge documents found above the similarity threshold."
        )

        shap_json = json.dumps(
            {k: round(v, 6) for k, v in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]},
            separators=(",", ":"),
        )

        prompt_template = self._load_prompt()
        prompt = self._assemble_prompt(prompt_template, prediction, shap_json, context, farmer_uid)

        explanation = self._call_gemini(prompt)

        # ── Store in Redis ────────────────────────────────────────────────────
        try:
            self._redis.set(cache_key, explanation.encode("utf-8"), ex=_CACHE_TTL_SECONDS)
            logger.debug("Cached explanation under key %s…", cache_key[-12:])
        except redis.RedisError:
            logger.warning("Redis SET failed — explanation not cached", exc_info=True)

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

        self._write_audit(
            query_text=query,
            retrieved_ids=[d.doc_id for d in docs],
            prediction=prediction,
            model_name=model_name,
            job_id=job_id,
            generated_text=explanation,
            latency_ms=elapsed_ms,
            cache_hit=False,
            prompt_version=prompt_version,
        )

        return ExplainResult(
            farmer_uid=farmer_uid,
            prediction=prediction,
            explanation=explanation,
            retrieved_doc_ids=[d.doc_id for d in docs],
            cache_hit=False,
            prompt_version=prompt_version,
            latency_ms=elapsed_ms,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_query(self, prediction: str, shap_dict: dict) -> str:
        """Construct a deterministic retrieval query from prediction + top SHAP features.

        Limits to the top 10 features by absolute SHAP value to keep the
        embedding query focused on the most influential drivers.

        Args:
            prediction: Predicted class label.
            shap_dict: Full feature-importance dictionary.

        Returns:
            Query string for embedding and pgvector retrieval.
        """
        top_shap = {
            k: round(v, 6)
            for k, v in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        }
        shap_json = json.dumps(top_shap, separators=(",", ":"))
        return f"Model predicted: {prediction}\nTop features: {shap_json}"

    def _build_cache_key(self, prediction: str, shap_dict: dict, prompt_version: str) -> str:
        """Build a deterministic SHA-256 cache key.

        The canonical JSON uses sorted keys and 6 dp float rounding to ensure
        the key is identical for the same logical inputs across environments.

        Args:
            prediction: Predicted class label.
            shap_dict: Feature-importance dictionary (values rounded to 6 dp).
            prompt_version: Active prompt version string (e.g. ``"v1"``).

        Returns:
            Redis key string prefixed with ``"rag:explain:"``.
        """
        rounded_shap = {k: round(v, 6) for k, v in shap_dict.items()}
        canonical = json.dumps(
            {"prediction": prediction, "shap": rounded_shap, "version": prompt_version},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"rag:explain:{digest}"

    def _load_prompt(self) -> dict:
        """Load and parse the active versioned prompt YAML file.

        Resolves the path as ``config.prompt_dir / f"{config.prompt_version}.yaml"``.

        Returns:
            Parsed YAML dictionary with prompt template sections.

        Raises:
            FileNotFoundError: If the versioned YAML file does not exist,
                with a message that includes the version and expected path.
        """
        prompt_path: Path = self._config.prompt_dir / f"{self._config.prompt_version}.yaml"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt version '{self._config.prompt_version}' not found at {prompt_path}. "
                f"Create backend/prompts/{self._config.prompt_version}.yaml or set PROMPT_VERSION "
                "to an existing version."
            )
        with open(prompt_path, encoding="utf-8") as fh:
            data: dict = yaml.safe_load(fh) or {}
        logger.debug("Loaded prompt version '%s' from %s", self._config.prompt_version, prompt_path)
        return data

    def _assemble_prompt(
        self,
        template: dict,
        prediction: str,
        shap_json: str,
        retrieved_context: str,
        farmer_uid: str,
    ) -> str:
        """Assemble the full prompt string from a versioned template dict.

        Substitutes ``{prediction}``, ``{shap_json}``, ``{retrieved_context}``,
        and ``{farmer_uid}`` in the template sections.

        Args:
            template: Parsed prompt YAML dictionary.
            prediction: Predicted class label.
            shap_json: JSON-serialised top SHAP features.
            retrieved_context: Concatenated retrieved document texts.
            farmer_uid: Unique farmer identifier.

        Returns:
            Fully assembled prompt string ready for Gemini.
        """
        subs = {
            "prediction": prediction,
            "shap_json": shap_json,
            "retrieved_context": retrieved_context,
            "farmer_uid": farmer_uid,
        }
        parts = [
            f"SYSTEM:\n{template.get('system', '').format_map(subs)}",
            f"CONTEXT:\n{template.get('context_header', '').format_map(subs)}\n{retrieved_context}",
            f"TASK:\n{template.get('task', '').format_map(subs)}",
            f"INPUT:\n{template.get('input_template', '').format_map(subs)}",
            f"RESPONSE:\n{template.get('response_directive', '').format_map(subs)}",
        ]
        return "\n\n".join(parts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_gemini(self, prompt: str) -> str:
        """Call the Gemini API and extract the generated text.

        Retried up to 3 times with exponential back-off (2–10 s).  If all
        attempts fail the original exception is re-raised so the caller can
        convert it to a 503 response.

        Args:
            prompt: Fully assembled prompt string.

        Returns:
            Stripped explanation text from the model response.
        """
        model_id = self._config.gemini_model_id
        assert model_id is not None, "GEMINI_MODEL must be set in config"
        t0 = time.perf_counter()
        response = self._gemini.models.generate_content(
            model=model_id,
            contents=prompt,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.debug("Gemini call completed in %d ms", elapsed)

        if hasattr(response, "text") and response.text:
            return response.text.strip()
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) or []
            if parts:
                text_val = getattr(parts[0], "text", None)
                if text_val:
                    return str(text_val).strip()
        return str(response).strip()

    def _write_audit(
        self,
        *,
        query_text: str,
        retrieved_ids: list[int],
        prediction: str | None,
        model_name: str | None,
        job_id: str | None,
        generated_text: str | None,
        latency_ms: int,
        cache_hit: bool,
        prompt_version: str | None,
    ) -> None:
        """Insert an append-only audit row into ``rag_audit_log``.

        Failures are logged at ERROR and silently swallowed — the retrieval
        or explanation result is always returned to the caller regardless.

        Args:
            query_text: Raw query string used for retrieval.
            retrieved_ids: List of ``rag_documents.id`` values returned.
            prediction: ML prediction label (may be None for retrieval-only).
            model_name: ML model name (may be None).
            job_id: Inference job UUID string (may be None).
            generated_text: AI-generated explanation (None for retrieval-only).
            latency_ms: Wall-clock latency in milliseconds.
            cache_hit: Whether the result was served from Redis cache.
            prompt_version: Active prompt version (None for retrieval-only).
        """
        import datetime

        job_uuid = None
        if job_id:
            try:
                import uuid as _uuid

                job_uuid = _uuid.UUID(job_id)
            except ValueError:
                logger.warning("Invalid job_id UUID format: %s", job_id)

        try:
            with Session(self._engine) as session:
                row = RagAuditLogDB(
                    query_text=query_text,
                    retrieved_ids=retrieved_ids,
                    prediction=prediction,
                    model_name=model_name,
                    job_id=job_uuid,
                    generated_text=generated_text,
                    latency_ms=latency_ms,
                    cache_hit=cache_hit,
                    prompt_version=prompt_version,
                    created_at=datetime.datetime.now(datetime.UTC),
                )
                session.add(row)
                session.commit()
            logger.info(
                "RAG audit written: docs=%d cache_hit=%s latency=%d ms",
                len(retrieved_ids),
                cache_hit,
                latency_ms,
            )
        except SQLAlchemyError:
            logger.error("Failed to write RAG audit row", exc_info=True)
            # Do NOT re-raise — result already returned to caller.
