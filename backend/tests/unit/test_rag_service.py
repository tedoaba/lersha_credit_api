"""Unit tests for backend.chat.rag_service.RagService.

All external I/O (PostgreSQL, Redis, LLM) is mocked — no real connections
are made. Tests follow the arrange / act / assert pattern and are named to
describe what they verify (not how).

Run:
    uv run pytest backend/tests/unit/test_rag_service.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.chat.rag_service import ExplainResult, RagService, RetrievedDoc

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_PREDICTION = "Eligible"
SAMPLE_SHAP: dict = {
    "yield_per_hectare": 0.45,
    "net_income": 0.31,
    "farmsizehectares": -0.12,
}
SAMPLE_FARMER_UID = "ETH-2024-001"
SAMPLE_EXPLANATION = "Fixed explanation for testing purposes."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_session_cls(mocker):
    """Return a patched sqlalchemy Session that yields a no-op session."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.execute.return_value.fetchall.return_value = [
        MagicMock(id=7, content="Feature: farm size.", similarity=0.82),
        MagicMock(id=12, content="Policy: income threshold.", similarity=0.79),
    ]
    mocker.patch("backend.chat.rag_service.Session", return_value=mock_session)
    return mock_session


@pytest.fixture()
def mock_redis(mocker):
    """Return a mocked redis.Redis instance with no stored keys."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None  # cache miss by default
    mocker.patch("backend.chat.rag_service.redis.Redis.from_url", return_value=redis_mock)
    return redis_mock


@pytest.fixture()
def mock_llm_response(mocker):
    """Patch _call_llm to return a fixed explanation string."""
    return mocker.patch(
        "backend.chat.rag_service.RagService._call_llm",
        return_value=SAMPLE_EXPLANATION,
    )


@pytest.fixture()
def mock_ollama_embed(mocker):
    """Patch the Ollama embed call."""
    import numpy as np

    mocker.patch(
        "backend.chat.rag_service.RagService._ollama_embed",
        return_value=np.zeros(1024).tolist(),
    )


@pytest.fixture()
def service(mock_session_cls, mock_redis, mock_ollama_embed, mocker):
    """Fully-mocked RagService instance."""
    mocker.patch("backend.chat.rag_service.db_engine", return_value=MagicMock())
    mocker.patch("backend.chat.rag_service.httpx.Client")
    svc = RagService()
    return svc


# ---------------------------------------------------------------------------
# T008 — retrieve tests
# ---------------------------------------------------------------------------


def test_retrieve_returns_docs_and_writes_audit(service, mock_session_cls):
    """retrieve() returns RetrievedDoc list and commits an audit row."""
    docs = service.retrieve("Model predicted: Eligible\nTop features: {}")

    assert len(docs) == 2  # noqa: PLR2004
    assert all(isinstance(d, RetrievedDoc) for d in docs)
    assert docs[0].doc_id == 7
    assert docs[0].similarity == pytest.approx(0.82)

    # Audit row must have been added and committed
    assert mock_session_cls.add.called
    assert mock_session_cls.commit.called


def test_retrieve_audit_row_has_no_cache_hit(service, mock_session_cls):
    """retrieve() writes an audit row with cache_hit=False and generated_text=None."""
    service.retrieve("some query")

    call_args = mock_session_cls.add.call_args[0][0]
    assert call_args.cache_hit is False
    assert call_args.generated_text is None


# ---------------------------------------------------------------------------
# T008 — explain tests
# ---------------------------------------------------------------------------


def test_explain_cache_miss_calls_gemini_and_caches(service, mock_redis, mock_llm_response):
    """On cache miss: LLM is called once and result is stored in Redis."""
    result = service.explain(SAMPLE_PREDICTION, SAMPLE_SHAP, SAMPLE_FARMER_UID)

    assert isinstance(result, ExplainResult)
    assert result.explanation == SAMPLE_EXPLANATION
    assert result.cache_hit is False
    assert result.prompt_version == "v1"

    mock_llm_response.assert_called_once()
    mock_redis.set.assert_called_once()  # stored in cache
    key_arg = mock_redis.set.call_args[0][0]
    assert key_arg.startswith("rag:explain:")


def test_explain_cache_hit_skips_gemini(service, mock_redis, mock_llm_response):
    """On cache hit: LLM is NOT called and cached text is returned."""
    mock_redis.get.return_value = SAMPLE_EXPLANATION.encode()

    result = service.explain(SAMPLE_PREDICTION, SAMPLE_SHAP, SAMPLE_FARMER_UID)

    assert result.explanation == SAMPLE_EXPLANATION
    assert result.cache_hit is True
    mock_llm_response.assert_not_called()


def test_explain_audit_log_written_on_cache_hit(service, mock_session_cls, mock_redis):
    """Even on cache hit an audit row is written with cache_hit=True."""
    mock_redis.get.return_value = b"Cached explanation."

    service.explain(SAMPLE_PREDICTION, SAMPLE_SHAP, SAMPLE_FARMER_UID)

    # Last add() call should be the explain audit row
    call_args = mock_session_cls.add.call_args[0][0]
    assert call_args.cache_hit is True
    assert call_args.generated_text == "Cached explanation."


# ---------------------------------------------------------------------------
# T008 — graceful Redis degradation
# ---------------------------------------------------------------------------


def test_explain_redis_error_degrades_gracefully(service, mock_redis, mock_llm_response, mocker):
    """When Redis raises RedisError, explain still returns a valid result (no 5xx)."""
    import redis

    mock_redis.get.side_effect = redis.RedisError("Connection refused")
    mock_redis.set.side_effect = redis.RedisError("Connection refused")

    # Should not raise — gracefully degrade
    result = service.explain(SAMPLE_PREDICTION, SAMPLE_SHAP, SAMPLE_FARMER_UID)

    assert result.explanation == SAMPLE_EXPLANATION
    assert result.cache_hit is False  # treated as miss
    mock_llm_response.assert_called_once()


# ---------------------------------------------------------------------------
# T020 — prompt versioning unit tests (Phase 4)
# ---------------------------------------------------------------------------


def test_load_prompt_raises_on_missing_version(service):
    """_load_prompt() raises FileNotFoundError with a descriptive message for unknown version."""
    original = service._config.prompt_version
    service._config.prompt_version = "v99"
    try:
        with pytest.raises(FileNotFoundError, match="v99"):
            service._load_prompt()
    finally:
        service._config.prompt_version = original


def test_cache_key_differs_across_prompt_versions(service):
    """_build_cache_key() produces distinct keys for different prompt versions."""
    key_v1 = service._build_cache_key(SAMPLE_PREDICTION, SAMPLE_SHAP, "v1")
    key_v2 = service._build_cache_key(SAMPLE_PREDICTION, SAMPLE_SHAP, "v2")
    assert key_v1 != key_v2
    assert key_v1.startswith("rag:explain:")
    assert key_v2.startswith("rag:explain:")


# ---------------------------------------------------------------------------
# T025 — retrieval-only audit (Phase 5)
# ---------------------------------------------------------------------------


def test_audit_log_retrieval_only_has_null_generated_text(service, mock_session_cls):
    """retrieve() audit row has generated_text=None (retrieval-only event)."""
    service.retrieve("some query", prediction="Eligible")

    audit_row = mock_session_cls.add.call_args[0][0]
    assert audit_row.generated_text is None
    assert audit_row.cache_hit is False
