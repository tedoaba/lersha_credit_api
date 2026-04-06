"""Integration tests for POST /v1/explain.

Uses TestClient (synchronous) + real test_lersha PostgreSQL DB + mocked Gemini.
All Gemini calls are replaced with a fixed return value via mocker.patch so that
no real API cost or rate-limit risk is incurred.

Prerequisites (CI + local):
  - DB_URI pointing at a test_lersha PostgreSQL database
  - uv run alembic upgrade head applied before the test session
  - Redis available at REDIS_URL (or mocked per test)

Run:
    uv run pytest backend/tests/integration/test_explain_endpoint.py -v
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.main import create_app
from backend.services.db_model import CreditScoringRecordDB, InferenceJobDB, RagAuditLogDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXED_EXPLANATION = "The farmer's strong yield and repayment history supported the Eligible decision."
API_KEY = "ci-test-key"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(scope="module")
def client(test_db_engine):  # noqa: ANN001
    """Synchronous TestClient backed by the real test_lersha DB engine."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def mock_rag_gemini(mocker: Any) -> Any:
    """Patch RagService._call_gemini to return a fixed string."""
    return mocker.patch(
        "backend.chat.rag_service.RagService._call_gemini",
        return_value=FIXED_EXPLANATION,
    )


@pytest.fixture()
def mock_redis_miss(mocker: Any) -> Any:
    """Force cache miss by patching Redis.get to return None."""
    redis_mock = mocker.patch("backend.chat.rag_service.redis.Redis.from_url")
    instance = redis_mock.return_value
    instance.get.return_value = None
    instance.set.return_value = True
    return instance


@pytest.fixture()
def seeded_job(test_db_engine) -> dict[str, Any]:
    """Insert a completed inference_jobs row + one candidate_result row and return their IDs."""
    job_id = str(uuid.uuid4())
    farmer_uid = f"ETH-TEST-{job_id[:8]}"

    with Session(test_db_engine) as session:
        job = InferenceJobDB(
            job_id=job_id,
            status="completed",
            result={"result_xgboost": {"evaluations": [{"predicted_class_name": "Eligible"}]}},
            created_at=__import__("datetime").datetime.utcnow(),
            completed_at=__import__("datetime").datetime.utcnow(),
        )
        session.add(job)

        record = CreditScoringRecordDB(
            farmer_uid=farmer_uid,
            first_name="Test",
            middle_name=None,
            last_name="Farmer",
            predicted_class_name="Eligible",
            top_feature_contributions=[
                {"feature": "yield_per_hectare", "value": 0.45},
                {"feature": "net_income", "value": 0.31},
            ],
            rag_explanation="placeholder",
            model_name="xgboost",
            timestamp=__import__("datetime").datetime.utcnow(),
        )
        session.add(record)
        session.commit()
        record_id = record.id

    yield {"job_id": job_id, "farmer_uid": farmer_uid, "record_index": 0, "record_id": record_id}

    # Teardown: remove inserted rows
    with Session(test_db_engine) as session:
        session.query(CreditScoringRecordDB).filter_by(id=record_id).delete()
        session.query(InferenceJobDB).filter_by(job_id=job_id).delete()
        session.commit()


# ---------------------------------------------------------------------------
# T009 / T019 — integration tests
# ---------------------------------------------------------------------------


def test_explain_returns_200_with_explanation(client, seeded_job, mock_rag_gemini, mock_redis_miss):
    """POST /v1/explain returns 200 with a non-empty explanation for a valid job."""
    payload = {
        "job_id": seeded_job["job_id"],
        "record_index": seeded_job["record_index"],
        "model_name": "xgboost",
    }
    response = client.post("/v1/explain/", json=payload, headers=HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] != ""
    assert body["farmer_uid"] == seeded_job["farmer_uid"]
    assert body["prediction"] == "Eligible"
    assert isinstance(body["retrieved_doc_ids"], list)
    assert isinstance(body["cache_hit"], bool)
    assert body["prompt_version"] == "v1"
    assert body["latency_ms"] >= 0


def test_explain_cache_hit_on_repeated_call(client, seeded_job, mock_rag_gemini, mocker):
    """Second call with identical inputs returns cache_hit=True and skips Gemini."""
    redis_mock = mocker.patch("backend.chat.rag_service.redis.Redis.from_url")
    instance = redis_mock.return_value

    call_count = 0

    def side_effect_get(key):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None  # first call: cache miss
        return FIXED_EXPLANATION.encode()  # second call: cache hit

    instance.get.side_effect = side_effect_get
    instance.set.return_value = True

    payload = {
        "job_id": seeded_job["job_id"],
        "record_index": seeded_job["record_index"],
        "model_name": "xgboost",
    }

    # First call — cache miss, Gemini called
    r1 = client.post("/v1/explain/", json=payload, headers=HEADERS)
    assert r1.status_code == 200
    assert r1.json()["cache_hit"] is False

    # Second call — cache hit
    r2 = client.post("/v1/explain/", json=payload, headers=HEADERS)
    assert r2.status_code == 200
    assert r2.json()["cache_hit"] is True
    assert r2.json()["explanation"] == FIXED_EXPLANATION

    # Gemini should only have been called once total
    assert mock_rag_gemini.call_count == 1


def test_explain_404_on_invalid_job_id(client):
    """POST /v1/explain returns 404 when job_id does not exist."""
    payload = {
        "job_id": str(uuid.uuid4()),
        "record_index": 0,
        "model_name": "xgboost",
    }
    response = client.post("/v1/explain/", json=payload, headers=HEADERS)
    assert response.status_code == 404


def test_explain_403_without_api_key(client, seeded_job):
    """POST /v1/explain returns 403 when X-API-Key header is absent."""
    payload = {
        "job_id": seeded_job["job_id"],
        "record_index": 0,
        "model_name": "xgboost",
    }
    response = client.post("/v1/explain/", json=payload)  # no headers
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# T022 — prompt version in response (Phase 4)
# ---------------------------------------------------------------------------


def test_prompt_version_in_response(client, seeded_job, mock_rag_gemini, mock_redis_miss, mocker):
    """Response prompt_version reflects the active PROMPT_VERSION config."""
    from backend.config.config import config

    original = config.prompt_version
    config.prompt_version = "v1"  # ensure deterministic
    try:
        payload = {
            "job_id": seeded_job["job_id"],
            "record_index": seeded_job["record_index"],
            "model_name": "xgboost",
        }
        response = client.post("/v1/explain/", json=payload, headers=HEADERS)
        assert response.status_code == 200
        assert response.json()["prompt_version"] == "v1"
    finally:
        config.prompt_version = original


# ---------------------------------------------------------------------------
# T023 / T024 — audit log presence (Phase 5)
# ---------------------------------------------------------------------------


def test_audit_log_entry_created_after_explain(client, seeded_job, mock_rag_gemini, mock_redis_miss, test_db_engine):
    """After a successful explain (cache miss), rag_audit_log contains one new row."""
    payload = {
        "job_id": seeded_job["job_id"],
        "record_index": seeded_job["record_index"],
        "model_name": "xgboost",
    }

    with Session(test_db_engine) as s:
        before = s.query(RagAuditLogDB).count()

    response = client.post("/v1/explain/", json=payload, headers=HEADERS)
    assert response.status_code == 200

    with Session(test_db_engine) as s:
        after = s.query(RagAuditLogDB).count()
        latest: RagAuditLogDB = s.query(RagAuditLogDB).order_by(RagAuditLogDB.id.desc()).first()

    assert after > before
    assert latest.cache_hit is False
    assert latest.prompt_version == "v1"
    assert latest.latency_ms is not None and latest.latency_ms >= 0


def test_audit_log_on_cache_hit(client, seeded_job, mock_rag_gemini, test_db_engine, mocker):
    """Two explain calls produce two audit rows; second has cache_hit=True."""
    redis_mock = mocker.patch("backend.chat.rag_service.redis.Redis.from_url")
    instance = redis_mock.return_value
    call_n = 0

    def get_side(key):
        nonlocal call_n
        call_n += 1
        return None if call_n == 1 else FIXED_EXPLANATION.encode()

    instance.get.side_effect = get_side
    instance.set.return_value = True

    payload = {
        "job_id": seeded_job["job_id"],
        "record_index": seeded_job["record_index"],
        "model_name": "xgboost",
    }

    with Session(test_db_engine) as s:
        before = s.query(RagAuditLogDB).count()

    client.post("/v1/explain/", json=payload, headers=HEADERS)
    client.post("/v1/explain/", json=payload, headers=HEADERS)

    with Session(test_db_engine) as s:
        after = s.query(RagAuditLogDB).count()
        rows = s.query(RagAuditLogDB).order_by(RagAuditLogDB.id.desc()).limit(2).all()

    assert after >= before + 2  # noqa: PLR2004
    cache_hits = [r.cache_hit for r in rows]
    assert True in cache_hits
