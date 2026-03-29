"""Integration tests for POST /v1/predict and GET /v1/predict/{job_id}.

These tests use an in-process ``httpx.AsyncClient`` backed by
``ASGITransport`` — no real HTTP server is needed.  They do require a live
``test_lersha`` PostgreSQL database because ``create_job()`` is called
synchronously inside the POST handler before the 202 response is returned.

Fixtures
--------
api_client      — async HTTPX client (from conftest.py)
test_db_engine  — session-scoped real PG engine (from conftest.py)

All tests are async and are collected automatically via
``asyncio_mode = "auto"`` in ``pyproject.toml``.
"""
from __future__ import annotations

import uuid

import pytest

VALID_KEY = "ci-test-key"
WRONG_KEY = "definitely-wrong-key"
PREDICT_BODY = {"source": "Batch Prediction", "number_of_rows": 2}
NONEXISTENT_JOB_ID = "00000000-0000-0000-0000-000000000000"


# ── Authentication tests ──────────────────────────────────────────────────────

async def test_no_api_key_returns_403(api_client) -> None:
    """POST /v1/predict with no X-API-Key header must return HTTP 403."""
    response = await api_client.post("/v1/predict", json=PREDICT_BODY)
    assert response.status_code == 403


async def test_wrong_api_key_returns_403(api_client) -> None:
    """POST /v1/predict with an incorrect X-API-Key must return HTTP 403."""
    response = await api_client.post(
        "/v1/predict",
        json=PREDICT_BODY,
        headers={"X-API-Key": WRONG_KEY},
    )
    assert response.status_code == 403


# ── Job submission tests ──────────────────────────────────────────────────────

async def test_valid_predict_returns_202_with_job_id(
    api_client, test_db_engine
) -> None:
    """POST /v1/predict with correct key and body must return HTTP 202.

    The response body must contain a ``job_id`` field that is a valid UUID v4.
    The ``test_db_engine`` fixture ensures the ``inference_jobs`` table exists
    so that ``create_job()`` inside the handler doesn't fail.
    """
    response = await api_client.post(
        "/v1/predict",
        json=PREDICT_BODY,
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body, f"Expected 'job_id' in response body, got: {body}"
    # Validate it is a parsable UUID — raises ValueError if not
    uuid.UUID(body["job_id"])


async def test_get_job_status_returns_200(api_client, test_db_engine) -> None:
    """GET /v1/predict/{job_id} with a valid key must return HTTP 200.

    First creates a job via POST /v1/predict, then polls the status endpoint
    with the returned ``job_id``.
    """
    # Create the job
    post_response = await api_client.post(
        "/v1/predict",
        json=PREDICT_BODY,
        headers={"X-API-Key": VALID_KEY},
    )
    assert post_response.status_code == 202
    job_id = post_response.json()["job_id"]

    # Poll for status
    get_response = await api_client.get(
        f"/v1/predict/{job_id}",
        headers={"X-API-Key": VALID_KEY},
    )
    assert get_response.status_code == 200
    body = get_response.json()
    assert "status" in body, f"Expected 'status' in body, got: {body}"


# ── 404 test ──────────────────────────────────────────────────────────────────

async def test_nonexistent_job_returns_404(api_client, test_db_engine) -> None:
    """GET /v1/predict/{job_id} for an unknown job_id must return HTTP 404."""
    response = await api_client.get(
        f"/v1/predict/{NONEXISTENT_JOB_ID}",
        headers={"X-API-Key": VALID_KEY},
    )
    assert response.status_code == 404
    body = response.json()
    assert "detail" in body
