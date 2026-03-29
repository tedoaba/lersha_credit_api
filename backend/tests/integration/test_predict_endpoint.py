"""Integration tests for POST /v1/predict and GET /v1/predict/{job_id}.

These tests use an in-process ``httpx.AsyncClient`` backed by
``ASGITransport`` — no real HTTP server is needed.

``run_inference_task.delay`` is patched so that no real Celery/Redis
connection is needed.  The ``test_db_engine`` fixture provides a shared
SQLite in-memory engine (or PostgreSQL when ``DB_URI`` is set).

Fixtures
--------
api_client      — async HTTPX client (from conftest.py)
test_db_engine  — session-scoped shared engine (from conftest.py)

All tests are async and are collected automatically via
``asyncio_mode = "auto"`` in ``pyproject.toml``.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

VALID_KEY = "ci-test-key"
WRONG_KEY = "definitely-wrong-key"
PREDICT_BODY = {"source": "Batch Prediction", "number_of_rows": 2}
NONEXISTENT_JOB_ID = "00000000-0000-0000-0000-000000000000"


# ── Authentication tests ──────────────────────────────────────────────────────


async def test_no_api_key_returns_422(api_client) -> None:
    """POST /v1/predict with no X-API-Key header returns HTTP 422 (validation error).

    FastAPI treats the X-API-Key header as a required field (declared with
    ``Header(...)``).  A missing required header returns 422 Unprocessable Entity,
    not 403 — 403 is returned when the key is present but incorrect.
    """
    response = await api_client.post("/v1/predict/", json=PREDICT_BODY)
    assert response.status_code == 422


async def test_wrong_api_key_returns_403(api_client) -> None:
    """POST /v1/predict with an incorrect X-API-Key must return HTTP 403."""
    response = await api_client.post(
        "/v1/predict/",
        json=PREDICT_BODY,
        headers={"X-API-Key": WRONG_KEY},
    )
    assert response.status_code == 403


# ── Job submission tests ──────────────────────────────────────────────────────


async def test_valid_predict_returns_202_with_job_id(api_client, test_db_engine) -> None:
    """POST /v1/predict with correct key and body must return HTTP 202.

    The response body must contain a ``job_id`` field that is a valid UUID v4.
    ``run_inference_task.delay`` is patched to a no-op so no Redis connection
    is required.
    """
    with patch("backend.api.routers.predict.run_inference_task") as mock_task:
        mock_task.delay.return_value = None
        response = await api_client.post(
            "/v1/predict/",
            json=PREDICT_BODY,
            headers={"X-API-Key": VALID_KEY},
        )
    assert response.status_code == 202, f"Expected 202, got {response.status_code}: {response.text}"
    body = response.json()
    assert "job_id" in body, f"Expected 'job_id' in response body, got: {body}"
    # Validate it is a parsable UUID — raises ValueError if not
    uuid.UUID(body["job_id"])


async def test_get_job_status_returns_200(api_client, test_db_engine) -> None:
    """GET /v1/predict/{job_id} with a valid key must return HTTP 200.

    First creates a job via POST /v1/predict (with Celery mocked), then polls
    the status endpoint with the returned ``job_id``.
    """
    # Create the job (Celery mocked to avoid Redis)
    with patch("backend.api.routers.predict.run_inference_task") as mock_task:
        mock_task.delay.return_value = None
        post_response = await api_client.post(
            "/v1/predict/",
            json=PREDICT_BODY,
            headers={"X-API-Key": VALID_KEY},
        )
    assert post_response.status_code == 202, f"POST failed: {post_response.text}"
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
