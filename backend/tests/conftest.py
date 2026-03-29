"""Pytest configuration and shared fixtures for backend tests.

Fixture overview
================
test_db_engine  (session)  — Real PostgreSQL engine for the ``test_lersha``
                              database.  Creates ``candidate_result`` and
                              ``inference_jobs`` tables before any test runs,
                              drops them after the session.

sample_farmer_df (function) — Synthetic 5-row DataFrame with all 29 raw
                              farmer schema columns; all field values are
                              realistic and safe for feature-engineering
                              transformations (farmsize > 0, binary flags ∈
                              {0,1}, etc.).

api_client       (function) — ``httpx.AsyncClient`` backed by
                              ``ASGITransport(app=create_app())``.  No real
                              HTTP server is started.  Required env vars are
                              set before the app is instantiated.

mock_gemini      (function) — Patches ``gemini_client.models.generate_content``
                              in ``backend.chat.rag_engine`` to return a fixed
                              explanation string.  Prevents any real API calls
                              in tests that trigger the RAG pipeline.
"""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ── test_db_engine ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_db_engine() -> Generator[Engine, None, None]:
    """Session-scoped SQLAlchemy engine connected to ``test_lersha`` PostgreSQL.

    Creates ``candidate_result`` and ``inference_jobs`` tables before any test
    runs.  Tears down by dropping both tables after the session, even if a test
    raises an exception.

    Requires the ``DB_URI`` environment variable to point at a reachable
    PostgreSQL 16 instance (e.g.
    ``postgresql://lersha:lersha@localhost:5432/test_lersha``).
    """
    db_uri = os.environ.get("DB_URI")
    if not db_uri:
        pytest.skip("DB_URI not set — skipping DB-dependent tests")

    engine = create_engine(db_uri)

    create_stmts = [
        """
        CREATE TABLE IF NOT EXISTS candidate_result (
            id SERIAL PRIMARY KEY,
            farmer_uid VARCHAR(100) NOT NULL,
            first_name VARCHAR(100),
            middle_name VARCHAR(100),
            last_name VARCHAR(100),
            predicted_class_name VARCHAR(100) NOT NULL,
            top_feature_contributions JSONB NOT NULL,
            rag_explanation TEXT NOT NULL,
            model_name TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS inference_jobs (
            job_id VARCHAR(36) PRIMARY KEY,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            result JSONB,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )
        """,
    ]

    with engine.connect() as conn:
        for stmt in create_stmts:
            conn.execute(text(stmt.strip()))
        conn.commit()

    try:
        yield engine
    finally:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS candidate_result CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS inference_jobs CASCADE"))
            conn.commit()
        engine.dispose()


# ── sample_farmer_df ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_farmer_df() -> pd.DataFrame:
    """Return a synthetic 5-row farmer DataFrame matching the raw schema.

    All numeric columns are non-null positives.  ``farmsizehectares > 0``
    ensures that ``yield_per_hectare`` and ``input_intensity`` computations
    in feature engineering are safe from division-by-zero.
    """
    return pd.DataFrame({
        "farmer_uid": ["F-001", "F-002", "F-003", "F-004", "F-005"],
        "first_name": ["Abebe", "Bekele", "Chaltu", "Dawit", "Eleni"],
        "middle_name": ["Hailu", "Girma", "Tigist", "Amare", "Lemma"],
        "last_name": ["Woldemariam", "Abebe", "Teshome", "Kebede", "Haile"],
        "gender": ["Male", "Male", "Female", "Male", "Female"],
        "age": [32, 45, 28, 38, 52],
        "family_size": [5, 7, 4, 6, 3],
        "estimated_income": [12000.0, 18000.0, 9000.0, 15000.0, 11000.0],
        "estimated_income_another_farm": [2000.0, 3000.0, 1000.0, 2500.0, 1500.0],
        "estimated_expenses": [4000.0, 6000.0, 3000.0, 5000.0, 3500.0],
        "estimated_cost": [2000.0, 3000.0, 1500.0, 2500.0, 1800.0],
        "agricultureexperience": [8, 15, 5, 12, 20],
        "hasmemberofmicrofinance": [1, 1, 0, 1, 0],
        "hascooperativeassociation": [0, 1, 1, 0, 1],
        "agriculturalcertificate": [1, 0, 0, 1, 1],
        "hascommunityhealthinsurance": [1, 1, 0, 1, 0],
        "farmsizehectares": [2.5, 4.0, 1.5, 3.0, 2.0],
        "expectedyieldquintals": [30.0, 50.0, 18.0, 40.0, 25.0],
        "seedquintals": [3.0, 5.0, 2.0, 4.0, 2.5],
        "ureafertilizerquintals": [1.5, 2.5, 1.0, 2.0, 1.2],
        "dapnpsfertilizerquintals": [1.0, 2.0, 0.5, 1.5, 0.8],
        "value_chain": ["maize", "wheat", "teff", "maize", "wheat"],
        "total_farmland_size": [3.0, 5.0, 2.0, 4.0, 2.5],
        "land_size": [2.5, 4.0, 1.5, 3.0, 2.0],
        "childrenunder12": [2, 3, 1, 2, 0],
        "elderlymembersover60": [1, 0, 0, 1, 1],
        "maincrops": ["maize", "wheat", "teff", "maize", "wheat"],
        "lastyearaverageprice": [500.0, 600.0, 450.0, 550.0, 580.0],
        "decision": ["Eligible", "Eligible", "Review", "Eligible", "Ineligible"],
    })


# ── api_client ───────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def api_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTPX client backed by ASGITransport — no real HTTP server.

    Sets required environment variables before importing ``create_app`` so the
    ``Config`` singleton can initialise without raising ``ValueError``.
    The ``DB_URI`` default points at an in-memory SQLite so unit tests that use
    ``api_client`` without ``test_db_engine`` do not require PostgreSQL.
    """
    os.environ.setdefault("API_KEY", "ci-test-key")
    os.environ.setdefault("GEMINI_MODEL", "gemini-1.5-pro")
    os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key-fixture")
    os.environ.setdefault("DB_URI", "sqlite:///:memory:")

    # Deferred import so env vars are set before Config instantiation.
    from backend.main import create_app  # noqa: PLC0415

    app = create_app()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ── mock_gemini ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_gemini(mocker: Any) -> Any:
    """Patch the Gemini API client so no real API calls are made.

    Patches ``backend.chat.rag_engine.gemini_client.models.generate_content``
    to return a ``MagicMock`` whose ``.text`` attribute is the fixed string
    ``"Fixed explanation for testing."``.  This makes any code path that calls
    ``get_rag_explanation()`` deterministic and free.
    """
    mock_response = mocker.MagicMock()
    mock_response.text = "Fixed explanation for testing."

    patched = mocker.patch(
        "backend.chat.rag_engine.gemini_client.models.generate_content",
        return_value=mock_response,
    )
    return patched
