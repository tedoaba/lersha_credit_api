"""Integration tests for database utility functions in backend/services/db_utils.py.

All tests run against a real ``test_lersha`` PostgreSQL database using the
``test_db_engine`` session-scoped fixture which creates and tears down the
required tables around the test session.

The ``db_utils`` functions internally call ``db_engine()`` which reads
``config.db_uri`` — set to the ``test_lersha`` connection string via the
``DB_URI`` environment variable (guaranteed by ``test_db_engine`` precondition).

Fixtures
--------
test_db_engine  — session-scoped engine (from conftest.py)
sample_farmer_df — synthetic 5-row DataFrame (from conftest.py)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import text

# Patch DB_URI early so Config initialises with the test database,
# not the default None that would raise ValueError.
# This is already handled by the test_db_engine fixture precondition,
# but we guard here for test isolation.
_TEST_DB_URI = os.environ.get("DB_URI", "")

pytestmark = pytest.mark.skipif(
    not _TEST_DB_URI or "sqlite" in _TEST_DB_URI,
    reason="Integration tests require a real PostgreSQL DB_URI (not SQLite)",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_FARMER_TABLE = "candidate_raw_data_test"  # isolated table for fetch tests


def _seed_farmer_table(engine, n: int = 3) -> None:
    """Insert ``n`` synthetic raw farmer rows into a temporary farmer table."""
    with engine.connect() as conn:
        conn.execute(
            text(f"""
            CREATE TABLE IF NOT EXISTS {_FARMER_TABLE} (
                farmer_uid VARCHAR(100) PRIMARY KEY,
                first_name VARCHAR(100),
                age INTEGER,
                farmsizehectares FLOAT
            )
        """)
        )
        conn.execute(text(f"TRUNCATE {_FARMER_TABLE}"))
        for i in range(1, n + 1):
            conn.execute(
                text(f"""
                INSERT INTO {_FARMER_TABLE} (farmer_uid, first_name, age, farmsizehectares)
                VALUES (:uid, :name, :age, :farm)
            """),
                {"uid": f"F-{i:03d}", "name": f"Farmer{i}", "age": 25 + i, "farm": float(i)},
            )
        conn.commit()


# ── fetch_raw_data ────────────────────────────────────────────────────────────


def test_fetch_raw_data_returns_matching_row(test_db_engine) -> None:
    """fetch_raw_data must return exactly 1 row matching the supplied farmer_uid.

    Seeds 3 rows into a temporary farmer table, then queries for a specific
    UID and confirms only that single row is returned.
    """
    import importlib

    _seed_farmer_table(test_db_engine, n=3)

    # Import after env is set so config reads the correct DB_URI
    from backend.services.db_utils import fetch_raw_data  # noqa: PLC0415

    result = fetch_raw_data(_FARMER_TABLE, "F-002")

    assert len(result) == 1, f"Expected 1 row, got {len(result)}"
    assert result["farmer_uid"].iloc[0] == "F-002"


# ── fetch_multiple_raw_data ───────────────────────────────────────────────────


def test_fetch_multiple_raw_data_returns_n_rows(test_db_engine) -> None:
    """fetch_multiple_raw_data(n=3) must return exactly 3 rows."""
    _seed_farmer_table(test_db_engine, n=5)  # seed ≥ 3 rows

    from backend.services.db_utils import fetch_multiple_raw_data  # noqa: PLC0415

    result = fetch_multiple_raw_data(_FARMER_TABLE, n_rows=3)

    assert len(result) == 3, f"Expected exactly 3 rows, got {len(result)}"


# ── save_batch_evaluations ────────────────────────────────────────────────────


def test_save_batch_evaluations_inserts_correct_count(test_db_engine) -> None:
    """save_batch_evaluations must insert the correct number of rows.

    Constructs a 2-row input DataFrame and 2 evaluation result dicts, calls
    ``save_batch_evaluations``, then counts rows in ``candidate_result``.
    """
    from backend.services.db_utils import save_batch_evaluations  # noqa: PLC0415

    input_df = pd.DataFrame(
        {
            "farmer_uid": ["F-001", "F-002"],
            "first_name": ["Abebe", "Bekele"],
            "middle_name": ["Hailu", "Girma"],
            "last_name": ["Woldemariam", "Abebe"],
        }
    )

    evaluation_results = [
        {
            "predicted_class_name": "Eligible",
            "top_feature_contributions": [{"feature": "net_income", "shap_value": 0.42}],
            "rag_explanation": "Farmer is eligible based on income and experience.",
            "model_name": "xgboost",
        },
        {
            "predicted_class_name": "Review",
            "top_feature_contributions": [{"feature": "yield_per_hectare", "shap_value": -0.15}],
            "rag_explanation": "Yield is below average for the region.",
            "model_name": "random_forest",
        },
    ]

    # Clear previous test data
    with test_db_engine.connect() as conn:
        conn.execute(text("TRUNCATE candidate_result RESTART IDENTITY"))
        conn.commit()

    save_batch_evaluations(input_df, evaluation_results)

    with test_db_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM candidate_result")).scalar()

    assert count == 2, f"Expected 2 rows in candidate_result, got {count}"


# ── create_job + get_job round-trip ──────────────────────────────────────────


def test_create_and_get_job_round_trip(test_db_engine) -> None:
    """create_job followed by get_job must return status 'pending'."""
    from backend.services.db_utils import create_job, get_job  # noqa: PLC0415

    job_id = str(uuid.uuid4())
    create_job(job_id)

    result = get_job(job_id)

    assert result is not None, f"get_job returned None for job_id={job_id}"
    assert result["status"] == "pending", f"Expected status 'pending', got '{result['status']}'"


# ── update_job_result sets status = completed ─────────────────────────────────


def test_update_job_result_sets_completed(test_db_engine) -> None:
    """update_job_result must transition status to 'completed' and store result."""
    from backend.services.db_utils import (  # noqa: PLC0415
        create_job,
        get_job,
        update_job_result,
    )

    job_id = str(uuid.uuid4())
    create_job(job_id)

    result_payload = {"result_xgboost": {"score": 0.85, "class": "Eligible"}}
    update_job_result(job_id, result_payload)

    job = get_job(job_id)

    assert job is not None
    assert job["status"] == "completed", f"Expected status 'completed', got '{job['status']}'"
    assert job["result"] is not None, "Result payload should be populated after update"
