"""Integration tests for the pgvector-backed RAG engine.

All tests run against the ``test_lersha`` PostgreSQL database (requires
migration 003 applied).  A clean fixture inserts synthetic documents before
each test function and removes them afterwards to guarantee isolation.

Tests:
    test_top3_similarity_above_threshold     — retrieved docs exceed similarity floor
    test_audit_log_populated_after_retrieval — retrieval writes an audit row
    test_audit_log_populated_on_empty_retrieval — empty result still writes audit
    test_latency_under_50ms                  — end-to-end retrieval within SLA
    test_alembic_upgrade_head_idempotent     — migration 003 is re-entrant safe

Prerequisites:
    - DB_URI env var pointing at test_lersha PostgreSQL (with pgvector extension)
    - Migration 003 applied: uv run alembic upgrade head
    - pgvector Python package: pgvector>=0.3.0
    - sentence-transformers model cached locally (all-MiniLM-L6-v2)
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime

import pytest
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.services.db_model import RagAuditLogDB, RagDocumentDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EMBEDDER = None  # Module-level cache to avoid reloading across tests


def _get_embedder() -> SentenceTransformer:
    """Load the sentence-transformer model once per test session."""
    global _EMBEDDER  # noqa: PLW0603
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _embed(text: str) -> list[float]:
    """Return a 384-float embedding for the given text string."""
    return _get_embedder().encode(text).tolist()


@pytest.fixture(scope="module")
def pg_engine():
    """Session-scoped engine for the test_lersha PostgreSQL database.

    Skips the entire module if DB_URI is not set or is SQLite (unit-test env).
    """
    db_uri = os.environ.get("DB_URI", "")
    if not db_uri or "postgresql" not in db_uri:
        pytest.skip("DB_URI not set to a PostgreSQL URI — skipping integration tests")
    engine = create_engine(db_uri)
    yield engine
    engine.dispose()


@pytest.fixture()
def seed_docs(pg_engine):
    """Insert 10 synthetic documents into rag_documents before each test.

    5 documents describe farm creditworthiness (feature_definition).
    5 documents describe eligibility policy rules (policy_rule).

    Removes all inserted rows after the test via DELETE on the known doc_ids.
    """
    now = datetime.utcnow()
    embedder = _get_embedder()

    feature_docs = [
        ("net_income_test", "Net income is total income minus total cost — the primary creditworthiness indicator."),
        ("farm_size_test", "Farm size in hectares directly influences yield capacity and loan repayment ability."),
        ("credit_score_test", "Credit score reflects the farmer's ability to repay loans based on income and assets."),
        ("yield_per_ha_test", "Yield per hectare measures productivity and is key to assessing loan eligibility."),
        ("asset_ownership_test", "Asset ownership includes land, equipment, and livestock that secure loan repayment."),
    ]
    policy_docs = [
        ("eligible_policy_test", "Farmers with net income above 10,000 ETB and land title are eligible for credit."),
        ("review_policy_test", "Farmers in the review category need additional income documentation before approval."),
        ("ineligible_policy_test", "Farmers with negative net income or high debt ratios are ineligible for credit."),
        ("cooperative_policy_test", "Membership in a RUSACCO cooperative is a positive signal for creditworthiness."),
        ("mechanized_policy_test", "Fully mechanized farms with verified yield are prioritised for credit approval."),
    ]

    all_docs = [(name, desc, "feature_definition") for name, desc in feature_docs] + [
        (name, desc, "policy_rule") for name, desc in policy_docs
    ]

    doc_ids = []
    rows = []
    for name, content, category in all_docs:
        doc_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"test.rag.{name}")
        doc_ids.append(doc_id)
        rows.append(
            RagDocumentDB(
                doc_id=doc_id,
                category=category,
                title=name.replace("_", " ").title(),
                content=content,
                embedding=embedder.encode(content).tolist(),
                created_at=now,
                updated_at=now,
            )
        )

    with Session(pg_engine) as session:
        session.add_all(rows)
        session.commit()

    yield doc_ids

    # Teardown: remove inserted rows and any audit rows that reference them
    with Session(pg_engine) as session:
        session.execute(
            text("DELETE FROM rag_documents WHERE doc_id = ANY(:ids)"),
            {"ids": [str(d) for d in doc_ids]},
        )
        session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTopKSimilarity:
    """Verify that retrieved documents exceed the similarity threshold."""

    @pytest.mark.usefixtures("seed_docs")
    def test_top3_similarity_above_threshold(self, pg_engine) -> None:
        """Top-3 documents for a creditworthiness query must all exceed 0.75 similarity."""
        from unittest.mock import patch

        from backend.chat.rag_engine import retrieve_docs

        with patch("backend.chat.rag_engine.db_engine", return_value=pg_engine):
            results = retrieve_docs(
                query="Model predicted: Eligible — net income positive, farm size large, land title held",
                k=3,
                prediction="Eligible",
            )

        assert len(results) >= 1, "Expected at least 1 document returned for a clear creditworthiness query"
        for doc_id, content, similarity in results[:3]:
            assert similarity > 0.75, (
                f"Similarity {similarity:.3f} is below threshold 0.75 for doc_id={doc_id}: '{content[:60]}...'"
            )


class TestAuditLogPopulatedAfterRetrieval:
    """Every successful retrieval must write exactly one audit log row."""

    @pytest.mark.usefixtures("seed_docs")
    def test_audit_log_populated_after_retrieval(self, pg_engine) -> None:
        """Call retrieve_docs() and assert one new rag_audit_log row is created."""
        from unittest.mock import patch

        from backend.chat.rag_engine import retrieve_docs

        # Count existing audit rows before the call
        with Session(pg_engine) as session:
            before_count = session.execute(text("SELECT count(*) FROM rag_audit_log")).scalar()

        with patch("backend.chat.rag_engine.db_engine", return_value=pg_engine):
            results = retrieve_docs(
                query="Farm assets and cooperative membership indicate strong creditworthiness",
                prediction="Eligible",
                model_name="xgboost",
            )

        with Session(pg_engine) as session:
            after_count = session.execute(text("SELECT count(*) FROM rag_audit_log")).scalar()
            latest = session.execute(
                text("SELECT * FROM rag_audit_log ORDER BY id DESC LIMIT 1")
            ).fetchone()

        assert after_count == before_count + 1, (
            f"Expected exactly 1 new audit row, got {after_count - before_count}"
        )
        assert latest is not None
        assert latest.latency_ms is not None and latest.latency_ms >= 0
        assert latest.prediction == "Eligible"
        assert latest.model_name == "xgboost"

        if results:
            assert latest.retrieved_ids is not None
            assert len(latest.retrieved_ids) > 0


class TestAuditLogOnEmptyRetrieval:
    """An empty retrieval (below-threshold query) must still produce an audit row."""

    @pytest.mark.usefixtures("seed_docs")
    def test_audit_log_populated_on_empty_retrieval(self, pg_engine) -> None:
        """Nonsense query → empty results → audit row still written with retrieved_ids=[]."""
        from unittest.mock import patch

        from backend.chat.rag_engine import retrieve_docs

        with Session(pg_engine) as session:
            before_count = session.execute(text("SELECT count(*) FROM rag_audit_log")).scalar()

        # Use threshold=0.9999 to guarantee no results exceed it
        with patch("backend.chat.rag_engine.db_engine", return_value=pg_engine):
            # Temporarily override threshold via config hyperparams patch
            with patch(
                "backend.chat.rag_engine.config.hyperparams",
                {"inference": {"rag_top_k": 5, "rag_similarity_threshold": 0.9999}},
            ):
                results = retrieve_docs(query="xyzzy frobnosticator lorem ipsum dolor sit amet")

        assert results == [], "Expected empty results when threshold is set to near-1.0"

        with Session(pg_engine) as session:
            after_count = session.execute(text("SELECT count(*) FROM rag_audit_log")).scalar()
            latest = session.execute(
                text("SELECT * FROM rag_audit_log ORDER BY id DESC LIMIT 1")
            ).fetchone()

        assert after_count == before_count + 1, "Audit row must be written even when retrieval returns empty results"
        assert latest is not None
        # retrieved_ids should be an empty list (not NULL) for empty retrievals
        assert latest.retrieved_ids is not None or latest.retrieved_ids == []


class TestLatencyBenchmark:
    """End-to-end retrieval must complete within the 50 ms p95 SLA (SC-002)."""

    @pytest.mark.usefixtures("seed_docs")
    def test_latency_under_50ms(self, pg_engine) -> None:
        """Measure wall-clock time of retrieve_docs(); assert < 50 ms."""
        from unittest.mock import patch

        from backend.chat.rag_engine import retrieve_docs

        with patch("backend.chat.rag_engine.db_engine", return_value=pg_engine):
            start = time.perf_counter()
            retrieve_docs(
                query="Eligible farmer with positive net income and land title",
                prediction="Eligible",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, (
            f"RAG retrieval latency {elapsed_ms:.1f} ms exceeds 50 ms SLA (SC-002). "
            "Check IVFFlat index presence and PostgreSQL connection pool settings."
        )


class TestAlembicMigrationIdempotent:
    """Running alembic upgrade head twice must not raise any exceptions (SC-007)."""

    def test_alembic_upgrade_head_idempotent(self, pg_engine) -> None:
        """Programmatically run alembic upgrade head twice; assert no exception raised."""
        import alembic.command
        import alembic.config

        # Locate alembic.ini — it lives at the project root
        alembic_cfg = alembic.config.Config(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend", "alembic.ini")
        )
        alembic_cfg.set_main_option("sqlalchemy.url", str(pg_engine.url))

        # First run (may already be at head — should be a no-op)
        try:
            alembic.command.upgrade(alembic_cfg, "head")
        except Exception as exc:
            pytest.fail(f"First alembic upgrade head raised an exception: {exc}")

        # Second run (must be idempotent)
        try:
            alembic.command.upgrade(alembic_cfg, "head")
        except Exception as exc:
            pytest.fail(f"Second alembic upgrade head raised an exception: {exc}")
