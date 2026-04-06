"""Unit tests for backend.chat.rag_engine (pgvector implementation).

These tests use unittest.mock to mock the SQLAlchemy session and the
SentenceTransformer encoder — no database or network connections are made.

Tests:
    test_retrieve_docs_returns_list_of_tuples     — happy-path return type
    test_retrieve_docs_empty_when_no_results      — below-threshold / empty DB case
    test_audit_log_write_called_once              — guarantees audit row written per call
    test_no_chromadb_import                       — regression: chromadb must not be imported
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_row(doc_id: int, content: str, similarity: float) -> Any:
    """Return a MagicMock that quacks like a SQLAlchemy Row."""
    row = MagicMock()
    row.id = doc_id
    row.content = content
    row.similarity = similarity
    return row


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetrieveDocsReturnType:
    """Happy-path: retrieve_docs returns a list of (int, str, float) tuples."""

    def test_retrieve_docs_returns_list_of_tuples(self) -> None:
        """Mocked DB returns two rows; function must return matching typed tuples."""
        mock_row_1 = _make_mock_row(doc_id=1, content="Farmsize description", similarity=0.91)
        mock_row_2 = _make_mock_row(doc_id=7, content="Net income description", similarity=0.83)

        mock_execute_result = MagicMock()
        mock_execute_result.fetchall.return_value = [mock_row_1, mock_row_2]

        mock_session_instance = MagicMock()
        mock_session_instance.__enter__ = MagicMock(return_value=mock_session_instance)
        mock_session_instance.__exit__ = MagicMock(return_value=False)
        mock_session_instance.execute.return_value = mock_execute_result

        with (
            patch("backend.chat.rag_engine.Session", return_value=mock_session_instance),
            patch("backend.chat.rag_engine._embedder") as mock_embedder,
            patch("backend.chat.rag_engine.db_engine"),
        ):
            mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.0] * 384)

            from backend.chat.rag_engine import retrieve_docs

            result = retrieve_docs("Model predicted: Eligible\nSHAP contributions: {}")

        assert isinstance(result, list), "retrieve_docs must return a list"
        assert len(result) == 2, "Expected exactly 2 results"
        for item in result:
            assert isinstance(item, tuple), f"Each result must be a tuple, got {type(item)}"
            assert len(item) == 3, f"Each tuple must have 3 elements, got {len(item)}"
            doc_id, content, similarity = item
            assert isinstance(doc_id, int), f"doc_id must be int, got {type(doc_id)}"
            assert isinstance(content, str), f"content must be str, got {type(content)}"
            assert isinstance(similarity, float), f"similarity must be float, got {type(similarity)}"

        assert result[0] == (1, "Farmsize description", 0.91)
        assert result[1] == (7, "Net income description", 0.83)


class TestRetrieveDocsEmptyResults:
    """Below-threshold case: retrieve_docs must return [] without raising."""

    def test_retrieve_docs_empty_when_no_results(self) -> None:
        """DB returns no rows; function must return [] and not raise any exception."""
        mock_execute_result = MagicMock()
        mock_execute_result.fetchall.return_value = []

        mock_session_instance = MagicMock()
        mock_session_instance.__enter__ = MagicMock(return_value=mock_session_instance)
        mock_session_instance.__exit__ = MagicMock(return_value=False)
        mock_session_instance.execute.return_value = mock_execute_result

        with (
            patch("backend.chat.rag_engine.Session", return_value=mock_session_instance),
            patch("backend.chat.rag_engine._embedder") as mock_embedder,
            patch("backend.chat.rag_engine.db_engine"),
        ):
            mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.0] * 384)

            from backend.chat.rag_engine import retrieve_docs

            result = retrieve_docs("xyzzy nonsense query that should not match anything")

        assert result == [], f"Expected empty list, got {result!r}"


class TestAuditLogWrittenOnce:
    """Audit log: retrieve_docs must write exactly one RagAuditLogDB row per call."""

    def test_audit_log_write_called_once(self) -> None:
        """Mock session; verify session.add() is called with a RagAuditLogDB instance exactly once."""
        from backend.services.db_model import RagAuditLogDB

        mock_row = _make_mock_row(doc_id=3, content="Land title explanation", similarity=0.88)
        mock_execute_result = MagicMock()
        mock_execute_result.fetchall.return_value = [mock_row]

        # We need two separate session context managers: one for retrieval, one for audit write.
        # Use side_effect to return a new mock for each Session() call.
        retrieval_session = MagicMock()
        retrieval_session.__enter__ = MagicMock(return_value=retrieval_session)
        retrieval_session.__exit__ = MagicMock(return_value=False)
        retrieval_session.execute.return_value = mock_execute_result

        audit_session = MagicMock()
        audit_session.__enter__ = MagicMock(return_value=audit_session)
        audit_session.__exit__ = MagicMock(return_value=False)

        session_calls = [retrieval_session, audit_session]

        with (
            patch("backend.chat.rag_engine.Session", side_effect=session_calls),
            patch("backend.chat.rag_engine._embedder") as mock_embedder,
            patch("backend.chat.rag_engine.db_engine"),
        ):
            mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.0] * 384)

            from backend.chat.rag_engine import retrieve_docs

            retrieve_docs(
                query="Model predicted: Review\nSHAP contributions: {}",
                prediction="Review",
                model_name="xgboost",
            )

        # Audit session must have had .add() called exactly once with a RagAuditLogDB instance
        audit_session.add.assert_called_once()
        added_arg = audit_session.add.call_args[0][0]
        assert isinstance(added_arg, RagAuditLogDB), (
            f"Expected RagAuditLogDB instance passed to session.add(), got {type(added_arg)}"
        )
        audit_session.commit.assert_called_once()


class TestNoChromadbImport:
    """Regression: the rag_engine module must not import chromadb at all."""

    def test_no_chromadb_import(self) -> None:
        """After importing rag_engine, 'chromadb' must not be in sys.modules."""
        # Reload the module to ensure we catch any top-level import
        import backend.chat.rag_engine as engine_module  # noqa: F401

        assert "chromadb" not in sys.modules, (
            "chromadb was imported by rag_engine — the ChromaDB dependency must be fully removed"
        )
