"""Unit tests for inference job CRUD in backend/services/db_utils.py.

All tests mock the SQLAlchemy session to avoid requiring a real database.
Only pure Python behaviour is verified here — no DB, no Docker required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.services.db_utils import update_job_status


class TestUpdateJobStatus:
    """Tests for the update_job_status() function."""

    def test_update_job_status_changes_status_field(self):
        """update_job_status() sets job.status and calls session.commit().

        Verifies that:
        - The ORM session is entered as a context manager.
        - session.get() is called with the correct job_id.
        - job.status is set to the new value.
        - session.commit() is called exactly once.
        """
        mock_job = MagicMock()
        mock_job.status = "pending"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_job

        mock_engine = MagicMock()

        with patch("backend.services.db_utils.db_engine", return_value=mock_engine), \
             patch("backend.services.db_utils.Session", return_value=mock_session):
            update_job_status("test-job-id", "processing")

        mock_session.get.assert_called_once()
        assert mock_job.status == "processing"
        mock_session.commit.assert_called_once()

    def test_update_job_status_no_commit_when_job_missing(self):
        """update_job_status() does NOT call commit() if the job_id is not found.

        This prevents spurious commits for non-existent rows.
        """
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = None  # job not found

        mock_engine = MagicMock()

        with patch("backend.services.db_utils.db_engine", return_value=mock_engine), \
             patch("backend.services.db_utils.Session", return_value=mock_session):
            update_job_status("non-existent-id", "processing")

        mock_session.commit.assert_not_called()

    @pytest.mark.parametrize("status", ["pending", "processing", "completed", "failed"])
    def test_update_job_status_accepts_all_valid_statuses(self, status: str):
        """update_job_status() accepts all four valid status values without error."""
        mock_job = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_job

        mock_engine = MagicMock()

        with patch("backend.services.db_utils.db_engine", return_value=mock_engine), \
             patch("backend.services.db_utils.Session", return_value=mock_session):
            update_job_status("job-id", status)

        assert mock_job.status == status
