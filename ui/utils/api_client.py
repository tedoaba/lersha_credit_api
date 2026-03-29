"""HTTP API client for the Lersha UI layer.

All Streamlit pages communicate with the backend exclusively via this
client. Direct imports from backend.* are strictly forbidden in ui/.

Usage:
    from ui.utils.api_client import LershaAPIClient
    client = LershaAPIClient()
    response = client.submit_prediction(source="Batch Prediction", number_of_rows=5)
"""

from __future__ import annotations

import os
import time

import requests
from dotenv import load_dotenv

# Load .env so API_KEY is available when Streamlit is launched via `make ui`
# (the shell does not automatically export variables from .env)
load_dotenv()


class LershaAPIClient:
    """HTTP client for the Lersha Credit Scoring API v1.

    Reads configuration from environment variables:
      - ``API_BASE_URL``: Base URL of the backend (default: http://localhost:8000)
      - ``API_KEY``: X-API-Key header value

    All requests automatically include the X-API-Key header.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = (base_url or os.getenv("API_BASE_URL", "http://localhost:8000")).rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.timeout = timeout
        self._connect_timeout = 5  # seconds to establish a TCP connection
        self._read_timeout = 60  # seconds to wait for the server to respond
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": self.api_key, "Content-Type": "application/json"})

    def health(self) -> dict:
        """Check the backend health status.

        Returns:
            dict: ``{"status": "ok"|"degraded", "dependencies": {...}}``
        """
        resp = self._session.get(f"{self.base_url}/health", timeout=(self._connect_timeout, self._read_timeout))
        resp.raise_for_status()
        return resp.json()

    def submit_prediction(
        self,
        source: str,
        farmer_uid: str | None = None,
        number_of_rows: int | None = None,
    ) -> dict:
        """Submit an inference job.

        Args:
            source: ``"Single Value"`` or ``"Batch Prediction"``.
            farmer_uid: Required for Single Value requests.
            number_of_rows: Required for Batch Prediction requests.

        Returns:
            dict: ``{"job_id": str, "status": "accepted"}``
        """
        payload: dict = {"source": source}
        if farmer_uid is not None:
            payload["farmer_uid"] = farmer_uid
        if number_of_rows is not None:
            payload["number_of_rows"] = number_of_rows

        resp = self._session.post(
            f"{self.base_url}/v1/predict/", json=payload, timeout=(self._connect_timeout, self._read_timeout)
        )
        resp.raise_for_status()
        return resp.json()

    def get_prediction_result(self, job_id: str) -> dict:
        """Poll the status of an inference job.

        Args:
            job_id: UUID string returned by ``submit_prediction``.

        Returns:
            dict: Job status response with ``status``, ``result``, and ``error`` fields.
        """
        resp = self._session.get(
            f"{self.base_url}/v1/predict/{job_id}", timeout=(self._connect_timeout, self._read_timeout)
        )
        resp.raise_for_status()
        return resp.json()

    def poll_until_complete(self, job_id: str, poll_interval: float = 2.0, max_wait: float = 300.0) -> dict:
        """Poll ``get_prediction_result`` until the job completes or times out.

        Args:
            job_id: UUID string of the job to poll.
            poll_interval: Seconds to wait between polls (default 2).
            max_wait: Maximum seconds to wait before raising ``TimeoutError``.

        Returns:
            dict: Completed job response (status ``"completed"`` or ``"failed"``).

        Raises:
            TimeoutError: If the job has not completed within ``max_wait`` seconds.
        """
        elapsed = 0.0
        while elapsed < max_wait:
            result = self.get_prediction_result(job_id)
            if result["status"] in ("completed", "failed"):
                return result
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(f"Job '{job_id}' did not complete within {max_wait}s")

    def get_results(self, limit: int = 500, model_name: str | None = None) -> dict:
        """Fetch evaluation history from the results endpoint.

        Args:
            limit: Maximum records to return (1–1000).
            model_name: Optional filter by model name.

        Returns:
            dict: ``{"total": int, "records": [...]}``
        """
        params: dict = {"limit": limit}
        if model_name:
            params["model_name"] = model_name
        resp = self._session.get(
            f"{self.base_url}/v1/results/", params=params, timeout=(self._connect_timeout, self._read_timeout)
        )
        resp.raise_for_status()
        return resp.json()
