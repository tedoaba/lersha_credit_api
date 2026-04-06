"""Locust load testing harness for the Lersha Credit Scoring API.

Run with::

    pip install locust
    locust -f backend/tests/load/locustfile.py --host http://localhost:8006

Then open http://localhost:8089 to configure and start the test.
"""

import os

from locust import HttpUser, between, task

API_KEY = os.getenv("API_KEY", "test-key")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


class CreditScoringUser(HttpUser):
    """Simulates a typical API consumer: submit predictions, poll results, query analytics."""

    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(3)
    def get_analytics(self):
        self.client.get("/v1/analytics/summary", headers=HEADERS)

    @task(5)
    def get_results_paginated(self):
        self.client.get("/v1/results/?page=1&per_page=20", headers=HEADERS)

    @task(2)
    def search_farmers(self):
        self.client.get("/v1/farmers/search?q=Ab&limit=10", headers=HEADERS)

    @task(1)
    def submit_batch_prediction(self):
        """Submit a small batch prediction (2 rows) and poll until complete."""
        with self.client.post(
            "/v1/predict/",
            json={"source": "Batch Prediction", "number_of_rows": 2},
            headers=HEADERS,
            catch_response=True,
        ) as resp:
            if resp.status_code == 202:
                resp.success()
                job_id = resp.json().get("job_id")
                if job_id:
                    # Poll once (don't block the load test waiting for completion)
                    self.client.get(f"/v1/predict/{job_id}", headers=HEADERS)
            elif resp.status_code == 429:
                resp.success()  # Rate limited is expected under load
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")
