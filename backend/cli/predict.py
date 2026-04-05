#!/usr/bin/env python
"""CLI: submit a prediction job and optionally poll until complete.

Usage:
    # Single farmer
    python -m backend.cli.predict --source "Single Value" --farmer-uid FRM-001

    # Batch (random sample)
    python -m backend.cli.predict --source "Batch Prediction" --rows 10

    # Submit and poll until done
    python -m backend.cli.predict --source "Batch Prediction" --rows 5 --wait
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from backend.cli._client import api_get, api_post, die


def submit(source: str, farmer_uid: str | None, rows: int | None) -> dict:
    payload: dict = {"source": source}
    if farmer_uid:
        payload["farmer_uid"] = farmer_uid
    if rows:
        payload["number_of_rows"] = rows
    return api_post("/v1/predict/", json=payload)


def poll(job_id: str, interval: float = 2.0, timeout: float = 300.0) -> dict:
    elapsed = 0.0
    while elapsed < timeout:
        result = api_get(f"/v1/predict/{job_id}")
        status = result.get("status", "unknown")
        print(f"  status: {status} ({elapsed:.0f}s)", file=sys.stderr)
        if status in ("completed", "failed"):
            return result
        time.sleep(interval)
        elapsed += interval
    die(f"job {job_id} did not complete within {timeout}s")
    return {}  # unreachable


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a prediction job via the API")
    parser.add_argument("--source", required=True, choices=["Single Value", "Batch Prediction"])
    parser.add_argument("--farmer-uid", default=None, help="Farmer UID (Single Value)")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows (Batch Prediction)")
    parser.add_argument("--wait", action="store_true", help="Poll until job completes")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between polls")
    args = parser.parse_args()

    try:
        resp = submit(args.source, args.farmer_uid, args.rows)
    except Exception as exc:
        die(str(exc))

    job_id = resp.get("job_id", "unknown")
    print(f"job accepted: {job_id}")

    if args.wait:
        result = poll(job_id, interval=args.poll_interval)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()
