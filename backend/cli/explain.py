#!/usr/bin/env python
"""CLI: request an AI-generated explanation for a prediction result.

Usage:
    python -m backend.cli.explain --job-id <uuid> --record-index 0
    python -m backend.cli.explain --job-id <uuid> --record-index 0 --model xgboost
"""

from __future__ import annotations

import argparse
import json

from backend.cli._client import api_post, die


def main() -> None:
    parser = argparse.ArgumentParser(description="Get AI explanation for a prediction")
    parser.add_argument("--job-id", required=True, help="Job UUID")
    parser.add_argument("--record-index", type=int, required=True, help="Record index within job")
    parser.add_argument("--model", default=None, help="Model name (e.g. xgboost)")
    args = parser.parse_args()

    payload: dict = {"job_id": args.job_id, "record_index": args.record_index}
    if args.model:
        payload["model_name"] = args.model

    try:
        data = api_post("/v1/explain/", json=payload)
    except Exception as exc:
        die(str(exc))

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
