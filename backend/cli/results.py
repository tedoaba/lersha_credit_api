#!/usr/bin/env python
"""CLI: fetch evaluation results from the API.

Usage:
    python -m backend.cli.results
    python -m backend.cli.results --limit 10 --model xgboost
    python -m backend.cli.results --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from backend.cli._client import api_get, die


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch evaluation results via the API")
    parser.add_argument("--limit", type=int, default=100, help="Max records (1-1000)")
    parser.add_argument("--model", default=None, help="Filter by model name")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    args = parser.parse_args()

    params: dict = {"limit": args.limit}
    if args.model:
        params["model_name"] = args.model

    try:
        data = api_get("/v1/results/", params=params)
    except Exception as exc:
        die(str(exc))

    if args.format == "csv":
        records = data.get("records", [])
        if not records:
            print("no records", file=sys.stderr)
            return
        writer = csv.DictWriter(sys.stdout, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
