#!/usr/bin/env python
"""CLI: check backend health status.

Usage:
    python -m backend.cli.health
"""

from __future__ import annotations

import json
import sys

from backend.cli._client import api_get, die


def main() -> None:
    try:
        data = api_get("/health")
    except Exception as exc:
        die(str(exc))

    all_ok = all(v == "ok" for v in data.values())
    print(json.dumps(data, indent=2))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
