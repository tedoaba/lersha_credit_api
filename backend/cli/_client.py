"""Shared HTTP client for CLI commands.

Reads API_BASE_URL and API_KEY from environment variables (or .env file).
No backend internals are imported — only stdlib + requests.
"""

from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
_API_KEY = os.getenv("API_KEY", "")
_TIMEOUT = (5, 60)  # (connect, read) seconds


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"X-API-Key": _API_KEY, "Content-Type": "application/json"})
    return s


def api_get(path: str, params: dict | None = None) -> dict:
    """GET request to the backend API."""
    resp = _session().get(f"{_BASE_URL}{path}", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json: dict | None = None) -> dict:
    """POST request to the backend API."""
    resp = _session().post(f"{_BASE_URL}{path}", json=json, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def die(msg: str) -> None:
    """Print an error and exit."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)
