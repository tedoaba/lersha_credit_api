"""FastAPI dependency functions for the Lersha Credit Scoring API."""

from fastapi import Header, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config.config import config


def _get_remote_address(request: Request) -> str:
    """Get the real client IP, respecting reverse proxy headers.

    Priority:
      1. ``X-Forwarded-For`` header (first IP — set by Caddy, ALB, etc.)
      2. Direct ``request.client.host``
      3. ``"127.0.0.1"`` fallback for tests (httpx ASGITransport has no client)
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is None:
        return "127.0.0.1"
    return get_remote_address(request)


# ── Rate limiter ──────────────────────────────────────────────────────────────
# Keyed by client remote IP address. Attached to app.state in create_app().
limiter = Limiter(key_func=_get_remote_address)


async def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """Validate the X-API-Key header against the configured API key.

    Raises:
        HTTPException: 403 Forbidden if the key is missing or does not match.
    """
    if x_api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
