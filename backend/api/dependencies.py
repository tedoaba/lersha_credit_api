"""FastAPI dependency functions for the Lersha Credit Scoring API."""
from fastapi import Header, HTTPException, status

from backend.config.config import config


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
