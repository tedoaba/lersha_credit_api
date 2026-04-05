"""Health check router — GET / and GET /health."""

import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.config.config import config
from backend.logger.logger import get_logger
from backend.services.db_utils import db_engine

logger = get_logger(__name__)
router = APIRouter()

_EAGER_MODE = os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() in ("1", "true", "yes")


@router.get("/", tags=["Health"])
async def root() -> dict:
    """Root endpoint — basic liveness check."""
    return {"message": "Lersha Credit Scoring API is running", "version": "1.0.0"}


@router.get("/health", tags=["Health"])
async def health_check() -> JSONResponse:
    """Live dependency health check.

    Performs real connectivity probes against PostgreSQL (SELECT 1),
    pgvector extension existence, and Redis (PING).  Returns HTTP 200
    when all dependencies are healthy, or HTTP 503 with the failing key
    showing the error reason.
    """
    health: dict[str, str] = {}
    all_ok = True

    engine = db_engine()

    # PostgreSQL connectivity check
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health["db"] = "ok"
    except Exception as exc:
        logger.error("PostgreSQL health check failed: %s", exc)
        health["db"] = f"error: {exc}"
        all_ok = False

    # pgvector extension check (validates the vector extension is installed and active)
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'")).fetchone()
        if row is None:
            raise RuntimeError("pgvector extension not found in pg_catalog.pg_extension")
        health["pgvector"] = "ok"
    except Exception as exc:
        logger.error("pgvector health check failed: %s", exc)
        health["pgvector"] = f"error: {exc}"
        all_ok = False

    # Redis connectivity check — non-fatal in eager mode (no Redis needed locally)
    try:
        import redis

        r = redis.Redis.from_url(config.redis_url, socket_connect_timeout=3)
        r.ping()
        health["redis"] = "ok"
    except Exception as exc:
        if _EAGER_MODE:
            logger.debug("Redis unavailable (eager mode, non-fatal): %s", exc)
            health["redis"] = "skipped (eager mode)"
        else:
            logger.error("Redis health check failed: %s", exc)
            health["redis"] = f"error: {exc}"
            all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(content=health, status_code=status_code)
