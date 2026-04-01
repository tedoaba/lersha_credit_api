"""Health check router — GET / and GET /health."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.logger.logger import get_logger
from backend.services.db_utils import db_engine

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", tags=["Health"])
async def root() -> dict:
    """Root endpoint — basic liveness check."""
    return {"message": "Lersha Credit Scoring API is running", "version": "1.0.0"}


@router.get("/health", tags=["Health"])
async def health_check() -> JSONResponse:
    """Live dependency health check.

    Performs a real ``SELECT 1`` against PostgreSQL and a pgvector extension
    existence check.  Returns HTTP 200 with ``{"db":"ok","pgvector":"ok"}``
    if all dependencies are healthy, or HTTP 503 with the failing key showing
    the error reason.

    Migrated in 006-migrate-chroma-pgvector (2026-04-01):
        Replaced ChromaDB heartbeat probe with pgvector extension check.
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
            row = conn.execute(
                text("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'")
            ).fetchone()
        if row is None:
            raise RuntimeError("pgvector extension not found in pg_catalog.pg_extension")
        health["pgvector"] = "ok"
    except Exception as exc:
        logger.error("pgvector health check failed: %s", exc)
        health["pgvector"] = f"error: {exc}"
        all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(content=health, status_code=status_code)
