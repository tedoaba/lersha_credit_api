"""Health check router — GET / and GET /health."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.config.config import config
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

    Performs a real ``SELECT 1`` against PostgreSQL and a ChromaDB heartbeat.
    Returns HTTP 200 with ``{"db":"ok","chroma":"ok"}`` if all dependencies
    are healthy, or HTTP 503 with the failing key showing the error reason.
    """
    import chromadb

    health: dict[str, str] = {}
    all_ok = True

    # PostgreSQL check
    try:
        engine = db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health["db"] = "ok"
    except Exception as exc:
        logger.error("PostgreSQL health check failed: %s", exc)
        health["db"] = f"error: {exc}"
        all_ok = False

    # ChromaDB check
    try:
        client = chromadb.PersistentClient(path=str(config.chroma_db_path))
        client.heartbeat()
        health["chroma"] = "ok"
    except Exception as exc:
        logger.error("ChromaDB health check failed: %s", exc)
        health["chroma"] = f"error: {exc}"
        all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(content=health, status_code=status_code)
