"""Health check router — GET / and GET /health."""
from fastapi import APIRouter
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
async def health_check() -> dict:
    """Live dependency health check.

    Performs a real ``SELECT 1`` against PostgreSQL and a ChromaDB heartbeat.
    Returns HTTP 200 if all dependencies are healthy, HTTP 503 if degraded.
    """
    import chromadb
    from fastapi.responses import JSONResponse

    dependencies: dict[str, str] = {}
    all_ok = True

    # PostgreSQL check
    try:
        engine = db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        dependencies["postgresql"] = "ok"
    except Exception as exc:
        logger.error("PostgreSQL health check failed: %s", exc)
        dependencies["postgresql"] = f"error: {exc}"
        all_ok = False

    # ChromaDB check
    try:
        client = chromadb.PersistentClient(path=str(config.chroma_db_path))
        client.heartbeat()
        dependencies["chromadb"] = "ok"
    except Exception as exc:
        logger.error("ChromaDB health check failed: %s", exc)
        dependencies["chromadb"] = f"error: {exc}"
        all_ok = False

    status_str = "ok" if all_ok else "degraded"
    status_code = 200 if all_ok else 503
    return JSONResponse(
        content={"status": status_str, "dependencies": dependencies},
        status_code=status_code,
    )
