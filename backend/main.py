"""Lersha Credit Scoring API — application factory.

Production startup (via Dockerfile CMD):
    alembic upgrade head && gunicorn backend.main:app \\
        --worker-class uvicorn.workers.UvicornWorker \\
        --workers 4 --bind 0.0.0.0:8006 --timeout 120

Development (hot reload):
    uvicorn backend.main:app --reload --reload-dir backend --port 8006

The factory pattern (``create_app()``) enables easy test client instantiation:
    from fastapi.testclient import TestClient
    client = TestClient(create_app())
"""

# ── Pickle compatibility patch ──────────────────────────────────────────────
# The pre-refactor sklearn Pipelines (.pkl files in backend/models/) were
# trained when ``logic`` was a root-level package containing ``replace_inf``.
# joblib.load() tries to import ``logic.smote_updated`` during
# deserialization, which fails with ModuleNotFoundError after the cleanup.
#
# This patch registers the old module paths in sys.modules before any model
# is loaded, pointing them at the canonical backend.core.preprocessing module.
# All code stays inside backend/ — no root-level shim folder needed.
#
# Safe to remove once all .pkl artifacts have been retrained and re-pickled
# using the backend.core.* import paths.
import sys
import types

import backend.core.preprocessing as _preprocessing  # noqa: E402

_logic_smote = types.ModuleType("logic.smote_updated")
_logic_smote.replace_inf = _preprocessing.replace_inf  # type: ignore[attr-defined]

_logic_pkg = types.ModuleType("logic")
_logic_pkg.smote_updated = _logic_smote  # type: ignore[attr-defined]

sys.modules.setdefault("logic", _logic_pkg)
sys.modules.setdefault("logic.smote_updated", _logic_smote)
# ── End compatibility patch ─────────────────────────────────────────────────

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from slowapi import _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402

from backend.api.dependencies import limiter  # noqa: E402
from backend.api.middleware import RequestIDMiddleware  # noqa: E402
from backend.api.routers import analytics, explain, farmers, health, jobs, predict, results  # noqa: E402
from backend.logger.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm heavy resources at server startup.

    Imports rag_engine eagerly so the sentence-transformers model is loaded
    from the local HuggingFace cache once at startup — not on the first
    inference request. This avoids the ~30 s first-request delay.
    """
    logger.info("Pre-warming RAG engine (loading sentence-transformers from cache)...")
    try:
        import backend.chat.rag_engine  # noqa: F401 — side-effect import warms module cache

        logger.info("RAG engine ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG engine pre-warm failed (non-fatal): %s", exc)
    yield
    # Shutdown: nothing to clean up.


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Registers all versioned routers, attaches the rate limiter to app state,
    registers the rate-limit exceeded exception handler, and adds the
    RequestIDMiddleware for per-request trace IDs.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(
        title="Lersha Credit Scoring API",
        version="1.0.0",
        description=(
            "Agricultural credit scoring for Ethiopian smallholder farmers. "
            "All inference routes require the X-API-Key header. "
            "POST /v1/predict is rate-limited to 10 requests/minute/IP."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Rate limiter ────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # ── Middleware ──────────────────────────────────────────────────────────
    app.add_middleware(RequestIDMiddleware)

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(health.router)  # GET /, GET /health
    app.include_router(predict.router, prefix="/v1/predict", tags=["v1 — Inference"])
    app.include_router(results.router, prefix="/v1/results", tags=["v1 — Results"])
    app.include_router(explain.router, prefix="/v1/explain", tags=["v1 — Explain"])
    app.include_router(analytics.router, prefix="/v1/analytics", tags=["v1 — Analytics"])
    app.include_router(jobs.router, prefix="/v1/jobs", tags=["v1 — Jobs"])
    app.include_router(farmers.router, prefix="/v1/farmers", tags=["v1 — Farmers"])

    return app


app = create_app()
