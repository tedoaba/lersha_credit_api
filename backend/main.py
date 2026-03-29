"""Lersha Credit Scoring API — application factory.

Production startup (via Dockerfile CMD):
    alembic upgrade head && gunicorn backend.main:app \\
        --worker-class uvicorn.workers.UvicornWorker \\
        --workers 4 --bind 0.0.0.0:8000 --timeout 120

Development (hot reload):
    uvicorn backend.main:app --reload --port 8000

The factory pattern (``create_app()``) enables easy test client instantiation:
    from fastapi.testclient import TestClient
    client = TestClient(create_app())
"""

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.api.dependencies import limiter
from backend.api.middleware import RequestIDMiddleware
from backend.api.routers import health, predict, results


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

    return app


app = create_app()
