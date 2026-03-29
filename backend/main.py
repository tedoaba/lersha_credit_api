"""Lersha Credit Scoring API — application factory.

Usage:
    uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI

from backend.api.routers import health, predict, results


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Registers all versioned routers and middleware. This factory pattern
    enables easy test client instantiation via ``TestClient(create_app())``.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(
        title="Lersha Credit Scoring API",
        version="1.0.0",
        description=(
            "Agricultural credit scoring for Ethiopian smallholder farmers. "
            "All inference routes require the X-API-Key header."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(health.router)                              # GET /, GET /health
    app.include_router(predict.router, prefix="/v1/predict", tags=["v1 — Inference"])
    app.include_router(results.router, prefix="/v1/results", tags=["v1 — Results"])

    return app


app = create_app()
