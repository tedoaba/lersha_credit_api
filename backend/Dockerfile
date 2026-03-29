# Lersha Credit Scoring API — Backend Dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install runtime dependencies only (no dev extras)
RUN uv sync --frozen --no-dev

# Copy backend source and model artifacts
COPY backend/ ./backend/

# Ensure models and chroma_db directories exist
RUN mkdir -p backend/models chroma_db output logs

EXPOSE 8000

ENV PYTHONPATH=/app

# Dev alternative (hot reload, single worker):
#   uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
#
# Production: run Alembic migrations first, then start Gunicorn with 4 Uvicorn workers.
# --workers 4       : one master + 4 worker processes
# --worker-class    : Uvicorn's async ASGI worker
# --timeout 120     : generous timeout for inference requests
CMD ["sh", "-c", "uv run alembic -c backend/alembic.ini upgrade head && uv run gunicorn backend.main:app --worker-class uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8000 --timeout 120"]
