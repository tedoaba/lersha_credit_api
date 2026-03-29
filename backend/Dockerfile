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

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
