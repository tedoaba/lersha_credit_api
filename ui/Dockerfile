# Lersha Credit Scoring UI — Streamlit Dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install runtime dependencies only (no dev extras)
RUN uv sync --frozen --no-dev

# Copy Streamlit UI source
COPY ui/ ./ui/

EXPOSE 8501

ENV PYTHONPATH=/app

CMD ["uv", "run", "streamlit", "run", "ui/Introduction.py", "--server.port=8501", "--server.address=0.0.0.0"]
