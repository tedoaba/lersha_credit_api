# Lersha Credit Scoring System — Makefile
# Usage: make <target>

.PHONY: help install setup-db setup-chroma lint format check-format ci-quality test coverage api ui mlflow docker-build docker-up docker-down clean

# Default target
help:
	@echo ""
	@echo "Lersha Credit Scoring System"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install all dependencies (uv sync + dev extras)"
	@echo "  make setup-db      Initialise PostgreSQL schema and load CSV data"
	@echo "  make setup-chroma  Populate ChromaDB credit_features collection"
	@echo ""
	@echo "Development:"
	@echo "  make api           Start the FastAPI backend on port 8000 (hot reload)"
	@echo "  make ui            Start the Streamlit UI on port 8501"
	@echo "  make mlflow        Start the MLflow tracking server on port 5000"
	@echo ""
	@echo "Quality:"
	@echo "  make lint          Run ruff linter on backend/ and ui/"
	@echo "  make format        Auto-format backend/ and ui/ with ruff"
	@echo "  make check-format  Check formatting without applying changes"
	@echo "  make ci-quality    Run lint + format check (CI quality gate)"
	@echo "  make test          Run the full test suite"
	@echo "  make coverage      Run tests with HTML coverage report"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build backend and ui Docker images"
	@echo "  make docker-up     Start the full Docker Compose stack"
	@echo "  make docker-down   Stop the Docker Compose stack"
	@echo ""
	@echo "  make clean         Remove __pycache__, .coverage, htmlcov/"

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	uv sync --extra dev

setup-db:
	uv run python backend/scripts/db_init.py

setup-chroma:
	uv run python backend/scripts/populate_chroma.py

# ── Development ────────────────────────────────────────────────────────────────

api:
	uv run uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0

ui:
	uv run streamlit run ui/Introduction.py --server.port 8501 --server.address 0.0.0.0

mlflow:
	uv run mlflow ui --backend-store-uri mlruns --port 5000

# ── Quality ────────────────────────────────────────────────────────────────────

lint:
	uv run ruff check backend/ ui/

format:
	uv run ruff format backend/ ui/

check-format:
	uv run ruff format --check backend/ ui/

ci-quality: lint check-format

test:
	uv run pytest backend/tests/ -v

coverage:
	uv run pytest backend/tests/ --cov=backend --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# ── Docker ─────────────────────────────────────────────────────────────────────

docker-build:
	docker build -f backend/Dockerfile -t lersha-backend:latest .
	docker build -f ui/Dockerfile -t lersha-ui:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/ .ruff_cache/
	@echo "Clean complete"
