# Lersha Credit Scoring System — Makefile
# Usage: make <target>

.PHONY: help install setup-db migrate db-stamp setup-chroma lint format check-format ci-quality typecheck pre-commit test coverage dev api ui mlflow docker-build docker-up docker-down restore-db clean

# Default target
help:
	@echo ""
	@echo "Lersha Credit Scoring System"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install all dependencies (uv sync + dev extras)"
	@echo "  make setup-db      Initialise PostgreSQL schema and load CSV data"
	@echo "  make migrate       Apply pending Alembic migrations (upgrade head)"
	@echo "  make db-stamp      Stamp DB at current head (use after manual schema creation)"
	@echo "  make setup-chroma  Populate ChromaDB credit_features collection"
	@echo ""
	@echo "Development:"
	@echo "  make dev           Start API (new window) + UI (current terminal)"
	@echo "  make api           Start the FastAPI backend on port 8000 (hot reload)"
	@echo "  make ui            Start the Streamlit UI on port 8501"
	@echo "  make mlflow        Start the MLflow tracking server on port 5000"
	@echo ""
	@echo "Quality:"
	@echo "  make lint          Run ruff linter on backend/ and ui/"
	@echo "  make format        Auto-format backend/ and ui/ with ruff"
	@echo "  make check-format  Check formatting without applying changes"
	@echo "  make ci-quality    Run lint + format check (CI quality gate)"
	@echo "  make typecheck     Run mypy type checker on backend/"
	@echo "  make pre-commit    Run all pre-commit hooks on every file"
	@echo "  make test          Run the full test suite"
	@echo "  make coverage      Run tests with HTML coverage report"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build backend and ui Docker images"
	@echo "  make docker-up     Start the full Docker Compose stack (dev)"
	@echo "  make docker-down   Stop the Docker Compose stack"
	@echo "  make restore-db    Restore PostgreSQL from backup (BACKUP_FILE=path required)"
	@echo ""
	@echo "  make clean         Remove __pycache__, .coverage, htmlcov/"

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	uv sync --extra dev

setup-db:
	uv run python backend/scripts/db_init.py

migrate:
	uv run alembic -c backend/alembic.ini upgrade head

db-stamp:
	uv run alembic -c backend/alembic.ini stamp head

setup-chroma:
	uv run python backend/scripts/populate_chroma.py

# ── Development ────────────────────────────────────────────────────────────────

# Starts the API in a new terminal window and the UI in the current terminal.
# Both processes run concurrently; Ctrl-C in the current terminal stops the UI.
# Close the API window separately to stop the backend.
dev:
	start "Lersha-API" cmd /k "uv run uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0"
	uv run streamlit run ui/Introduction.py --server.port 8501 --server.address 0.0.0.0

api:
	uv run uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0 --reload

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

typecheck:
	uv run mypy backend/

pre-commit:
	uv run pre-commit run --all-files

test:
	uv run pytest backend/tests/ -v

coverage:
	uv run pytest backend/tests/ --cov=backend --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# ── Docker ─────────────────────────────────────────────────────────────────────

docker-build:
	@echo "NOTE: For production, use:  docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d"
	docker build -f backend/Dockerfile -t lersha-backend:latest .
	docker build -f ui/Dockerfile -t lersha-ui:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

restore-db:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore-db BACKUP_FILE=./backups/<date>.sql.gz"; \
		exit 1; \
	fi
	@echo "Restoring database from $(BACKUP_FILE) ..."
	gunzip -c "$(BACKUP_FILE)" | psql "$(DB_URI)"
	@echo "Restore complete."

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/ .ruff_cache/
	@echo "Clean complete"
