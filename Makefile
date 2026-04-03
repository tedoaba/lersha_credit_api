# Lersha Credit Scoring System — Makefile
# Usage: make <target>

.PHONY: help install setup-db migrate db-stamp setup-rag lint format check-format ci-quality typecheck pre-commit test coverage dev dev-next api ui mlflow docker-build docker-up docker-down restore-db clean frontend-dev frontend-build frontend-up

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
	@echo "  make setup-rag      Populate rag_documents table (pgvector knowledge base)"
	@echo ""
	@echo "Development:"
	@echo "  make dev           Start API + Streamlit UI (legacy)"
	@echo "  make dev-next      Start API + Next.js frontend together (recommended)"
	@echo "  make api           Start the FastAPI backend on port 8000 (hot reload)"
	@echo "  make ui            Start the Streamlit UI on port 8501"
	@echo "  make mlflow        Start the MLflow tracking server on port 5000"
	@echo "  make frontend-dev  Start the Next.js frontend only on port 3000"
	@echo "  make frontend-build Build the Next.js frontend for production"
	@echo "  make frontend-up   Start the Next.js frontend Docker service"
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

setup-rag:
	uv run python -m backend.scripts.populate_pgvector

# ── Development ────────────────────────────────────────────────────────────────

# Starts the API in a new terminal window and the UI in the current terminal.
# Both processes run concurrently; Ctrl-C in the current terminal stops the UI.
# Close the API window separately to stop the backend.
dev: migrate
	uv run uvicorn backend.main:app --reload --reload-dir backend --port 8000 --host 0.0.0.0 & \
	API_PID=$$!; \
	trap "kill $$API_PID 2>/dev/null; exit" INT TERM; \
	echo "Waiting for API to be ready..."; \
	until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 1; done; \
	echo "API ready — starting UI..."; \
	uv run streamlit run ui/Introduction.py --server.port 8501 --server.address 0.0.0.0; \
	kill $$API_PID 2>/dev/null

api:
	uv run uvicorn backend.main:app --reload --reload-dir backend --port 8000 --host 0.0.0.0

ui:
	uv run streamlit run ui/Introduction.py --server.port 8501 --server.address 0.0.0.0

mlflow:
	uv run mlflow ui --backend-store-uri mlruns --port 5000

# Starts FastAPI backend + Next.js frontend concurrently.
# API runs in the background; Ctrl-C stops both.
# Frontend is available at http://localhost:3000, API at http://localhost:8000
dev-next: migrate
	uv run uvicorn backend.main:app --reload --reload-dir backend --port 8000 --host 0.0.0.0 & \
	API_PID=$$!; \
	trap "kill $$API_PID 2>/dev/null; exit" INT TERM; \
	echo "Waiting for API to be ready..."; \
	until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 1; done; \
	echo "API ready — starting Next.js frontend on http://localhost:3000 ..."; \
	cd frontend && npm run dev; \
	kill $$API_PID 2>/dev/null

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-up:
	docker compose up frontend

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
