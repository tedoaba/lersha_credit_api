# CLAUDE.md — Lersha Credit Scoring API

## Project Overview

Agricultural credit scoring API for Ethiopian smallholder farmers. Classifies farmers as **Eligible**, **Review**, or **Not Eligible** using an ensemble of ML models (XGBoost, Random Forest, CatBoost) with SHAP explainability and RAG-powered natural-language explanations via Google Gemini.

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy 2.x, Alembic, Celery + Redis
- **ML**: XGBoost, Random Forest, CatBoost, SHAP, scikit-learn, MLflow
- **RAG**: pgvector (PostgreSQL), sentence-transformers (all-MiniLM-L6-v2), Google Gemini, LangChain
- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui, TanStack Query, Zustand
- **Infra**: Docker Compose (base + dev/prod overlays), Caddy (HTTPS), Gunicorn, uv (package manager)
- **Quality**: Ruff (lint + format), MyPy, Pytest (>=80% coverage gate), pre-commit hooks

## Quick Commands

```bash
make install          # Install all deps (uv sync --extra dev)
make setup-db         # Load CSV data into PostgreSQL
make migrate          # Apply Alembic migrations
make setup-rag        # Populate pgvector knowledge base
make dev              # Start API (port 8006) + Next.js frontend (port 3007)
make api              # Backend only with hot reload
make frontend-dev     # Next.js frontend only
make test             # Run full test suite
make coverage         # Tests + HTML coverage report (must reach >=80%)
make lint             # Ruff linter
make format           # Ruff auto-format
make typecheck        # MyPy type checker
make ci-quality       # lint + check-format (CI gate)
make docker-build     # Build Docker images
make docker-up        # Dev Docker stack
make docker-prod-up   # Production stack (Caddy, Gunicorn, backup)
```

## Project Structure

```
backend/
  api/           HTTP layer: routers (health, predict, results, explain), schemas, auth, middleware
  core/          ML pipeline: feature engineering, preprocessing, inference, SHAP computation
  services/      Data layer: all PostgreSQL CRUD (single source of truth for DB queries)
  chat/          RAG engine: pgvector retrieval + Gemini generation
  config/        Configuration singleton + hyperparams.yaml
  cli/           Thin HTTP-only CLI wrappers (zero backend imports)
  logger/        Rotating file + console logging factory
  models/        Pre-trained ML artifacts (.pkl files, git-tracked)
  prompts/       Versioned RAG prompt templates (v1.yaml, v2.yaml)
  scripts/       Bootstrap scripts (db_init, populate_pgvector, register_model)
  tests/         Unit + integration tests
  alembic/       Database migrations (4 versions)
  main.py        FastAPI app factory + lifespan
  worker.py      Celery app + inference task

frontend/
  app/           Next.js App Router pages (dashboard, predict, results)
  components/    React components (NavBar, PredictionForm, SHAPChart, etc.)
  lib/           API client, TypeScript types, TanStack Query hooks

specs/           Feature specifications (001-008)
docs/            Architecture and planning docs
```

## Architecture Key Points

- **Async inference**: POST /v1/predict returns 202 + job_id; client polls GET /v1/predict/{job_id}. Uses Celery in prod, BackgroundTasks in dev (CELERY_TASK_ALWAYS_EAGER=true).
- **UI-backend separation**: Frontend communicates exclusively via HTTP. Zero backend imports in UI code.
- **DB access**: All PostgreSQL queries live in `backend/services/db_utils.py` — no SQLAlchemy calls elsewhere.
- **Config singleton**: `from backend.config.config import config` — reads Docker secrets (/run/secrets/) with env var fallback. Fails fast on missing required values.
- **RAG pipeline**: Embeds query -> pgvector cosine search -> Gemini generation with versioned prompts. All retrievals logged to rag_audit_log for compliance.

## API Endpoints

All `/v1/` routes require `X-API-Key` header.

| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Health check (+ Redis probe) |
| POST | /v1/predict/ | Submit inference job (returns 202 + job_id) |
| GET | /v1/predict/{job_id} | Poll job status/result |
| GET | /v1/results | Paginated evaluation history |
| POST | /v1/explain/ | RAG explanation only (no prediction) |

## Database

PostgreSQL 16 with pgvector extension. Managed by Alembic migrations.

**Tables**: farmer_data_all (raw data), candidate_result (predictions), inference_jobs (async jobs), rag_documents (vector store), rag_audit_log (compliance trail).

```bash
make migrate                           # Apply migrations
uv run alembic -c backend/alembic.ini revision --autogenerate -m "desc"  # New migration
```

## Testing

```bash
make test              # All tests
make coverage          # With HTML report + 80% gate
uv run pytest backend/tests/unit/test_feature_engineering.py -v  # Single file
```

- Tests in `backend/tests/unit/` and `backend/tests/integration/`
- Shared fixtures in `conftest.py` (test DB engine, sample data, mock Gemini)
- Async mode: `asyncio_mode = "auto"`
- Gemini API is mocked in tests; DB uses real PostgreSQL or in-memory SQLite

## Code Style

- **Ruff**: line-length 120, target py312, rules: E, F, I, UP, B, SIM
- **MyPy**: strict=false, ignore_missing_imports=true, check_untyped_defs=true
- Run `make format` before committing; pre-commit hooks enforce this automatically

## Environment Variables

Required secrets (Docker secrets or env vars): `DB_URI`, `API_KEY`, `GEMINI_API_KEY`, `GEMINI_MODEL`

See `.env.example` for the full list of 40+ variables with documentation.

## Docker

```bash
# Dev: uses docker-compose.yml + docker-compose.override.yml (auto-loaded)
make docker-up

# Prod: explicit overlay with Caddy, Gunicorn, backup service
make docker-prod-up

# Secrets (prod only — never use .env in production):
echo -n "key" > secrets/api_key
echo -n "key" > secrets/gemini_api_key
```
