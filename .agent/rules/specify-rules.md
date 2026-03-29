# lersha_credit_api Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-29

## Active Technologies
- Python 3.12 + FastAPI 0.115+, Pydantic v2, SQLAlchemy 2.x, Streamlit ~1.40, requests, PyYAML, python-dotenv (002-fastapi-backend-http-wire)
- PostgreSQL (via SQLAlchemy 2.x engine; `inference_jobs` + `candidate_result` tables) (002-fastapi-backend-http-wire)
- Python 3.12 + FastAPI 0.121, SQLAlchemy 2.x, Alembic ≥1.13, Celery ≥5.3, Redis ≥5.0, python-json-logger ≥2.0, slowapi ≥0.1.9, gunicorn ≥21.2, tenacity ≥8.2, starlette BaseHTTPMiddleware, pre-commit ≥3.7, mypy ≥1.9 (004-harden-app-security)
- PostgreSQL 16 (SQLAlchemy engine), ChromaDB (PersistentClient), Redis (Celery broker) (004-harden-app-security)
- Python 3.12 (backend), YAML (compose/CI), Caddyfile DSL + Docker Compose v2, Caddy 2, Gunicorn, Uvicorn, MLflow, Celery, Redis, postgres-backup-local (005-infra-prod-hardening)
- PostgreSQL 16 (app + MLflow tracking), S3/GCS (MLflow artifacts), `./backups/` (pg_dump) (005-infra-prod-hardening)

- Python 3.12 + FastAPI 0.121, SQLAlchemy 2.x, pandas 2.x, XGBoost 3.x, scikit-learn 1.7, CatBoost 1.2, SHAP 0.50, MLflow 3.6, chromadb, sentence-transformers, google-generativeai, Streamlit (UI), uvicorn, pydantic v2 (001-monorepo-refactor)

## Project Structure

```text
backend/
frontend/
tests/
```

## Commands

cd src; pytest; ruff check .

## Code Style

Python 3.12: Follow standard conventions

## Recent Changes
- 005-infra-prod-hardening: Added Python 3.12 (backend), YAML (compose/CI), Caddyfile DSL + Docker Compose v2, Caddy 2, Gunicorn, Uvicorn, MLflow, Celery, Redis, postgres-backup-local
- 004-harden-app-security: Added Python 3.12 + FastAPI 0.121, SQLAlchemy 2.x, Alembic ≥1.13, Celery ≥5.3, Redis ≥5.0, python-json-logger ≥2.0, slowapi ≥0.1.9, gunicorn ≥21.2, tenacity ≥8.2, starlette BaseHTTPMiddleware, pre-commit ≥3.7, mypy ≥1.9
- 002-fastapi-backend-http-wire: Added Python 3.12 + FastAPI 0.115+, Pydantic v2, SQLAlchemy 2.x, Streamlit ~1.40, requests, PyYAML, python-dotenv


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
