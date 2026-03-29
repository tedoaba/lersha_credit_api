# lersha_credit_api Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-29

## Active Technologies
- Python 3.12 + FastAPI 0.115+, Pydantic v2, SQLAlchemy 2.x, Streamlit ~1.40, requests, PyYAML, python-dotenv (002-fastapi-backend-http-wire)
- PostgreSQL (via SQLAlchemy 2.x engine; `inference_jobs` + `candidate_result` tables) (002-fastapi-backend-http-wire)

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
- 002-fastapi-backend-http-wire: Added Python 3.12 + FastAPI 0.115+, Pydantic v2, SQLAlchemy 2.x, Streamlit ~1.40, requests, PyYAML, python-dotenv

- 001-monorepo-refactor: Added Python 3.12 + FastAPI 0.121, SQLAlchemy 2.x, pandas 2.x, XGBoost 3.x, scikit-learn 1.7, CatBoost 1.2, SHAP 0.50, MLflow 3.6, chromadb, sentence-transformers, google-generativeai, Streamlit (UI), uvicorn, pydantic v2

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
