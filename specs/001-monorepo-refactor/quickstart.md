# Quickstart: Lersha Credit Scoring System (Post-Refactor)

**Branch**: `001-monorepo-refactor`  
**Date**: 2026-03-29

---

## Prerequisites

- Python 3.12
- PostgreSQL 16 running locally (or via Docker)
- `uv` installed: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows) or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Google Gemini API key

---

## 1. Clone and Configure

```bash
git clone <repo-url>
cd lersha_credit_api
git checkout 001-monorepo-refactor

# Copy environment template
cp .env.example .env
# Edit .env and fill in required values (see below)
```

**Required `.env` values**:
```bash
GEMINI_API_KEY=<your-key>
GEMINI_MODEL=gemini-1.5-pro
DB_URI=postgresql://user:password@localhost:5432/lersha
FARMER_DATA_ALL=farmer_data_all
API_KEY=<your-secret-api-key>
```

---

## 2. Install Dependencies

```bash
uv sync              # installs all runtime deps from uv.lock
uv sync --extra dev  # adds pytest, ruff, httpx, pytest-mock
```

---

## 3. Bootstrap Database

```bash
make setup-db
# Equivalent: uv run python backend/scripts/db_init.py
```

This creates tables `farmer_data_all`, `candidate_raw_data_table`, `candidate_result`, and `inference_jobs`, then loads CSV data.

---

## 4. Populate ChromaDB

```bash
make setup-chroma
# Equivalent: uv run python backend/scripts/populate_chroma.py
```

Must be run once before the first prediction. Loads feature definitions into the `credit_features` ChromaDB collection.

---

## 5. Run the API

```bash
make api
# Equivalent: uv run uvicorn backend.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health` → `{"status":"ok","dependencies":{...}}`

---

## 6. Run the UI

```bash
make ui
# Equivalent: uv run streamlit run ui/Introduction.py --server.port 8501
```

Open: `http://localhost:8501`

---

## 7. Run Tests

```bash
make test
# Equivalent: uv run pytest backend/tests/ -v

make coverage
# Equivalent: uv run pytest backend/tests/ --cov=backend --cov-report=html
```

---

## 8. Lint and Format

```bash
make lint        # uv run ruff check backend/ ui/
make format      # uv run ruff format backend/ ui/
```

---

## 9. Docker (full stack)

```bash
make docker-up
# Starts: postgres, backend (port 8000), ui (port 8501), mlflow (port 5000)

make docker-down
```

---

## Smoke Test Commands

```bash
# Verify config loads correctly
uv run python -c "from backend.config.config import config; print(config.db_uri)"

# Verify feature engineering extraction
uv run python -c "from backend.core.feature_engineering import apply_feature_engineering; print('OK')"

# Verify no legacy imports remain
grep -r "from config.config" backend/ ui/  # must return nothing
grep -r "from src.logger" backend/ ui/     # must return nothing
grep -r "from services.db" backend/ ui/    # must return nothing
```

---

## MLflow UI

```bash
make mlflow
# Equivalent: uv run mlflow ui --backend-store-uri mlruns --port 5000
```

Open: `http://localhost:5000` → Experiment: `Credit Scoring Model`
