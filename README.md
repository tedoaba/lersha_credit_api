# Lersha Credit Scoring System

A production-grade credit scoring platform for Ethiopian smallholder farmers. Classifies each farmer as **Eligible**, **Review**, or **Not Eligible** using an ensemble of ML models with SHAP explainability and RAG-powered natural-language explanations.

---

## Architecture

```
Next.js Frontend  ──HTTP──►  FastAPI Backend  ──►  ML Pipeline + PostgreSQL + Ollama LLM
     (3007)                     (8006)                    (pgvector + Redis)
```

The frontend communicates with the backend **exclusively over HTTP** via Next.js API route proxies. No backend Python modules are imported in the frontend. The backend runs inference asynchronously — predictions return immediately with a job ID, and clients poll until completion.

```
POST /v1/predict  →  202 Accepted + {job_id}
                      │
                      └─► BackgroundTasks (dev) / Celery (prod)
                              └─► Models run in parallel (ThreadPoolExecutor)
                                    ├─► XGBoost  ──► SHAP + RAG explanation
                                    └─► Random Forest ──► SHAP + RAG explanation
                                          └─► Results saved to PostgreSQL

GET /v1/predict/{job_id}  →  poll until completed / failed
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12+, FastAPI, SQLAlchemy 2.x, Alembic, Celery + Redis |
| **ML** | XGBoost, Random Forest, CatBoost, SHAP, scikit-learn, MLflow |
| **RAG** | pgvector (PostgreSQL), Ollama (mxbai-embed-large + LLM), Redis cache |
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS 4, shadcn/ui, TanStack Query, Zustand |
| **Infra** | Docker Compose, Caddy (HTTPS), Gunicorn, uv (package manager) |
| **Quality** | Ruff (lint + format), MyPy, Pytest (>=80% coverage gate), pre-commit hooks |

---

## Project Structure

```
lersha_credit_api/
├── backend/
│   ├── api/                    # HTTP layer — routers, schemas, auth, middleware
│   │   └── routers/            # health, predict, results, explain, analytics, jobs, farmers
│   ├── core/                   # ML pipeline — feature engineering, preprocessing, inference, SHAP
│   ├── services/               # Data layer — all PostgreSQL CRUD (single source of truth)
│   ├── chat/                   # RAG engine — pgvector retrieval + Ollama generation
│   │   ├── rag_engine.py       # Legacy RAG (backward compatibility)
│   │   └── rag_service.py      # Production RAG — Redis caching, circuit breaker, versioned prompts
│   ├── config/                 # Configuration singleton + hyperparams.yaml
│   ├── cli/                    # Thin HTTP-only CLI wrappers (zero backend imports)
│   ├── logger/                 # Rotating file + console logging factory
│   ├── models/                 # Pre-trained ML artifacts (.pkl files, git-tracked)
│   ├── prompts/                # Versioned RAG prompt templates (v1.yaml, v2.yaml)
│   ├── scripts/                # Bootstrap scripts (db_init, populate_pgvector, register_model)
│   ├── tests/                  # Unit + integration tests
│   ├── alembic/                # Database migrations (8 versions)
│   ├── main.py                 # FastAPI app factory + lifespan
│   └── worker.py               # Celery app + inference task
├── frontend/
│   ├── app/                    # Next.js App Router pages (dashboard, predict, results)
│   │   └── api/                # Server-side API route proxies (adds auth headers)
│   ├── components/             # React components (DashboardPanel, FarmersPanel, SHAPChart, etc.)
│   └── lib/                    # API client, TypeScript types, TanStack Query hooks, Zustand stores
├── docs/                       # Architecture and planning documentation
├── specs/                      # Feature specifications (001–008)
├── docker-compose.yml          # Base service definitions
├── docker-compose.override.yml # Dev overrides (auto-loaded)
├── docker-compose.prod.yml     # Production overlay (Caddy, Gunicorn, backup)
├── Makefile                    # All development and deployment commands
└── pyproject.toml              # Python dependencies and tool config
```

---

## Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js >= 20
- PostgreSQL 16+ with [pgvector](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.com/) (for embeddings and LLM generation)
- Redis 7+ (for RAG caching; optional in dev)

**For production:**
- Docker + Docker Compose

---

## Quick Start

### 1. Install dependencies

```bash
make install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set the required values:

```bash
# PostgreSQL
DB_URI=postgresql://postgres:yourpassword@localhost:5432/credit_api_db

# Ollama (LLM + embeddings)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL_NAME=llama3
OLLAMA_EMBEDDER_MODEL=mxbai-embed-large

# API security
API_KEY=dev-local-api-key

# Local dev — runs inference in-process without Celery worker
CELERY_TASK_ALWAYS_EAGER=true
```

### 3. Pull Ollama models

```bash
ollama pull mxbai-embed-large
ollama pull llama3
```

### 4. Start Redis (for RAG caching)

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### 5. Bootstrap the database

```bash
make setup-db    # Load CSV data into PostgreSQL
make migrate     # Apply Alembic migrations (creates all tables + pgvector extension)
make setup-rag   # Populate pgvector knowledge base (400+ documents)
```

### 6. Start the services

```bash
make dev
```

| Service | URL |
|---------|-----|
| FastAPI backend | http://localhost:8006 |
| OpenAPI docs | http://localhost:8006/docs |
| Next.js frontend | http://localhost:3007 |

To stop: press `Ctrl-C`.

> Start individually: `make api` (backend only) or `make frontend-dev` (frontend only).

---

## API Reference

**Base URL:** `http://localhost:8006`
**Auth:** All `/v1/` routes require header `X-API-Key: <value>`
**Interactive docs:** http://localhost:8006/docs

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | No | Root liveness check |
| `GET` | `/health` | No | Dependency health (DB, Redis, pgvector) |
| `POST` | `/v1/predict/` | Yes | Submit async inference job (returns 202 + job_id) |
| `GET` | `/v1/predict/{job_id}` | Yes | Poll job status/result |
| `GET` | `/v1/results/` | Yes | Paginated evaluation history |
| `POST` | `/v1/explain/` | Yes | RAG explanation for a prediction |
| `GET` | `/v1/analytics/summary` | Yes | Analytics dashboard data |
| `GET` | `/v1/jobs/` | Yes | List recent inference jobs |
| `GET` | `/v1/farmers/search` | Yes | Autocomplete farmer lookup |

### Submit a Prediction

```bash
# Batch prediction (random farmers)
curl -X POST http://localhost:8006/v1/predict/ \
  -H "X-API-Key: dev-local-api-key" \
  -H "Content-Type: application/json" \
  -d '{"source": "Batch Prediction", "number_of_rows": 5}'

# Single farmer
curl -X POST http://localhost:8006/v1/predict/ \
  -H "X-API-Key: dev-local-api-key" \
  -H "Content-Type: application/json" \
  -d '{"source": "Single Value", "farmer_uid": "F-001"}'
```

**Response `202 Accepted`:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted"
}
```

### Poll Job Status

```bash
curl http://localhost:8006/v1/predict/<job_id> \
  -H "X-API-Key: dev-local-api-key"
```

**Response `200 OK` (completed):**
```json
{
  "job_id": "...",
  "status": "completed",
  "result": {
    "result_xgboost": {
      "records_processed": 5,
      "evaluations": [
        {
          "predicted_class_name": "Eligible",
          "confidence_score": 0.87,
          "top_feature_contributions": [
            {"feature": "net_income", "value": 0.412}
          ],
          "rag_explanation": "The farmer was classified as Eligible...",
          "model_name": "xgboost"
        }
      ]
    },
    "result_random_forest": { "..." }
  }
}
```

Job `status` values: `pending` -> `processing` -> `completed` | `failed`

---

## Features

### ML Ensemble Pipeline
- **3 models**: XGBoost, Random Forest, CatBoost (configurable via `hyperparams.yaml`)
- **Parallel execution**: Models run concurrently via `ThreadPoolExecutor`
- **Cached model loading**: `lru_cache` eliminates repeated disk I/O after first load
- **36 engineered features** from raw farmer data (demographics, assets, farm operations, income)

### SHAP Explainability
- Top 10 feature contributions per prediction with direction (increases/reduces risk)
- `TreeExplainer` cached per model for fast recomputation
- Summary plots logged to MLflow per inference run

### RAG Explanations
- **Vector store**: pgvector with 1024-dimensional embeddings (mxbai-embed-large)
- **Knowledge base**: 400+ documents — feature definitions, policy rules, domain knowledge
- **Recursive chunking with overlap** for optimal retrieval on long documents
- **Redis cache**: 24-hour TTL on explanations (SHA-256 cache key)
- **Circuit breaker**: Falls back gracefully after 5 consecutive LLM failures
- **Versioned prompts**: Switch via `PROMPT_VERSION` env var (v1, v2)
- **Audit trail**: Every retrieval logged to `rag_audit_log` for compliance

### Analytics Dashboard
- KPI tiles: total farmers, eligible/review/not-eligible counts with percentages
- Charts: consensus pie, gender breakdown, model comparison, confidence distribution
- Top risk factors: horizontal SHAP bar chart showing feature impact direction
- Model agreement rate across ensemble

### Async Inference
- POST returns 202 immediately; client polls for completion
- Dev mode: `BackgroundTasks` (no Redis/Celery needed)
- Prod mode: Celery workers with configurable concurrency

---

## Environment Variables

Copy `.env.example` to `.env`. The application fails at startup if required variables are missing.

### Required

| Variable | Description |
|----------|-------------|
| `DB_URI` | PostgreSQL connection string |
| `API_KEY` | Secret key for `X-API-Key` header on all `/v1/` routes |
| `OLLAMA_HOST` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL_NAME` | Ollama LLM model for RAG generation |
| `OLLAMA_EMBEDDER_MODEL` | Embedding model (default: `mxbai-embed-large`) |

### Key Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for RAG caching + Celery broker |
| `CELERY_TASK_ALWAYS_EAGER` | `false` | `true` = run inference in-process (dev) |
| `PROMPT_VERSION` | `v1` | Active RAG prompt template version |
| `API_BASE_URL` | `http://localhost:8006` | Backend URL for CLI and Next.js server-side |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow backend store URI |

> `rag_top_k` and `rag_similarity_threshold` are tuned in `backend/config/hyperparams.yaml`, not as env vars.

See `.env.example` for the full list of 40+ variables with documentation.

---

## Database

PostgreSQL 16 with pgvector extension. Schema managed by Alembic (8 migration versions).

| Table | Purpose |
|-------|---------|
| `farmer_data_all` | Raw farmer input records (bootstrapped from CSV) |
| `candidate_result` | Prediction outputs with SHAP contributions and RAG explanations |
| `inference_jobs` | Async job tracking (pending/processing/completed/failed) |
| `rag_documents` | Knowledge base with `VECTOR(1024)` embeddings |
| `rag_audit_log` | Immutable RAG retrieval audit trail for compliance |

### Schema Changes

```bash
# Generate a new migration from ORM model changes
uv run alembic -c backend/alembic.ini revision --autogenerate -m "describe_change"

# Apply
make migrate
```

---

## Testing

```bash
make test        # Run all tests
make coverage    # Run with HTML coverage report (htmlcov/)
```

Coverage target: **>= 80%** (enforced in `pyproject.toml`).

```
backend/tests/
├── conftest.py              # Shared fixtures: test DB, sample data, mock Ollama
├── unit/
│   ├── test_feature_engineering.py
│   ├── test_preprocessing.py
│   ├── test_rag_engine.py
│   └── test_rag_service.py
└── integration/
    ├── test_predict_endpoint.py
    ├── test_explain_endpoint.py
    └── test_rag_pgvector.py
```

- Async mode: `asyncio_mode = "auto"`
- Ollama/LLM calls are mocked in tests
- DB uses real PostgreSQL or in-memory SQLite

---

## Docker / Production Deployment

### Dev stack

```bash
make docker-up
```

| Service | Port | Description |
|---------|------|-------------|
| `postgres` | 5432 | pgvector/pgvector:pg16 with persistent volume |
| `redis` | 6379 | Celery broker + RAG cache |
| `backend` | 8006 | FastAPI with hot reload |
| `frontend` | 3007 | Next.js with hot reload |
| `mlflow` | 5000 | Experiment tracking UI |

### Production stack

```bash
make docker-prod-up
```

Production additions:
- **Caddy** terminates TLS (Let's Encrypt) and reverse-proxies to backend/frontend
- **Gunicorn** with 4 Uvicorn workers
- **Docker Secrets** for API keys (mounted at `/run/secrets/`)
- **Backup service** runs nightly `pg_dump` with 30-day retention

```bash
# Create secrets before deploying
echo -n "your-api-key"    > secrets/api_key
echo -n "your-gemini-key" > secrets/gemini_api_key
```

---

## All Make Commands

```bash
# ── Setup ──────────────────────────────────────────────
make install          # Install all deps (uv sync --extra dev)
make setup-db         # Load CSV data into PostgreSQL
make migrate          # Apply Alembic migrations
make setup-rag        # Populate pgvector knowledge base

# ── Development ────────────────────────────────────────
make dev              # Start API (8006) + Next.js frontend (3007)
make api              # Backend only with hot reload
make frontend-dev     # Next.js frontend only
make mlflow           # MLflow UI on :5000

# ── Quality ────────────────────────────────────────────
make lint             # Ruff linter
make format           # Ruff auto-format
make typecheck        # MyPy type checker
make ci-quality       # lint + check-format (CI gate)
make test             # Run full test suite
make coverage         # Tests + HTML coverage report (>=80% gate)

# ── Docker ─────────────────────────────────────────────
make docker-build     # Build Docker images
make docker-up        # Dev Docker stack
make docker-prod-up   # Production stack (Caddy, Gunicorn, backup)
make docker-down      # Stop Docker stack
make restore-db       # Restore from backup

# ── Cleanup ────────────────────────────────────────────
make clean            # Remove __pycache__, .coverage, htmlcov/, .ruff_cache/
```

---

## Observability

### Logs

All modules use `get_logger(__name__)` from `backend/logger/logger.py`. Logs are written to both stdout and `logs/credit_scoring_model.log` (rotating, 10 MB x 5 files). Structured JSON format in production.

### MLflow

Every inference run logs a nested MLflow experiment with prediction metrics, SHAP JSON, and summary plots.

```bash
make mlflow   # http://localhost:5000
```

### Rate Limiting

POST `/v1/predict` is rate-limited to 10 requests/minute per IP via slowapi.

---

## Contributing

1. Create a feature branch from `main`
2. Run `make ci-quality` before pushing (lint + format check)
3. Ensure `make test` passes with >= 80% coverage
4. All UI changes go through Next.js API route proxies — no direct backend imports in `frontend/`
5. All DB operations go through `backend/services/db_utils.py` — no SQLAlchemy calls elsewhere
