# Lersha Credit Scoring System — Production Refactoring Plan

> **Goal:** Reorganize the codebase into a clean `backend/` / `ui/` monorepo layout, migrate the package manager from Poetry to **uv**, adopt **ruff** for linting and formatting, introduce a **Makefile** as the single developer entry point, fix all known bugs, close production gaps (auth, async, testing, containerization), and establish a CI/CD baseline.
>
> **Reference:** All structure and conventions described here align with `docs/ARCHITECTURE.md`, which is the authoritative target-state document.

---

## Table of Contents

1. [Current Problems (Diagnosis)](#1-current-problems-diagnosis)
2. [Target Architecture](#2-target-architecture)
3. [Target Directory Tree](#3-target-directory-tree)
4. [Phase 0 — Preparation](#phase-0--preparation)
5. [Phase 1 — Structural Reorganization](#phase-1--structural-reorganization)
6. [Phase 2 — Fix Existing Bugs](#phase-2--fix-existing-bugs)
7. [Phase 3 — Production Hardening](#phase-3--production-hardening)
8. [Phase 4 — Testing Suite](#phase-4--testing-suite)
9. [Phase 5 — Containerization](#phase-5--containerization)
10. [Phase 6 — CI/CD Pipeline, Ruff, and Makefile](#phase-6--cicd-pipeline-ruff-and-makefile)
11. [Phase 7 — Documentation and Finalization](#phase-7--documentation-and-finalization)
12. [Phase 8 — Production Hardening](#phase-8--production-hardening)
13. [Execution Order and Checklist](#execution-order-and-checklist)
14. [Risk Register](#risk-register)

---

## 1. Current Problems (Diagnosis)

| # | Problem | Severity | Phase to Fix |
|---|---|---|---|
| P1 | Root-level clutter: `app.py`, `infer.py`, `db_init.py`, `Introduction.py` all at project root | High | Phase 1 |
| P2 | Streamlit imports `src/inference_pipeline.py` directly, bypassing FastAPI | High | Phase 3 |
| P3 | `app.py` returns stale field names (`result_18`, `result_44`) that don't match `infer()` output | High | Phase 2 |
| P4 | RAG explanation is disabled; `rag_explanation` field is hardcoded to the literal string `"rag_explanation"` | High | Phase 2 |
| P5 | `rf_model_36` and `cab_model_36` both read the env var `XGB_MODEL_36` | Medium | Phase 2 |
| P6 | `fetch_raw_data` and `fetch_multiple_raw_data` do full table scans, filter in Python | Medium | Phase 2 |
| P7 | Inference loop is synchronous; LLM calls block per row | High | Phase 3 |
| P8 | No authentication on any API or UI surface | High | Phase 3 |
| P9 | No test suite | High | Phase 4 |
| P10 | No containerization (no Dockerfile, no compose) | Medium | Phase 5 |
| P11 | No CI/CD pipeline | Medium | Phase 6 |
| P12 | Dead code (`generate_shap_value_summary_plotsss`) and unused model variants (`_34_`) | Low | Phase 2 |
| P13 | ChromaDB collection is never pre-populated; RAG context is always empty | Medium | Phase 2 |
| P14 | `logic/smote_updated.py` is a single-function module with no clear home | Low | Phase 1 |
| P15 | `src/utils.py` overlaps with `src/infer_utils.py`; responsibilities are blurred | Medium | Phase 1 |
| P16 | Package manager is Poetry — migrate to **uv** for faster installs, simpler CI, and native lockfile support | Medium | Phase 0 |
| P17 | No single developer entry point — commands are scattered across README | Low | Phase 6 |
| P18 | No unified linter/formatter — add **ruff** (replaces flake8 + isort + black) | Low | Phase 6 |

> **Production gaps** (DB migrations, job queue, rate limiting, HTTPS, etc.) are documented separately in `docs/ARCHITECTURE.md` Section 16.

---

## 2. Target Architecture

After the refactor, the system follows a clean **API-first** architecture. The Streamlit UI calls the FastAPI backend over HTTP, eliminating direct imports of ML logic from the UI layer.

```
+----------------------------------------------------------+
|                        ui/                               |
|   Streamlit pages call FastAPI via HTTP (requests lib)   |
|   No ML imports. Pure presentation.                      |
+---------------------------+------------------------------+
                            | HTTP (localhost:8000)
+---------------------------v------------------------------+
|                      backend/                            |
|  +------------------+  +------------------------------+ |
|  |   api/           |  |   core/                      | |
|  |  routers/        |  |   pipeline.py                | |
|  |  schemas/        |  |   infer_utils.py             | |
|  |  dependencies/   |  |   feature_engineering.py     | |
|  +------------------+  +------------------------------+ |
|  +------------------+  +------------------------------+ |
|  |   services/      |  |   chat/                      | |
|  |   db_utils.py    |  |   rag_engine.py              | |
|  |   db_model.py    |  |   chroma_loader.py (NEW)     | |
|  |   schema.py      |  +------------------------------+ |
|  +------------------+                                   |
|  +------------------+  +------------------------------+ |
|  |   config/        |  |   logger/                    | |
|  |   config.py      |  |   logger.py                  | |
|  +------------------+  +------------------------------+ |
+----------------------------------------------------------+
                            |
+---------------------------v------------------------------+
|           PostgreSQL + ChromaDB + MLflow                 |
+----------------------------------------------------------+
```

---

## 3. Target Directory Tree

```
lersha_credit_api/
│
├── backend/                          # Everything server-side
│   ├── Dockerfile                    # Backend container build
│   ├── __init__.py
│   ├── main.py                       # FastAPI app factory
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py           # X-API-Key auth guard
│   │   ├── schemas.py                # Pydantic request/response models
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── health.py             # GET /, GET /health
│   │       ├── predict.py            # POST /predict, GET /predict/{job_id}
│   │       └── results.py            # GET /results
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py               # match_inputs(), run_inferences()
│   │   ├── infer_utils.py            # Model load, SHAP, contribution table
│   │   ├── feature_engineering.py    # apply_feature_engineering() — extracted
│   │   └── preprocessing.py          # OHE, feature alignment, inf/nan sanitizer
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── db_utils.py               # All DB CRUD (SQL-fix applied)
│   │   ├── db_model.py               # SQLAlchemy ORM models
│   │   └── schema.py                 # Pydantic validation schemas
│   │
│   ├── chat/
│   │   ├── __init__.py
│   │   ├── rag_engine.py             # ChromaDB + Gemini RAG, fully wired
│   │   └── chroma_loader.py          # NEW: populates ChromaDB collection
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py                 # Env-var singleton, bug fixed
│   │   └── hyperparams.yaml          # NEW: externalized hyperparameters
│   │
│   ├── logger/
│   │   ├── __init__.py
│   │   └── logger.py                 # Centralized rotating logger
│   │
│   ├── scripts/                      # One-shot operational scripts
│   │   ├── db_init.py                # Database bootstrap
│   │   ├── populate_chroma.py        # NEW: loads feature defs into ChromaDB
│   │   └── run_inference.py          # CLI batch inference (renamed from infer.py)
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py               # Shared fixtures
│   │   ├── unit/
│   │   │   ├── test_feature_engineering.py
│   │   │   ├── test_preprocessing.py
│   │   │   └── test_contribution_table.py
│   │   └── integration/
│   │       ├── test_predict_endpoint.py
│   │       └── test_db_utils.py
│   │
│   ├── models/                       # Pre-trained model artifacts
│   ├── data/                         # Raw farmer data CSVs
│   └── prompts/                      # prompts.yaml — LLM system instructions
│
├── ui/                               # Everything Streamlit — zero backend imports
│   ├── Dockerfile                    # UI container build
│   ├── Introduction.py               # Homepage
│   ├── utils/
│   │   ├── eda.py
│   │   ├── plots.py
│   │   └── api_client.py             # NEW: shared HTTP client for FastAPI
│   └── pages/
│       ├── Dashboard.py              # Reads results via GET /results
│       └── New_Prediction.py         # Triggers inference via POST /predict
│
├── docs/
│   ├── ARCHITECTURE.md               # Target-state architecture (authoritative)
│   └── REFACTOR_PLAN.md              # This file — execution playbook
│
├── .env
├── .env.example
├── .gitignore
├── .dockerignore
├── pyproject.toml                    # uv-compatible, ruff configured
├── uv.lock
├── Makefile
└── docker-compose.yml
```

---

## Phase 0 — Preparation

> Prerequisite steps before touching any code.

### Step 0.1 — Create a Git Branch

```bash
git checkout -b refactor/production-layout
```

Never perform structural changes on `main`. This branch will be merged via PR after all phases pass.

### Step 0.2 — Verify Current State (Baseline)

```bash
poetry run python db_init.py
poetry run python infer.py
poetry run uvicorn app:app --reload
```

Document the baseline behavior. Note any errors that already exist.

### Step 0.3 — Migrate from Poetry to uv (P16)

**uv** is a drop-in, Rust-based Python package manager that is 10–100× faster than pip/Poetry, supports `pyproject.toml` natively, and generates a single `uv.lock` file.

#### Install uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Update `pyproject.toml`

Remove `[tool.poetry.*]` sections. uv uses the standard `[project]` table:

```toml
[project]
name = "lersha-credit-api"
version = "0.1.0"
description = "Agricultural credit scoring for Ethiopian smallholder farmers"
requires-python = ">=3.12, <3.14"

dependencies = [
    # ... same list as before ...
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "pytest-asyncio",
    "httpx",
    "pytest-mock",
    "ruff",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Ruff replaces flake8, isort, and black
[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"backend/tests/*" = ["S101", "F401"]

[tool.pytest.ini_options]
testpaths = ["backend/tests"]
pythonpath = ["."]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["backend"]
omit = ["backend/tests/*", "backend/scripts/*"]
```

#### Generate lockfile and clean up

```bash
uv sync
uv sync --extra dev
del poetry.lock   # Windows
```

Add `poetry.lock` to `.gitignore`. Add `uv.lock` to version control.

#### Verify

```bash
uv run python -c "import fastapi; print('FastAPI OK')"
uv run uvicorn app:app --reload
```

---

## Phase 1 — Structural Reorganization

> Move files into the new layout. No logic changes — only file moves and import path updates.

### Step 1.1 — Create the New Directory Skeleton

```bash
mkdir -p backend/api/routers
mkdir -p backend/core
mkdir -p backend/services
mkdir -p backend/chat
mkdir -p backend/config
mkdir -p backend/logger
mkdir -p backend/scripts
mkdir -p backend/tests/unit
mkdir -p backend/tests/integration
mkdir -p backend/models
mkdir -p backend/data
mkdir -p backend/prompts
mkdir -p ui/pages
mkdir -p ui/utils
```

Create `__init__.py` in every `backend/` package directory.

### Step 1.2 — File Move Manifest

| Source (current) | Destination (new) | Notes |
|---|---|---|
| `config/config.py` | `backend/config/config.py` | Update `BASE_DIR` — see Step 1.7 |
| `src/logger.py` | `backend/logger/logger.py` | No logic change |
| `services/db_model.py` | `backend/services/db_model.py` | No logic change |
| `services/schema.py` | `backend/services/schema.py` | No logic change |
| `services/db_utils.py` | `backend/services/db_utils.py` | SQL fix applied in Phase 2 |
| `chat/rag_engine.py` | `backend/chat/rag_engine.py` | RAG re-wired in Phase 2 |
| `src/infer_utils.py` | `backend/core/infer_utils.py` | Dead code removed in Phase 2 |
| `src/inference_pipeline.py` | `backend/core/pipeline.py` | Feature engineering extracted in Step 1.3 |
| `src/utils.py` | `backend/core/data_utils.py` | Renamed for clarity |
| `logic/smote_updated.py` | `backend/core/preprocessing.py` | Merged with OHE helpers from infer_utils |
| `app.py` | `backend/main.py` | Refactored via app factory pattern in Step 1.4 |
| `infer.py` | `backend/scripts/run_inference.py` | Renamed for clarity |
| `db_init.py` | `backend/scripts/db_init.py` | No logic change |
| `models/` | `backend/models/` | All .pkl files |
| `data/` | `backend/data/` | All CSVs |
| `prompts/` | `backend/prompts/` | prompts.yaml |
| `Introduction.py` | `ui/Introduction.py` | No logic change |
| `pages/Dashboard.py` | `ui/pages/Dashboard.py` | HTTP client swap in Phase 3 |
| `pages/New_Prediction.py` | `ui/pages/New_Prediction.py` | HTTP client swap in Phase 3 |
| `utils/eda.py` | `ui/utils/eda.py` | Streamlit-specific, stays in UI layer |
| `utils/plots.py` | `ui/utils/plots.py` | Streamlit-specific, stays in UI layer |

### Step 1.3 — Extract `feature_engineering.py`

Extract `apply_feature_engineering()` from `src/infer_utils.py` into `backend/core/feature_engineering.py`.

This function is pure data transformation — no model artifacts, SHAP, or MLflow dependency. Isolating it:
- Makes unit testing trivial
- Eliminates importing the entire ML harness just for data prep
- Creates a single findable place for all feature derivation logic

After extraction, `backend/core/infer_utils.py` imports from `backend/core/feature_engineering.py`.

Move `replace_inf()` from `logic/smote_updated.py` and `preprocessing_categorical_features()` from `infer_utils.py` into `backend/core/preprocessing.py`.

### Step 1.4 — Refactor `backend/main.py` Using the App Factory Pattern

```python
# backend/main.py
from fastapi import FastAPI
from backend.api.routers import predict, results, health

def create_app() -> FastAPI:
    app = FastAPI(
        title="Lersha Credit Scoring API",
        version="1.0.0",
        description="Agricultural credit evaluation for Ethiopian smallholder farmers",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, prefix="/v1/predict", tags=["v1 — Inference"])
    app.include_router(results.router, prefix="/v1/results", tags=["v1 — Results"])
    return app

app = create_app()
```

> Note the `/v1/` prefix — all routes are versioned from the start. See `docs/ARCHITECTURE.md` §16.6.

### Step 1.5 — Split Routers

- `backend/api/routers/health.py` — `GET /` and `GET /health` (real dependency ping — see ARCHITECTURE.md §16.16)
- `backend/api/routers/predict.py` — `POST /v1/predict` and `GET /v1/predict/{job_id}`
- `backend/api/routers/results.py` — `GET /v1/results`

Move `UserData` request model to `backend/api/schemas.py` and expand with explicit response models.

### Step 1.6 — Update All Import Paths

| Old Import | New Import |
|---|---|
| `from config.config import config` | `from backend.config.config import config` |
| `from src.logger import get_logger` | `from backend.logger.logger import get_logger` |
| `from src.inference_pipeline import ...` | `from backend.core.pipeline import ...` |
| `from src.infer_utils import ...` | `from backend.core.infer_utils import ...` |
| `from services.db_utils import ...` | `from backend.services.db_utils import ...` |
| `from services.schema import ...` | `from backend.services.schema import ...` |
| `from services.db_model import ...` | `from backend.services.db_model import ...` |
| `from chat.rag_engine import ...` | `from backend.chat.rag_engine import ...` |
| `from utils.eda import ...` | `from ui.utils.eda import ...` |
| `from utils.plots import ...` | `from ui.utils.plots import ...` |

### Step 1.7 — Fix `BASE_DIR` in Config

```python
# backend/config/config.py
# Before (1 level from config/)
BASE_DIR = Path(__file__).resolve().parents[1]

# After (now 2 levels from backend/config/)
BASE_DIR = Path(__file__).resolve().parents[2]
```

`BASE_DIR` still resolves to the project root where `backend/models/`, `backend/data/`, `backend/prompts/` are referenced from.

> **Startup assertion:** Add `assert Path(config.xgb_model_36).exists()` at startup to catch a wrong `BASE_DIR` immediately.

### Step 1.8 — Update `pyproject.toml` Entry Points

```toml
[project.scripts]
serve-api = "backend.main:app"
```

All commands migrate from `poetry run` to `uv run`:

```bash
# FastAPI
uv run uvicorn backend.main:app --reload

# Streamlit
uv run streamlit run ui/Introduction.py

# DB init
uv run python backend/scripts/db_init.py

# ChromaDB population
uv run python backend/scripts/populate_chroma.py

# MLflow UI
uv run mlflow ui --backend-store-uri mlruns
```

### Phase 1 Verification

```bash
uv run python -c "from backend.config.config import config; print(config.db_uri)"
uv run python -c "from backend.core.pipeline import match_inputs"
uv run python -c "from backend.core.feature_engineering import apply_feature_engineering"
uv run uvicorn backend.main:app --reload  # must start on :8000
uv run streamlit run ui/Introduction.py  # must start on :8501
```

---

## Phase 2 — Fix Existing Bugs

> Apply all P3–P6, P12–P13 fixes with fresh import paths in place.

### Step 2.1 — Fix API Response Field Names (P3)

**File:** `backend/api/routers/predict.py`

```python
# Before
result_18, result_44, result_featured = infer(...)
return {"result_18": result_18, "result_44": result_44, "result_featured": result_featured}

# After
result_xgboost, result_random_forest = infer(...)
return {"result_xgboost": result_xgboost, "result_random_forest": result_random_forest}
```

### Step 2.2 — Fix Config Env Var Bug (P5)

**File:** `backend/config/config.py`

```python
# Before — all three read XGB_MODEL_36
self.rf_model_36  = os.getenv("XGB_MODEL_36", ...)
self.cab_model_36 = os.getenv("XGB_MODEL_36", ...)

# After — each has its own env var
self.xgb_model_36 = os.getenv("XGB_MODEL_36", BASE_DIR / "backend/models/xgboost_36_credit_score.pkl")
self.rf_model_36  = os.getenv("RF_MODEL_36",  BASE_DIR / "backend/models/random_forest_36_credit_score.pkl")
self.cab_model_36 = os.getenv("CAB_MODEL_36", BASE_DIR / "backend/models/catboost_36_credit_score.pkl")
```

Add to `.env.example`:
```bash
XGB_MODEL_36=backend/models/xgboost_36_credit_score.pkl
RF_MODEL_36=backend/models/random_forest_36_credit_score.pkl
CAB_MODEL_36=backend/models/catboost_36_credit_score.pkl
```

### Step 2.3 — Fix Full Table Scan Queries (P6)

**File:** `backend/services/db_utils.py`

```python
# fetch_raw_data — Before
query = text(f"SELECT * FROM {table_name}")
df = pd.read_sql(query, con=engine)
df = df[df["farmer_uid"] == filters]

# After
query = text(f"SELECT * FROM {table_name} WHERE farmer_uid = :uid")
df = pd.read_sql(query, con=engine, params={"uid": filters})

# fetch_multiple_raw_data — Before
query = text(f"SELECT * FROM {table_name}")
df = pd.read_sql(query, con=engine)
sample_df = df.sample(n=n_rows, replace=False)

# After
query = text(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT :n")
df = pd.read_sql(query, con=engine, params={"n": n_rows})
```

### Step 2.4 — Re-enable the RAG Engine (P4)

**File:** `backend/core/pipeline.py`

```python
# Before (commented out)
# from backend.chat.rag_engine import get_rag_explanation
# rag_explanation = get_rag_explanation(prediction_class_name, shap_dict)
"rag_explanation": "rag_explanation",  # placeholder

# After
from backend.chat.rag_engine import get_rag_explanation
rag_explanation = get_rag_explanation(prediction_class_name, shap_dict)
"rag_explanation": rag_explanation,
```

Switch ChromaDB to persistent client in `backend/chat/rag_engine.py`:
```python
import chromadb
client = chromadb.PersistentClient(path=str(config.chroma_db_path))
```

Add `CHROMA_DB_PATH=chroma_db/` to config and `.env.example`.

### Step 2.5 — Create ChromaDB Population Script (P13)

**New file:** `backend/scripts/populate_chroma.py`

Structure:
1. Define feature definitions — one entry per feature in `columns_36`
2. Embed each definition using the configured SentenceTransformer model
3. Upsert into ChromaDB collection `credit_features`

### Step 2.6 — Remove Dead Code (P12)

**File:** `backend/core/infer_utils.py`

- Delete `generate_shap_value_summary_plotsss` (triple-`s` duplicate)
- Delete `load_prediction_model()` (singular) — references unused `_34_` model variants

### Phase 2 Verification

```bash
uv run python -c "
from backend.config.config import config
assert 'random_forest' in str(config.rf_model_36), 'rf_model_36 points at wrong file'
assert 'catboost' in str(config.cab_model_36), 'cab_model_36 points at wrong file'
print('Config env var bug: FIXED')
"

uv run python backend/scripts/populate_chroma.py

# rag_explanation must not be the literal string 'rag_explanation'
uv run python backend/scripts/run_inference.py
```

---

## Phase 3 — Production Hardening

> API-first Streamlit, authentication, async inference, and shared HTTP client.

### Step 3.1 — Make Streamlit API-First (P2)

**New file:** `ui/utils/api_client.py`

```python
import os
import requests
from requests import Session

class LershaAPIClient:
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.session = Session()
        self.session.headers.update({
            "X-API-Key": os.getenv("API_KEY", ""),
            "Content-Type": "application/json",
        })

    def submit_prediction(self, source: str, farmer_uid=None, number_of_rows=None) -> dict:
        payload = {"source": source, "farmer_uid": farmer_uid, "number_of_rows": number_of_rows}
        resp = self.session.post(f"{self.base_url}/v1/predict", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_prediction_result(self, job_id: str) -> dict:
        resp = self.session.get(f"{self.base_url}/v1/predict/{job_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_results(self, limit: int = 500) -> dict:
        resp = self.session.get(f"{self.base_url}/v1/results", params={"limit": limit}, timeout=30)
        resp.raise_for_status()
        return resp.json()
```

Update `ui/pages/New_Prediction.py`:
```python
# Before
from src.inference_pipeline import match_inputs, run_inferences

# After
from ui.utils.api_client import LershaAPIClient
client = LershaAPIClient()
response = client.submit_prediction(source=source, number_of_rows=number_of_rows)
job_id = response["job_id"]
result = client.get_prediction_result(job_id)
```

Update `ui/pages/Dashboard.py`:
```python
# Before
engine = create_engine(config.db_uri)
df = pd.read_sql(query, conn)

# After
client = LershaAPIClient()
data = client.get_results(limit=500)
df = pd.DataFrame(data["records"])
```

### Step 3.2 — Add API Key Authentication (P8)

**New file:** `backend/api/dependencies.py`

```python
from fastapi import Header, HTTPException, status
from backend.config.config import config

async def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
```

Apply to all non-health routers:
```python
router = APIRouter(dependencies=[Depends(require_api_key)])
```

Add to `backend/config/config.py`:
```python
self.api_key = os.getenv("API_KEY")
if not self.api_key:
    raise ValueError("API_KEY environment variable not set")
```

### Step 3.3 — Async Background Task Inference (P7)

**Endpoint design:**
```
POST /v1/predict          -> 202 Accepted  + {"job_id": "uuid", "status": "accepted"}
GET  /v1/predict/{job_id} -> 200 OK        + full result (when done)
                          -> 202 Accepted  + {"status": "processing"} (while running)
```

Add `inference_jobs` table to `backend/scripts/db_init.py`:
```sql
CREATE TABLE IF NOT EXISTS inference_jobs (
    job_id UUID PRIMARY KEY,
    status VARCHAR(20) DEFAULT 'pending',
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
```

Add job CRUD to `backend/services/db_utils.py`:
- `create_job(job_id)` — inserts pending row
- `update_job_result(job_id, result)` — sets status + result + completed_at
- `update_job_error(job_id, error)` — sets status=failed + error
- `get_job(job_id)` — returns current job state

Router implementation:
```python
@router.post("/", status_code=202)
async def submit_prediction(item: PredictRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    db_utils.create_job(job_id)
    background_tasks.add_task(_run_prediction_background, job_id, item)
    return {"job_id": job_id, "status": "accepted"}
```

### Step 3.4 — Externalize Hyperparameters

**New file:** `backend/config/hyperparams.yaml`

```yaml
inference:
  default_batch_size: 10
  max_batch_size: 100
  shap_max_samples: 100
  rag_top_k: 5

models:
  active:
    - xgboost
    - random_forest
```

Load in `config.py` and expose as `config.hyperparams`.

---

## Phase 4 — Testing Suite

> All critical logic must have coverage before this refactor is production-grade.

### Step 4.1 — Configure pytest

Already in `pyproject.toml` (added in Phase 0):
```toml
[tool.pytest.ini_options]
testpaths = ["backend/tests"]
pythonpath = ["."]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["backend"]
omit = ["backend/tests/*", "backend/scripts/*"]
```

Install dev dependencies:
```bash
uv sync --extra dev
```

### Step 4.2 — Unit Tests

**`backend/tests/unit/test_feature_engineering.py`**
- Asserts `net_income` = total income minus total cost (exact value)
- Asserts `institutional_support_score` equals sum of 4 binary flags
- Asserts every dropped column is absent from output
- Asserts fallback `age_group` binning triggers without error on small DataFrame

**`backend/tests/unit/test_preprocessing.py`**
- Asserts output columns exactly match the fixture `.pkl` feature list
- Asserts columns missing from input are filled with `0`
- Asserts output shape matches the canonical feature list length

**`backend/tests/unit/test_contribution_table.py`**
- CatBoost: list of 2D arrays (one per class)
- XGBoost multiclass: 3D ndarray (samples × features × classes)
- Binary: 2D ndarray
- Asserts length mismatch raises `ValueError`
- Asserts output is sorted descending by absolute SHAP value

### Step 4.3 — Integration Tests

**`backend/tests/conftest.py`** — shared fixtures:
- `test_db_engine`: connects to `test_lersha` PostgreSQL, creates tables, yields, tears down
- `sample_farmer_df`: 3-row synthetic DataFrame matching raw farmer schema
- `api_client`: `httpx.AsyncClient` against a TestClient of `create_app()`

**`backend/tests/integration/test_predict_endpoint.py`**:
- `POST /v1/predict` without `X-API-Key` → HTTP 403
- `POST /v1/predict` with valid key → HTTP 202 + `job_id`
- `GET /v1/predict/{job_id}` → HTTP 200 with `evaluations` after task completes

**`backend/tests/integration/test_db_utils.py`**:
- `fetch_raw_data` returns only the matching `farmer_uid` row
- `fetch_multiple_raw_data` returns exactly `n` rows
- `save_batch_evaluations` inserts correct row count into `candidate_result`

### Step 4.4 — Coverage Target

```bash
uv run pytest backend/tests/ -v --cov=backend --cov-report=term-missing
```

**Target: 80%+ on `backend/core/` and `backend/services/`.**

---

## Phase 5 — Containerization

### Step 5.1 — `backend/Dockerfile`

```dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy entire backend/ (includes models/, data/, prompts/, scripts/)
COPY backend/ ./backend/

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 5.2 — `ui/Dockerfile`

```dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY ui/ ./ui/

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "ui/Introduction.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 5.3 — `docker-compose.yml`

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_DATABASE}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    env_file: .env
    ports:
      - "8000:8000"
    depends_on:
      - postgres
    volumes:
      - ./backend/models:/app/backend/models
      - ./mlruns:/app/mlruns
      - ./output:/app/output
      - ./logs:/app/logs
      - ./chroma_db:/app/chroma_db

  ui:
    build:
      context: .
      dockerfile: ui/Dockerfile
    env_file: .env
    environment:
      API_BASE_URL: http://backend:8000
    ports:
      - "8501:8501"
    depends_on:
      - backend

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow.db

volumes:
  postgres_data:
```

### Step 5.4 — `.dockerignore`

```
__pycache__/
*.pyc
*.pyo
.git/
.env
backend/tests/
*.egg-info/
.mypy_cache/
.pytest_cache/
```

---

## Phase 6 — CI/CD Pipeline, Ruff, and Makefile

### Step 6.1 — GitHub Actions Workflow

**New file:** `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, "refactor/**"]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { version: "latest" }
      - run: uv sync --extra dev
      - run: uv run ruff check backend/ ui/
      - run: uv run ruff format --check backend/ ui/

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
          POSTGRES_DB: test_lersha
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { version: "latest" }
      - run: uv sync --extra dev
      - run: uv run pytest backend/tests/ --cov=backend --cov-fail-under=80
    env:
      DB_URI: postgresql://test_user:test_pass@localhost:5432/test_lersha
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      GEMINI_MODEL: gemini-1.5-flash
      FARMER_DATA_ALL: farmer_data_all
      API_KEY: ci-test-key

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - run: docker build -f backend/Dockerfile -t lersha-backend .
      - run: docker build -f ui/Dockerfile -t lersha-ui .
```

### Step 6.2 — Ruff Configuration

Configured in `pyproject.toml` (added in Phase 0). Replaces three tools:

| Replaced tool | ruff equivalent |
|---|---|
| flake8 | `ruff check` |
| isort | `I` rule set in `ruff check` |
| black / autopep8 | `ruff format` |

```bash
uv run ruff check backend/ ui/           # lint
uv run ruff check --fix backend/ ui/     # auto-fix
uv run ruff format backend/ ui/          # format
uv run ruff format --check backend/ ui/  # CI mode
```

No `.flake8` file needed — all config is in `[tool.ruff]` in `pyproject.toml`.

### Step 6.3 — Makefile

**New file:** `Makefile` at project root.

```makefile
# ============================================================
# Lersha Credit Scoring System — Developer Makefile
# ============================================================
.PHONY: help install lint format check-format ci-quality test coverage \
        api ui mlflow setup-db setup-chroma \
        docker-up docker-down docker-build clean

help:
	@echo ""
	@echo "  Lersha Credit Scoring — Available Commands"
	@echo "  ============================================"
	@echo ""
	@echo "  Setup"
	@echo "    make install        Install all dependencies (uv sync --extra dev)"
	@echo "    make setup-db       Initialise database tables"
	@echo "    make setup-chroma   Populate ChromaDB with feature definitions"
	@echo ""
	@echo "  Code Quality"
	@echo "    make lint           Run ruff linter"
	@echo "    make format         Auto-format code with ruff"
	@echo "    make check-format   Check formatting without writing (CI mode)"
	@echo ""
	@echo "  Testing"
	@echo "    make test           Run full pytest suite"
	@echo "    make coverage       Run tests with HTML coverage report"
	@echo ""
	@echo "  Run Locally"
	@echo "    make api            Start FastAPI backend (port 8000, hot reload)"
	@echo "    make ui             Start Streamlit UI (port 8501)"
	@echo "    make mlflow         Start MLflow tracking UI (port 5000)"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-build   Build all Docker images"
	@echo "    make docker-up      Start all services via docker-compose"
	@echo "    make docker-down    Stop and remove docker-compose services"
	@echo ""
	@echo "  Maintenance"
	@echo "    make clean          Remove __pycache__, .pytest_cache, htmlcov"
	@echo ""

# Setup
install:
	uv sync --extra dev

setup-db:
	uv run python backend/scripts/db_init.py

setup-chroma:
	uv run python backend/scripts/populate_chroma.py

# Code Quality
lint:
	uv run ruff check backend/ ui/

format:
	uv run ruff format backend/ ui/

check-format:
	uv run ruff format --check backend/ ui/

ci-quality: lint check-format

# Testing
test:
	uv run pytest backend/tests/ -v

coverage:
	uv run pytest backend/tests/ --cov=backend --cov-report=html --cov-report=term-missing
	@echo "HTML report: htmlcov/index.html"

# Run Locally
api:
	uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

ui:
	uv run streamlit run ui/Introduction.py --server.port 8501

mlflow:
	uv run mlflow ui --backend-store-uri mlruns --port 5000

# Docker
docker-build:
	docker build -f backend/Dockerfile -t lersha-backend .
	docker build -f ui/Dockerfile -t lersha-ui .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

# Maintenance
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete."
```

> **Windows users:** Install GNU make via `winget install GnuWin32.Make` or `choco install make`.

---

## Phase 7 — Documentation and Finalization

### Step 7.1 — Update `README.md`

Rewrite with the following sections:
- Prerequisites (Python 3.12, **uv**, **GNU make**, PostgreSQL, Docker)
- Installation (`make install`)
- Environment setup (copy `.env.example` → `.env`)
- One-time setup (`make setup-db && make setup-chroma`)
- Running locally: `make api`, `make ui`, `make mlflow`
- Running with Docker: `make docker-up`
- Linting and formatting: `make lint`, `make format`
- Running tests: `make test`, `make coverage`
- Quick-reference table of all `make` targets

### Step 7.2 — Verify `docs/ARCHITECTURE.md` is Up to Date

`docs/ARCHITECTURE.md` is the authoritative specification. After Phase 7, verify it accurately reflects the implemented system. Specifically confirm:
- Module paths match the implemented `backend/` structure
- API routes include the `/v1/` prefix
- Section 16 production gaps are flagged as future work items

### Step 7.3 — Delete Obsolete Directories

After all phases are complete and the full verification suite passes:

```bash
git rm -r src/ services/ chat/ config/ utils/ pages/ logic/ models/ data/ prompts/
git add .
git commit -m "chore: remove relocated legacy directories"
```

> **Warning:** Only run this after Phase 1 verification passes and `make test` is clean.

---

## Execution Order and Checklist

```
[ ] Phase 0 — Preparation
      [ ] 0.1  Git branch created: refactor/production-layout
      [ ] 0.2  Baseline run verified (infer.py and uvicorn both work)
      [ ] 0.3  uv installed (astral.sh/uv)
      [ ] 0.3  pyproject.toml updated: [tool.poetry] removed, hatchling build backend, dev extras, ruff config
      [ ] 0.3  testpaths = ["backend/tests"] configured in pyproject.toml
      [ ] 0.3  uv sync run successfully, uv.lock generated
      [ ] 0.3  poetry.lock deleted and added to .gitignore
      [ ] 0.3  Verification: uv run uvicorn app:app --reload starts cleanly

[ ] Phase 1 — Structural Reorganization
      [ ] 1.1  Directory skeleton created (backend/ + ui/ only at top level)
      [ ] 1.2  All files moved per manifest (including models/, data/, prompts/ → backend/)
      [ ] 1.3  feature_engineering.py and preprocessing.py extracted
      [ ] 1.4  backend/main.py app factory with /v1/ route prefixes
      [ ] 1.5  Routers split across health.py, predict.py, results.py
      [ ] 1.6  All import paths updated across entire codebase
      [ ] 1.7  BASE_DIR corrected to parents[2] + startup assertion added
      [ ] 1.8  pyproject.toml [project.scripts] updated, all poetry run → uv run
      [ ] Verification: uv run uvicorn backend.main:app --reload starts cleanly
      [ ] Verification: uv run streamlit run ui/Introduction.py starts cleanly

[ ] Phase 2 — Bug Fixes
      [ ] 2.1  API response field names corrected (result_xgboost, result_random_forest)
      [ ] 2.2  Config env var bug fixed (rf_model_36, cab_model_36 use distinct vars)
      [ ] 2.3  Full table scan queries replaced with parameterized WHERE / LIMIT
      [ ] 2.4  RAG engine call uncommented in pipeline.py
      [ ] 2.5  backend/scripts/populate_chroma.py created and run successfully
      [ ] 2.5  chroma_db/ path added to config, PersistentClient used
      [ ] 2.6  Dead code removed (triple-s duplicate, _34_ model references)
      [ ] Verification: inference run produces a real RAG explanation string
      [ ] Verification: .env.example updated with 3 new model path vars + CHROMA_DB_PATH

[ ] Phase 3 — Production Hardening
      [ ] 3.1  ui/utils/api_client.py created (uses /v1/ routes)
      [ ] 3.1  ui/pages/New_Prediction.py uses HTTP client only — no backend imports
      [ ] 3.1  ui/pages/Dashboard.py uses GET /v1/results — no direct DB calls
      [ ] 3.2  backend/api/dependencies.py created with require_api_key
      [ ] 3.2  All non-health routers require X-API-Key header
      [ ] 3.2  API_KEY added to config.py and .env.example
      [ ] 3.3  inference_jobs table added to backend/scripts/db_init.py
      [ ] 3.3  POST /v1/predict returns 202 + job_id
      [ ] 3.3  GET /v1/predict/{job_id} returns result or processing status
      [ ] 3.4  backend/config/hyperparams.yaml created and loaded in config
      [ ] Verification: POST /v1/predict without key returns 403
      [ ] Verification: POST /v1/predict returns 202, job polling returns result

[ ] Phase 4 — Testing
      [ ] 4.1  pytest and coverage configured in pyproject.toml (testpaths = backend/tests)
      [ ] 4.1  uv sync --extra dev installs all test deps
      [ ] 4.2  Unit tests: feature_engineering, preprocessing, contribution_table
      [ ] 4.3  Integration tests: predict endpoint, db_utils
      [ ] 4.4  make test passes, make coverage shows >= 80% on backend/core/ and backend/services/

[ ] Phase 5 — Containerization
      [ ] 5.1  backend/Dockerfile uses ghcr.io/astral-sh/uv:latest and uv sync --frozen
      [ ] 5.1  COPY backend/ ./backend/ (includes models, data, prompts, scripts)
      [ ] 5.1  Docker image builds and starts on :8000
      [ ] 5.2  ui/Dockerfile uses ghcr.io/astral-sh/uv:latest and uv sync --frozen
      [ ] 5.2  Docker image builds and starts on :8501
      [ ] 5.3  docker-compose.yml uses backend/Dockerfile and ui/Dockerfile build contexts
      [ ] 5.3  docker-compose up starts all 4 services (postgres, backend, ui, mlflow)
      [ ] 5.4  .dockerignore excludes backend/tests/ instead of tests/

[ ] Phase 6 — CI/CD, Ruff, Makefile
      [ ] 6.1  .github/workflows/ci.yml uses astral-sh/setup-uv@v5
      [ ] 6.1  CI lint: uv run ruff check backend/ ui/ passes
      [ ] 6.1  CI format: uv run ruff format --check backend/ ui/ passes
      [ ] 6.1  CI test: uv run pytest backend/tests/ --cov-fail-under=80 passes
      [ ] 6.1  CI build: docker build -f backend/Dockerfile and ui/Dockerfile pass
      [ ] 6.2  [tool.ruff] config in pyproject.toml — no .flake8 file
      [ ] 6.3  Makefile at project root with all targets
      [ ] 6.3  make help displays all commands
      [ ] 6.3  Windows: GNU make verified (winget install GnuWin32.Make)

[ ] Phase 7 — Finalization
      [ ] 7.1  README.md rewritten with make commands
      [ ] 7.2  docs/ARCHITECTURE.md verified as up to date
      [ ] 7.3  Legacy directories deleted: src/ services/ chat/ config/ utils/ pages/ logic/ models/ data/ prompts/
      [ ] PR reviewed and merged to main
```

---

## Phase 8 — Production Hardening

> Implements all requirements from `docs/ARCHITECTURE.md §16`. These steps transform the application from a working service into a production-grade system.

### Step 8.1 — Database Migrations (Alembic)

Replace raw DDL in `backend/scripts/db_init.py` with versioned Alembic migrations.

```bash
# Install Alembic
# Add alembic to [project.dependencies] in pyproject.toml, then:
uv sync

# Initialise Alembic inside backend/
cd backend && uv run alembic init alembic
```

Configure `backend/alembic/env.py` to point at `config.db_uri` and import `backend.services.db_model`.

Generate the initial migration from the existing ORM models:
```bash
uv run alembic revision --autogenerate -m "initial_schema"
uv run alembic upgrade head
```

Add to `backend/Dockerfile` CMD (run migrations before start):
```dockerfile
CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
```

All future schema changes (new columns, renamed fields) are added as new Alembic revision files — never by editing `db_init.py` directly.

### Step 8.2 — Persistent Job Queue (Celery + Redis)

Replace FastAPI `BackgroundTasks` with Celery so inference jobs survive server restarts.

Add to `pyproject.toml` dependencies:
```toml
"celery>=5.3",
"redis>=5.0",
```

**New file:** `backend/worker.py`
```python
from celery import Celery
from backend.config.config import config

celery_app = Celery(
    "lersha",
    broker=config.redis_url,
    backend=config.redis_url,
)

@celery_app.task(name="run_inference")
def run_inference_task(job_id: str, payload: dict):
    from backend.core import pipeline
    from backend.services import db_utils
    try:
        db_utils.update_job_status(job_id, "processing")
        result = pipeline.run_full_pipeline(payload)
        db_utils.update_job_result(job_id, result)
    except Exception as e:
        db_utils.update_job_error(job_id, str(e))
```

Update `backend/api/routers/predict.py`:
```python
from backend.worker import run_inference_task

@router.post("/", status_code=202)
async def submit_prediction(item: PredictRequest):
    job_id = str(uuid.uuid4())
    db_utils.create_job(job_id)
    run_inference_task.delay(job_id, item.dict())  # Celery, not BackgroundTasks
    return {"job_id": job_id, "status": "accepted"}
```

Add to `docker-compose.yml`:
```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"

worker:
  build:
    context: .
    dockerfile: backend/Dockerfile
  command: uv run celery -A backend.worker worker --loglevel=info
  env_file: .env
  depends_on:
    - redis
    - postgres
  volumes:
    - ./backend/models:/app/backend/models
    - ./chroma_db:/app/chroma_db
    - ./mlruns:/app/mlruns
```

Add `REDIS_URL=redis://redis:6379/0` to `.env.example` and `config.py`.

### Step 8.3 — Rate Limiting (slowapi)

Add to `pyproject.toml`: `"slowapi>=0.1.9"`

**File:** `backend/api/dependencies.py` — add limiter:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
```

**File:** `backend/main.py` — register limiter:
```python
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from backend.api.dependencies import limiter

def create_app() -> FastAPI:
    app = FastAPI(...)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    ...
```

**File:** `backend/api/routers/predict.py`:
```python
from backend.api.dependencies import limiter

@router.post("/")
@limiter.limit("10/minute")
async def submit_prediction(request: Request, item: PredictRequest, ...):
    ...
```

### Step 8.4 — HTTPS / TLS Termination (Caddy)

**New file:** `Caddyfile` at project root:
```caddy
your-domain.com {
    reverse_proxy /v1/* backend:8000
    reverse_proxy /* ui:8501
}
```

Add to `docker-compose.prod.yml`:
```yaml
caddy:
  image: caddy:2-alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./Caddyfile:/etc/caddy/Caddyfile
    - caddy_data:/data
    - caddy_config:/config
  depends_on:
    - backend
    - ui

volumes:
  caddy_data:
  caddy_config:
```

Caddy auto-provisions Let's Encrypt TLS certificates. For cloud deployments without a domain, use Nginx with a self-signed cert or cloud load balancer TLS termination.

### Step 8.5 — Multi-Worker Uvicorn (Gunicorn)

Add to `pyproject.toml`: `"gunicorn>=21.2"`

Split `backend/Dockerfile` CMD by environment:
```dockerfile
# Dev (keep --reload in Makefile `make api`)
# Prod CMD:
CMD ["sh", "-c", \
  "uv run alembic upgrade head && \
   uv run gunicorn backend.main:app \
   --worker-class uvicorn.workers.UvicornWorker \
   --workers 4 \
   --bind 0.0.0.0:8000 \
   --timeout 120"]
```

### Step 8.6 — API Versioning (already in Phase 1)

All routes use `/v1/` prefix — implemented in `create_app()` in Phase 1.4. When a breaking change is needed, add a `/v2/` router in parallel without removing `/v1/`.

### Step 8.7 — MLflow Production Backend

Update `docker-compose.prod.yml`:
```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  command: >
    mlflow server
    --backend-store-uri postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/mlflow
    --default-artifact-root s3://${MLFLOW_S3_BUCKET}/artifacts
    --host 0.0.0.0
    --port 5000
  environment:
    AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
    AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
  ports:
    - "5000:5000"
```

Add to `.env.example`:
```bash
MLFLOW_S3_BUCKET=lersha-mlflow-artifacts
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

For GCS: use `gs://bucket/artifacts` and `GOOGLE_APPLICATION_CREDENTIALS`.

### Step 8.8 — SQLAlchemy Connection Pool

**File:** `backend/services/db_utils.py`
```python
engine = create_engine(
    config.db_uri,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,      # test connections before use
    pool_recycle=3600,       # recycle stale connections after 1 hour
)
```

### Step 8.9 — Retry Logic (tenacity)

Add to `pyproject.toml`: `"tenacity>=8.2"`

**File:** `backend/chat/rag_engine.py`
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def get_rag_explanation(prediction: str, shap_dict: dict) -> str:
    ...
```

Also wrap `GeminiClient.models.generate_content` calls with the same decorator.

In `ui/utils/api_client.py` add explicit timeouts:
```python
self.session.request = functools.partial(
    self.session.request, timeout=(5, 60)  # (connect, read)
)
```

### Step 8.10 — Structured JSON Logging

Add to `pyproject.toml`: `"python-json-logger>=2.0"`

**File:** `backend/logger/logger.py`
```python
import logging
from pythonjsonlogger import jsonlogger

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

Every module: `logger = get_logger(__name__)`. Log lines become structured JSON, ingested automatically by Datadog, CloudWatch, GCP Logging, or Loki.

### Step 8.11 — Request ID Middleware

**New file:** `backend/api/middleware.py`
```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

Register in `backend/main.py`:
```python
from backend.api.middleware import RequestIDMiddleware
app.add_middleware(RequestIDMiddleware)
```

Inject `request_id` into log calls: `logger.info("Inference complete", extra={"request_id": request.state.request_id})`

### Step 8.12 — Pre-commit Hooks

**New file:** `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
```

Install once per developer:
```bash
uv run pre-commit install
```

Add `make pre-commit` target to Makefile:
```makefile
pre-commit:
	uv run pre-commit run --all-files
```

### Step 8.13 — Secrets Management

For Docker Compose (self-hosted):
```yaml
# docker-compose.prod.yml
secrets:
  api_key:
    file: ./secrets/api_key.txt
  gemini_api_key:
    file: ./secrets/gemini_api_key.txt

services:
  backend:
    secrets:
      - api_key
      - gemini_api_key
```

Update `backend/config/config.py` to read from Docker Secrets file path if env var is absent:
```python
def _read_secret(name: str, env_var: str) -> str:
    secret_path = Path(f"/run/secrets/{name}")
    if secret_path.exists():
        return secret_path.read_text().strip()
    return os.getenv(env_var, "")

self.api_key = _read_secret("api_key", "API_KEY")
```

For cloud deployments: inject secrets via AWS Secrets Manager, GCP Secret Manager, or Kubernetes Secrets at the infrastructure level — `config.py` reads from env vars regardless.

### Step 8.14 — Environment Separation

**Split compose files:**

`docker-compose.yml` — base definitions (no env values, no ports exposed externally)

`docker-compose.override.yml` — dev overrides (bind mounts, hot reload, exposed ports):
```yaml
services:
  backend:
    command: uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
    volumes:
      - ./backend:/app/backend
```

`docker-compose.prod.yml` — production overrides (Gunicorn, Caddy, no bind mounts, MLflow PG):
```yaml
services:
  backend:
    command: >-
      sh -c "uv run alembic upgrade head &&
             uv run gunicorn backend.main:app
             --worker-class uvicorn.workers.UvicornWorker
             --workers 4 --bind 0.0.0.0:8000"
```

Run production: `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d`

### Step 8.15 — Static Type Checking (mypy)

Add to `[project.optional-dependencies] dev`: `"mypy>=1.9"`

Add to `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.12"
strict = false
ignore_missing_imports = true
check_untyped_defs = true
exclude = ["backend/tests", "backend/scripts"]
```

Add to `.github/workflows/ci.yml` lint job:
```yaml
- run: uv run mypy backend/
```

Add Makefile target:
```makefile
typecheck:
	uv run mypy backend/
```

### Step 8.16 — Live Health Check

**File:** `backend/api/routers/health.py` — replace static response:
```python
from fastapi.responses import JSONResponse
from sqlalchemy import text
from backend.services.db_utils import engine
from backend.chat.rag_engine import chroma_client

@router.get("/health")
async def health():
    checks = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception:
        checks["db"] = "unreachable"

    try:
        chroma_client.heartbeat()
        checks["chroma"] = "ok"
    except Exception:
        checks["chroma"] = "unreachable"

    status_code = 200 if all(v == "ok" for v in checks.values()) else 503
    return JSONResponse(content=checks, status_code=status_code)
```

Add to `docker-compose.yml` backend service:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Step 8.17 — Model Versioning (MLflow Model Registry)

At training time, register the model:
```python
import mlflow
mlflow.sklearn.log_model(
    pipeline,
    artifact_path="model",
    registered_model_name="lersha-xgboost",
)
```

Update `backend/core/infer_utils.py` to load from registry by stage:
```python
import mlflow.sklearn

def load_prediction_models(model_name: str):
    registry_uri = f"models:/lersha-{model_name}/Production"
    return mlflow.sklearn.load_model(registry_uri)
```

Promotion from `Staging` → `Production` is done explicitly via the MLflow UI (`http://localhost:5000`), giving full audit trail and instant rollback by promoting the previous version back to `Production`.

### Step 8.18 — PostgreSQL Backups

Add to `docker-compose.prod.yml`:
```yaml
backup:
  image: prodrigestivill/postgres-backup-local
  environment:
    POSTGRES_HOST: postgres
    POSTGRES_DB: ${DB_DATABASE}
    POSTGRES_USER: ${DB_USER}
    POSTGRES_PASSWORD: ${DB_PASSWORD}
    SCHEDULE: "@daily"
    BACKUP_KEEP_DAYS: 30
    BACKUP_KEEP_WEEKS: 4
    BACKUP_KEEP_MONTHS: 6
  volumes:
    - ./backups:/backups
  depends_on:
    - postgres
```

For cloud deployments (RDS, Cloud SQL): enable point-in-time recovery (PITR) in the managed database console — no sidecar needed.

Add `make restore-db` to Makefile:
```makefile
restore-db:
	@read -p "Backup file path: " f; \
	 psql $$(uv run python -c "from backend.config.config import config; print(config.db_uri)") < $$f
```

### Phase 8 Verification

```bash
# 8.1 Alembic
uv run alembic upgrade head                        # must complete with no errors

# 8.2 Celery
docker-compose up worker redis -d
uv run celery -A backend.worker inspect active     # must show worker online

# 8.3 Rate limiting
curl -X POST http://localhost:8000/v1/predict ...   # 11th request must return 429

# 8.5 Gunicorn
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up backend
curl http://localhost:8000/health                   # must return {"db": "ok", "chroma": "ok"}

# 8.10 JSON logs
docker-compose logs backend | python -m json.tool | head -20  # must parse as JSON

# 8.12 Pre-commit
uv run pre-commit run --all-files                  # must pass with no errors

# 8.15 mypy
uv run mypy backend/                               # 0 errors

# 8.16 Health check
curl http://localhost:8000/health                   # {"db": "ok", "chroma": "ok"} with 200
```

---

## Execution Order and Checklist

```
[ ] Phase 0 — Preparation
      [ ] 0.1  Git branch created: refactor/production-layout
      [ ] 0.2  Baseline run verified (infer.py and uvicorn both work)
      [ ] 0.3  uv installed (astral.sh/uv)
      [ ] 0.3  pyproject.toml updated: [tool.poetry] removed, hatchling build backend, dev extras, ruff config
      [ ] 0.3  testpaths = ["backend/tests"] configured in pyproject.toml
      [ ] 0.3  uv sync run successfully, uv.lock generated
      [ ] 0.3  poetry.lock deleted and added to .gitignore
      [ ] 0.3  Verification: uv run uvicorn app:app --reload starts cleanly

[ ] Phase 1 — Structural Reorganization
      [ ] 1.1  Directory skeleton created (backend/ + ui/ only at top level)
      [ ] 1.2  All files moved per manifest (including models/, data/, prompts/ → backend/)
      [ ] 1.3  feature_engineering.py and preprocessing.py extracted
      [ ] 1.4  backend/main.py app factory with /v1/ route prefixes
      [ ] 1.5  Routers split across health.py, predict.py, results.py
      [ ] 1.6  All import paths updated across entire codebase
      [ ] 1.7  BASE_DIR corrected to parents[2] + startup assertion added
      [ ] 1.8  pyproject.toml [project.scripts] updated, all poetry run → uv run
      [ ] Verification: uv run uvicorn backend.main:app --reload starts cleanly
      [ ] Verification: uv run streamlit run ui/Introduction.py starts cleanly

[ ] Phase 2 — Bug Fixes
      [ ] 2.1  API response field names corrected (result_xgboost, result_random_forest)
      [ ] 2.2  Config env var bug fixed (rf_model_36, cab_model_36 use distinct vars)
      [ ] 2.3  Full table scan queries replaced with parameterized WHERE / LIMIT
      [ ] 2.4  RAG engine call uncommented in pipeline.py
      [ ] 2.5  backend/scripts/populate_chroma.py created and run successfully
      [ ] 2.5  chroma_db/ path added to config, PersistentClient used
      [ ] 2.6  Dead code removed (triple-s duplicate, _34_ model references)
      [ ] Verification: inference run produces a real RAG explanation string
      [ ] Verification: .env.example updated with 3 new model path vars + CHROMA_DB_PATH

[ ] Phase 3 — Production Hardening (core)
      [ ] 3.1  ui/utils/api_client.py created (uses /v1/ routes)
      [ ] 3.1  ui/pages/New_Prediction.py uses HTTP client only — no backend imports
      [ ] 3.1  ui/pages/Dashboard.py uses GET /v1/results — no direct DB calls
      [ ] 3.2  backend/api/dependencies.py created with require_api_key
      [ ] 3.2  All non-health routers require X-API-Key header
      [ ] 3.2  API_KEY added to config.py and .env.example
      [ ] 3.3  inference_jobs table added to backend/scripts/db_init.py
      [ ] 3.3  POST /v1/predict returns 202 + job_id
      [ ] 3.3  GET /v1/predict/{job_id} returns result or processing status
      [ ] 3.4  backend/config/hyperparams.yaml created and loaded in config
      [ ] Verification: POST /v1/predict without key returns 403
      [ ] Verification: POST /v1/predict returns 202, job polling returns result

[ ] Phase 4 — Testing
      [ ] 4.1  pytest and coverage configured in pyproject.toml (testpaths = backend/tests)
      [ ] 4.1  uv sync --extra dev installs all test deps
      [ ] 4.2  Unit tests: feature_engineering, preprocessing, contribution_table
      [ ] 4.3  Integration tests: predict endpoint, db_utils
      [ ] 4.4  make test passes, make coverage shows >= 80% on backend/core/ and backend/services/

[ ] Phase 5 — Containerization
      [ ] 5.1  backend/Dockerfile uses ghcr.io/astral-sh/uv:latest and uv sync --frozen
      [ ] 5.1  COPY backend/ ./backend/ (includes models, data, prompts, scripts)
      [ ] 5.1  Docker image builds and starts on :8000
      [ ] 5.2  ui/Dockerfile uses ghcr.io/astral-sh/uv:latest and uv sync --frozen
      [ ] 5.2  Docker image builds and starts on :8501
      [ ] 5.3  docker-compose.yml uses backend/Dockerfile and ui/Dockerfile build contexts
      [ ] 5.3  docker-compose up starts all 4 services (postgres, backend, ui, mlflow)
      [ ] 5.4  .dockerignore excludes backend/tests/ instead of tests/

[ ] Phase 6 — CI/CD, Ruff, Makefile
      [ ] 6.1  .github/workflows/ci.yml uses astral-sh/setup-uv@v5
      [ ] 6.1  CI lint: uv run ruff check backend/ ui/ passes
      [ ] 6.1  CI format: uv run ruff format --check backend/ ui/ passes
      [ ] 6.1  CI test: uv run pytest backend/tests/ --cov-fail-under=80 passes
      [ ] 6.1  CI build: docker build -f backend/Dockerfile and ui/Dockerfile pass
      [ ] 6.2  [tool.ruff] config in pyproject.toml — no .flake8 file
      [ ] 6.3  Makefile at project root with all targets
      [ ] 6.3  make help displays all commands
      [ ] 6.3  Windows: GNU make verified (winget install GnuWin32.Make)

[ ] Phase 7 — Finalization
      [ ] 7.1  README.md rewritten with make commands
      [ ] 7.2  docs/ARCHITECTURE.md verified as up to date
      [ ] 7.3  Legacy directories deleted: src/ services/ chat/ config/ utils/ pages/ logic/ models/ data/ prompts/
      [ ] PR reviewed and merged to main

[ ] Phase 8 — Production Hardening (infrastructure)
      [ ] 8.1   Alembic initialised in backend/alembic/, initial migration generated
      [ ] 8.1   backend/Dockerfile CMD runs alembic upgrade head before uvicorn
      [ ] 8.2   celery and redis added to pyproject.toml dependencies
      [ ] 8.2   backend/worker.py created with run_inference_task
      [ ] 8.2   predict.py updated to use run_inference_task.delay() not BackgroundTasks
      [ ] 8.2   redis + worker services added to docker-compose.yml
      [ ] 8.3   slowapi added, limiter registered in main.py, /v1/predict limited to 10/min
      [ ] 8.4   Caddyfile created, caddy service added to docker-compose.prod.yml
      [ ] 8.5   gunicorn added, backend/Dockerfile prod CMD uses 4 uvicorn workers
      [ ] 8.7   docker-compose.prod.yml MLflow uses PostgreSQL + S3 artifact store
      [ ] 8.8   SQLAlchemy engine has pool_size=10, max_overflow=20, pool_pre_ping=True
      [ ] 8.9   tenacity @retry added to get_rag_explanation and Gemini generate_content
      [ ] 8.10  python-json-logger added, get_logger() emits structured JSON
      [ ] 8.11  RequestIDMiddleware created and registered, X-Request-ID in all responses
      [ ] 8.12  .pre-commit-config.yaml created, uv run pre-commit install run
      [ ] 8.13  Docker Secrets or cloud secrets manager used for API_KEY, GEMINI_API_KEY
      [ ] 8.14  docker-compose.override.yml (dev) and docker-compose.prod.yml (prod) created
      [ ] 8.15  mypy added to dev deps, [tool.mypy] in pyproject.toml, CI mypy step passes
      [ ] 8.16  GET /health pings DB + ChromaDB, returns 503 if either unreachable
      [ ] 8.16  Docker healthcheck: points at GET /health
      [ ] 8.17  Models registered in MLflow Model Registry, load_prediction_models() uses registry URI
      [ ] 8.18  postgres-backup-local service in docker-compose.prod.yml, daily backups verified
      [ ] Verification: all Phase 8 verification commands pass
```

| Risk | Mitigation |
|---|---|
| `BASE_DIR` resolves incorrectly after file move | Add startup assertion: `assert Path(config.xgb_model_36).exists()` before any inference |
| Import cycle after `feature_engineering.py` extraction | Keep `feature_engineering.py` dependency-free — no imports from `infer_utils` or `pipeline` |
| Streamlit session state breaks during HTTP migration | Test both Dashboard and New_Prediction pages manually at end of Phase 3 |
| Gemini API rate limit during integration tests | Mock `GeminiClient.models.generate_content` using `pytest-mock` in all test fixtures |
| Random Forest `.pkl` (222 MB) causes slow Docker builds | Layer cache by copying `pyproject.toml` + `uv.lock` before `COPY backend/`; mount models as volume in compose |
| ChromaDB in-memory client loses data on backend restart | Use `chromadb.PersistentClient(path=...)` in Step 2.5; mount `chroma_db/` path in docker-compose |
| Background task fails silently | Wrap task body in try/except; call `db_utils.update_job_error(job_id, str(e))` on failure |
| `CANDIDATE_RESULT` table name env var not set | Default to `"candidate_result"` in config; assert table exists at startup |
| Streamlit UI broken if backend is down after Phase 3 | UI must show a graceful error message when `requests` raises `ConnectionError` |
| Test paths not found by pytest | Confirm `testpaths = ["backend/tests"]` in `pyproject.toml` before running `make test` |
