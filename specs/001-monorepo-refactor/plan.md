# Implementation Plan: Lersha Monorepo Refactor

**Branch**: `001-monorepo-refactor` | **Date**: 2026-03-29 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/001-monorepo-refactor/spec.md`

---

## Summary

Migrate the Lersha Credit Scoring System from a flat legacy layout to a clean `backend/` / `ui/` monorepo, simultaneously migrating the build system from Poetry to uv, extracting two pure-function modules (`feature_engineering.py`, `preprocessing.py`), fixing four confirmed bugs (response field names, config env vars, SQL full-table scans, disabled RAG), and deleting two dead functions. The spec defines 27 functional requirements and 12 measurable success criteria. This plan decomposes that into 8 ordered phases, each independently verifiable.

---

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: FastAPI 0.121, SQLAlchemy 2.x, pandas 2.x, XGBoost 3.x, scikit-learn 1.7, CatBoost 1.2, SHAP 0.50, MLflow 3.6, chromadb, sentence-transformers, google-generativeai, Streamlit (UI), uvicorn, pydantic v2  
**Storage**: PostgreSQL (primary), ChromaDB (vector store, persistent client), MLflow (experiment tracking, SQLite dev / PostgreSQL prod)  
**Testing**: pytest 8+, pytest-asyncio (`asyncio_mode=auto`), httpx, pytest-mock; coverage target ≥ 80% on `backend/core/` and `backend/services/`  
**Target Platform**: Linux server (Docker), developer workstation (Windows/macOS for local dev)  
**Project Type**: Backend REST API (FastAPI) + Streamlit web UI — two-service containerized monorepo  
**Performance Goals**: API responses (non-inference) < 300ms p95; inference jobs async, non-blocking  
**Constraints**: `asyncio_mode="auto"` required throughout; `BASE_DIR` must resolve to project root; no flat-layout imports in any committed file  
**Scale/Scope**: ~3 models, ~36 features, batch sizes up to 100 rows per job, single-region deployment

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status |
|---|---|---|
| **P1 – MODULAR** | `backend/` and `ui/` two-folder layout; each sub-package has a single concern | ✅ Enforced by FR-001–FR-004 |
| **P2 – PEP** | ruff with `line-length=120`, `target-version="py312"`, `select=["E","F","I","UP","B","SIM"]` | ✅ FR-008 |
| **P3 – LOG** | `from backend.logger.logger import get_logger` in every module | ✅ enforced by import path plan; P3 governs all modules |
| **P4 – EXC** | Background task boundary must call `update_job_error()`; external calls wrapped with tenacity | ✅ Phase 3 (Step 3.3); tenacity wrap is Phase 8 (deferred, documented) |
| **P5 – CONFIG** | `BASE_DIR = Path(__file__).resolve().parents[2]`; startup assertion on `xgb_model_36` | ✅ FR-018, FR-019 |
| **P6 – API** | Versioned `/v1/` prefix; Pydantic schemas; `result_xgboost`/`result_random_forest` fields | ✅ FR-022; app factory registers routers with `/v1/` |
| **P7 – TEST** | `testpaths=["backend/tests"]`, `pythonpath=["."]`, `asyncio_mode="auto"` | ✅ FR-009 |
| **P8 – DB** | `WHERE farmer_uid = :uid` parameterized; `ORDER BY RANDOM() LIMIT :n`; Pydantic before ORM | ✅ FR-023, FR-024 |
| **P9 – SEC** | `API_KEY` hard-fail at startup; `.env` gitignored | ✅ Phase 3, Step 3.2 |
| **P10 – OBS** | RAG uncommented; ChromaDB PersistentClient; MLflow tracking retained | ✅ FR-025 |
| **P11 – CONT** | uv replaces Poetry; `uv.lock` committed; Dockerfiles reference `python:3.12-slim` | ✅ FR-005–FR-007 |
| **P12 – CI** | Makefile targets; ruff + pytest in CI pipeline | ✅ Phase 6 |

**Gate result: PASS** — no violations. All 12 principles are addressed within the 8-phase plan.

---

## Project Structure

### Documentation (this feature)

```text
specs/001-monorepo-refactor/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── api-contract.md  # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created by /speckit-plan)
```

### Source Code (repository root — target state)

```text
lersha_credit_api/
├── backend/
│   ├── __init__.py
│   ├── main.py                       # app factory — create_app()
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py           # require_api_key dependency
│   │   ├── schemas.py                # Pydantic I/O models
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── health.py             # GET /health — real DB + ChromaDB ping
│   │       ├── predict.py            # POST /v1/predict, GET /v1/predict/{job_id}
│   │       └── results.py            # GET /v1/results
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py               # match_inputs(), run_inferences()
│   │   ├── infer_utils.py            # load_prediction_models(), SHAP, build_contribution_table()
│   │   ├── feature_engineering.py    # apply_feature_engineering() — extracted, zero ML deps
│   │   └── preprocessing.py          # preprocessing_categorical_features(), replace_inf()
│   ├── services/
│   │   ├── __init__.py
│   │   ├── db_utils.py               # All PostgreSQL CRUD; SQL bugs fixed
│   │   ├── db_model.py               # SQLAlchemy ORM models
│   │   └── schema.py                 # Pydantic validation schemas
│   ├── chat/
│   │   ├── __init__.py
│   │   ├── rag_engine.py             # ChromaDB PersistentClient + Gemini
│   │   └── chroma_loader.py          # renamed from populate_chroma.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py                 # BASE_DIR fixed; RF/CAB env vars fixed; api_key added
│   │   └── hyperparams.yaml          # externalized tuning knobs
│   ├── logger/
│   │   ├── __init__.py
│   │   └── logger.py                 # rotating file + console handler
│   ├── scripts/
│   │   ├── db_init.py
│   │   ├── populate_chroma.py
│   │   └── run_inference.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── unit/
│   │   │   ├── test_feature_engineering.py
│   │   │   ├── test_preprocessing.py
│   │   │   └── test_contribution_table.py
│   │   └── integration/
│   │       ├── test_predict_endpoint.py
│   │       └── test_db_utils.py
│   ├── models/
│   ├── data/
│   └── prompts/
├── ui/
│   ├── Dockerfile
│   ├── Introduction.py
│   ├── utils/
│   │   ├── eda.py
│   │   ├── plots.py
│   │   └── api_client.py             # NEW — all HTTP calls centralized here
│   └── pages/
│       ├── Dashboard.py
│       └── New_Prediction.py
├── .env
├── .env.example
├── .gitignore
├── .dockerignore
├── pyproject.toml
├── uv.lock
├── Makefile
└── docker-compose.yml
```

**Structure Decision**: Two-service monorepo (Option 2 variant fitted to this project). `backend/` holds the FastAPI API server, ML pipeline, and all operational scripts. `ui/` holds the Streamlit application. Both share a single `pyproject.toml` / `uv.lock` at the repository root. The `ui/` container receives no `backend/` Python package — communication is exclusively over HTTP.

---

## Implementation Phases

### Phase 0 — Toolchain Migration (uv + pyproject.toml)

**Goal**: Replace Poetry with uv; set up ruff, pytest, and coverage config. No application code changed.

**Steps**:
1. Remove `[tool.poetry.*]` from `pyproject.toml`. Preserve the existing `[project]` table but convert it to the standard form with `hatchling` build backend.
2. Add `[project.optional-dependencies]` dev group: `pytest>=8.0`, `pytest-cov`, `pytest-asyncio`, `httpx`, `pytest-mock`, `ruff`, `mypy`.
3. Add `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.pytest.ini_options]`, `[tool.coverage.run]` sections as specified in FR-008–FR-010.
4. Run `uv sync`. Verify `uv.lock` is generated.
5. Delete `poetry.lock`. Add `poetry.lock` to `.gitignore` (ensure `uv.lock` is NOT in `.gitignore`).

**Verification**: `uv run python -c "import fastapi; print('OK')"`

---

### Phase 1 — Create Directory Skeleton

**Goal**: Create all `backend/` and `ui/` directories with empty `__init__.py` files. No file moves yet.

**Steps**:
1. Create all directories per the target tree above.
2. Create empty `__init__.py` in every `backend/` package (`api/`, `api/routers/`, `core/`, `services/`, `chat/`, `config/`, `logger/`, `scripts/`, `tests/`, `tests/unit/`, `tests/integration/`).
3. Create empty `__init__.py` in `backend/` root itself.

**Verification**: `python -c "import pathlib; [print(p) for p in pathlib.Path('backend').rglob('__init__.py')]"`

---

### Phase 2 — File Moves + Import Path Updates

**Goal**: Move every legacy file to its new location and update all intra-project import paths. No logic changes in this phase.

**File move manifest** (source → destination):

| Legacy Path | New Path | Note |
|---|---|---|
| `config/config.py` | `backend/config/config.py` | `BASE_DIR` fixed in Phase 3 |
| `src/logger.py` | `backend/logger/logger.py` | no logic change |
| `services/db_model.py` | `backend/services/db_model.py` | no logic change |
| `services/schema.py` | `backend/services/schema.py` | no logic change |
| `services/db_utils.py` | `backend/services/db_utils.py` | SQL fix in Phase 4 |
| `chat/rag_engine.py` | `backend/chat/rag_engine.py` | RAG fix in Phase 4 |
| `src/infer_utils.py` | `backend/core/infer_utils.py` | extraction + dead code in Phase 3 |
| `src/inference_pipeline.py` | `backend/core/pipeline.py` | RAG re-wire in Phase 4 |
| `src/utils.py` | `backend/core/data_utils.py` | no logic change |
| `logic/smote_updated.py` | `backend/core/preprocessing.py` | merge in Phase 3 |
| `app.py` | `backend/main.py` | app factory rewrite in Phase 4 |
| `infer.py` | `backend/scripts/run_inference.py` | no logic change |
| `db_init.py` | `backend/scripts/db_init.py` | no logic change |
| `models/` (dir) | `backend/models/` | binary artifacts, copy whole dir |
| `data/` (dir) | `backend/data/` | CSVs, copy whole dir |
| `prompts/` (dir) | `backend/prompts/` | prompts.yaml, copy whole dir |
| `Introduction.py` | `ui/Introduction.py` | no logic change |
| `pages/Dashboard.py` | `ui/pages/Dashboard.py` | HTTP client swap in Phase 5 |
| `pages/New_Prediction.py` | `ui/pages/New_Prediction.py` | HTTP client swap in Phase 5 |
| `utils/eda.py` | `ui/utils/eda.py` | no logic change |
| `utils/plots.py` | `ui/utils/plots.py` | no logic change |

**Import path rewrites** (global search-and-replace across all moved files):

| Old import | New import |
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

**Verification**: `uv run python -c "from backend.logger.logger import get_logger"` (no ModuleNotFoundError)

---

### Phase 3 — Module Extractions + Dead Code Removal

**Goal**: Extract `apply_feature_engineering()` and preprocessing helpers into their dedicated modules. Delete dead functions.

**Steps**:

**3a — Create `backend/core/feature_engineering.py`**
- Copy `apply_feature_engineering()` verbatim from `backend/core/infer_utils.py` (lines 269–323 in original source)
- Imports needed: `numpy`, `pandas` only — zero ML framework imports
- Add `from backend.logger.logger import get_logger; logger = get_logger(__name__)`
- Add PEP 257 module docstring

**3b — Create / populate `backend/core/preprocessing.py`**
- Copy `preprocessing_categorical_features()` from `backend/core/infer_utils.py`
- Merge `replace_inf()` from `logic/smote_updated.py` (if it exists there)
- Add `load_features()` helper (needed by `preprocessing_categorical_features`)
- All imports: `pandas`, `pickle`, `pathlib`, logger — no ML framework imports

**3c — Update `backend/core/infer_utils.py`**
- Delete `apply_feature_engineering()` definition (moved to feature_engineering.py)
- Delete `preprocessing_categorical_features()` definition (moved to preprocessing.py)
- Delete `generate_shap_value_summary_plotsss` (triple-s dead function) — **FR-026**
- Delete `load_prediction_model()` singular function (references unused `_34_` models) — **FR-027**
- Add imports: `from backend.core.feature_engineering import apply_feature_engineering` and `from backend.core.preprocessing import preprocessing_categorical_features, load_features`

**3d — Fix `BASE_DIR` in `backend/config/config.py`** (FR-018, FR-019)
```python
# Before
BASE_DIR = Path(__file__).resolve().parents[1]
# After
BASE_DIR = Path(__file__).resolve().parents[2]
```
Add startup assertion after `config = Config()`:
```python
assert Path(config.xgb_model_36).exists(), f"XGB model not found at {config.xgb_model_36} — check BASE_DIR"
```

**Verification**:
- `uv run python -c "from backend.core.feature_engineering import apply_feature_engineering; print('OK')"`
- `uv run ruff check backend/core/infer_utils.py` exits 0 (no dead code complaints)

---

### Phase 4 — Bug Fixes

**Goal**: Fix all 4 confirmed bugs from the problem log (P3, P4, P5, P6).

**Bug Fix 4a — Config env var bug (P5, FR-020, FR-021)**  
File: `backend/config/config.py`
```python
# BEFORE (both read XGB_MODEL_36 — confirmed in source lines 35-36)
self.rf_model_36  = os.getenv("XGB_MODEL_36", ...)
self.cab_model_36 = os.getenv("XGB_MODEL_36", ...)

# AFTER
self.xgb_model_36 = os.getenv("XGB_MODEL_36", BASE_DIR / "backend" / "models" / "xgboost_36_credit_score.pkl")
self.rf_model_36  = os.getenv("RF_MODEL_36",  BASE_DIR / "backend" / "models" / "random_forest_36_credit_score.pkl")
self.cab_model_36 = os.getenv("CAB_MODEL_36", BASE_DIR / "backend" / "models" / "catboost_36_credit_score.pkl")
self.chroma_db_path = os.getenv("CHROMA_DB_PATH", BASE_DIR / "chroma_db")
self.api_key = os.getenv("API_KEY")
if not self.api_key:
    raise ValueError("API_KEY environment variable not set")
```
Update `.env.example` to add `RF_MODEL_36`, `CAB_MODEL_36`, `CHROMA_DB_PATH`, `API_KEY`.

**Bug Fix 4b — SQL full-table scans (P6, FR-023, FR-024)**  
File: `backend/services/db_utils.py`
```python
# fetch_raw_data — BEFORE (confirmed: lines 88-90 do full scan + Python filter)
query = text(f"SELECT * FROM {table_name}")
df = pd.read_sql(query, con=engine)
df = df[df["farmer_uid"] == filters]

# AFTER
query = text(f"SELECT * FROM {table_name} WHERE farmer_uid = :uid")
df = pd.read_sql(query, con=engine, params={"uid": filters})

# fetch_multiple_raw_data — BEFORE (confirmed: lines 96-99 do full scan + df.sample())
query = text(f"SELECT * FROM {table_name}")
df = pd.read_sql(query, con=engine)
sample_df = df.sample(n=n_rows, replace=False)

# AFTER
query = text(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT :n")
df = pd.read_sql(query, con=engine, params={"n": n_rows})
```

**Bug Fix 4c — API response field names (P3, FR-022)**  
File: `backend/main.py` → `backend/api/routers/predict.py` (after app factory refactor)
```python
# BEFORE (confirmed: app.py lines 27-32)
result_18, result_44, result_featured = infer(...)
return {"result_18": result_18, "result_44": result_44, "result_featured": result_featured}

# AFTER
result_xgboost, result_random_forest = ...
return {"result_xgboost": result_xgboost, "result_random_forest": result_random_forest}
```
Update `backend/api/schemas.py` Pydantic response model to use these field names.

**Bug Fix 4d — Re-enable RAG (P4, FR-025)**  
File: `backend/core/pipeline.py`
```python
# BEFORE (confirmed: inference_pipeline.py lines 13, 91-92, 96)
# from chat.rag_engine import get_rag_explanation
# rag_explanation = get_rag_explanation(...)
"rag_explanation": "rag_explanation",   # ← literal string placeholder

# AFTER
from backend.chat.rag_engine import get_rag_explanation
rag_explanation = get_rag_explanation(prediction_class_name, shap_dict)
"rag_explanation": rag_explanation,
```

File: `backend/chat/rag_engine.py`
```python
# BEFORE (ephemeral/in-memory client)
import chromadb
client = chromadb.Client()

# AFTER
client = chromadb.PersistentClient(path=str(config.chroma_db_path))
```

**Verification** (Phase 4):
```bash
uv run python -c "
from backend.config.config import config
assert 'random_forest' in str(config.rf_model_36)
assert 'catboost' in str(config.cab_model_36)
print('Config bug: FIXED')
"
# No search hit for full table scan:
# rg "df\.sample\(" backend/
# rg "df\[df\[.farmer_uid.\]" backend/
```

---

### Phase 5 — App Factory + API Authentication + Async Inference

**Goal**: Rewrite `backend/main.py` as an app factory. Add versioned routers, API key auth, and async background-task inference pattern.

**5a — App factory** (`backend/main.py`)
```python
from fastapi import FastAPI
from backend.api.routers import health, predict, results

def create_app() -> FastAPI:
    app = FastAPI(title="Lersha Credit Scoring API", version="1.0.0")
    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, prefix="/v1/predict", tags=["v1 — Inference"])
    app.include_router(results.router, prefix="/v1/results", tags=["v1 — Results"])
    return app

app = create_app()
```

**5b — Authentication** (`backend/api/dependencies.py`)
```python
from fastapi import Header, HTTPException, status
from backend.config.config import config

async def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if x_api_key != config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key")
```
Apply `router = APIRouter(dependencies=[Depends(require_api_key)])` to `predict.py` and `results.py` routers.

**5c — Async inference** (`backend/api/routers/predict.py`)
Add `inference_jobs` table DDL to `backend/scripts/db_init.py`.
Add job CRUD to `backend/services/db_utils.py`:  
`create_job()`, `update_job_result()`, `update_job_error()`, `get_job()`

Router pattern:
```python
@router.post("/", status_code=202)
async def submit_prediction(item: PredictRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    db_utils.create_job(job_id)
    background_tasks.add_task(_run_prediction_background, job_id, item)
    return {"job_id": job_id, "status": "accepted"}
```

Background task error boundary:
```python
async def _run_prediction_background(job_id: str, item: PredictRequest) -> None:
    try:
        result = _run_full_pipeline(item)
        db_utils.update_job_result(job_id, result)
    except Exception as exc:
        logger.error("Job %s failed", job_id, exc_info=True)
        db_utils.update_job_error(job_id, str(exc))
```

**5d — UI API client** (`ui/utils/api_client.py`)  
New file. Full `LershaAPIClient` implementation with `submit_prediction()`, `get_prediction_result()`, `get_results()`. Attaches `X-API-Key` header from `API_KEY` env var.

Replace direct `backend/` imports in `ui/pages/New_Prediction.py` and `ui/pages/Dashboard.py` with `LershaAPIClient` HTTP calls.

**Verification**: `uv run uvicorn backend.main:app --reload` starts without error. `curl http://localhost:8000/health` returns 200.

---

### Phase 6 — Containerization

**Goal**: Write `backend/Dockerfile`, `ui/Dockerfile`, and `docker-compose.yml`.

**`backend/Dockerfile`**:
```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY backend/ ./backend/
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`ui/Dockerfile`**:
```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY ui/ ./ui/
EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "ui/Introduction.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**`docker-compose.yml`** services: `postgres:16`, `backend`, `ui`, `mlflow`.

---

### Phase 7 — Testing Suite

**Goal**: Write unit and integration tests to reach ≥ 80% coverage on `backend/core/` and `backend/services/`.

**`backend/tests/conftest.py`**: fixtures for `test_db_engine`, `sample_farmer_df` (3-row synthetic), `api_client` (httpx against TestClient of `create_app()`).

**Unit tests** (`backend/tests/unit/`):
- `test_feature_engineering.py` — `net_income` formula, `institutional_support_score` sum, dropped columns, fallback binning
- `test_preprocessing.py` — output columns match `.pkl`, missing columns filled with `0`, shape assertion
- `test_contribution_table.py` — CatBoost list, XGB 3D ndarray, 2D binary; length mismatch raises `ValueError`; descending SHAP sort

**Integration tests** (`backend/tests/integration/`):
- `test_predict_endpoint.py` — 403 without key, 202 with key, job polling to completion
- `test_db_utils.py` — `fetch_raw_data` returns single-farmer row (no full scan), `fetch_multiple_raw_data` returns exactly `n` rows, `save_batch_evaluations` correct row count

**Verification**: `uv run pytest backend/tests/ --cov=backend --cov-fail-under=80`

---

### Phase 8 — Makefile + CI + Lint

**Goal**: Single developer entry point, CI pipeline, and clean lint pass.

**Makefile targets**: `install`, `setup-db`, `setup-chroma`, `lint`, `format`, `check-format`, `test`, `coverage`, `api`, `ui`, `mlflow`, `docker-build`, `docker-up`, `docker-down`, `clean`.

**`.github/workflows/ci.yml`**: `lint` job → `test` job (requires postgres service) → `build` job (requires lint + test).

**Final lint gate**: `uv run ruff check backend/ ui/` must exit 0.

---

## Complexity Tracking

> No constitution violations requiring justification. All 12 principles satisfied by the standard two-service monorepo pattern. No Complexity Tracking entries needed.

---

## Execution Order

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8
 uv/toml    dirs    file moves  extract   bugs       api+auth   docker    tests     CI+lint
```

Each phase has a verification gate. Phases 3 and 4 must run before Phase 5 (app factory needs clean imports and fixed config). Phases 5 and 7 can partially overlap (test scaffolding can be written while app factory is in progress).

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `BASE_DIR` parents count wrong after move | High (known bug class) | High (all model paths break) | Startup assertion catches at import time |
| `chromadb.PersistentClient` API change across plugin versions | Medium | Medium | Pin `chromadb>=0.6,<0.7` in pyproject.toml |
| `session.add()` + `Session(bind=engine)` deprecated in SQLAlchemy 2.x | High (legacy pattern confirmed in db_utils.py line 155) | Medium | Migrate to `with Session(engine) as session:` context manager pattern |
| `pd.read_sql` with `text()` requires `engine.connect()` in pandas 2.x | Medium | Medium | Always use `with engine.connect() as conn: pd.read_sql(..., conn)` |
| RAG Gemini API quota exceeded during integration tests | High | Low | Mock `google.generativeai.GenerativeModel.generate_content` in all tests |
| `asyncio_mode="auto"` requires `pytest-asyncio>=0.23` | Low | Low | Pin in dev dependencies |
| Poetry lockfile accidentally re-generated by IDE | Low | Low | Add `poetry.lock` to `.gitignore`; remove `[tool.poetry.*]` blocks |
