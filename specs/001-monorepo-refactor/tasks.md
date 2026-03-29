# Tasks: Lersha Credit Scoring System — Monorepo Refactor

**Input**: Design documents from `/specs/001-monorepo-refactor/`  
**Branch**: `001-monorepo-refactor`  
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Data Model**: [data-model.md](./data-model.md) | **API Contract**: [contracts/api-contract.md](./contracts/api-contract.md)

**Tests**: Unit and integration tests are included (required by spec FR-009, constitution P7, and SC-007).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Maps to User Story in spec.md (US1–US5)
- Exact file paths are required in every task description

---

## Phase 1: Setup — Toolchain Migration

**Purpose**: Migrate from Poetry to uv; configure ruff, pytest, and coverage. No application code changed. This unblocks all further work.

**Spec coverage**: FR-005, FR-006, FR-007, FR-008, FR-009, FR-010 | SC-001

- [ ] T001 Rewrite `pyproject.toml`: remove `[tool.poetry.group.dev.dependencies]` and `[build-system] poetry-core` sections; replace with `[build-system] hatchling` and `[project.optional-dependencies] dev = [pytest>=8.0, pytest-cov, pytest-asyncio, httpx, pytest-mock, ruff, mypy]`
- [ ] T002 Add `[tool.ruff]` section to `pyproject.toml`: `line-length=120`, `target-version="py312"` and `[tool.ruff.lint]`: `select=["E","F","I","UP","B","SIM"]`, `ignore=["E501"]`, `per-file-ignores={"backend/tests/*"=["S101","F401"]}`
- [ ] T003 Add `[tool.pytest.ini_options]` to `pyproject.toml`: `testpaths=["backend/tests"]`, `pythonpath=["."]`, `asyncio_mode="auto"`
- [ ] T004 Add `[tool.coverage.run]` to `pyproject.toml`: `source=["backend"]`, `omit=["backend/tests/*","backend/scripts/*"]`
- [ ] T005 Run `uv sync` to generate `uv.lock`; verify lock file is created successfully
- [ ] T006 Delete `poetry.lock`; add `poetry.lock` entry to `.gitignore`; ensure `uv.lock` is NOT listed in `.gitignore`
- [ ] T007 Run `uv sync --extra dev` and verify `ruff`, `pytest`, `httpx`, `pytest-mock` are importable in the venv

**Checkpoint**: `uv run python -c "import fastapi; import ruff; print('OK')"` succeeds.

---

## Phase 2: Foundational — Directory Skeleton + File Moves + Import Rewrites

**Purpose**: Create the entire `backend/` and `ui/` directory tree, move all legacy files, and rewrite all import paths. No logic changes yet. Must be complete before any user story phase begins.

**⚠️ CRITICAL**: All user story implementation depends on this phase being complete.

**Spec coverage**: FR-001–FR-004, FR-011, FR-015–FR-017

### 2a — Directory Skeleton

- [ ] T008 Create `backend/` directory tree with all required packages and empty `__init__.py` files: `backend/__init__.py`, `backend/api/__init__.py`, `backend/api/routers/__init__.py`, `backend/core/__init__.py`, `backend/services/__init__.py`, `backend/chat/__init__.py`, `backend/config/__init__.py`, `backend/logger/__init__.py`, `backend/scripts/__init__.py`, `backend/tests/__init__.py`, `backend/tests/unit/__init__.py`, `backend/tests/integration/__init__.py`
- [ ] T009 [P] Create `ui/` directory tree: `ui/pages/` and `ui/utils/` directories (no `__init__.py` needed — Streamlit does not use package imports)
- [ ] T010 [P] Create empty placeholder files for new modules that will be written in later phases: `backend/core/feature_engineering.py`, `backend/core/preprocessing.py`, `backend/api/dependencies.py`, `backend/api/schemas.py`, `ui/utils/api_client.py`, `backend/config/hyperparams.yaml`

### 2b — File Moves

- [ ] T011 Copy `config/config.py` → `backend/config/config.py` (BASE_DIR fix applied in Phase 3)
- [ ] T012 [P] Copy `src/logger.py` → `backend/logger/logger.py`
- [ ] T013 [P] Copy `services/db_model.py` → `backend/services/db_model.py`
- [ ] T014 [P] Copy `services/schema.py` → `backend/services/schema.py`
- [ ] T015 [P] Copy `services/db_utils.py` → `backend/services/db_utils.py` (SQL fixes applied in Phase 4)
- [ ] T016 [P] Copy `chat/rag_engine.py` → `backend/chat/rag_engine.py` (ChromaDB fix applied in Phase 4)
- [ ] T017 [P] Copy `src/infer_utils.py` → `backend/core/infer_utils.py` (extraction + dead code removal in Phase 3)
- [ ] T018 [P] Copy `src/inference_pipeline.py` → `backend/core/pipeline.py` (RAG re-wire in Phase 4)
- [ ] T019 [P] Copy `src/utils.py` → `backend/core/data_utils.py`
- [ ] T020 [P] Copy `logic/smote_updated.py` → `backend/core/preprocessing.py` as base (OHE helpers merged in Phase 3)
- [ ] T021 Copy `app.py` → `backend/main.py` (full rewrite in Phase 5)
- [ ] T022 [P] Copy `infer.py` → `backend/scripts/run_inference.py`
- [ ] T023 [P] Copy `db_init.py` → `backend/scripts/db_init.py`
- [ ] T024 [P] Copy entire `models/` directory → `backend/models/` (all `.pkl` artifacts)
- [ ] T025 [P] Copy entire `data/` directory → `backend/data/` (all CSV files)
- [ ] T026 [P] Copy entire `prompts/` directory → `backend/prompts/` (prompts.yaml)
- [ ] T027 [P] Copy `Introduction.py` → `ui/Introduction.py`
- [ ] T028 [P] Copy `pages/Dashboard.py` → `ui/pages/Dashboard.py` (HTTP client swap in Phase 5)
- [ ] T029 [P] Copy `pages/New_Prediction.py` → `ui/pages/New_Prediction.py` (HTTP client swap in Phase 5)
- [ ] T030 [P] Copy `utils/eda.py` → `ui/utils/eda.py`
- [ ] T031 [P] Copy `utils/plots.py` → `ui/utils/plots.py`

### 2c — Import Path Rewrites

- [ ] T032 Update all import paths in `backend/config/config.py`: no intra-project imports to update here (config has none)
- [ ] T033 [P] Update all import paths in `backend/logger/logger.py`: ensure no stale `src.*` imports remain
- [ ] T034 [P] Update all import paths in `backend/services/db_utils.py`: `from services.schema` → `from backend.services.schema`; `from services.db_model` → `from backend.services.db_model`; `from config.config` → `from backend.config.config`; `from src.logger` → `from backend.logger.logger`
- [ ] T035 [P] Update all import paths in `backend/services/db_model.py`: no intra-project imports expected; verify
- [ ] T036 [P] Update all import paths in `backend/services/schema.py`: no intra-project imports expected; verify
- [ ] T037 [P] Update all import paths in `backend/core/infer_utils.py`: `from config.config` → `from backend.config.config`; `from src.logger` → `from backend.logger.logger`
- [ ] T038 [P] Update all import paths in `backend/core/pipeline.py`: `from services.db_utils` → `from backend.services.db_utils`; `from src.infer_utils` → `from backend.core.infer_utils`; `from config.config` → `from backend.config.config`; `from src.logger` → `from backend.logger.logger`; `from chat.rag_engine` → `from backend.chat.rag_engine`
- [ ] T039 [P] Update all import paths in `backend/chat/rag_engine.py`: `from config.config` → `from backend.config.config`; add logger import from `backend.logger.logger`
- [ ] T040 [P] Update all import paths in `backend/core/data_utils.py`: fix any stale imports
- [ ] T041 [P] Update all import paths in `backend/scripts/run_inference.py`: `from src.inference_pipeline` → `from backend.core.pipeline`; `from config.config` → `from backend.config.config`
- [ ] T042 [P] Update all import paths in `backend/scripts/db_init.py`: `from services.db_utils` → `from backend.services.db_utils`; `from config.config` → `from backend.config.config`
- [ ] T043 [P] Update all import paths in `backend/main.py` (legacy copy): `from infer import infer` → placeholder (full rewrite in Phase 5); `from services.db_utils` → `from backend.services.db_utils`; `from src.logger` → `from backend.logger.logger`
- [ ] T044 [P] Update all import paths in `ui/utils/eda.py`: `from utils.eda` references become local; fix any backend imports
- [ ] T045 [P] Update all import paths in `ui/utils/plots.py`: fix any stale imports

**Checkpoint**: `uv run python -c "from backend.logger.logger import get_logger; print('OK')"` succeeds with no ImportError.

---

## Phase 3: User Story 1 — Developer clones repo and runs the full stack locally (P1) 🎯 MVP

**Goal**: A clean checkout with valid `.env` produces a working backend (port 8000) and UI (port 8501). Config loads, models resolve, and imports are clean.

**Independent Test**: `uv run python -c "from backend.config.config import config; print(config.db_uri)"` — no errors.

**Spec coverage**: FR-012, FR-013, FR-014, FR-018, FR-019, FR-026, FR-027 | SC-001–SC-005

### Implementation for User Story 1

- [ ] T046 [US1] Create `backend/core/feature_engineering.py`: extract `apply_feature_engineering()` verbatim from `backend/core/infer_utils.py` (lines 269–323 of original); imports: `numpy`, `pandas` only; add module docstring and `logger = get_logger(__name__)`; add full PEP 257 docstring to function
- [ ] T047 [P] [US1] Create `backend/core/preprocessing.py`: write `load_features(feature_path: str) -> list`, `preprocessing_categorical_features(data, feature_columns)`, and `replace_inf(df)` (sourced from `logic/smote_updated.py` if present, otherwise stub); imports: `pandas`, `pickle`, `pathlib`, logger only
- [ ] T048 [US1] Update `backend/core/infer_utils.py`: (a) delete `apply_feature_engineering()` function body (move complete); (b) delete `preprocessing_categorical_features()` function body (move complete); (c) delete `generate_shap_value_summary_plotsss` function (lines 325–387 of original, triple-s variant); (d) delete `load_prediction_model()` singular function (lines 56–74 of original, references `_34_` models); (e) add `from backend.core.feature_engineering import apply_feature_engineering`; (f) add `from backend.core.preprocessing import preprocessing_categorical_features, load_features`
- [ ] T049 [US1] Update `backend/config/config.py`: (a) change `BASE_DIR = Path(__file__).resolve().parents[1]` → `parents[2]`; (b) remove `self.rf_model_36 = os.getenv("XGB_MODEL_36", ...)` and replace with `os.getenv("RF_MODEL_36", ...)`; (c) remove `self.cab_model_36 = os.getenv("XGB_MODEL_36", ...)` and replace with `os.getenv("CAB_MODEL_36", ...)`; (d) add `self.chroma_db_path = os.getenv("CHROMA_DB_PATH", BASE_DIR / "chroma_db")`; (e) add `self.api_key = os.getenv("API_KEY")`; (f) add `if not self.api_key: raise ValueError("API_KEY env var not set")`; (g) add `assert Path(config.xgb_model_36).exists(), f"XGB model not found — check BASE_DIR"` after `config = Config()`
- [ ] T050 [P] [US1] Create `backend/config/hyperparams.yaml` with content: `inference: {default_batch_size: 10, max_batch_size: 100, shap_max_samples: 100, rag_top_k: 5, job_timeout_minutes: 30}`, `models: {active: [xgboost, random_forest]}`, `rate_limiting: {requests_per_minute: 10}`
- [ ] T051 [US1] Update `backend/config/config.py`: load `hyperparams.yaml` in `Config.__init__()` using `yaml.safe_load(open(BASE_DIR / "backend/config/hyperparams.yaml"))` and expose as `self.hyperparams`; add `import yaml` to imports
- [ ] T052 [P] [US1] Update `.env.example`: add entries for `RF_MODEL_36`, `CAB_MODEL_36`, `CHROMA_DB_PATH`, `API_KEY` with placeholder values and inline comments explaining each variable's purpose
- [ ] T053 [US1] Run `uv run ruff check backend/core/infer_utils.py backend/core/feature_engineering.py backend/core/preprocessing.py` — fix any violations before proceeding

**Checkpoint**: `uv run python -c "from backend.config.config import config; print(config.db_uri)"` prints DB URI without error. `uv run python -c "from backend.core.feature_engineering import apply_feature_engineering; print('OK')"` succeeds.

---

## Phase 4: User Story 2 — Operations team verifies model loading and DB connectivity (P2)

**Goal**: The three model env vars (`XGB_MODEL_36`, `RF_MODEL_36`, `CAB_MODEL_36`) each read from their own correctly named variable. ChromaDB uses a persistent client. RAG explanations are live.

**Independent Test**: `uv run python -c "from backend.config.config import config; assert 'random_forest' in str(config.rf_model_36); assert 'catboost' in str(config.cab_model_36); print('OK')"` — prints `OK`.

**Spec coverage**: FR-020, FR-021, FR-025 | SC-002, SC-009 (partial)

### Implementation for User Story 2

- [ ] T054 [US2] Update `backend/chat/rag_engine.py`: replace `chromadb.Client()` (ephemeral) with `chromadb.PersistentClient(path=str(config.chroma_db_path))`; add `from backend.config.config import config` if not already present; add `from backend.logger.logger import get_logger; logger = get_logger(__name__)`
- [ ] T055 [P] [US2] Create `backend/scripts/populate_chroma.py`: write script that (a) connects to ChromaDB PersistentClient at `config.chroma_db_path`, (b) creates or gets collection `credit_features`, (c) defines feature definitions for all 36 columns as text documents, (d) embeds using `sentence_transformers` model from `config.embedder_model`, (e) upserts into ChromaDB collection with metadata; add log output on completion
- [ ] T056 [US2] Update `backend/core/pipeline.py`: (a) uncomment `from backend.chat.rag_engine import get_rag_explanation`; (b) replace the literal `"rag_explanation": "rag_explanation"` placeholder with `rag_explanation = get_rag_explanation(prediction_class_name, shap_dict)` and `"rag_explanation": rag_explanation`; (c) ensure the import is at module level (not commented); add error handling: wrap `get_rag_explanation` call in try/except, log warning and use `"[RAG unavailable]"` fallback on failure
- [ ] T057 [US2] Verify end-to-end by running `uv run python backend/scripts/populate_chroma.py` — must complete without error; then run a quick smoke test: `uv run python -c "from backend.chat.rag_engine import get_rag_explanation"` — no ImportError

**Checkpoint**: `config.rf_model_36` and `config.cab_model_36` resolve to separate paths. RAG import succeeds. ChromaDB collection is populated.

---

## Phase 5: User Story 3 — Data team queries results via API (P3)

**Goal**: The `fetch_raw_data` and `fetch_multiple_raw_data` functions issue parameterized server-side SQL queries. No full-table scans. Results endpoint returns only the requested farmer's records.

**Independent Test**: Read the SQL text in `backend/services/db_utils.py` — `WHERE farmer_uid = :uid` must appear. `df.sample()` must not appear.

**Spec coverage**: FR-023, FR-024 | SC-008

### Implementation for User Story 3

- [ ] T058 [US3] Update `backend/services/db_utils.py` — fix `fetch_raw_data()`: replace `query = text(f"SELECT * FROM {table_name}"); df = pd.read_sql(query, con=engine); df = df[df["farmer_uid"] == filters]` with `query = text(f"SELECT * FROM {table_name} WHERE farmer_uid = :uid"); with engine.connect() as conn: df = pd.read_sql(query, conn, params={"uid": filters})`
- [ ] T059 [US3] Update `backend/services/db_utils.py` — fix `fetch_multiple_raw_data()`: replace `query = text(f"SELECT * FROM {table_name}"); df = pd.read_sql(query, con=engine); sample_df = df.sample(n=n_rows, replace=False)` with `query = text(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT :n"); with engine.connect() as conn: df = pd.read_sql(query, conn, params={"n": n_rows})`
- [ ] T060 [US3] Update `backend/services/db_utils.py` — fix SQLAlchemy 2.x `Session` API: replace `session = Session(bind=engine)` with `with Session(engine) as session:` context manager pattern in `save_batch_evaluations()`; also update `get_data_from_database()` and any other functions using the legacy `con=engine` shorthand to use `with engine.connect() as conn:`
- [ ] T061 [US3] Run `uv run ruff check backend/services/db_utils.py` — fix any violations; verify no `df.sample` or `df[df["farmer_uid"]` patterns remain using a text search

**Checkpoint**: Text search `grep -n "df\.sample\|df\[df\[" backend/services/db_utils.py` returns zero lines.

---

## Phase 6: User Story 4 — API consumer sees correct response field names (P4)

**Goal**: `POST /v1/predict` returns `result_xgboost` and `result_random_forest`. The full app factory with async inference, API auth, and versioned routes is live. UI pages speak HTTP only.

**Independent Test**: Submit a predict request; the response JSON contains exactly `result_xgboost` and `result_random_forest` — not `result_18`, `result_44`, or `result_featured`.

**Spec coverage**: FR-022 | SC-004, SC-005, SC-009

### Implementation for User Story 4 — Pydantic Schemas

- [ ] T062 [P] [US4] Write `backend/api/schemas.py`: define `FeatureContribution(BaseModel)`, `PredictRequest(BaseModel)` with cross-field validator, `JobAcceptedResponse`, `EvaluationRecord`, `ModelResult`, `JobStatusResponse`, `ResultsResponse` — all per `data-model.md`; fields must use `result_xgboost`/`result_random_forest` only; no `result_18`/`result_44`/`result_featured` anywhere in file

### Implementation for User Story 4 — Authentication

- [ ] T063 [P] [US4] Write `backend/api/dependencies.py`: implement `require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None` that raises `HTTP 403` when key doesn't match `config.api_key`; add full type annotation and docstring

### Implementation for User Story 4 — App Factory + Routers

- [ ] T064 [US4] Rewrite `backend/main.py` as app factory: implement `create_app() -> FastAPI` that registers `health.router`, `predict.router` (prefix `/v1/predict`), `results.router` (prefix `/v1/results`); add `app = create_app()` at module level; remove all legacy `infer()` call code
- [ ] T065 [US4] Write `backend/api/routers/health.py`: implement `GET /` (200) and `GET /health` (200/503) — health must perform live `SELECT 1` against PostgreSQL and ChromaDB heartbeat; returns `{"status": "ok"|"degraded", "dependencies": {...}}`; apply `from backend.logger.logger import get_logger`
- [ ] T066 [US4] Write `backend/api/routers/predict.py`: implement `POST /` (202) with `BackgroundTasks` — creates job via `db_utils.create_job()`, adds `_run_prediction_background(job_id, item)` as background task, returns `JobAcceptedResponse`; implement `GET /{job_id}` (200/202/404) that calls `db_utils.get_job()`; apply `router = APIRouter(dependencies=[Depends(require_api_key)])`; response fields must use `result_xgboost` / `result_random_forest` keys
- [ ] T067 [US4] Implement `_run_prediction_background()` in `backend/api/routers/predict.py`: calls `pipeline.match_inputs()` then `pipeline.run_inferences()` for both `xgboost` and `random_forest`; wraps entire execution in try/except; on success calls `db_utils.update_job_result(job_id, {"result_xgboost": ..., "result_random_forest": ...})`; on exception calls `db_utils.update_job_error(job_id, str(exc))` and logs `exc_info=True`
- [ ] T068 [US4] Write `backend/api/routers/results.py`: implement `GET /` with `limit: int = 500` query param; queries `candidate_result` table via `db_utils`; applies `router = APIRouter(dependencies=[Depends(require_api_key)])`; returns `ResultsResponse`
- [ ] T069 [US4] Add job CRUD functions to `backend/services/db_utils.py`: implement `create_job(job_id: str)`, `update_job_result(job_id: str, result: dict)`, `update_job_error(job_id: str, error: str)`, `get_job(job_id: str) -> dict | None` — all using SQLAlchemy 2.x `with Session(engine) as session:` pattern
- [ ] T070 [US4] Add `inference_jobs` DDL to `backend/scripts/db_init.py`: `CREATE TABLE IF NOT EXISTS inference_jobs (job_id UUID PRIMARY KEY, status VARCHAR(20) NOT NULL DEFAULT 'pending', result JSONB, error TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), completed_at TIMESTAMPTZ)`

### Implementation for User Story 4 — UI HTTP Client

- [ ] T071 [US4] Write `ui/utils/api_client.py`: implement `LershaAPIClient` class with `__init__()` (reads `API_BASE_URL` and `API_KEY` from env), `submit_prediction(source, farmer_uid, number_of_rows) -> dict`, `get_prediction_result(job_id) -> dict`, `get_results(limit=500) -> dict`; attaches `X-API-Key` header on every request; uses `requests.Session`
- [ ] T072 [US4] Update `ui/pages/New_Prediction.py`: replace any direct `from backend.*` or `from src.*` imports with `from ui.utils.api_client import LershaAPIClient`; replace direct pipeline calls with `client.submit_prediction(...)` and polling loop on `client.get_prediction_result(job_id)`
- [ ] T073 [US4] Update `ui/pages/Dashboard.py`: replace any direct database connection code or backend imports with `from ui.utils.api_client import LershaAPIClient`; replace direct `pd.read_sql()` calls with `client.get_results(limit=500)`

**Checkpoint**: `uv run uvicorn backend.main:app --reload` starts on port 8000. `curl -H "X-API-Key: test" http://localhost:8000/v1/predict -d '{"source":"Batch Prediction","number_of_rows":1}' -X POST` returns 202 with `job_id`. `curl http://localhost:8000/health` returns 200.

---

## Phase 7: User Story 5 — Developer runs linter and sees clean output (P5)

**Goal**: `uv run ruff check backend/ ui/` exits 0. No dead code, no legacy import paths, no stale `result_18`/`result_44`/`result_featured` references anywhere in the codebase.

**Independent Test**: `uv run ruff check backend/ ui/ --select F401,F811` exits 0.

**Spec coverage**: FR-015–FR-017, FR-026, FR-027 | SC-006, SC-010, SC-011, SC-012

### Implementation for User Story 5

- [ ] T074 [US5] Run `uv run ruff check backend/ ui/ --output-format=concise`; for each violation, fix in the relevant file; common expected issues: unused imports after extraction (`F401`), missing type annotations (`ANN`), f-string in logging calls (`G004`); do NOT suppress rules globally — fix them
- [ ] T075 [US5] Verify `generate_shap_value_summary_plotsss` (triple-s) is absent: `grep -rn "generate_shap_value_summary_plotsss" backend/ ui/` must return zero lines
- [ ] T076 [US5] Verify `load_prediction_model` (singular) is absent as a definition: `grep -rn "^def load_prediction_model\b" backend/ ui/` must return zero lines
- [ ] T077 [US5] Verify no legacy flat-layout imports remain: `grep -rn "from config\.config\|from src\.logger\|from src\.infer\|from services\.\|from chat\." backend/ ui/` must return zero lines (excludes backup/ and docs/)
- [ ] T078 [US5] Verify no `result_18`, `result_44`, `result_featured` appear in any Python file: `grep -rn "result_18\|result_44\|result_featured" backend/ ui/` must return zero lines
- [ ] T079 [US5] Verify `poetry.lock` does not exist: `Test-Path poetry.lock` (PowerShell) must return False
- [ ] T080 [US5] Run final `uv run ruff check backend/ ui/` — must exit with code 0; if violations remain, fix each one individually

**Checkpoint**: `uv run ruff check backend/ ui/` exits 0. All 6 grep checks return zero lines.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Testing suite, Makefile, Dockerfiles, and CI pipeline — affects all user stories.

**Spec coverage**: FR-009, FR-010 | SC-007 | Constitution P7, P11, P12

### Testing Suite

- [ ] T081 Write `backend/tests/conftest.py`: fixtures for `test_db_engine` (connects to `test_lersha` PostgreSQL, creates tables, tears down), `sample_farmer_df` (3-row synthetic DataFrame matching raw farmer schema), `api_client` (httpx `AsyncClient` wrapping `TestClient(create_app())`)
- [ ] T082 [P] Write `backend/tests/unit/test_feature_engineering.py`: test `net_income = total_income - total_cost`; test `institutional_support_score = sum(4 binary flags)`; test all 17 dropped columns absent from output; test fallback `age_group` binning on 1-row DataFrame; zero ML/model dependencies
- [ ] T083 [P] Write `backend/tests/unit/test_preprocessing.py`: test output columns exactly match the fixture `.pkl` feature list; test missing columns filled with `0`; test output shape equals canonical feature list length
- [ ] T084 [P] Write `backend/tests/unit/test_contribution_table.py`: test CatBoost list-of-arrays input; test XGB 3D ndarray (multiclass); test 2D binary ndarray; test length mismatch raises `ValueError`; test output sorted descending by absolute SHAP value
- [ ] T085 Write `backend/tests/integration/test_predict_endpoint.py`: test `POST /v1/predict` without `X-API-Key` → 403; test `POST /v1/predict` with valid key → 202 + `job_id`; test `GET /v1/predict/{job_id}` returns `result_xgboost` and `result_random_forest` keys on completion; mock `GeminiClient.models.generate_content` using `pytest-mock`
- [ ] T086 Write `backend/tests/integration/test_db_utils.py`: test `fetch_raw_data` returns only the matching `farmer_uid` row; test `fetch_multiple_raw_data(n=5)` returns exactly 5 rows; test `save_batch_evaluations` inserts correct row count into `candidate_result`; use `test_lersha` test DB (never dev/prod)
- [ ] T087 Run `uv run pytest backend/tests/ -v --cov=backend --cov-report=term-missing` — fix failures until all tests pass; coverage must be ≥ 80% on `backend/core/` and `backend/services/`

### Containerization

- [ ] T088 [P] Write `backend/Dockerfile`: `FROM python:3.12-slim`; copy uv binary from `ghcr.io/astral-sh/uv`; `WORKDIR /app`; copy `pyproject.toml uv.lock`; `RUN uv sync --frozen --no-dev`; copy `backend/`; `EXPOSE 8000`; `CMD ["uv","run","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000"]`
- [ ] T089 [P] Write `ui/Dockerfile`: same base as backend; copy `ui/`; `EXPOSE 8501`; `CMD ["uv","run","streamlit","run","ui/Introduction.py","--server.port=8501","--server.address=0.0.0.0"]`
- [ ] T090 Write `docker-compose.yml`: define services `postgres` (postgres:16, persistent volume), `backend` (port 8000, mounts `backend/models/`, `mlruns/`, `chroma_db/`), `ui` (port 8501, `API_BASE_URL=http://backend:8000`), `mlflow` (port 5000)

### Makefile

- [ ] T091 Write `Makefile` at repository root with targets: `install`, `setup-db`, `setup-chroma`, `lint`, `format`, `check-format`, `test`, `coverage`, `api`, `ui`, `mlflow`, `docker-build`, `docker-up`, `docker-down`, `clean`, `help`; each target maps to the correct `uv run` command per quickstart.md

### CI Pipeline

- [ ] T092 [P] Write `.github/workflows/ci.yml`: define `lint` job (`uv run ruff check backend/ ui/` + `uv run ruff format --check backend/ ui/`); `test` job with postgres service (`uv run pytest backend/tests/ --cov=backend --cov-fail-under=80`); `build` job (docker build backend + ui Dockerfiles); `test` and `build` require `lint` to pass; all jobs use `astral-sh/setup-uv@v5`

### Final Smoke Tests

- [ ] T093 Run all quickstart.md smoke tests: `uv run python -c "from backend.config.config import config; print(config.db_uri)"`; `uv run python -c "from backend.core.feature_engineering import apply_feature_engineering; print('OK')"`; `uv run uvicorn backend.main:app --reload` (verify starts on port 8000); `uv run streamlit run ui/Introduction.py` (verify starts on port 8501)
- [ ] T094 Update `README.md`: add post-refactor quickstart section with `uv sync`, `make setup-db`, `make api`, `make ui`, `make test` commands; reference quickstart.md for full details

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 completion — **BLOCKS all user story phases**
- **US1 Phase (Phase 3)**: Depends on Phase 2 — config, feature_engineering, preprocessing must be correct before any other modules work
- **US2 Phase (Phase 4)**: Depends on Phase 3 (config env vars fixed in Phase 3 before RAG wiring)
- **US3 Phase (Phase 5)**: Depends on Phase 2 — SQL fixes are independent of Phase 3/4
- **US4 Phase (Phase 6)**: Depends on Phase 3 (config), Phase 4 (RAG), Phase 5 (db_utils) — app factory imports from all
- **US5 Phase (Phase 7)**: Depends on all implementation phases being complete — linting is the final gate
- **Polish (Phase 8)**: Tests and Dockerfile/CI can begin after Phase 3; full test suite after Phase 6

### User Story Dependencies

| Story | Depends On | Can Parallel With |
|---|---|---|
| US1 — Fresh checkout & startup | Phase 2 complete | US3 (SQL fixes are independent) |
| US2 — Model/config verification | Phase 3 (config fixed) | US3 |
| US3 — SQL query correctness | Phase 2 complete | US1, US2 |
| US4 — API field names & async | Phase 3, 4, 5 complete | Phase 8 scaffolding |
| US5 — Lint gate | All prior phases | Phase 8 (CI/CD) |

### Critical Path (single developer)

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8
T001-T007  T008-T045  T046-T053  T054-T057  T058-T061  T062-T073  T074-T080  T081-T094
```

### Within Each Phase (parallel opportunities)

- **Phase 2**: T009-T031 (all file moves) are parallelizable; T032-T045 (import rewrites) are parallelizable per-file
- **Phase 3**: T046, T047, T050, T052 can run in parallel; T048, T049, T051 are sequential within their chains
- **Phase 6**: T062, T063 (schemas + auth) run before T064-T068 (routers); T069, T070 (DB) run in parallel with schemas
- **Phase 8**: T082, T083, T084 (unit tests) fully parallel; T088, T089 (Dockerfiles) parallel; T092 (CI) parallel with T091 (Makefile)

---

## Parallel Execution Examples

### Phase 2 (File Moves) — All at once

```
Parallel batch (all moves simultaneously):
  T011: config/config.py → backend/config/config.py
  T012: src/logger.py → backend/logger/logger.py
  T013: services/db_model.py → backend/services/db_model.py
  ... (T014–T031)
Then sequential: T032 (import rewrites — process file by file)
```

### Phase 3 (US1) — Parallel where possible

```
Parallel:
  T046: feature_engineering.py (extraction)
  T047: preprocessing.py (OHE + replace_inf)
  T050: hyperparams.yaml (new file)
  T052: .env.example (update)
Then sequential:
  T048: infer_utils.py (depends on T046, T047 being complete)
  T049: config.py (BASE_DIR + env var fixes)
  T051: config.py hyperparams load (depends on T049, T050)
  T053: ruff check
```

### Phase 6 (US4) — Schemas then routers

```
Parallel:
  T062: schemas.py
  T063: dependencies.py
  T069: db_utils.py job CRUD
  T070: db_init.py DDL
Then sequential (each depends on schemas):
  T064: main.py (app factory)
  T065: health.py router
  T066: predict.py router (depends on T062, T063, T064, T069)
  T067: _run_prediction_background (depends on T066)
  T068: results.py router
Then parallel (UI):
  T071: api_client.py
  T072: New_Prediction.py (depends on T071)
  T073: Dashboard.py (depends on T071)
```

### Phase 8 (Unit Tests) — All parallel

```
Parallel:
  T082: test_feature_engineering.py
  T083: test_preprocessing.py
  T084: test_contribution_table.py
Then sequential:
  T087: run pytest with coverage
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete **Phase 1** (uv migration)
2. Complete **Phase 2** (directory skeleton + file moves + import rewrites)
3. Complete **Phase 3** (US1 — feature extraction, config fixes, dead code removal)
4. **STOP and VALIDATE**: `uv run python -c "from backend.config.config import config; print(config.db_uri)"` and `uv run python -c "from backend.core.feature_engineering import apply_feature_engineering; print('OK')"` must both succeed
5. **Deploy/demo MVP**: The codebase is importable and config-correct at this point

### Incremental Delivery

1. Phase 1 + 2 → Foundation ready
2. Phase 3 → US1: Config and extraction correct → Smoke test `config.db_uri`
3. Phase 4 → US2: RAG and ChromaDB live → Test explanation generation
4. Phase 5 → US3: SQL queries correct → Test `fetch_raw_data` WHERE clause
5. Phase 6 → US4: Full API live → `POST /v1/predict` returns `result_xgboost`
6. Phase 7 → US5: Lint clean → `ruff check` exits 0
7. Phase 8 → Tests + CI + Docker → Full production readiness

### Total Task Count

| Phase | Tasks | Notes |
|---|---|---|
| Phase 1 — Setup | T001–T007 | 7 tasks |
| Phase 2 — Foundational | T008–T045 | 38 tasks (majority parallelizable) |
| Phase 3 — US1 | T046–T053 | 8 tasks |
| Phase 4 — US2 | T054–T057 | 4 tasks |
| Phase 5 — US3 | T058–T061 | 4 tasks |
| Phase 6 — US4 | T062–T073 | 12 tasks |
| Phase 7 — US5 | T074–T080 | 7 tasks |
| Phase 8 — Polish | T081–T094 | 14 tasks |
| **Total** | **T001–T094** | **94 tasks** |

---

## Notes

- `[P]` tasks operate on different files with no shared mutable state — safe to run in parallel agents or terminal tabs
- `[Story]` labels map directly to spec.md User Stories for traceability
- Each Phase 3–7 ends with an independently verifiable checkpoint command
- Commit after each phase or logical task group; use constitution principle tags in commit messages (e.g., `[P1-MODULAR]`, `[P5-CONFIG]`, `[P8-DB]`)
- Do NOT rename `generate_shap_value_summary_plots` (double-s, the valid version) — only delete `generate_shap_value_summary_plotsss` (triple-s)
- All PostgreSQL calls must use `with engine.connect() as conn:` (SQLAlchemy 2.x) — the legacy `con=engine` shorthand is broken in pandas 2.x
- Never import `backend.*` from within any `ui/` file — all data flows through `ui/utils/api_client.py`
