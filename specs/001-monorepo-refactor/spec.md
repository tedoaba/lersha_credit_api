# Feature Specification: Lersha Credit Scoring System — Monorepo Refactor

**Feature Branch**: `001-monorepo-refactor`  
**Created**: 2026-03-29  
**Status**: Draft  
**Input**: User description: "Refactor the Lersha Credit Scoring System codebase from its legacy flat layout into a clean two-folder monorepo."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Developer clones repo and runs the full stack locally (Priority: P1)

A developer (or automated CI runner) clones the repository, copies `.env.example` to `.env`, fills in credentials, runs `uv sync`, and starts both services with two commands (`make api` and `make ui`). The backend starts on port 8000 and the Streamlit UI starts on port 8501 without any import errors, missing-file errors, or manual path manipulation.

**Why this priority**: This is the foundational developer-experience requirement. If a fresh checkout does not work out of the box, every subsequent workflow is blocked. It validates that the monorepo layout, import paths, and toolchain migration are all correct simultaneously.

**Independent Test**: Run `uv run python -c "from backend.config.config import config; print(config.db_uri)"` in a fresh virtualenv — it must print the DB URI without any `ModuleNotFoundError`.

**Acceptance Scenarios**:

1. **Given** a fresh clone with a valid `.env`, **When** a developer runs `uv sync`, **Then** all dependencies resolve from `uv.lock` with no errors and no reference to Poetry.
2. **Given** a valid environment, **When** `uv run uvicorn backend.main:app --reload` is executed, **Then** the FastAPI application starts on port 8000, the `/health` endpoint returns `200`, and the startup assertion `assert Path(config.xgb_model_36).exists()` passes silently.
3. **Given** a valid environment, **When** `uv run streamlit run ui/Introduction.py` is executed, **Then** the Streamlit app starts on port 8501 without importing any `backend/` module directly.
4. **Given** a valid environment, **When** `uv run pytest backend/tests/` is executed, **Then** all existing tests pass and coverage meets the defined threshold.

---

### User Story 2 — Operations team verifies correct model loading and DB connectivity (Priority: P2)

An operations engineer performs a post-deployment smoke test. They confirm that the XGBoost, Random Forest, and CatBoost models load from `backend/models/`, that the database connection string is read from the correct environment variable, and that the RAG explanation system queries ChromaDB from a persistent path.

**Why this priority**: Config regressions (wrong model path, wrong env var key) cause silent failures in production — predictions may succeed but use wrong models. This story validates the three critical bug fixes in config and RAG.

**Independent Test**: Run `uv run python -c "from backend.config.config import config; assert config.rf_model_36; assert config.cab_model_36; print('OK')"` — must print `OK`.

**Acceptance Scenarios**:

1. **Given** `.env` contains `RF_MODEL_36`, `CAB_MODEL_36`, and `XGB_MODEL_36`, **When** the config module is imported, **Then** `config.rf_model_36` reads from `RF_MODEL_36` and `config.cab_model_36` reads from `CAB_MODEL_36` (not both reading `XGB_MODEL_36`).
2. **Given** a ChromaDB collection exists at `CHROMA_DB_PATH`, **When** the prediction pipeline generates an explanation, **Then** the RAG engine connects to the persistent ChromaDB path and returns a non-empty explanation string.
3. **Given** `backend/config/config.py` is imported, **When** the startup assertion runs, **Then** `assert Path(config.xgb_model_36).exists()` passes, confirming `BASE_DIR` resolves to the project root (2 levels above `backend/config/`).

---

### User Story 3 — Data team queries prediction results via the API (Priority: P3)

A data team member calls `GET /v1/results/{farmer_uid}` to retrieve credit scoring results for a specific farmer. The query runs efficiently — it fetches only that farmer's records from the database using a targeted server-side filter rather than loading the full table into memory.

**Why this priority**: The SQL full-table-scan bug (load all rows, filter in Python) causes memory exhaustion and slow responses under production data volumes. Fixing this is necessary for production viability, but the system still functions (slowly) without it.

**Independent Test**: Inspect the SQL emitted by `db_utils.fetch_results_for_farmer(uid)` — the `WHERE farmer_uid = :uid` clause must appear in the query string.

**Acceptance Scenarios**:

1. **Given** a database with 10,000 candidate records, **When** `GET /v1/results/{farmer_uid}` is called for a single farmer, **Then** the database receives a `SELECT … WHERE farmer_uid = :uid` parameterized query and returns only that farmer's rows without loading the full table.
2. **Given** a request for a random sample of results, **When** the sampling query is executed, **Then** the database uses `ORDER BY RANDOM() LIMIT :n` rather than Python `df.sample()`.
3. **Given** a request for a non-existent farmer UID, **When** the results endpoint is called, **Then** the API returns a structured `404` response with `{"detail": "No results found for farmer_uid", "type": "not_found"}`.

---

### User Story 4 — Developer verifies API response field naming (Priority: P4)

A developer or API consumer calls `POST /v1/predict` and inspects the response. The prediction result fields in the JSON response use the canonical model names `result_xgboost` and `result_random_forest` rather than the legacy cryptic names `result_18`, `result_44`, or `result_featured`.

**Why this priority**: Incorrect field names in the API response cause all downstream consumers (Streamlit UI, any external integrations) to silently receive `null` for prediction values. This is a breaking correctness bug.

**Independent Test**: Submit a valid predict request and assert `"result_xgboost"` and `"result_random_forest"` are present as top-level keys in the response JSON.

**Acceptance Scenarios**:

1. **Given** a valid inference payload, **When** `POST /v1/predict` completes, **Then** the response JSON contains `result_xgboost` and `result_random_forest` fields and does NOT contain `result_18`, `result_44`, or `result_featured`.
2. **Given** the Pydantic response schema in `backend/api/schemas.py`, **When** the schema is inspected, **Then** field names `result_xgboost` and `result_random_forest` are defined and `result_18`/`result_44`/`result_featured` are absent.

---

### User Story 5 — Developer runs the linter on the refactored codebase (Priority: P5)

A developer runs `uv run ruff check backend/ ui/` after the refactor. The command exits with code 0. The codebase contains no dead code (no `generate_shap_value_summary_plotsss` triple-s function, no `load_prediction_model()` singular function), no unused imports, and all import paths reference the new `backend.*` / `ui.*` namespaces.

**Why this priority**: Dead code and stale imports are the canary for an incomplete refactor. CI enforces this gate before any merge.

**Independent Test**: `uv run ruff check backend/ ui/ --select F401,F811` exits with code 0.

**Acceptance Scenarios**:

1. **Given** the refactored codebase, **When** `uv run ruff check backend/ ui/` is run, **Then** it exits with code `0` and reports zero violations.
2. **Given** the codebase, **When** a search for `generate_shap_value_summary_plotsss` (triple-s) is performed, **Then** no matches are found in any Python file.
3. **Given** the codebase, **When** a search for `load_prediction_model` (singular) is performed, **Then** no definition of this function exists (the plural form used elsewhere may remain if valid).
4. **Given** any `backend/` module, **When** its imports are inspected, **Then** all intra-project imports use the `backend.*` namespace (e.g., `from backend.config.config import config`), not legacy paths like `from config.config import config`.

---

### Edge Cases

- What happens when `BASE_DIR` is miscalculated (e.g., only 1 `parents` level)? → The startup assertion `assert Path(config.xgb_model_36).exists()` must raise `AssertionError` and prevent the server from starting.
- What happens when `RF_MODEL_36` or `CAB_MODEL_36` is missing from `.env`? → `config.py` must raise `ValueError` at import time, preventing silent fallback to the XGBoost path.
- What happens when `CHROMA_DB_PATH` does not exist on disk? → ChromaDB `PersistentClient` must raise an error at connection time, which the RAG engine must catch and log as `ERROR`.
- What happens when a legacy flat-layout import path (`from config.config import config`) is accidentally left in a file? → `ruff` rule `F401`/`E401` or a Python `ModuleNotFoundError` must surface it immediately — it must never reach production silently.
- What happens when `poetry.lock` still exists after migration? → `uv sync` must be the only lock-file mechanism. `poetry.lock` must be deleted so developers cannot accidentally run `poetry install`.
- What happens when a `ui/` module imports from `backend/` directly? → The import must fail with `ModuleNotFoundError` when `backend/` is not in the Streamlit container's Python path, enforcing the API-only communication constraint.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Directory Structure

- **FR-001**: The repository MUST contain exactly two top-level code folders: `backend/` and `ui/`. No Python application code MUST reside at the repository root.
- **FR-002**: The repository root MUST contain only: `.env`, `.env.example`, `.gitignore`, `.dockerignore`, `pyproject.toml`, `uv.lock`, `Makefile`, `docker-compose.yml`, and the `specs/`, `docs/`, `.specify/`, `.agent/` support directories.
- **FR-003**: The `backend/` directory MUST contain the sub-packages: `api/`, `core/`, `services/`, `chat/`, `config/`, `logger/`, `scripts/`, `tests/`, `models/`, `data/`, `prompts/` — each with an `__init__.py` where applicable.
- **FR-004**: The `ui/` directory MUST contain: `Introduction.py`, `pages/Dashboard.py`, `pages/New_Prediction.py`, `utils/eda.py`, `utils/plots.py`, `utils/api_client.py`, and a `Dockerfile`.

#### Toolchain Migration

- **FR-005**: The project MUST use `uv` as the sole dependency and virtual-environment manager. Poetry configuration (`[tool.poetry.*]` sections and `poetry.lock`) MUST be removed entirely.
- **FR-006**: `pyproject.toml` MUST contain a `[project]` table with `hatchling` as the build backend and all runtime dependencies declared under `[project.dependencies]`.
- **FR-007**: `uv.lock` MUST be generated by `uv sync` and committed to version control.
- **FR-008**: `pyproject.toml` MUST configure `ruff` with: `line-length = 120`, `target-version = "py312"`, `select = ["E","F","I","UP","B","SIM"]`, `ignore = ["E501"]`.
- **FR-009**: `pyproject.toml` MUST configure `pytest` with: `testpaths = ["backend/tests"]`, `pythonpath = ["."]`, `asyncio_mode = "auto"`.
- **FR-010**: `pyproject.toml` MUST configure coverage with: `source = ["backend"]`, `omit = ["backend/tests/*","backend/scripts/*"]`.

#### File Migrations & Module Extraction

- **FR-011**: All server-side Python files MUST be migrated to their prescribed paths under `backend/` as defined in the FILE MOVES specification (see Assumptions for the complete mapping).
- **FR-012**: `apply_feature_engineering()` MUST be extracted into `backend/core/feature_engineering.py` as a pure function with zero ML framework imports.
- **FR-013**: `replace_inf()` and `preprocessing_categorical_features()` MUST reside in `backend/core/preprocessing.py` (merged from `logic/smote_updated.py`).
- **FR-014**: `backend/core/pipeline.py` MUST import from `backend/core/feature_engineering` and `backend/core/preprocessing` — it MUST NOT contain these functions inline.

#### Import Path Updates

- **FR-015**: All intra-project imports across `backend/` MUST use the `backend.*` namespace prefix (e.g., `from backend.config.config import config`).
- **FR-016**: All intra-project imports across `ui/` MUST use the `ui.*` namespace prefix where applicable.
- **FR-017**: No legacy flat-layout import paths (e.g., `from config.config`, `from src.logger`, `from services.db_utils`) MUST remain in any committed file.

#### Config Fixes

- **FR-018**: `BASE_DIR` in `backend/config/config.py` MUST be `Path(__file__).resolve().parents[2]` (the project root, two levels above `backend/config/`).
- **FR-019**: A startup assertion `assert Path(config.xgb_model_36).exists()` MUST be present in `backend/config/config.py` to validate `BASE_DIR` at import time.
- **FR-020**: `config.rf_model_36` MUST read from the environment variable `RF_MODEL_36`. `config.cab_model_36` MUST read from the environment variable `CAB_MODEL_36`. Neither MUST read from `XGB_MODEL_36`.
- **FR-021**: `RF_MODEL_36`, `CAB_MODEL_36`, and `CHROMA_DB_PATH` MUST be added to `backend/config/config.py` and documented in `.env.example`.

#### Bug Fixes

- **FR-022**: The API response schema MUST use field names `result_xgboost` and `result_random_forest`. Fields named `result_18`, `result_44`, and `result_featured` MUST NOT appear in any response schema or Pydantic model.
- **FR-023**: Database queries that retrieve records for a farmer MUST use a server-side `WHERE farmer_uid = :uid` parameterized clause. Full-table `SELECT *` followed by Python-side filtering MUST be replaced.
- **FR-024**: Random sampling queries MUST use `ORDER BY RANDOM() LIMIT :n` at the database level. Python `df.sample()` on a full table result MUST be replaced.
- **FR-025**: The RAG explanation call (`get_rag_explanation`) MUST be uncommented and active in `backend/core/pipeline.py`. ChromaDB MUST use `chromadb.PersistentClient(path=config.chroma_db_path)`.

#### Dead Code Removal

- **FR-026**: The function `generate_shap_value_summary_plotsss` (with triple trailing `s`) MUST be deleted from the codebase.
- **FR-027**: The function `load_prediction_model()` (singular, referencing unused `_34_` model files) MUST be deleted from the codebase.

### Key Entities

- **`backend/config/config.py` Config singleton**: Central configuration object exposing all environment variable values and hyperparameters. Single instance imported by all `backend/` modules.
- **`backend/core/pipeline.py` Inference pipeline**: Orchestrates the end-to-end credit scoring workflow: data fetch → feature engineering → preprocessing → model inference → RAG explanation → DB write.
- **`backend/core/feature_engineering.py`**: Pure Python module containing `apply_feature_engineering()` — no ML framework dependencies.
- **`backend/core/preprocessing.py`**: Module containing `replace_inf()` and `preprocessing_categorical_features()`.
- **`backend/api/schemas.py` Pydantic models**: Defines all API request/response shapes including the corrected `result_xgboost` / `result_random_forest` field names.
- **`uv.lock`**: Deterministic dependency lockfile replacing `poetry.lock`. Must be committed.
- **`.env.example`**: Documents all required environment variables including `RF_MODEL_36`, `CAB_MODEL_36`, `CHROMA_DB_PATH`.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A developer can run `uv sync` on a fresh clone and get a fully resolved environment in under 2 minutes, with zero manual steps beyond copying `.env.example`.
- **SC-002**: The statement `uv run python -c "from backend.config.config import config; print(config.db_uri)"` executes without errors on any developer machine that has a valid `.env`.
- **SC-003**: The statement `uv run python -c "from backend.core.feature_engineering import apply_feature_engineering"` executes without errors, confirming the extraction is complete and import path is correct.
- **SC-004**: `uv run uvicorn backend.main:app --reload` starts the backend within 10 seconds with zero import errors, and `GET /health` returns `200`.
- **SC-005**: `uv run streamlit run ui/Introduction.py` starts the UI within 15 seconds with zero import errors and without importing any `backend/` module.
- **SC-006**: `uv run ruff check backend/ ui/` exits with code `0` — no lint violations remain after refactor.
- **SC-007**: `uv run pytest backend/tests/` passes all tests with the `asyncio_mode = "auto"` setting active.
- **SC-008**: A call to `GET /v1/results/{farmer_uid}` for any farmer issues exactly one SQL query containing `WHERE farmer_uid = :uid` — no full-table scan occurs, as verified by query logging.
- **SC-009**: A call to `POST /v1/predict` returns a response containing `result_xgboost` and `result_random_forest` fields — `result_18`, `result_44`, and `result_featured` do not appear.
- **SC-010**: A text search for `generate_shap_value_summary_plotsss` across the entire codebase returns zero matches.
- **SC-011**: A text search for `from config.config`, `from src.logger`, `from services.db_utils`, `from src.inference_pipeline`, `from src.infer_utils` across the codebase returns zero matches in any non-backup, non-docs file.
- **SC-012**: `poetry.lock` does not exist in the repository root after the refactor is complete.

---

## Assumptions

- The legacy flat-layout files (`app.py`, `Introduction.py`, `pages/`, `utils/`, `src/`, `services/`, `chat/`, `config/`, `logic/`, `infer.py`, `db_init.py`) exist at the repository root or in their current locations as observed in the codebase at time of specification.
- The `backend/models/` directory will contain the `.pkl` files referenced by environment variables `XGB_MODEL_36`, `RF_MODEL_36`, and `CAB_MODEL_36` after migration of the `models/` directory.
- The Streamlit UI will communicate with the backend exclusively through `ui/utils/api_client.py` over HTTP. No direct Python-level imports from `backend/` into `ui/` are permitted — this is enforced by container isolation and import path boundaries.
- Migration of `logic/smote_updated.py` to `backend/core/preprocessing.py` includes only `replace_inf()` and `preprocessing_categorical_features()`. SMOTE training code (if present) is out of scope for this refactor.
- The `data/` and `prompts/` directories are moved as-is from the root to `backend/data/` and `backend/prompts/` without restructuring their internal content.
- Alembic migration setup is out of scope for this refactor. `backend/scripts/db_init.py` handles initial schema bootstrapping only.
- The `mlruns/` directory is not migrated — it is excluded from the monorepo structure as a runtime artifact and remains `.gitignore`d.
- The `backup/` directory at the root is not migrated — it is documentation/reference material only.
- All Python application code is written for Python 3.12. No backward compatibility with earlier versions is required.
- Docker Compose service definitions (`docker-compose.yml`) are in scope for updating to reference the new `backend/` and `ui/` paths, but creating `docker-compose.override.yml` and `docker-compose.prod.yml` is deferred to a subsequent phase.
- The `chroma_loader.py` script (populating ChromaDB) moves to `backend/scripts/populate_chroma.py` as specified.
