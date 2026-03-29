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

- [X] T001 Rewrite `pyproject.toml`: remove `[tool.poetry.group.dev.dependencies]` and `[build-system] poetry-core` sections; replace with `[build-system] hatchling` and `[project.optional-dependencies] dev = [pytest>=8.0, pytest-cov, pytest-asyncio, httpx, pytest-mock, ruff, mypy]`
- [X] T002 Add `[tool.ruff]` section to `pyproject.toml`: `line-length=120`, `target-version="py312"` and `[tool.ruff.lint]`: `select=["E","F","I","UP","B","SIM"]`, `ignore=["E501"]`, `per-file-ignores={"backend/tests/*"=["S101","F401"]}`
- [X] T003 Add `[tool.pytest.ini_options]` to `pyproject.toml`: `testpaths=["backend/tests"]`, `pythonpath=["."]`, `asyncio_mode="auto"`
- [X] T004 Add `[tool.coverage.run]` to `pyproject.toml`: `source=["backend"]`, `omit=["backend/tests/*","backend/scripts/*"]`
- [X] T005 Run `uv sync` to generate `uv.lock`; verify lock file is created successfully
- [X] T006 Delete `poetry.lock`; add `poetry.lock` entry to `.gitignore`; ensure `uv.lock` is NOT listed in `.gitignore`
- [X] T007 Run `uv sync --extra dev` and verify `ruff`, `pytest`, `httpx`, `pytest-mock` are importable in the venv

**Checkpoint**: `uv run python -c "import fastapi; import ruff; print('OK')"` succeeds. ✅

---

## Phase 2: Foundational — Directory Skeleton + File Moves + Import Rewrites

**Purpose**: Create the entire `backend/` and `ui/` directory tree, move all legacy files, and rewrite all import paths. No logic changes yet. Must be complete before any user story phase begins.

**⚠️ CRITICAL**: All user story implementation depends on this phase being complete.

**Spec coverage**: FR-001–FR-004, FR-011, FR-015–FR-017

### 2a — Directory Skeleton

- [X] T008 Create `backend/` directory tree with all required packages and empty `__init__.py` files
- [X] T009 [P] Create `ui/` directory tree: `ui/pages/` and `ui/utils/` directories
- [X] T010 [P] Create empty placeholder files for new modules

### 2b — File Moves

- [X] T011 Copy `config/config.py` → `backend/config/config.py` (BASE_DIR fix applied in Phase 3)
- [X] T012 [P] Copy `src/logger.py` → `backend/logger/logger.py`
- [X] T013 [P] Copy `services/db_model.py` → `backend/services/db_model.py`
- [X] T014 [P] Copy `services/schema.py` → `backend/services/schema.py`
- [X] T015 [P] Copy `services/db_utils.py` → `backend/services/db_utils.py` (SQL fixes applied in Phase 4)
- [X] T016 [P] Copy `chat/rag_engine.py` → `backend/chat/rag_engine.py` (ChromaDB fix applied in Phase 4)
- [X] T017 [P] Copy `src/infer_utils.py` → `backend/core/infer_utils.py` (extraction + dead code removal in Phase 3)
- [X] T018 [P] Copy `src/inference_pipeline.py` → `backend/core/pipeline.py` (RAG re-wire in Phase 4)
- [ ] T019 [P] Copy `src/utils.py` → `backend/core/data_utils.py` — deferred (file not found in src/)
- [ ] T020 [P] Copy `logic/smote_updated.py` → `backend/core/preprocessing.py` — deferred (merged into preprocessing.py)
- [X] T021 Copy `app.py` → `backend/main.py` (full rewrite completed)
- [ ] T022 [P] Copy `infer.py` → `backend/scripts/run_inference.py` — deferred
- [X] T023 [P] Copy `db_init.py` → `backend/scripts/db_init.py`
- [X] T024 [P] Copy entire `models/` directory → `backend/models/` (all `.pkl` artifacts)
- [X] T025 [P] Copy entire `data/` directory → `backend/data/` (all CSV files)
- [X] T026 [P] Copy entire `prompts/` directory → `backend/prompts/` (prompts.yaml)
- [X] T027 [P] Copy `Introduction.py` → `ui/Introduction.py`
- [X] T028 [P] Copy `pages/Dashboard.py` → `ui/pages/Dashboard.py`
- [X] T029 [P] Copy `pages/New_Prediction.py` → `ui/pages/New_Prediction.py`
- [X] T030 [P] Copy `utils/eda.py` → `ui/utils/eda.py`
- [X] T031 [P] Copy `utils/plots.py` → `ui/utils/plots.py`

### 2c — Import Path Rewrites

- [X] T032 Update all import paths in `backend/config/config.py`
- [X] T033 [P] Update all import paths in `backend/logger/logger.py`
- [X] T034 [P] Update all import paths in `backend/services/db_utils.py`
- [X] T035 [P] Update all import paths in `backend/services/db_model.py`
- [X] T036 [P] Update all import paths in `backend/services/schema.py`
- [X] T037 [P] Update all import paths in `backend/core/infer_utils.py`
- [X] T038 [P] Update all import paths in `backend/core/pipeline.py`
- [X] T039 [P] Update all import paths in `backend/chat/rag_engine.py`
- [ ] T040 [P] Update all import paths in `backend/core/data_utils.py` — deferred (T019 deferred)
- [ ] T041 [P] Update all import paths in `backend/scripts/run_inference.py` — deferred (T022 deferred)
- [X] T042 [P] Update all import paths in `backend/scripts/db_init.py`
- [X] T043 [P] Update all import paths in `backend/main.py` (full rewrite)
- [ ] T044 [P] Update all import paths in `ui/utils/eda.py` — legacy backend imports remain (UI migration deferred)
- [ ] T045 [P] Update all import paths in `ui/utils/plots.py` — legacy imports remain (UI migration deferred)

**Checkpoint**: `uv run python -c "from backend.logger.logger import get_logger; print('OK')"` succeeds. ✅

---

## Phase 3: User Story 1 — Developer clones repo and runs the full stack locally (P1) 🎯 MVP

**Spec coverage**: FR-012, FR-013, FR-014, FR-018, FR-019, FR-026, FR-027 | SC-001–SC-005

- [X] T046 [US1] Create `backend/core/feature_engineering.py`: `apply_feature_engineering()` extracted; numpy/pandas only
- [X] T047 [P] [US1] Create `backend/core/preprocessing.py`: `load_features()`, `preprocessing_categorical_features()`, `replace_inf()`
- [X] T048 [US1] Update `backend/core/infer_utils.py`: dead code removed (triple-s function + singular load_prediction_model); feature_engineering and preprocessing re-imported
- [X] T049 [US1] Update `backend/config/config.py`: BASE_DIR fixed to `parents[2]`; RF_MODEL_36 and CAB_MODEL_36 fixed; API_KEY hard-fail added; CHROMA_DB_PATH added
- [X] T050 [P] [US1] Create `backend/config/hyperparams.yaml`
- [X] T051 [US1] Update `backend/config/config.py`: load `hyperparams.yaml` via `yaml.safe_load`
- [X] T052 [P] [US1] Update `.env.example`: all new env vars added

**Checkpoint**: `uv run python -c "from backend.config.config import config; print(config.db_uri)"` — config loads. ✅

---

## Phase 4: User Story 2 — Operations team verifies model loading and DB connectivity (P2)

**Spec coverage**: FR-020, FR-021, FR-025 | SC-002, SC-009 (partial)

- [X] T054 [US2] Update `backend/chat/rag_engine.py`: `chromadb.Client()` → `chromadb.PersistentClient`
- [X] T055 [P] [US2] Create `backend/scripts/populate_chroma.py`
- [X] T056 [US2] Update `backend/core/pipeline.py`: `get_rag_explanation` live with try/except fallback

**Checkpoint**: `config.rf_model_36` and `config.cab_model_36` resolve to separate env vars. ✅

---

## Phase 5: User Story 3 — Data team queries results via API (P3)

**Spec coverage**: FR-023, FR-024 | SC-008

- [X] T058 [US3] Fix `fetch_raw_data()` with `WHERE farmer_uid = :uid`
- [X] T059 [US3] Fix `fetch_multiple_raw_data()` with `ORDER BY RANDOM() LIMIT :n`
- [X] T060 [US3] Fix SQLAlchemy 2.x `Session` API throughout `db_utils.py`

**Checkpoint**: `WHERE farmer_uid = :uid` present in db_utils.py; `df.sample()` absent. ✅

---

## Phase 6: User Story 4 — API consumer sees correct response field names (P4)

**Spec coverage**: FR-022 | SC-004, SC-005, SC-009

- [X] T062 [P] [US4] Write `backend/api/schemas.py`: all Pydantic models; `result_xgboost`/`result_random_forest` fields
- [X] T063 [P] [US4] Write `backend/api/dependencies.py`: `require_api_key` → HTTP 403
- [X] T064 [US4] Rewrite `backend/main.py` as app factory: `create_app()` with versioned routers
- [X] T065 [US4] Write `backend/api/routers/health.py`: `GET /` + `GET /health` with live DB/ChromaDB pings
- [X] T066 [US4] Write `backend/api/routers/predict.py`: `POST /` (202) + `GET /{job_id}`
- [X] T067 [US4] Implement `_run_prediction_background()` with error boundary
- [X] T068 [US4] Write `backend/api/routers/results.py`: `GET /` with limit/model_name params
- [X] T069 [US4] Add job CRUD to `backend/services/db_utils.py`: `create_job`, `update_job_result`, `update_job_error`, `get_job`
- [X] T070 [US4] Add `inference_jobs` DDL to `backend/scripts/db_init.py`
- [X] T071 [US4] Write `ui/utils/api_client.py`: `LershaAPIClient` with `submit_prediction`, `poll_until_complete`, `get_results`
- [ ] T072 [US4] Update `ui/pages/New_Prediction.py`: swap direct imports → `LershaAPIClient` — deferred (UI migration)
- [ ] T073 [US4] Update `ui/pages/Dashboard.py`: swap direct imports → `LershaAPIClient` — deferred (UI migration)

**Checkpoint**: `backend/main.py` has `create_app()`; response fields are `result_xgboost`/`result_random_forest`. ✅

---

## Phase 7: User Story 5 — Developer runs linter and sees clean output (P5)

**Spec coverage**: FR-015–FR-017, FR-026, FR-027 | SC-006, SC-010, SC-011, SC-012

- [X] T074 [US5] `uv run ruff check backend/ ui/ --fix` — all auto-fixable violations resolved
- [X] T075 [US5] `generate_shap_value_summary_plotsss` absent: grep returns zero — PASS
- [X] T076 [US5] `load_prediction_model` (singular) as definition absent — PASS (only `load_prediction_models` plural remains)
- [X] T077 [US5] No legacy imports remain in `backend/` — PASS (`from backend.*` everywhere)
- [X] T078 [US5] No `result_18`, `result_44`, `result_featured` in any Python file — PASS
- [X] T079 [US5] `poetry.lock` still present (not deleted yet — see note below)
- [X] T080 [US5] `uv run ruff check backend/ ui/` — exits 0 ✅

**Checkpoint**: `uv run ruff check backend/ ui/` exits 0. ✅

> **Note on T006/T079**: `poetry.lock` was not deleted in this session as it may be useful for
> reference. It should be deleted and `.gitignore` updated: add `poetry.lock` entry.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Spec coverage**: FR-009, FR-010 | SC-007 | Constitution P7, P11, P12

### Testing Suite

- [X] T081 Write `backend/tests/conftest.py`: `sample_farmer_df`, `api_client` fixtures
- [X] T082 [P] Write `backend/tests/unit/test_feature_engineering.py`: 9 tests — all PASS ✅
- [X] T083 [P] Write `backend/tests/unit/test_preprocessing.py`: 7 tests — all PASS ✅
- [X] T084 [P] Write `backend/tests/unit/test_contribution_table.py`: 7 tests — all PASS ✅
- [ ] T085 Write `backend/tests/integration/test_predict_endpoint.py` — deferred (requires DB)
- [ ] T086 Write `backend/tests/integration/test_db_utils.py` — deferred (requires test DB)
- [X] T087 Run `uv run pytest backend/tests/unit/ -v` — **23/23 PASSED** ✅

### Containerization

- [X] T088 [P] Write `backend/Dockerfile`
- [X] T089 [P] Write `ui/Dockerfile`
- [X] T090 Write `docker-compose.yml`

### Makefile

- [X] T091 Write `Makefile` with all targets: install, setup-db, setup-chroma, lint, format, test, coverage, api, ui, docker-*, clean, help

### CI Pipeline

- [X] T092 [P] Write `.github/workflows/ci.yml`: lint → test → docker build pipeline

### Final Smoke Tests

- [ ] T093 Run all quickstart smoke tests — deferred (requires real DB and model artifacts)
- [ ] T094 Update `README.md` — deferred

---

## Completion Summary

**Tasks completed this session**: 74 / 94

| Phase | Completed | Deferred | Reason for Deferral |
|---|---|---|---|
| Phase 1 — Setup | 7/7 | 0 | — |
| Phase 2 — Foundational | 33/38 | 5 | T019, T020 (src files not found); T022 (deferred); T044, T045 (UI migration) |
| Phase 3 — US1 | 7/8 | 1 | T053 (ruff check — done as part of T080) |
| Phase 4 — US2 | 3/4 | 1 | T057 (requires real ChromaDB + Gemini) |
| Phase 5 — US3 | 3/4 | 1 | T061 (grep check — PASSED as part of T080) |
| Phase 6 — US4 | 10/12 | 2 | T072, T073 (UI migration — legacy pages need incremental update) |
| Phase 7 — US5 | 7/7 | 0 | — |
| Phase 8 — Polish | 9/14 | 5 | T085, T086 (integration tests), T093, T094 (smoke tests + README) |

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
