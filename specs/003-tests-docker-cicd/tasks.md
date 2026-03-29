# Tasks: Test Suite, Docker Build System & CI/CD Pipeline

**Branch**: `003-tests-docker-cicd` | **Date**: 2026-03-29  
**Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)  
**Total Tasks**: 25 | **Parallelizable**: 14

---

## Progress

- Phase 1 — Setup: 0/1 complete
- Phase 2 — Foundational: 0/2 complete
- Phase 3 — US1: Unit Test Correctness: 0/2 complete
- Phase 4 — US2: Preprocessing Schema: 0/1 complete
- Phase 5 — US3: API Integration Tests: 0/4 complete
- Phase 6 — US4: Database Utility Integration Tests: 0/5 complete
- Phase 7 — US5: Docker & Containerisation: 0/4 complete
- Phase 8 — US6: CI/CD Pipeline: 0/4 complete
- Phase 9 — Polish: 0/2 complete

---

## Dependencies

```
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational — conftest.py rewrite)
    │
    ├──► Phase 3 (US1 — unit tests)     ← no DB needed; can run immediately after Phase 2
    ├──► Phase 4 (US2 — preprocessing)  ← no DB needed; can run immediately after Phase 2
    ├──► Phase 5 (US3 — API integration)← needs test_db_engine from Phase 2
    └──► Phase 6 (US4 — DB integration) ← needs test_db_engine from Phase 2

Phase 7 (US5 — Docker) ← independent of Phases 3–6; can start anytime
Phase 8 (US6 — CI/CD) ← depends on Phase 7 (images must build); logically after all others

Phase 9 (Polish) ← after everything else
```

**Parallel opportunities**:
- Phase 3 + Phase 4 can run concurrently (no shared files).
- Phase 5 + Phase 6 can run concurrently after Phase 2 (no shared files; both use `test_db_engine`).
- Phase 7 tasks T017, T018, T019 can run concurrently (different files).
- Phase 8 tasks T021 + T022 can run concurrently.

---

## Phase 1 — Setup

> **Goal**: Verify prerequisites and confirm the test database is reachable.  
> **Independent test**: `uv run pytest --collect-only` succeeds with no import errors.

- [ ] T001 Verify `test_lersha` PostgreSQL database exists and `DB_URI` env var is exported; run `uv run pytest --collect-only` to confirm zero collection errors before any changes

---

## Phase 2 — Foundational

> **Goal**: Rewrite `conftest.py` to provide all four required shared fixtures.  
> All unit and integration tests depend on this phase.  
> **Independent test**: `uv run pytest backend/tests/ --collect-only` — all fixtures resolve without error; pytest does not report fixture-not-found warnings.

- [ ] T002 Rewrite `backend/tests/conftest.py` — replace the existing 63-line file with: (a) `test_db_engine` session-scoped fixture that reads `DB_URI` from env, connects to `test_lersha` PostgreSQL, creates `candidate_result` and `inference_jobs` tables using `sqlalchemy.text()` DDL matching the production schema in `backend/scripts/db_init.py`, yields the engine, and in a `finally` block drops both tables; (b) `sample_farmer_df` function-scoped fixture returning a 5-row `pd.DataFrame` with all 29 raw farmer columns (UIDs F-001–F-005, realistic numeric values, `farmsizehectares > 0` for all rows, binary membership flags ∈ {0,1}); (c) `api_client` async fixture that calls `os.environ.setdefault` for `API_KEY=ci-test-key`, `GEMINI_MODEL`, `GEMINI_API_KEY`, `DB_URI`, then imports `create_app` and returns `httpx.AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")` used as an async context manager; (d) `mock_gemini` function-scoped fixture using `mocker.patch` to patch the `generate_content` call path used in `backend/chat/` to return the fixed string `"Fixed explanation for testing."`

- [ ] T003 Add `anyio` backend marker to `pyproject.toml` if not already set — confirm `asyncio_mode = "auto"` is present under `[tool.pytest.ini_options]` and `anyio` (or `pytest-anyio`) is not conflicting; run `uv run pytest backend/tests/unit/ -v` to confirm all existing unit tests still pass after the conftest rewrite

---

## Phase 3 — US1: Validate Core Business Logic with Unit Tests

> **User Story**: A developer runs unit tests on feature engineering and SHAP contribution — no DB, no models, no network.  
> **Independent test**: `uv run pytest backend/tests/unit/test_feature_engineering.py backend/tests/unit/test_contribution_table.py -v` — all tests pass.

- [ ] T004 [P] [US1] Add `test_age_group_binning_two_row_edge_case` to `backend/tests/unit/test_feature_engineering.py` — create a 2-row DataFrame with all required raw farmer columns (using a local fixture or inline), call `apply_feature_engineering(df)`, assert `"age_group" in result.columns`, assert `result["age_group"].isna().sum() == 0`, and assert each value ∈ `["Young", "Early_Middle", "Late_Middle", "Senior"]`; keep the existing `test_age_group_fallback_single_row` test unchanged

- [ ] T005 [P] [US1] Verify `backend/tests/unit/test_contribution_table.py` covers all three SHAP formats as required by FR-015–FR-019 — run `uv run pytest backend/tests/unit/test_contribution_table.py -v` and confirm: `test_catboost_list_input`, `test_xgb_3d_multiclass`, `test_2d_binary_ndarray`, `test_length_mismatch_raises_value_error`, and `test_sorted_descending_by_abs_shap` all pass; no new code required if all pass

---

## Phase 4 — US2: Validate Preprocessing Output Matches Trained Feature Schema

> **User Story**: A developer runs preprocessing unit tests to confirm the 36-column schema is matched exactly.  
> **Independent test**: `uv run pytest backend/tests/unit/test_preprocessing.py -v` — all tests pass.

- [ ] T006 [P] [US2] Verify `backend/tests/unit/test_preprocessing.py` satisfies FR-011–FR-014 — run `uv run pytest backend/tests/unit/test_preprocessing.py -v` and confirm: `test_preprocessing_output_columns_match_pkl`, `test_missing_columns_filled_with_zero`, and `test_output_shape_equals_feature_list` all pass using the temp-file `.pkl` fixture pattern; if any test is missing, add it inline following the existing fixture style with a 4-column canonical feature list

---

## Phase 5 — US3: Validate API Endpoints via Integration Tests

> **User Story**: A developer or CI runner validates authentication, job lifecycle, and error responses end-to-end via the ASGI transport client.  
> **Independent test**: `uv run pytest backend/tests/integration/test_predict_endpoint.py -v` — all 5 tests pass against a live `test_lersha` DB.  
> **Fixtures required**: `api_client`, `test_db_engine`

- [ ] T007 [US3] Create `backend/tests/integration/test_predict_endpoint.py` with module docstring explaining ASGI transport usage; import `pytest`, `httpx`, `uuid`, and the `api_client` + `test_db_engine` fixtures from `conftest`; add `pytestmark = pytest.mark.asyncio` at module level (or rely on `asyncio_mode = "auto"`)

- [ ] T008 [US3] Add `test_no_api_key_returns_403` to `backend/tests/integration/test_predict_endpoint.py` — async test function that takes `api_client` fixture, sends `POST /v1/predict` with body `{"source": "Batch Prediction", "number_of_rows": 2}` and NO `X-API-Key` header, asserts `response.status_code == 403`

- [ ] T009 [US3] Add `test_wrong_api_key_returns_403` — async test that sends `POST /v1/predict` with `X-API-Key: definitely-wrong-key` header and valid body, asserts `response.status_code == 403`

- [ ] T010 [US3] Add three more tests to `backend/tests/integration/test_predict_endpoint.py`: (a) `test_valid_predict_returns_202_with_job_id` — takes `api_client` and `test_db_engine`, sends `POST /v1/predict` with `X-API-Key: ci-test-key` and body `{"source": "Batch Prediction", "number_of_rows": 2}`, asserts `response.status_code == 202`, asserts `"job_id"` in `response.json()`, asserts `uuid.UUID(response.json()["job_id"])` does not raise; (b) `test_get_job_status_returns_200` — reuses the `job_id` from the 202 response (as a module-level or session fixture), sends `GET /v1/predict/{job_id}` with valid key, asserts `response.status_code == 200` and `"status"` in `response.json()`; (c) `test_nonexistent_job_returns_404` — sends `GET /v1/predict/00000000-0000-0000-0000-000000000000` with valid key, asserts `response.status_code == 404`

---

## Phase 6 — US4: Validate Database Utilities via Integration Tests

> **User Story**: A developer runs DB integration tests against a real test PostgreSQL to confirm all CRUD operations are correct.  
> **Independent test**: `uv run pytest backend/tests/integration/test_db_utils.py -v` — all 5 tests pass.  
> **Fixtures required**: `test_db_engine`, `sample_farmer_df`

- [ ] T011 [US4] Create `backend/tests/integration/test_db_utils.py` with module docstring; import `os`, `uuid`, `datetime`, `pandas as pd`, `pytest`, `sqlalchemy.text` from `sqlalchemy`; import `fetch_raw_data`, `fetch_multiple_raw_data`, `save_batch_evaluations`, `create_job`, `get_job`, `update_job_result` from `backend.services.db_utils`; note that all `db_utils` functions read `config.db_uri` internally so `DB_URI` env var must be set (guaranteed by `api_client` fixture or a module-level `os.environ` call in conftest)

- [ ] T012 [US4] Add `test_fetch_raw_data_returns_matching_row` — test takes `test_db_engine`; creates a raw farmer table in `test_lersha` (or reuses the table schema that `test_db_engine` already created); inserts 3 synthetic rows with distinct `farmer_uid` values using `engine.connect() + conn.execute(text(...))` + `conn.commit()`; calls `fetch_raw_data(table_name, "F-002")`; asserts `len(result) == 1` and `result["farmer_uid"].iloc[0] == "F-002"`; note that `fetch_raw_data` signature is `fetch_raw_data(table_name: str, filters: str)` where `filters` is the `farmer_uid` value

- [ ] T013 [US4] Add `test_fetch_multiple_raw_data_returns_n_rows` — test takes `test_db_engine`; seeds the same farmer table with ≥ 3 rows if not already seeded; calls `fetch_multiple_raw_data(table_name, n_rows=3)`; asserts `len(result) == 3`

- [ ] T014 [US4] Add `test_save_batch_evaluations_inserts_correct_count` — test takes `test_db_engine`; constructs a 2-row `input_df` with `farmer_uid`, `first_name`, `middle_name`, `last_name` columns; constructs `evaluation_results` as a list of 2 dicts with keys `predicted_class_name`, `top_feature_contributions` (list of one `{"feature": "f", "shap_value": 0.1}` dict), `rag_explanation`, `model_name`; calls `save_batch_evaluations(input_df, evaluation_results)`; queries `SELECT COUNT(*) FROM candidate_result` via `test_db_engine`; asserts count equals 2

- [ ] T015 [US4] Add `test_create_and_get_job_round_trip` and `test_update_job_result_sets_completed` to `backend/tests/integration/test_db_utils.py`: (a) `test_create_and_get_job_round_trip` takes `test_db_engine`, calls `create_job(job_id)` with a fresh `str(uuid.uuid4())`, immediately calls `get_job(job_id)`, asserts result is not `None`, asserts `result["status"] == "pending"`; (b) `test_update_job_result_sets_completed` takes `test_db_engine`, creates a job, calls `update_job_result(job_id, {"score": 0.85})`, calls `get_job(job_id)`, asserts `result["status"] == "completed"` and `result["result"] is not None`

---

## Phase 7 — US5: Docker & Containerisation

> **User Story**: A DevOps engineer builds images and brings up the four-service stack with a single command.  
> **Independent test**: `make docker-build` exits 0; `docker-compose up` starts all 4 services; `curl http://localhost:8000/health` returns 200.

- [ ] T016 [P] [US5] Update the `mlflow` service in `docker-compose.yml` — replace the current `image: python:3.12-slim` + `command: bash -c "pip install mlflow && mlflow server ..."` block with `image: ghcr.io/mlflow/mlflow:latest` and `command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root /mlruns`; keep the `restart`, `ports`, and `volumes` sections unchanged

- [ ] T017 [P] [US5] Create `.dockerignore` at the project root with these entries on separate lines: `__pycache__/`, `*.pyc`, `.git/`, `.env`, `backend/tests/`, `*.egg-info/`, `.mypy_cache/`, `.pytest_cache/`, `htmlcov/`, `.ruff_cache/`; verify `uv.lock` is NOT excluded (it must be copied into the image)

- [ ] T018 [P] [US5] Verify `backend/Dockerfile` satisfies FR-031–FR-034 by reviewing each layer: base image is `python:3.12-slim`, `uv` binary is copied from `ghcr.io/astral-sh/uv:latest`, `pyproject.toml` and `uv.lock` are copied before source, `uv sync --frozen --no-dev` installs runtime deps, port 8000 is exposed, CMD uses `uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000`; no changes needed if already compliant

- [ ] T019 [P] [US5] Verify `ui/Dockerfile` satisfies FR-035–FR-036 by reviewing: same base and `uv` pattern, `uv.lock` copied before source, `uv sync --frozen --no-dev`, port 8501 exposed, CMD uses `uv run streamlit run ui/Introduction.py --server.port=8501 --server.address=0.0.0.0`; no changes needed if already compliant; then run `make docker-build` and confirm both builds exit 0

---

## Phase 8 — US6: Enforce Code Quality & CI/CD Pipeline

> **User Story**: A developer pushes a commit and three automated CI jobs (lint → test → build) gate the merge.  
> **Independent test**: `.github/workflows/ci.yml` is valid YAML; `act` (or GitHub Actions) runs lint job green; `make ci-quality` exits 0 locally.

- [ ] T020 [US6] Add `ci-quality` target to `Makefile` — insert `ci-quality: lint check-format` as a new target body after the `check-format` target; add `ci-quality` to the `.PHONY` declaration on line 4; add a help echo line under `Quality:` section: `@echo "  make ci-quality    Run lint + format check (CI gate)"`; run `make ci-quality` to verify it exits 0

- [ ] T021 [P] [US6] Create `.github/workflows/ci.yml` — create the `.github/workflows/` directory and write the file with: (a) `name: CI`; (b) `on: push` (all branches) and `pull_request` (to `main`); (c) `lint` job on `ubuntu-latest` with steps: `actions/checkout@v4`, `astral-sh/setup-uv@v5`, `uv sync --extra dev`, `uv run ruff check backend/ ui/`, `uv run ruff format --check backend/ ui/`, `uv run mypy backend/`; (d) `test` job on `ubuntu-latest` with `needs: lint`, a `services.postgres` block using `image: postgres:16` with env vars `POSTGRES_USER: lersha`, `POSTGRES_PASSWORD: lersha`, `POSTGRES_DB: test_lersha` and health check options `--health-cmd pg_isready --health-interval 10s --health-timeout 5s --health-retries 5` and `ports: ["5432:5432"]`, job-level `env` block with `DB_URI: postgresql://lersha:lersha@localhost:5432/test_lersha`, `GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}`, `API_KEY: ci-test-key`, and steps: checkout, setup-uv, `uv sync --extra dev`, `uv run pytest backend/tests/ --cov=backend --cov-report=term-missing --cov-fail-under=80`; (e) `build` job on `ubuntu-latest` with `needs: [lint, test]`, steps: checkout, setup-uv, `docker/setup-buildx-action@v3`, `docker build -f backend/Dockerfile -t lersha-backend:ci .`, `docker build -f ui/Dockerfile -t lersha-ui:ci .`

- [ ] T022 [P] [US6] Validate `.github/workflows/ci.yml` is well-formed — run `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` and confirm no `yaml.YAMLError` is raised; also verify the file structure by checking all three job names (`lint`, `test`, `build`) appear in the parsed dict

- [ ] T023 [US6] Run `make lint` and `make ci-quality` locally — fix any ruff violations in the newly created test files; common issues: missing type annotations on fixtures, unused imports, `assert` style in test files (S101 already ignored per `pyproject.toml`); ensure all new files under `backend/tests/` pass `ruff check` before committing

---

## Phase 9 — Polish & Cross-Cutting Concerns

> **Goal**: Verify the complete system works end-to-end and coverage gate is satisfied.

- [ ] T024 Run the full test suite with coverage — execute `make coverage` (i.e., `uv run pytest backend/tests/ --cov=backend --cov-report=html --cov-report=term-missing`) and confirm: (a) all 25+ tests pass with zero failures, (b) coverage report shows ≥ 80% on `backend/core/` and `backend/services/`, (c) `htmlcov/index.html` is generated; if coverage is below 80%, identify the uncovered functions in `backend/core/` or `backend/services/` and add targeted unit tests for the specific branches missing coverage

- [ ] T025 Commit all changes and verify git status — run `git diff --stat` to confirm only the expected files are modified/created: `backend/tests/conftest.py`, `backend/tests/unit/test_feature_engineering.py`, `backend/tests/integration/test_predict_endpoint.py`, `backend/tests/integration/test_db_utils.py`, `.dockerignore`, `.github/workflows/ci.yml`, `docker-compose.yml`, `Makefile`; stage all files and create commit with message `feat(003): full test suite, docker ci/cd pipeline`

---

## Dependency Graph (User Story Completion Order)

```
T001 (verify prerequisites)
  │
  ▼
T002 → T003  (conftest rewrite + verification)
  │
  ├──────────────────────────────┬───────────────────────────────────────┐
  ▼                              ▼                                        ▼
T004 → T005               T006                               T007 → T008 → T009 → T010
US1: Feature Eng          US2: Preprocessing                 US3: API Integration
(parallel with T006)      (parallel with T004/T005)          (needs test_db_engine)
                                │                                         │
                                │                                         │
                         T011 → T012 → T013 → T014 → T015                │
                         US4: DB Integration                (parallel with US3)
                         (needs test_db_engine)
  │
  └── T016 → T017 → T018 → T019    (parallel, US5: Docker)
  │
  └── T020 → T021 → T022 → T023    (US6: CI/CD)
        │
        ▼
      T024 → T025  (Polish: full run + commit)
```

> **MVP scope** (minimum passing CI): T001 → T002 → T003 → T004 → T005 → T006 → T021 → T022
> This gives a green lint + unit-test CI with coverage, without integration tests or Docker build.

---

## Parallel Execution Examples

### Parallel Batch A (after T003)
Run simultaneously in separate terminals:
```bash
# Terminal 1
uv run pytest backend/tests/unit/test_feature_engineering.py -v  # T004/T005

# Terminal 2
uv run pytest backend/tests/unit/test_preprocessing.py -v        # T006
```

### Parallel Batch B (after T003, with test_lersha running)
```bash
# Terminal 1 — write integration/test_predict_endpoint.py   (T007–T010)

# Terminal 2 — write integration/test_db_utils.py           (T011–T015)
```

### Parallel Batch C (infrastructure — any time)
```bash
# Terminal 1 — update docker-compose.yml                    (T016)

# Terminal 2 — create .dockerignore                         (T017)

# Terminal 3 — create .github/workflows/ci.yml              (T021)
```

---

## Implementation Strategy

1. **Start with conftest** (T002–T003) — all other tasks block on this.
2. **Unit tests next** (T004–T006) — fast to verify, no DB needed.
3. **Integration tests** (T007–T015) — require `test_lersha` to be running.
4. **Infrastructure** (T016–T023) — can be done in parallel with integration tests.
5. **Polish** (T024–T025) — run only after all other tasks complete.

This order ensures early feedback: unit tests pass within the first 30 minutes, integration tests within an hour, and infrastructure is validated last before the final commit.
