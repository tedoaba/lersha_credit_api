# Feature Specification: Test Suite, Docker Build System & CI/CD Pipeline

**Feature Branch**: `003-tests-docker-cicd`  
**Created**: 2026-03-29  
**Status**: Draft  
**Input**: User description: "Implement the full test suite, Docker build system, and CI/CD pipeline for the Lersha Credit Scoring System."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Validate Core Business Logic with Unit Tests (Priority: P1)

A developer working on the Lersha Credit Scoring System runs the test suite locally to confirm that the feature-engineering transformations, preprocessing pipeline, and SHAP contribution table produce mathematically correct outputs. Tests run in isolation without a database or external service.

**Why this priority**: Core algorithmic correctness is the highest-risk area; a silent regression in income calculation or SHAP formatting would corrupt every credit score silently.

**Independent Test**: Fully testable by running unit tests against synthetic DataFrames — no database, no model artefact, no network call required.

**Acceptance Scenarios**:

1. **Given** a synthetic farmer DataFrame with known column values, **When** the feature-engineering function is applied, **Then** `net_income` equals `total_estimated_income − total_estimated_cost` to exact arithmetic precision.
2. **Given** a farmer with membership flags for four support organisations, **When** `institutional_support_score` is computed, **Then** the result equals the exact integer sum of those four binary columns.
3. **Given** the feature-engineering output DataFrame, **When** inspected for columns listed in `columns_to_drop_after_feature_engineering`, **Then** none of those columns are present.
4. **Given** a DataFrame with only 2 rows (edge case), **When** `age_group` binning is applied, **Then** the function completes without error and returns a valid categorical column.
5. **Given** known `expectedyieldquintals` and `farmsizehectares` values, **When** `yield_per_hectare` is computed, **Then** the result equals the exact quotient to floating-point precision.
6. **Given** known seed, urea, DAP, and farm-size values, **When** `input_intensity` is computed, **Then** the result equals `(seeds + urea + dap) / farmsizehectares` exactly.
7. **Given** SHAP values in CatBoost format (list of three 2-D arrays), **When** the contribution table is built, **Then** the output is sorted descending by absolute SHAP value without error.
8. **Given** SHAP values in XGBoost multiclass format (3-D ndarray), **When** the contribution table is built, **Then** the output is sorted correctly and all feature rows are present.
9. **Given** SHAP values in binary format (2-D ndarray), **When** the contribution table is built, **Then** the output is sorted correctly.
10. **Given** SHAP value rows that do not match the length of `feature_names`, **When** the contribution table is built, **Then** a `ValueError` is raised immediately.

---

### User Story 2 — Validate Preprocessing Output Matches Trained Feature Schema (Priority: P1)

A developer runs the preprocessing unit tests to ensure that the categorical preprocessing step produces a DataFrame whose columns exactly match the 36-feature schema saved alongside the shipped models — guaranteeing no silent feature-mismatch errors at inference time.

**Why this priority**: A column mismatch between preprocessing output and the trained model's expected features causes silent or hard-to-diagnose prediction failures in production.

**Independent Test**: Fully testable by loading the `36_feature_columns.pkl` fixture and comparing preprocessing output columns.

**Acceptance Scenarios**:

1. **Given** the `36_feature_columns.pkl` fixture is loaded, **When** a synthetic farmer DataFrame is passed through `preprocessing_categorical_features()`, **Then** the output column set exactly matches the fixture's column list.
2. **Given** a farmer DataFrame missing columns that appear in the feature schema, **When** preprocessing completes, **Then** the missing columns are present in the output filled with zero.
3. **Given** the preprocessing output, **When** `output.shape[1]` is compared to `len(feature_columns)`, **Then** they are equal.

---

### User Story 3 — Validate API Endpoints via Integration Tests (Priority: P1)

A developer or CI runner executes the integration test suite, which spins up the FastAPI application in-process (no real HTTP server) and a real test PostgreSQL database, to confirm that authentication, job lifecycle, and error responses all behave correctly end-to-end.

**Why this priority**: Integration tests are the only layer that catches wiring bugs — a correctly implemented function still fails if the router dependency or the database schema is wrong.

**Independent Test**: Testable using an in-process ASGI transport client and a dedicated test schema on a real PostgreSQL instance.

**Acceptance Scenarios**:

1. **Given** a `POST /v1/predict` call with no `X-API-Key` header, **When** the request is processed, **Then** the response is HTTP 403 Forbidden.
2. **Given** a `POST /v1/predict` call with an incorrect `X-API-Key` value, **When** the request is processed, **Then** the response is HTTP 403 Forbidden.
3. **Given** a valid API key and a request body `{"source": "Batch Prediction", "number_of_rows": 2}`, **When** `POST /v1/predict` is called, **Then** the response is HTTP 202 with a `job_id` field containing a valid UUID.
4. **Given** a `job_id` obtained from a successful submission, **When** `GET /v1/predict/{job_id}` is called with a valid API key, **Then** the response is HTTP 200 with a `status` field present.
5. **Given** a `job_id` that does not exist in the database, **When** `GET /v1/predict/{job_id}` is called, **Then** the response is HTTP 404.

---

### User Story 4 — Validate Database Utilities via Integration Tests (Priority: P1)

A developer runs database integration tests against a real test PostgreSQL instance to confirm that every CRUD operation on `candidate_result` and `inference_jobs` tables behaves correctly — including row filtering, batch inserts, and job status transitions.

**Why this priority**: Database utility bugs (e.g., returning the full table instead of a single row) directly corrupt the business logic that depends on them.

**Independent Test**: Testable using a test-scoped PostgreSQL engine that creates and tears down tables around each test run.

**Acceptance Scenarios**:

1. **Given** a database seeded with multiple farmer records, **When** `fetch_raw_data(farmer_uid)` is called, **Then** exactly one row matching that `farmer_uid` is returned — not the full table.
2. **Given** a database seeded with at least 3 rows, **When** `fetch_multiple_raw_data(n=3)` is called, **Then** exactly 3 rows are returned.
3. **Given** a batch of evaluation records prepared in memory, **When** `save_batch_evaluations()` is called, **Then** the correct number of rows appear in the `candidate_result` table.
4. **Given** `create_job(job_id)` is called followed by `get_job(job_id)`, **When** the job is retrieved immediately, **Then** its status is `"pending"`.
5. **Given** a job in `"pending"` status, **When** `update_job_result(job_id, result)` is called, **Then** `get_job(job_id)` returns status `"completed"` and the result payload is populated.

---

### User Story 5 — Build & Ship Services via Docker (Priority: P2)

A DevOps engineer or a developer runs a single command to build production-ready container images for the backend and UI, then brings the full four-service stack (PostgreSQL, backend, UI, MLflow) online with `docker-compose up`.

**Why this priority**: Containerisation is the delivery mechanism for production; if the images don't build or the services don't reach each other, nothing ships.

**Independent Test**: Testable by running `docker build` for each Dockerfile and verifying all four services pass health checks after `docker-compose up`.

**Acceptance Scenarios**:

1. **Given** the project root and `backend/Dockerfile`, **When** `docker build` is run, **Then** the image builds without errors and the final layer contains the application entry point.
2. **Given** the project root and `ui/Dockerfile`, **When** `docker build` is run, **Then** the image builds without errors and the final layer contains the Streamlit entry point.
3. **Given** `docker-compose up --build`, **When** all services start, **Then** all four services (postgres, backend, ui, mlflow) reach a healthy/running state and the backend is reachable on port 8000.
4. **Given** the backend container, **When** `GET /health` is called, **Then** HTTP 200 is returned, confirming the service is running inside the container.
5. **Given** the UI container, **When** port 8501 is polled, **Then** the Streamlit server responds, confirming the UI is running inside the container.
6. **Given** the `ui` service definition in `docker-compose.yml`, **When** the container starts, **Then** the `API_BASE_URL` environment variable is set to `http://backend:8000` and the UI client uses it.

---

### User Story 6 — Enforce Code Quality & Pass CI/CD Pipeline (Priority: P2)

A developer pushes a commit to the repository. Three automated CI jobs run in sequence: linting/type-checking, test suite with coverage gate, and Docker image builds. The pipeline passes only when all three jobs succeed, giving the team a reliable quality gate before any merge.

**Why this priority**: A CI gate prevents regressions from being merged; without it every team member must manually re-run checks, and human error will eventually allow broken code into the main branch.

**Independent Test**: Testable by observing the GitHub Actions workflow status on the commit.

**Acceptance Scenarios**:

1. **Given** a commit with a ruff lint violation, **When** the `lint` CI job runs, **Then** the job fails and the subsequent `test` and `build` jobs do not execute.
2. **Given** a commit where test coverage on `backend/core/` and `backend/services/` falls below 80%, **When** the `test` CI job runs, **Then** the job fails with a coverage gate error.
3. **Given** a commit that passes linting and achieves ≥ 80% coverage, **When** the `build` CI job runs, **Then** both Docker images build successfully and the job is marked green.
4. **Given** the `test` CI job, **When** it runs, **Then** a PostgreSQL 16 service container is available to the tests and `DB_URI`, `GEMINI_API_KEY`, and `API_KEY` are injected from repository secrets.
5. **Given** the `lint` CI job, **When** it runs, **Then** `ruff check`, `ruff format --check`, and `mypy` all pass without errors.

---

### Edge Cases

- What if the test database already has leftover data from a previous failed run? → The `test_db_engine` fixture MUST drop all tables after the test session, ensuring a clean slate on every run regardless of prior state.
- What happens if `36_feature_columns.pkl` is missing from the test fixtures directory? → The test should fail with a clear `FileNotFoundError` pointing to the missing fixture, not a cryptic `AttributeError`.
- What if a SHAP value array has a shape inconsistent with the number of features? → `build_contribution_table` MUST raise `ValueError` with a descriptive message before any further processing.
- What if `GEMINI_API_KEY` is not set in the CI environment? → The `mock_gemini` fixture patches the client at the module level so no real API call is made, making the test independent of the secret's presence during unit test runs.
- What if the Docker build cache is stale? → The `Dockerfile` uses `COPY pyproject.toml uv.lock ./` before `COPY` of source code to maximise layer-cache reuse while ensuring dependency changes always invalidate the install layer.
- What if `docker-compose up` is run without the PostgreSQL volume having been initialised? → The `postgres` service starts fresh; `db_init.py` must be run inside the backend container as a post-start step (or as a Makefile target) before production traffic.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Test Fixtures

- **FR-001**: `backend/tests/conftest.py` MUST define a `test_db_engine` session-scoped fixture that connects to a test PostgreSQL database (`test_lersha`), creates `candidate_result` and `inference_jobs` tables, yields the engine, and drops all tables after the test session.
- **FR-002**: `conftest.py` MUST define a `sample_farmer_df` fixture returning a 5-row synthetic Pandas DataFrame with all required raw farmer schema columns populated with realistic values.
- **FR-003**: `conftest.py` MUST define an `api_client` fixture returning an `httpx.AsyncClient` using `ASGITransport(app=create_app())` — no real HTTP server.
- **FR-004**: `conftest.py` MUST define a `mock_gemini` fixture (using `pytest-mock`) that patches `GeminiClient.models.generate_content` to return a fixed explanation string, making AI calls deterministic and free.

#### Unit Tests — Feature Engineering

- **FR-005**: `backend/tests/unit/test_feature_engineering.py` MUST assert that `net_income = total_estimated_income − total_estimated_cost` holds to exact arithmetic.
- **FR-006**: The file MUST assert that `institutional_support_score` equals the integer sum of the four binary membership columns.
- **FR-007**: The file MUST assert that no column named in `columns_to_drop_after_feature_engineering` appears in the output DataFrame.
- **FR-008**: The file MUST assert that `age_group` binning completes without error on a 2-row DataFrame (edge-case boundary).
- **FR-009**: The file MUST assert that `yield_per_hectare = expectedyieldquintals / farmsizehectares` holds to floating-point precision.
- **FR-010**: The file MUST assert that `input_intensity = (seeds + urea + dap) / farmsizehectares` holds to floating-point precision.

#### Unit Tests — Preprocessing

- **FR-011**: `backend/tests/unit/test_preprocessing.py` MUST load `36_feature_columns.pkl` and pass a synthetic DataFrame through `preprocessing_categorical_features()`.
- **FR-012**: The test MUST assert that output columns exactly match the `.pkl` list (order-insensitive set comparison).
- **FR-013**: The test MUST assert that columns absent from the input DataFrame are present in the output filled with zero.
- **FR-014**: The test MUST assert that `output.shape[1] == len(feature_columns)`.

#### Unit Tests — Contribution Table

- **FR-015**: `backend/tests/unit/test_contribution_table.py` MUST test the CatBoost format: `shap_values` as a list of three 2-D arrays (one per class).
- **FR-016**: The file MUST test the XGBoost multiclass format: `shap_values` as a 3-D ndarray (samples × features × classes).
- **FR-017**: The file MUST test the binary format: `shap_values` as a 2-D ndarray.
- **FR-018**: The file MUST assert that a `ValueError` is raised when `shap_values` row count does not match `len(feature_names)`.
- **FR-019**: The file MUST assert that the output DataFrame is sorted descending by absolute SHAP value.

#### Integration Tests — Predict Endpoint

- **FR-020**: `backend/tests/integration/test_predict_endpoint.py` MUST assert that `POST /v1/predict` without `X-API-Key` returns HTTP 403.
- **FR-021**: The file MUST assert that `POST /v1/predict` with a wrong key returns HTTP 403.
- **FR-022**: The file MUST assert that `POST /v1/predict` with a valid key and `{"source":"Batch Prediction","number_of_rows":2}` returns HTTP 202 with a `job_id` UUID.
- **FR-023**: The file MUST assert that `GET /v1/predict/{job_id}` with a valid key returns HTTP 200 with a `status` field.
- **FR-024**: The file MUST assert that `GET /v1/predict/nonexistent-id` returns HTTP 404.

#### Integration Tests — Database Utilities

- **FR-025**: `backend/tests/integration/test_db_utils.py` MUST assert that `fetch_raw_data(farmer_uid)` returns only the row matching the supplied UID.
- **FR-026**: The file MUST assert that `fetch_multiple_raw_data(n=3)` returns exactly 3 rows.
- **FR-027**: The file MUST assert that `save_batch_evaluations()` inserts the correct number of rows into `candidate_result`.
- **FR-028**: The file MUST assert that `create_job(job_id)` followed by `get_job(job_id)` returns status `"pending"`.
- **FR-029**: The file MUST assert that `update_job_result(job_id, result)` transitions status to `"completed"` and populates the result column.

#### Coverage Gate

- **FR-030**: The test suite MUST achieve ≥ 80% line coverage on `backend/core/` and `backend/services/` as reported by `pytest-cov`.

#### Containerisation — Backend Dockerfile

- **FR-031**: `backend/Dockerfile` MUST use `python:3.12-slim` as the base image and copy the `uv` binary from `ghcr.io/astral-sh/uv:latest`.
- **FR-032**: The Dockerfile MUST copy `pyproject.toml` and `uv.lock` before source code to maximise layer-cache efficiency.
- **FR-033**: The Dockerfile MUST run `uv sync --frozen --no-dev` to install only production dependencies.
- **FR-034**: The Dockerfile MUST expose port 8000 and set `CMD ["uv","run","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000"]`.

#### Containerisation — UI Dockerfile

- **FR-035**: `ui/Dockerfile` MUST use `python:3.12-slim` and the same `uv` binary copy pattern.
- **FR-036**: The Dockerfile MUST expose port 8501 and set `CMD ["uv","run","streamlit","run","ui/Introduction.py","--server.port=8501","--server.address=0.0.0.0"]`.

#### docker-compose

- **FR-037**: `docker-compose.yml` MUST define four services: `postgres` (postgres:16 with a persistent named volume), `backend` (port 8000, mounts for `backend/models`, `mlruns`, `chroma_db`, `logs`, `output`), `ui` (port 8501, `API_BASE_URL=http://backend:8000`, `depends_on: backend`), `mlflow` (ghcr.io/mlflow/mlflow, port 5000, SQLite backend store for development).
- **FR-038**: `docker-compose.yml` MUST declare a `.dockerignore` sibling at the project root that excludes `__pycache__/`, `*.pyc`, `.git/`, `.env`, `backend/tests/`, `*.egg-info/`, `.mypy_cache/`, `.pytest_cache/`.

#### Makefile

- **FR-039**: The project-root `Makefile` MUST expose: `install`, `setup-db`, `setup-chroma`, `lint`, `format`, `check-format`, `ci-quality`, `test`, `coverage`, `api`, `ui`, `mlflow`, `docker-build`, `docker-up`, `docker-down`, `clean`.
- **FR-040**: `make test` MUST invoke `pytest backend/tests/ -v`.
- **FR-041**: `make coverage` MUST invoke `pytest backend/tests/ --cov=backend --cov-report=html --cov-report=term-missing`.
- **FR-042**: `make ci-quality` MUST run `lint` and `check-format` in sequence.
- **FR-043**: `make clean` MUST remove `__pycache__`, `.pytest_cache`, and `htmlcov` directories.

#### CI/CD Pipeline

- **FR-044**: `.github/workflows/ci.yml` MUST define three jobs: `lint`, `test`, `build`.
- **FR-045**: The `lint` job MUST run `ruff check backend/ ui/`, `ruff format --check backend/ ui/`, and `mypy backend/`.
- **FR-046**: The `test` job MUST provision a `postgres:16` service container and run `uv run pytest backend/tests/ --cov-fail-under=80`, with `DB_URI`, `GEMINI_API_KEY` (from repository secrets), and `API_KEY=ci-test-key` injected as environment variables.
- **FR-047**: The `build` job MUST run `docker build -f backend/Dockerfile` and `docker build -f ui/Dockerfile`, and MUST declare `needs: [lint, test]` so it only runs when both prior jobs pass.
- **FR-048**: Every job MUST use the `astral-sh/setup-uv@v5` GitHub Action to install `uv`.

### Key Entities

- **TestDatabaseEngine**: A session-scoped test fixture that owns the lifecycle of the `test_lersha` PostgreSQL schema — create tables before tests, drop tables after.
- **SampleFarmerDataFrame**: A synthetic 5-row Pandas DataFrame conforming to the raw farmer schema; serves as the canonical input for all unit and integration tests.
- **FeatureColumns (36-column schema)**: The list of 36 post-preprocessing feature names persisted in `36_feature_columns.pkl`; the ground truth for preprocessing output validation.
- **CIWorkflow**: The three-job GitHub Actions pipeline (lint → test → build) that gates merges into the main branch.
- **DockerService**: One of the four compose services (postgres, backend, ui, mlflow); each has defined ports, volumes, and environment requirements.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `make test` completes with all tests passing (zero failures, zero errors) on a developer machine with a running `test_lersha` PostgreSQL instance.
- **SC-002**: `make coverage` reports ≥ 80% line coverage on `backend/core/` and `backend/services/` in both the HTML report and terminal output.
- **SC-003**: `make docker-build` completes successfully for both the backend and UI images with no build errors, in under 5 minutes on a standard developer machine.
- **SC-004**: `docker-compose up` brings all four services to a healthy/running state within 60 seconds, with `GET /health` on the backend returning 200.
- **SC-005**: The CI pipeline passes end-to-end (lint → test → build) on every green commit, with the `build` job gated behind both `lint` and `test` succeeding.
- **SC-006**: A commit introducing a lint violation causes the CI `lint` job to fail and prevents the `build` job from executing, with a visible failure in the GitHub Actions UI.
- **SC-007**: A commit that drops coverage below 80% causes the CI `test` job to fail, preventing the `build` job from executing.
- **SC-008**: The preprocessing unit test verifying the 36-column schema passes without modification on a fresh checkout — no manual fixture generation required.

---

## Assumptions

- A PostgreSQL 16 instance named `test_lersha` is available on `localhost` during local test runs; connection details are supplied via the `DB_URI` environment variable.
- The `36_feature_columns.pkl` file is already present in the repository under `backend/models/` or a `backend/tests/fixtures/` directory; no additional generation step is needed before test collection.
- `pytest-cov`, `pytest-mock`, `httpx`, and `anyio` (or equivalent async test backend) are added as dev dependencies in `pyproject.toml`.
- The CI runner has Docker available and able to build multi-stage images (standard GitHub-hosted runners satisfy this).
- `GEMINI_API_KEY` is stored as a GitHub Actions repository secret; the `mock_gemini` fixture ensures unit tests never call the real API, but the environment variable must be present for integration tests that exercise the live pipeline.
- MLflow in `docker-compose.yml` uses a file-backed SQLite store for development convenience; a production deployment would substitue a managed PostgreSQL backend for MLflow.
- The `backend/` and `ui/` Dockerfiles assume a monorepo layout where `pyproject.toml` and `uv.lock` live at the project root.
- `mypy` is configured via `pyproject.toml` for strict-mode type checking limited to the `backend/` package; the `ui/` package is excluded from type checking in the first iteration.
- The `clean` Makefile target removes only generated artefacts (`__pycache__`, `.pytest_cache`, `htmlcov`) and does NOT delete model files, database volumes, or environment files.
