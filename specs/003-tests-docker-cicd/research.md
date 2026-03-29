# Research: Test Suite, Docker Build System & CI/CD Pipeline

**Branch**: `003-tests-docker-cicd` | **Date**: 2026-03-29  
**Phase**: 0 — Gap Analysis & Decisions

---

## Component-by-Component Gap Analysis

### 1. `backend/tests/conftest.py`

**Current state**: Exists. Provides `sample_farmer_df` (3 rows, missing some columns), `api_client` (uses sync `TestClient`, not `httpx.AsyncClient`). No `test_db_engine` fixture. No `mock_gemini` fixture.

**Gaps**:
- `test_db_engine`: Entirely missing. Needs session-scoped fixture that creates `candidate_result` + `inference_jobs` tables in `test_lersha` PostgreSQL, yields engine, drops tables after session.
- `sample_farmer_df`: Exists but only 3 rows. Spec requires 5 rows. All columns present ✅.
- `api_client`: Uses `fastapi.testclient.TestClient` (sync). Spec requires `httpx.AsyncClient` with `ASGITransport`. Since `pytest-asyncio` is already in dev deps and `asyncio_mode = "auto"` is set, this is a straightforward change.
- `mock_gemini`: Entirely missing. `pytest-mock` is already in dev deps. Fixture must patch `GeminiClient.models.generate_content`.

**Decision**: Rewrite `conftest.py` fully to satisfy all four fixtures. Keep the existing fixtures working for existing unit tests (the sync `api_client` can remain as a compatibility alias or be replaced — since existing unit tests don't use it, replacing is safe).

---

### 2. Unit Tests — Feature Engineering (`test_feature_engineering.py`)

**Current state**: Exists. Covers all 6 spec requirements:
- `test_net_income_formula` ✅ (FR-005)
- `test_institutional_support_score` ✅ (FR-006)
- `test_dropped_columns_absent` ✅ (FR-007)
- `test_age_group_fallback_single_row` ⚠️ (FR-008 — spec wants 2-row, tests 1-row — functionally equivalent but should match spec precisely)
- `test_yield_per_hectare` ✅ (FR-009)
- `test_input_intensity` ✅ (FR-010)

**Decision**: Add a `test_age_group_binning_two_row_edge_case` test on a 2-row DataFrame to exactly satisfy FR-008. The existing 1-row test can remain as-is (it tests the harder edge case).

---

### 3. Unit Tests — Preprocessing (`test_preprocessing.py`)

**Current state**: Exists. Covers FR-011 through FR-014 fully using a temp `.pkl` fixture. The spec says "Load 36_feature_columns.pkl fixture" but the current implementation already uses a proper `.pkl` fixture pattern (just with 4 columns instead of 36). This is intentional and correct — loading the actual 36-column model artifact would create a coupling between tests and shipped model files.

**Decision**: No changes required. Current implementation satisfies all spec requirements. The temp `.pkl` approach is superior to loading the real model artifact because it makes tests hermetic (the real pkl is only tested indirectly at pipeline integration time).

---

### 4. Unit Tests — Contribution Table (`test_contribution_table.py`)

**Current state**: Exists. Covers all 5 spec requirements:
- CatBoost list format ✅ (FR-015)
- XGBoost 3D multiclass ✅ (FR-016)
- Binary 2D format ✅ (FR-017)
- ValueError on length mismatch ✅ (FR-018)
- Sorted descending by abs SHAP ✅ (FR-019)

**Decision**: No changes required.

---

### 5. Integration Tests — Predict Endpoint (`test_predict_endpoint.py`)

**Current state**: File `backend/tests/integration/test_predict_endpoint.py` does not exist.

**Key design decisions**:

- **Transport**: Use `httpx.AsyncClient(transport=ASGITransport(app=create_app()))` — no real HTTP server needed. Requires `pytest-asyncio` (`asyncio_mode = "auto"` already set in `pyproject.toml`).
- **API key for tests**: Inject `API_KEY=ci-test-key` via `os.environ` inside the fixture or use monkeypatch.
- **Background tasks**: The `POST /v1/predict` test only needs to assert 202 + job_id UUID. The background task will attempt to run the pipeline, which will fail due to no real DB/models. The `_run_prediction_background` will call `update_job_error` — the test doesn't need to wait for it. However, `create_job` is called synchronously before the response, so it WILL fail if no real DB is connected. **Solution**: The `api_client` fixture must mock `db_utils.create_job` and `db_utils.get_job` for the predict endpoint tests, OR use the `test_db_engine` to create the `inference_jobs` table in a real test DB. Given the spec requires both fixtures to co-exist, the integration tests should use the real DB via `test_db_engine`.
- **404 test**: Use a hardcoded UUID that was never inserted.

**Decision**: Integration predict tests require both `api_client` and `test_db_engine` fixtures. The `test_db_engine` fixture ensures the `inference_jobs` table exists. The `api_client` uses `ASGITransport` with env vars pointing at the test DB.

---

### 6. Integration Tests — Database Utilities (`test_db_utils.py`)

**Current state**: `backend/tests/unit/test_db_utils.py` exists (mocks SQLAlchemy session). `backend/tests/integration/test_db_utils.py` does not exist.

**Key design decisions**:
- Must use `test_db_engine` fixture for all tests.
- `fetch_raw_data(table_name, farmer_uid)` signature differs from spec description ("supplied farmer_uid"). The current signature is `fetch_raw_data(table_name, filters)` — the `filters` parameter IS the `farmer_uid` string. Tests must use the correct signature.
- `fetch_multiple_raw_data(table_name, n_rows)` — current signature confirmed.
- `save_batch_evaluations(input_df, evaluation_results)` — requires constructing valid result dicts with required keys.
- `create_job` + `get_job` round-trip — straightforward.
- `update_job_result` sets `status="completed"` — verified from source.

**Decision**: Integration DB tests use a real `test_lersha` PostgreSQL instance seeded via `test_db_engine`. Tests are marked with a custom `@pytest.mark.integration` marker (optional) and use the `DB_URI` env var.

---

### 7. Docker — `backend/Dockerfile`

**Current state**: Fully implemented. Matches spec exactly.

**Decision**: No changes required.

---

### 8. Docker — `ui/Dockerfile`

**Current state**: Fully implemented. Matches spec exactly.

**Decision**: No changes required.

---

### 9. `docker-compose.yml`

**Current state**: Exists. Services: postgres ✅, backend ✅, ui ✅, mlflow ⚠️ (uses `python:3.12-slim` + pip install instead of `ghcr.io/mlflow/mlflow`).

**Gaps**:
- `mlflow` service should use the official `ghcr.io/mlflow/mlflow` image per spec (FR-037). Current implementation works but doesn't match spec.

**Decision**: Update the `mlflow` service to use `ghcr.io/mlflow/mlflow` image with proper `mlflow server` command.

---

### 10. `.dockerignore`

**Current state**: Does not exist.

**Decision**: Create at project root. Exclude: `__pycache__/`, `*.pyc`, `.git/`, `.env`, `backend/tests/`, `*.egg-info/`, `.mypy_cache/`, `.pytest_cache/`.

---

### 11. `Makefile`

**Current state**: Exists. All required targets present except `ci-quality`.

**Gap**: `ci-quality` target (runs `lint` + `check-format` in sequence) is absent from the `.PHONY` declaration and the target body.

**Decision**: Add `ci-quality` target that depends on `lint` and `check-format`.

---

### 12. `.github/workflows/ci.yml`

**Current state**: `.github/` directory does not exist.

**Decision**: Create `.github/workflows/ci.yml` with three jobs:
- `lint`: `astral-sh/setup-uv@v5` + ruff check + ruff format --check + mypy
- `test`: `astral-sh/setup-uv@v5` + postgres:16 service + `uv run pytest --cov-fail-under=80` with env vars
- `build`: `astral-sh/setup-uv@v5` + docker build backend + docker build ui + `needs: [lint, test]`

**Research finding**: `astral-sh/setup-uv@v5` is the current stable version (as of 2026-03). The `github.com/astral-sh/uv` repo confirms v5 is the latest major version of the action.

---

## Key Decisions Summary

| # | Decision | Rationale |
|---|----------|-----------|
| D-01 | Rewrite `conftest.py` completely | Current version is missing 2 of 4 required fixtures and has wrong row count |
| D-02 | Add 2-row binning test without removing 1-row test | Satisfies FR-008 precisely; 1-row test is a valid additional edge case |
| D-03 | Integration tests use real PostgreSQL (`test_db_engine`) not mocks | Mocking SQLAlchemy at integration level defeats the purpose of integration tests |
| D-04 | `api_client` uses `httpx.AsyncClient + ASGITransport` | Matches spec; required for async endpoint testing without a real server |
| D-05 | `mock_gemini` patches `GeminiClient.models.generate_content` | Prevents real API calls in CI; deterministic explanation output |
| D-06 | Update `mlflow` compose service to official image | Matches spec requirement FR-037 |
| D-07 | Add `ci-quality` Makefile target | Required by FR-042; delegates to `lint` and `check-format` |
| D-08 | Create `.dockerignore` at project root | Required by FR-038; reduces image size and prevents test leakage into containers |
| D-09 | `mypy` in CI runs on `backend/` only | Matches assumption in spec; `ui/` excluded in first iteration |
| D-10 | `API_KEY=ci-test-key` hardcoded in CI workflow | Secret rotation avoidance; non-sensitive for test runner |

---

## Dependency Verification

All required packages confirmed present in `pyproject.toml`:

| Package | Section | Required For |
|---------|---------|--------------|
| `pytest>=8.0` | dev | All tests |
| `pytest-cov>=5.0` | dev | Coverage gate |
| `pytest-asyncio>=0.23` | dev | Async integration tests |
| `httpx>=0.27` | dev | `ASGITransport` for `api_client` |
| `pytest-mock>=3.14` | dev | `mock_gemini` fixture |
| `ruff>=0.4.0` | dev | Lint + format CI |
| `mypy>=1.10` | dev | Type check CI |

**No new dependencies need to be added.**

---

## File Inventory: New Files to Create

| File | Status |
|------|--------|
| `backend/tests/conftest.py` | Rewrite (partial → complete) |
| `backend/tests/integration/test_predict_endpoint.py` | Create |
| `backend/tests/integration/test_db_utils.py` | Create |
| `.dockerignore` | Create |
| `.github/workflows/ci.yml` | Create (directory + file) |

## File Inventory: Existing Files to Modify

| File | Change |
|------|--------|
| `backend/tests/unit/test_feature_engineering.py` | Add 2-row binning test |
| `docker-compose.yml` | Update mlflow service image |
| `Makefile` | Add `ci-quality` target |
