# Implementation Plan: Test Suite, Docker Build System & CI/CD Pipeline

**Branch**: `003-tests-docker-cicd` | **Date**: 2026-03-29 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/003-tests-docker-cicd/spec.md`

---

## Summary

The code foundations are in an advanced state: all three Dockerfiles, the Makefile, `docker-compose.yml`, and three of the five test files are already fully implemented. The remaining work is:

1. **Rewrite `conftest.py`** — add `test_db_engine` (real PG, session-scoped), expand `sample_farmer_df` to 5 rows, switch `api_client` to `httpx.AsyncClient + ASGITransport`, add `mock_gemini`.
2. **Create two integration test files** — `test_predict_endpoint.py` and `test_db_utils.py` in `backend/tests/integration/`.
3. **Add 2-row binning test** — one test added to `test_feature_engineering.py`.
4. **Four infrastructure files** — update `docker-compose.yml` (mlflow service), add `ci-quality` Makefile target, create `.dockerignore`, create `.github/workflows/ci.yml`.

See `research.md` for the full component-by-component gap analysis.

---

## Technical Context

**Language/Version**: Python 3.12  
**Test Framework**: pytest 8.x + pytest-asyncio (auto mode) + pytest-cov + pytest-mock  
**HTTP Test Client**: httpx 0.27+ with `ASGITransport` for async ASGI testing  
**Database**: PostgreSQL 16 (real instance for integration tests; `test_lersha` database)  
**Container Runtime**: Docker + Docker Compose v2  
**CI Platform**: GitHub Actions (ubuntu-latest runners)  
**Package Manager**: uv (astral-sh/setup-uv@v5 in CI)  
**Linter/Formatter**: ruff 0.4.0+  
**Type Checker**: mypy 1.10+ (backend/ only)  
**Coverage Target**: ≥ 80% on `backend/core/` and `backend/services/`  

---

## Constitution Check

*Gate: Must pass before Phase 1 design. Re-checked after Phase 2 implementation.*

| Principle | Rule | Status |
|-----------|------|--------|
| `[P2-PEP]` | ruff check + format; type annotations on all public functions | ✅ New test files use type annotations; conftest fixtures are annotated |
| `[P3-LOG]` | `get_logger(__name__)` in every module; no `print()` | ✅ Tests use `logger` only where needed; conftest uses no logger (acceptable) |
| `[P4-EXC]` | Background task catches all exceptions; calls `update_job_error` | ✅ Already implemented in `predict.py` |
| `[P7-TEST]` | New functions require unit tests; 80% coverage gate | ✅ This feature IS the test suite; coverage gate enforced in CI and Makefile |
| `[P8-DB]` | All DB ops in `db_utils.py`; parameterised queries only | ✅ Integration tests call `db_utils` functions, not raw SQL |
| `[P11-CONT]` | docker-compose.yml exists; Python 3.12-slim | ✅ Both Dockerfiles already implement this |
| `[P12-CI]` | Makefile targets; CI workflow | ⚠️ **Gap**: `ci-quality` target missing; `.github/workflows/ci.yml` missing — **fixed in Phase 2** |

**Violations requiring resolution** (blocking):
1. `[P12-CI]` — `ci-quality` target and CI workflow missing → **fixed in Tasks 4 and 5**

---

## Project Structure

### Documentation (this feature)

```text
specs/003-tests-docker-cicd/
├── plan.md              ← This file
├── spec.md              ← Feature specification
├── research.md          ← Phase 0: gap analysis and decisions
├── data-model.md        ← Phase 1: entities and schemas
├── quickstart.md        ← Phase 1: local dev guide
├── contracts/
│   └── test-contracts.md  ← Phase 1: test interface contracts
└── checklists/
    └── requirements.md  ← Spec quality checklist
```

### Source Code Changes (repository root)

```text
backend/tests/
├── conftest.py                           ← REWRITE (add test_db_engine, mock_gemini; fix api_client, sample_farmer_df)
├── unit/
│   ├── test_feature_engineering.py       ← MODIFY (add 2-row binning test)
│   ├── test_preprocessing.py             ← ✅ No changes
│   └── test_contribution_table.py        ← ✅ No changes
└── integration/
    ├── test_predict_endpoint.py          ← CREATE
    └── test_db_utils.py                  ← CREATE

.dockerignore                             ← CREATE (project root)
.github/
└── workflows/
    └── ci.yml                            ← CREATE

docker-compose.yml                        ← MODIFY (mlflow service: use ghcr.io/mlflow/mlflow)
Makefile                                  ← MODIFY (add ci-quality target)
```

---

## Phase 0: Research

✅ **Complete** — see [research.md](research.md)

Key findings:
- 3 of 5 unit test files already fully implemented
- `conftest.py` exists but requires 3 additions and 1 modification
- 2 integration test files must be created from scratch
- 4 infrastructure changes needed (docker-compose mlflow, Makefile ci-quality, .dockerignore, CI workflow)
- No new Python dependencies required

---

## Phase 1: Design & Contracts

✅ **Complete** — see artifacts:
- [data-model.md](data-model.md) — Fixture entities, table schemas, state transitions, CI job model
- [contracts/test-contracts.md](contracts/test-contracts.md) — Test interface contracts for all test files
- [quickstart.md](quickstart.md) — Local dev guide covering test setup, Docker, and CI

---

## Phase 2: Implementation Tasks

> **Order**: Dependencies flow top-to-bottom. Tasks 1–3 build on each other;
> Tasks 4–6 are independent infrastructure tasks that can be done in parallel.

---

### Task 1 — Rewrite `backend/tests/conftest.py` [FR-001 – FR-004]

**File**: `backend/tests/conftest.py`  
**Change type**: Full rewrite

**What changes**:
1. Add `test_db_engine` — session-scoped fixture. Uses `DB_URI` env var. Creates `candidate_result` and `inference_jobs` tables (raw SQL DDL). Yields engine. Drops all tables via `metadata.drop_all()` in teardown.
2. Expand `sample_farmer_df` from 3 rows → 5 rows. All existing columns preserved. Five farmer UIDs: F-001 through F-005.
3. Replace `api_client` — switch from `fastapi.testclient.TestClient` to `httpx.AsyncClient(transport=ASGITransport(app=create_app()))`. Mark as async fixture with `scope="function"`.
4. Add `mock_gemini` — function-scoped fixture using `pytest-mock`. Patches `google.generativeai.GenerativeModel.generate_content` (or `GeminiClient.models.generate_content` — exact path determined by `backend/chat/` module structure) to return `"Fixed explanation for testing."`.

**Key implementation notes**:
- `test_db_engine` must use `sqlalchemy.text()` for raw DDL
- `api_client` fixture must `os.environ.setdefault` all required env vars before importing `create_app()`
- The `httpx` fixture must be async (`async def api_client()`) — `asyncio_mode = "auto"` in `pyproject.toml` handles this automatically

**Tests guaranteed by Task 1**: `conftest.py` itself is not tested, but all integration tests in Tasks 2–3 depend on it.

---

### Task 2 — Create `backend/tests/integration/test_predict_endpoint.py` [FR-020 – FR-024]

**File**: `backend/tests/integration/test_predict_endpoint.py`  
**Change type**: New file

**Test map**:

| Test function | FR | What it asserts |
|---------------|-----|----------------|
| `test_no_api_key_returns_403` | FR-020 | `POST /v1/predict` with no header → HTTP 403 |
| `test_wrong_api_key_returns_403` | FR-021 | `POST /v1/predict` with wrong key → HTTP 403 |
| `test_valid_predict_returns_202_with_job_id` | FR-022 | Valid key + body → HTTP 202 + `job_id` UUID |
| `test_get_job_status_returns_200` | FR-023 | `GET /v1/predict/{job_id}` with valid key → HTTP 200 + `status` field |
| `test_nonexistent_job_returns_404` | FR-024 | `GET /v1/predict/nonexistent-id` → HTTP 404 |

**Key implementation notes**:
- All tests are `async def` (auto-collected by `asyncio_mode = "auto"`)
- `api_client` fixture used in all tests
- `test_valid_predict_returns_202_with_job_id` uses the `test_db_engine` fixture to ensure `inference_jobs` table exists before the request fires (so `create_job()` doesn't fail)
- Job ID from the 202 response is reused in `test_get_job_status_returns_200`
- Valid API key value = `"ci-test-key"` (set via `os.environ` in `api_client` fixture)

---

### Task 3 — Create `backend/tests/integration/test_db_utils.py` [FR-025 – FR-029]

**File**: `backend/tests/integration/test_db_utils.py`  
**Change type**: New file

**Test map**:

| Test function | FR | What it asserts |
|---------------|-----|----------------|
| `test_fetch_raw_data_returns_matching_row` | FR-025 | Only row with matching `farmer_uid` returned |
| `test_fetch_multiple_raw_data_returns_n_rows` | FR-026 | Exactly 3 rows returned when `n=3` |
| `test_save_batch_evaluations_inserts_correct_count` | FR-027 | Correct row count in `candidate_result` |
| `test_create_and_get_job_round_trip` | FR-028 | Status is `"pending"` immediately after creation |
| `test_update_job_result_sets_completed` | FR-029 | Status = `"completed"`, result populated |

**Key implementation notes**:
- All tests use the `test_db_engine` fixture
- The `db_utils` functions use `config.db_uri` internally — the fixture must set `DB_URI` env var pointing at `test_lersha` before `db_utils` imports `config`
- `save_batch_evaluations` requires a valid `input_df` with farmer name columns and an `evaluation_results` list of dicts with keys: `predicted_class_name`, `top_feature_contributions` (list of dicts), `rag_explanation`, `model_name`
- Seeding: tests that read data must first insert seed rows (using the engine directly or calling `db_utils` write functions)

---

### Task 4 — Add 2-row binning test to `test_feature_engineering.py` [FR-008]

**File**: `backend/tests/unit/test_feature_engineering.py`  
**Change type**: Add one test function

```python
def test_age_group_binning_two_row_edge_case():
    """age_group binning must succeed on a 2-row DataFrame without raising."""
    df = pd.DataFrame({
        "farmer_uid": ["F-001", "F-002"],
        "gender": ["Male", "Female"],
        "age": [25, 55],  # two distinct age values
        "family_size": [4, 6],
        # ... all required columns with 2 rows
    })
    result = apply_feature_engineering(df)
    assert "age_group" in result.columns
    assert result["age_group"].isna().sum() == 0  # no NaN values
```

**Key notes**: The existing `test_age_group_fallback_single_row` already tests the 1-row path. The 2-row test exercises `pd.qcut` which fails and falls back to `pd.cut`. Keep both tests.

---

### Task 5 — Infrastructure: `.dockerignore`, `Makefile`, `docker-compose.yml` [FR-037, FR-038, FR-039, FR-042]

**Files**: 3 files modified/created

**5a — Create `.dockerignore`** (project root):
```
__pycache__/
*.pyc
.git/
.env
backend/tests/
*.egg-info/
.mypy_cache/
.pytest_cache/
htmlcov/
.ruff_cache/
```

**5b — Add `ci-quality` target to `Makefile`**:
```makefile
ci-quality: lint check-format
```
Add `ci-quality` to both the `.PHONY` declaration and the `Quality` section of the help text.

**5c — Update `mlflow` service in `docker-compose.yml`**:
```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root /mlruns
  restart: unless-stopped
  ports:
    - "5000:5000"
  volumes:
    - mlruns_data:/mlruns
```
Replace the `python:3.12-slim` + pip install approach with the official MLflow image.

---

### Task 6 — Create `.github/workflows/ci.yml` [FR-044 – FR-048]

**File**: `.github/workflows/ci.yml`  
**Change type**: New file (new directory)

**Full workflow structure**:

```yaml
name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["main"]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev
      - run: uv run ruff check backend/ ui/
      - run: uv run ruff format --check backend/ ui/
      - run: uv run mypy backend/

  test:
    runs-on: ubuntu-latest
    needs: lint
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: lersha
          POSTGRES_PASSWORD: lersha
          POSTGRES_DB: test_lersha
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    env:
      DB_URI: postgresql://lersha:lersha@localhost:5432/test_lersha
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      API_KEY: ci-test-key
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev
      - run: uv run pytest backend/tests/ --cov=backend --cov-report=term-missing --cov-fail-under=80

  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - uses: docker/setup-buildx-action@v3
      - run: docker build -f backend/Dockerfile -t lersha-backend:ci .
      - run: docker build -f ui/Dockerfile -t lersha-ui:ci .
```

---

### Task 7 — Constitution Re-check (Post-Implementation)

After Tasks 1–6 are complete, verify:

| Check | Pass Condition |
|-------|----------------|
| `[P12-CI]` | `ci-quality` target exists in Makefile; `.github/workflows/ci.yml` is present and valid YAML |
| `[P7-TEST]` | `uv run pytest backend/tests/ --cov-fail-under=80` passes locally with `test_lersha` DB running |
| `[P2-PEP]` | `uv run ruff check backend/ ui/` passes with 0 errors on all new test files |
| Docker | `docker build -f backend/Dockerfile .` and `docker build -f ui/Dockerfile .` succeed |
| Compose | `docker-compose up` starts all 4 services; `curl http://localhost:8000/health` → 200 |

---

## Complexity Tracking

| Decision | Why Needed | Simpler Alternative Rejected Because |
|----------|------------|--------------------------------------|
| `api_client` uses `httpx.AsyncClient + ASGITransport` instead of `TestClient` | Spec explicitly requires this pattern; `asyncio_mode = "auto"` already configured | `TestClient` doesn't support async fixtures properly; new integration tests would need workarounds |
| `test_db_engine` uses raw SQL DDL instead of SQLAlchemy `metadata.create_all()` | Table schemas include `JSONB` and `TIMESTAMPTZ` which SQLAlchemy ORM maps differently in test; raw DDL matches production `db_init.py` exactly | ORM-based creation could diverge from production schema silently |
| MLflow updated to official image | Spec FR-037 specifies `ghcr.io/mlflow/mlflow`; pip-install-in-container is not reproducible | Custom image adds maintenance burden |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `conftest.py` `test_db_engine` conflicts with unit tests (unit tests don't need real PG) | Medium | Low | Make `test_db_engine` session-scoped; unit tests never request it, so it never runs for unit-only runs |
| `mock_gemini` patch path changes if Gemini SDK API changes | Low | Medium | Pin `google-generativeai>=0.5.0,<1.0` in pyproject.toml; patch the exact module path |
| `test_predict_endpoint.py` `POST /v1/predict` triggers background task that calls real pipeline | High | Low | Background task will fail gracefully via `update_job_error`; test only checks the 202 response, not the background outcome |
| Coverage < 80% after new integration tests added (paradoxically) | Low | High | Integration tests add coverage to `db_utils.py` and `predict.py` which are currently low-coverage; net coverage increases |
| GitHub Actions `docker build` fails due to missing `uv.lock` | Low | High | `uv.lock` is committed to the repo; `.dockerignore` must not exclude it |
