<!--
SYNC IMPACT REPORT
==================
Version change: N/A → 1.0.0 (initial ratification)
Added sections:
  - Principle 1: Modularity & Separation of Concerns
  - Principle 2: PEP Standards & Code Style
  - Principle 3: Structured Logging
  - Principle 4: Exception Handling
  - Principle 5: Configuration Management
  - Principle 6: API-First & Clean Layer Boundaries
  - Principle 7: Testing Discipline
  - Principle 8: Database Integrity
  - Principle 9: Security
  - Principle 10: Observability & MLflow Tracking
  - Principle 11: Containerization & Environment Parity
  - Principle 12: CI/CD & Automation
  - Governance
Templates requiring updates:
  ✅ .specify/memory/constitution.md — created (this file)
  ⚠  .specify/templates/plan-template.md — pending creation
  ⚠  .specify/templates/spec-template.md — pending creation
  ⚠  .specify/templates/tasks-template.md — pending creation
Deferred TODOs:
  - TODO(RATIFICATION_DATE): Confirm exact adoption date with team; 2026-03-29 used as first-commit date.
-->

# Lersha Credit Scoring System — Project Constitution

**Version:** 1.0.0
**Ratification Date:** 2026-03-29
**Last Amended:** 2026-03-29
**Status:** Active — applies to all development on the `folder-refactor` branch and all future branches.

---

## Purpose

This constitution defines the non-negotiable governing principles for all development work on the
**Lersha Credit Scoring System** codebase. It exists to ensure that every code change — whether a
bug fix, feature addition, or refactoring task — adheres to a shared standard of quality,
maintainability, and production-readiness.

All contributors (human or AI) MUST read and comply with this document before writing or reviewing
code. The architecture reference (`docs/ARCHITECTURE.md`) and execution playbook
(`docs/REFACTOR_PLAN.md`) are subordinate to these principles.

---

## Principle 1 — Modularity & Separation of Concerns

**Name:** MODULAR BOUNDARIES

Every module MUST have a single, clearly stated responsibility. Cross-cutting concerns (logging,
config, DB access) MUST be implemented once and imported everywhere — never duplicated.

### Non-negotiable rules

- The codebase MUST be organized into the `backend/` / `ui/` monorepo layout defined in
  `docs/ARCHITECTURE.md §2`.
- Each `backend/` sub-package MUST own exactly one concern:

  | Package | Single Responsibility |
  |---|---|
  | `backend/api/` | HTTP routing, request/response shapes — zero business logic |
  | `backend/core/` | ML pipeline orchestration and pure data transformations |
  | `backend/services/` | All PostgreSQL CRUD — nothing else touches SQLAlchemy |
  | `backend/chat/` | RAG retrieval + Gemini generation — isolated from core pipeline |
  | `backend/config/` | Environment variable loading and hyperparameter exposure |
  | `backend/logger/` | Logger factory only — no business logic |
  | `backend/scripts/` | One-shot operational scripts — not imported by the application |
  | `ui/` | Streamlit presentation only — ZERO imports from `backend/` |

- Feature engineering (`apply_feature_engineering`) and preprocessing (`preprocessing_categorical_features`)
  MUST remain in separate modules. They MUST NOT be merged into `pipeline.py`.
- The `ui/` layer MUST communicate with the backend **exclusively** via `ui/utils/api_client.py`
  over HTTP. Direct imports of any `backend/` module from `ui/` are **forbidden**.

**Rationale:** Modular boundaries allow each layer to be tested, scaled, and replaced in isolation.
A Streamlit UI that imports ML code cannot be containerized independently or tested without a full
ML environment.

---

## Principle 2 — PEP Standards & Code Style

**Name:** PEP COMPLIANCE

All Python code MUST conform to PEP 8 style, PEP 257 docstring conventions, and PEP 484/526 type
annotation standards. Compliance is enforced automatically — manual review is not a substitute.

### Non-negotiable rules

- **Linter + formatter:** `ruff` is the single tool for linting and formatting. It replaces
  `flake8`, `isort`, and `black`. No other style tool is permitted.
- **Line length:** 120 characters maximum (`line-length = 120` in `pyproject.toml`).
- **Target version:** Python 3.12 (`target-version = "py312"`).
- **Active ruff rule sets:** `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `UP` (upgrade
  to modern syntax), `B` (bugbear), `SIM` (simplify). These MUST NOT be disabled globally.
- **Type annotations:** All public functions MUST have return type annotations. All function
  parameters MUST be type-annotated. Use `Optional[T]` or `T | None` (Python 3.10+ union style).
- **Docstrings:** Every module, class, and public function MUST have a PEP 257-compliant docstring
  describing its purpose, parameters, and return values.
- **Naming conventions:**
  - Modules: `snake_case`
  - Classes: `PascalCase`
  - Functions and variables: `snake_case`
  - Constants: `SCREAMING_SNAKE_CASE`
- **Dead code:** No commented-out code blocks, unused imports, or orphan functions MUST exist in
  any committed file. Remove them or log them as tracked TODOs in issue tickets.
- **Pre-commit enforcement:** `ruff check` and `ruff format` MUST run as pre-commit hooks via
  `.pre-commit-config.yaml`. CI will also enforce these — a failing lint job blocks merge.

**Rationale:** Consistent style reduces cognitive load during code review and prevents style-related
merge conflicts. Automated enforcement removes the burden from reviewers.

---

## Principle 3 — Structured Logging

**Name:** STRUCTURED LOGGING

Every module MUST emit structured, levelled log events. Ad-hoc `print()` statements are forbidden
in any non-script file.

### Non-negotiable rules

- **Logger acquisition:** Every module MUST obtain its logger at the top of the file using:
  ```python
  from backend.logger.logger import get_logger
  logger = get_logger(__name__)
  ```
  This ensures log lines carry the fully-qualified module name (e.g., `backend.core.pipeline`).

- **Log format:** Log lines MUST include: timestamp, log level, module name, and message.
  The `backend/logger/logger.py` factory MUST configure both:
  - A rotating file handler: `logs/credit_scoring_model.log`, 10 MB per file, 5 rotations.
  - A console (stdout) handler.
  Format token: `[YYYY-MM-DD HH:MM:SS] [LEVEL] [module.name] message`

- **Structured JSON logging (production):** In containerized production deployments, the formatter
  MUST be replaced with `python-json-logger` so log aggregators (Datadog, CloudWatch, Loki) can
  parse individual fields. Each JSON log line MUST include `request_id` when available.

- **Log levels — correct usage:**
  | Level | When to use |
  |---|---|
  | `DEBUG` | Detailed trace data useful during development only |
  | `INFO` | Normal operational events (job started, model loaded, records processed) |
  | `WARNING` | Degraded but recoverable state (retry triggered, fallback path taken) |
  | `ERROR` | Failures that were caught and handled (job failed, DB write error) |
  | `CRITICAL` | Unrecoverable application state (config missing, model file not found) |

- **What MUST be logged:**
  - Inference job start and completion (with `job_id` and `records_processed`)
  - Every external call to Gemini or ChromaDB (with timing if feasible)
  - All caught exceptions (with full `exc_info=True` traceback)
  - Configuration validation failures at startup

- **What MUST NOT be logged:**
  - Raw `farmer_uid` or PII in `INFO`/`DEBUG` messages (only in `DEBUG` with a note)
  - API keys, DB passwords, or any secret values — ever

- **`print()` is forbidden** in `backend/` and `ui/` application code. Scripts in
  `backend/scripts/` may use `print()` for operator feedback only.

**Rationale:** Structured logs make production incident response tractable. Without correlated,
levelled log lines, debugging a failed inference job requires code archaeology rather than a log
query.

---

## Principle 4 — Exception Handling

**Name:** EXPLICIT EXCEPTION HANDLING

All exceptions MUST be caught, logged with full context, and re-raised or converted to typed errors.
Silent swallowing of exceptions is forbidden. Naked `except:` clauses are forbidden.

### Non-negotiable rules

- **Catch specifically:** Always catch the most specific exception type available. Use bare
  `except Exception` only as a last-resort boundary (e.g., background task wrappers, health checks)
  and always log the exception with `exc_info=True`.

- **Never silence:** A caught exception MUST be either:
  1. Logged with `logger.error("...", exc_info=True)` and re-raised, or
  2. Logged and converted to a structured error response or job failure record.

- **FastAPI error responses:** API route handlers MUST NOT leak raw Python exceptions to callers.
  All unhandled errors in routers MUST be caught by a registered FastAPI exception handler and
  returned as structured JSON with a consistent `{"detail": "...", "type": "..."}` shape.

- **Background task error boundary:** The `_run_prediction_background` wrapper MUST catch all
  exceptions, call `db_utils.update_job_error(job_id, str(exc))`, and log with `exc_info=True`.
  It MUST NOT silently exit, leaving the job row in `pending` state forever.

- **External call resilience:** Calls to Gemini API and ChromaDB MUST be wrapped with `tenacity`
  retry decorators (3 attempts, exponential back-off, min 2s, max 10s). After exhausting retries,
  the exception MUST be re-raised for the background task boundary to catch.

- **Startup validation:** `backend/config/config.py` MUST raise `ValueError` at import time if any
  required environment variable (`GEMINI_API_KEY`, `GEMINI_MODEL`, `DB_URI`, `API_KEY`) is absent.
  It MUST NOT fall back to a default value for secrets.

- **Type-annotated custom exceptions:** Where domain-specific error semantics are needed, define
  typed exception classes:
  ```python
  class InferenceError(RuntimeError): ...
  class DatabaseError(RuntimeError): ...
  ```
  These MUST be placed in `backend/core/exceptions.py` and `backend/services/exceptions.py`
  respectively.

**Rationale:** In a credit scoring system, a silently failed inference is more dangerous than a
visible error. An unlogged exception means no audit trail, no alert, and no ability to replay or
reprocess the affected farmers' evaluations.

---

## Principle 5 — Configuration Management

**Name:** SINGLE-SOURCE CONFIG

All configuration (environment variables, file paths, hyperparameters) MUST be read in exactly one
place: `backend/config/config.py`. Every other module imports the pre-built singleton.

### Non-negotiable rules

- **Singleton import:** All modules MUST import `from backend.config.config import config`.
  No module MUST instantiate `Config()` directly. No module MUST call `os.getenv()` directly.

- **Required variable hard-fail:** Missing required variables (`GEMINI_API_KEY`, `GEMINI_MODEL`,
  `DB_URI`, `API_KEY`) MUST raise a `ValueError` with a clear message at startup — not silently
  default to `None` or an empty string.

- **Hyperparameters in YAML:** All tuning knobs (batch sizes, SHAP sample counts, RAG top-K)
  MUST live in `backend/config/hyperparams.yaml`, not hard-coded in business logic. These MUST be
  loaded once in `config.py` and exposed via `config.hyperparams`.

- **No hard-coded paths:** File paths to models, data, ChromaDB, MLflow, and output directories
  MUST be resolved via `config`, not as string literals inside business logic.

- **`.env.example` must stay current:** Every new environment variable added to `config.py` MUST
  be simultaneously documented in `.env.example` with a placeholder value and comment explaining
  its purpose.

- **`BASE_DIR` contract:** `BASE_DIR` in `config.py` MUST resolve to the project root
  (`Path(__file__).resolve().parents[2]`). A startup assertion `assert Path(config.xgb_model_36).exists()`
  MUST be present to catch a wrong `BASE_DIR` immediately.

**Rationale:** Scattered `os.getenv()` calls make it impossible to audit what environment variables
the application depends on without reading every file. A single config object makes the dependency
surface explicit and testable.

---

## Principle 6 — API-First & Clean Layer Boundaries

**Name:** API-FIRST ARCHITECTURE

The Streamlit UI and any external consumer MUST interact with the backend exclusively through the
versioned FastAPI REST API. No component MUST bypass the API layer to call business logic directly.

### Non-negotiable rules

- **Versioned routes:** All API routes MUST carry a version prefix. Current version: `/v1/`.
  Example: `POST /v1/predict`, `GET /v1/results`. Unversioned routes break consumers silently.

- **API routers contain zero business logic:** Route handler functions MUST only: validate the
  incoming request (via Pydantic), delegate to a service/core function, and return a typed response.
  Inference logic, DB queries, and LLM calls MUST live in `backend/core/` or `backend/services/`.

- **Pydantic for all I/O boundaries:** Every API request and response shape MUST be defined as a
  Pydantic model in `backend/api/schemas.py`. No raw `dict` types allowed as function signatures
  at the HTTP boundary.

- **Authentication on all non-health routes:** The `require_api_key` dependency MUST be applied
  via `router = APIRouter(dependencies=[Depends(require_api_key)])` on all routers except health.
  Individual route-level overrides to bypass auth are **forbidden** without a documented rationale.

- **Async inference pattern:** `POST /v1/predict` MUST return `202 Accepted` immediately with a
  `job_id`. Inference MUST run in a `BackgroundTask`. `GET /v1/predict/{job_id}` MUST poll job
  state from the DB. Synchronous inference that blocks the HTTP response is **forbidden**.

- **Rate limiting:** The `/v1/predict` route MUST apply a `slowapi` rate limit (configurable via
  `hyperparams.yaml`). The default is 10 requests per minute per IP.

- **API versioning sunset policy:** When a breaking change is introduced, the old version MUST
  remain operational for a minimum of 30 days. A `Deprecation` response header MUST be added to the
  old-version routes announcing the sunset date.

**Rationale:** The API-first constraint is what allows the UI and backend to be containerized,
scaled, and tested in complete isolation. Bypassing it creates invisible coupling that defeats the
entire architectural design.

---

## Principle 7 — Testing Discipline

**Name:** TEST COVERAGE

All business logic MUST be covered by automated tests. A feature is not complete until its tests
pass. No code that reduces coverage below threshold MUST be merged.

### Non-negotiable rules

- **Minimum coverage:** 80% line coverage on `backend/core/` and `backend/services/`. Enforced
  by `--cov-fail-under=80` in CI. The coverage gate blocks merge — it is not advisory.

- **Test structure MUST follow the source layout:**
  ```
  backend/tests/
  ├── conftest.py          # shared fixtures only
  ├── unit/               # pure function tests — no DB, no HTTP, no models
  └── integration/        # real DB or real HTTP (TestClient), no real LLM
  ```

- **Unit tests are pure:** Unit tests MUST NOT connect to PostgreSQL, the real Gemini API, or load
  `.pkl` model files. They test pure Python functions with synthetic inputs.

- **Integration test isolation:** Integration tests MUST use a dedicated `test_lersha` PostgreSQL
  database (never the dev/prod DB). Tables are created in `conftest.py` fixtures and torn down
  after each test session.

- **LLM mocking is mandatory:** All integration tests MUST mock `GeminiClient.models.generate_content`
  using `pytest-mock`. Real Gemini API calls in tests are **forbidden** (API cost, non-determinism,
  rate limits).

- **Test naming convention:** Test functions MUST describe what they verify, not how:
  - ✅ `test_fetch_raw_data_returns_only_matching_farmer()`
  - ❌ `test_db_utils_1()`

- **conftest.py for shared fixtures:** DB connections, API clients, and sample DataFrames MUST be
  defined as pytest fixtures in `conftest.py`. Duplicating fixture setup inside individual tests
  is forbidden.

- **New features require new tests:** A PR that adds a function without a corresponding test
  MUST NOT be merged unless the function is explicitly excluded from coverage (with a documented
  reason via `# pragma: no cover`).

**Rationale:** The credit scoring system operates in a regulated financial domain. Untested
inference logic can produce systematically wrong eligibility decisions affecting farmers' access to
credit with no detection mechanism.

---

## Principle 8 — Database Integrity

**Name:** DATABASE INTEGRITY

All database interactions MUST go through the services layer. Schema migrations MUST be versioned
and reversible. Raw SQL outside designated modules is forbidden.

### Non-negotiable rules

- **Single DB module:** All PostgreSQL operations MUST be implemented in `backend/services/db_utils.py`.
  No other module MUST call SQLAlchemy `Session`, `engine.connect()`, or `pd.read_sql()` directly.

- **Parameterized queries only:** All SQL queries MUST use SQLAlchemy's `:param` style with
  `params={"key": value}`. String interpolation into SQL (f-strings, `%` formatting) is **forbidden**
  as it enables SQL injection.

- **Alembic for schema migrations:** Once integrated, ALL schema changes (add column, rename table,
  change type) MUST be implemented as Alembic migration scripts in `backend/alembic/versions/`.
  DDL in `db_init.py` is for initial bootstrapping only — it MUST NOT be extended for schema evolution.

- **Pydantic before ORM writes:** All writes to `candidate_result` MUST pass through the
  `CreditScoringRecord` Pydantic model for validation before mapping to the `CreditScoringRecordDB`
  SQLAlchemy ORM model.

- **Connection pool settings:** The SQLAlchemy engine MUST be configured with explicit pool settings:
  `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`, `pool_recycle=3600`. Default pool
  settings MUST NOT be used in production.

- **Job state management:** Background task boundaries MUST call `update_job_error()` on any
  exception to prevent rows from being stranded in `pending` state. Orphaned `pending` rows older
  than `config.hyperparams.job_timeout_minutes` MUST be detectable via a monitoring query.

- **Backups for production:** The `candidate_result` table contains the audit trail for all
  credit evaluations. Production deployments MUST have automated daily `pg_dump` backups with
  minimum 30-day retention stored in object storage.

**Rationale:** All credit scoring decisions are persisted in PostgreSQL. A DB layer violation
(direct SQL from the pipeline, unparameterized queries) can corrupt the audit trail or expose the
system to injection attacks — neither is acceptable in a financial scoring context.

---

## Principle 9 — Security

**Name:** SECURITY BY DEFAULT

Security controls are architectural requirements, not optional hardening. They MUST be implemented
from the first deployed version.

### Non-negotiable rules

- **API key authentication:** All non-health routes MUST require the `X-API-Key` header.
  The key MUST be loaded from environment variable `API_KEY`. Absence of `API_KEY` at startup MUST
  raise `ValueError` and prevent the server from starting.

- **No secrets in code or logs:** API keys, DB passwords, and Gemini credentials MUST NOT appear
  in source code, log files, or CI output. Use `.env` locally; use a secrets manager in production.

- **`.env` is git-ignored:** The `.env` file MUST be listed in `.gitignore`. Only `.env.example`
  (with placeholder values) MUST be committed to version control.

- **HTTPS for all production traffic:** All production HTTP traffic MUST be terminated at a reverse
  proxy (Nginx or Caddy) that enforces HTTPS. Direct access to `backend:8000` or `ui:8501` from
  outside the Docker network is **forbidden** in production.

- **Rate limiting on inference endpoints:** The `POST /v1/predict` route MUST be rate-limited
  (10 requests/minute/IP by default via `slowapi`). This protects Gemini API quota and prevents
  resource exhaustion.

- **Secrets management path:** Local dev uses `.env`. Production deployments MUST use an appropriate
  secrets manager (Docker Secrets, AWS Secrets Manager, GCP Secret Manager, or Kubernetes Secrets).

- **No PII in logs or responses:** Farmer PII (names, UIDs) MUST NOT appear in API error responses
  returned to callers. Log lines at `INFO` or above MUST NOT include raw PII field values.

**Rationale:** The system scores Ethiopian smallholder farmers for agricultural credit eligibility.
Exposure of farmer PII or scoring results to unauthorized callers violates both data ethics and
likely local data protection regulations. Security is not a post-launch concern.

---

## Principle 10 — Observability & MLflow Tracking

**Name:** OBSERVABLE SYSTEM

Every inference run MUST be tracked in MLflow. Every module MUST emit structured log events. The
system's operational state MUST be queryable without reading source code.

### Non-negotiable rules

- **MLflow experiment tracking:** Every call to `run_inferences()` MUST open a nested MLflow run
  under the `"Credit Scoring Model"` experiment. The following MUST be logged:
  - Params: `records_processed`, `num_columns`, `model_name`
  - Metrics: `prediction_{idx}` (float class index) per row
  - Artifacts: SHAP JSON and PNG summary plots

- **Live health check:** `GET /health` MUST check real dependencies (PostgreSQL `SELECT 1` and
  ChromaDB heartbeat). A static `{"status": "ok"}` response is **forbidden**. The endpoint MUST
  return `503` with a dependency breakdown when any dependency is unreachable.

- **Request ID propagation:** The `RequestIDMiddleware` in `backend/api/middleware.py` MUST
  generate or forward a `X-Request-ID` UUID per request and inject it into all log lines emitted
  during that request's lifetime.

- **MLflow production backend:** In production, the MLflow tracking store MUST use PostgreSQL
  (not SQLite) and the artifact store MUST use object storage (S3/GCS). SQLite is unsafe for
  concurrent write access from multiple Celery workers.

- **Model versioning:** Model `.pkl` files MUST eventually be registered in the MLflow Model
  Registry and loaded by stage (`models:/lersha-xgboost/Production`), not by filename. This is
  a Phase 8 requirement and MUST be planned for before adding new model versions.

**Rationale:** Without MLflow tracking, there is no way to compare prediction distributions across
model versions or replay an inference to investigate a disputed credit decision. Observability is
the audit trail for a regulated system.

---

## Principle 11 — Containerization & Environment Parity

**Name:** CONTAINER PARITY

The application MUST run identically in development, CI, and production via Docker. No environment-
specific code paths are permitted in application modules.

### Non-negotiable rules

- **Docker for all services:** Backend, UI, PostgreSQL, ChromaDB (via volume mount), and MLflow
  MUST all be defined in `docker-compose.yml`. There MUST be no manual setup steps not captured
  in a `make` target or Dockerfile.

- **uv for all dependency management:** `pyproject.toml` is the single source of truth for
  dependencies. `uv.lock` MUST be committed and cannot be stale. No dependency MUST be installed
  via `pip install` directly — use `uv add`.

- **Multi-stage compose files:**
  - `docker-compose.yml` — base definitions (no env values hard-coded)
  - `docker-compose.override.yml` — dev overrides (bind mounts, hot reload)
  - `docker-compose.prod.yml` — prod overrides (Gunicorn workers, prod DB URIs)

- **Production startup:** Production containers MUST use Gunicorn as the process manager with
  Uvicorn workers: `--workers 4 --worker-class uvicorn.workers.UvicornWorker`. Single-worker
  `uvicorn --reload` is for development only.

- **Persistent volumes:** ChromaDB (`chroma_db/`), MLflow runs (`mlruns/`), model artifacts
  (`backend/models/`), and output files (`output/`) MUST be mounted as Docker volumes to survive
  container restarts.

- **Python version pinned:** All environments MUST use Python 3.12. The base image in all
  Dockerfiles MUST be `python:3.12-slim`.

**Rationale:** "Works on my machine" failures are eliminated when the CI environment and production
environment run the same container image built from the same lockfile.

---

## Principle 12 — CI/CD & Automation

**Name:** AUTOMATED QUALITY GATES

All quality checks MUST be automated. No code that fails lint, format, type checks, or tests MUST
reach the `main` branch.

### Non-negotiable rules

- **CI runs on every push and PR to `main` or `refactor/**` branches.**
- **CI jobs (in order):**
  1. `lint` — `uv run ruff check backend/ ui/` and `uv run ruff format --check backend/ ui/`
  2. `mypy` — `uv run mypy backend/` (non-blocking initially; becomes blocking after type annotation
     coverage reaches 80%)
  3. `test` — `uv run pytest backend/tests/ --cov=backend --cov-fail-under=80` (requires postgres
     service in CI)
  4. `build` — Docker image builds for `backend/` and `ui/` (requires lint + test to pass)

- **Merge to `main` is blocked** if any of jobs 1, 3, or 4 fail.
- **Pre-commit hooks:** `ruff check --fix`, `ruff format`, and `mypy` MUST run as pre-commit hooks.
  Developers MUST run `uv run pre-commit install` after cloning the repository.
- **Makefile is the developer entry point:** All common tasks (lint, format, test, coverage, api,
  ui, docker-up, setup-db, setup-chroma) MUST be accessible via `make <target>`. Running raw
  `uv run` commands is acceptable but the Makefile target MUST always exist as the canonical form.
- **Secrets in CI:** `GEMINI_API_KEY` and real DB credentials MUST be stored as GitHub Actions
  secrets. They MUST NOT appear in `ci.yml` as plain text.

**Rationale:** Automated gates remove dependency on reviewer vigilance for mechanical quality
checks. They enforce the same bar on every contribution regardless of contributor experience level.

---

## Governance

### Amendment Procedure

1. Propose an amendment via a GitHub Pull Request that modifies this file.
2. The PR description MUST explain: what changed, why, and what version bump applies
   (MAJOR/MINOR/PATCH per semantic versioning rules below).
3. At least one other contributor MUST review and approve the PR.
4. On merge, `LAST_AMENDED_DATE` is updated to the merge date and `CONSTITUTION_VERSION` is
   incremented.

### Versioning Policy

| Change type | Bump |
|---|---|
| Principle removal or backward-incompatible redefinition | MAJOR (2.0.0) |
| New principle added or material expansion of existing principle | MINOR (1.1.0) |
| Wording clarification, typo fix, non-semantic refinement | PATCH (1.0.1) |

### Compliance Review

- This constitution MUST be reviewed and re-confirmed (or amended) at the start of each new
  major development phase (Phase 0, 1, 2, ... as defined in `docs/REFACTOR_PLAN.md`).
- Any AI agent or contributor implementing tasks from `tasks.md` MUST cite the relevant principle
  in commit messages when the change is directly governed by a principle.
  Example: `fix(db): parameterize all SQL queries [P8-DB-INTEGRITY]`
- Principle violations discovered during code review MUST be treated as blocking comments, not
  suggestions.

---

## Quick Reference — Principle Tags

| Tag | Principle |
|---|---|
| `[P1-MODULAR]` | Modularity & Separation of Concerns |
| `[P2-PEP]` | PEP Standards & Code Style |
| `[P3-LOG]` | Structured Logging |
| `[P4-EXC]` | Exception Handling |
| `[P5-CONFIG]` | Configuration Management |
| `[P6-API]` | API-First & Clean Layer Boundaries |
| `[P7-TEST]` | Testing Discipline |
| `[P8-DB]` | Database Integrity |
| `[P9-SEC]` | Security |
| `[P10-OBS]` | Observability & MLflow Tracking |
| `[P11-CONT]` | Containerization & Environment Parity |
| `[P12-CI]` | CI/CD & Automation |
