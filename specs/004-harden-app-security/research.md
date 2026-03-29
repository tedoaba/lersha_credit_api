# Research: Application-Level Security & Reliability Hardening

**Phase 0 — Unknowns Resolved**
**Date**: 2026-03-29

---

## R-001 — Alembic with Existing SQLAlchemy ORM Models

**Decision**: Use `alembic init alembic` from inside `backend/` so `alembic.ini` is co-located. Configure `env.py` to import `config.db_uri` and `Base.metadata` from existing `db_model.py`. Use `--autogenerate` to derive migrations from ORM.

**Rationale**: The codebase already has `CreditScoringRecordDB` and `InferenceJobDB` as SQLAlchemy ORM models with `declarative_base()`. Alembic reads `Base.metadata` directly — no migration SQL needs to be written by hand. The two-phased autogenerate approach (`initial_schema` then `add_inference_jobs`) matches the spec precisely.

**Key finding**: `alembic.ini` must have `sqlalchemy.url` commented out or set to a placeholder — the actual URL is injected in `env.py` via `config.db_uri` to avoid hardcoding credentials.

**Alternatives considered**:
- Manual Alembic migration scripts: Rejected — autogenerate from ORM is more accurate and less error-prone.
- Keeping `create_candidate_result_table()` in `db_init.py`: Rejected — spec FR-003 requires DDL to be removed from the data-loading script. Alembic owns all DDL.

---

## R-002 — Celery vs FastAPI BackgroundTasks

**Decision**: Replace `BackgroundTasks` with Celery + Redis for inference task execution.

**Rationale**: FastAPI `BackgroundTasks` run in the same Uvicorn event loop as the HTTP server. A long-running ML pipeline (XGBoost + RAG) blocks async tasks. Celery workers run in separate processes, giving true isolation, retry/dead-letter support, and visibility into task state.

**Key finding**: The existing inference logic in `_run_prediction_background()` maps 1:1 to a Celery task. The `db_utils` CRUD functions (`update_job_status`, `update_job_result`, `update_job_error`) are already in place — only the invocation mechanism changes.

**Celery task naming**: Use `name="run_inference_task"` explicitly to avoid auto-generated names that include the module path, ensuring stable task IDs across refactors.

**Alternatives considered**:
- RQ (Redis Queue): Simpler but lacks retry decorators, rate-limit support, and monitoring. Celery is already specified.
- Keeping BackgroundTasks: Does not meet the spec requirement for a persistent, independently-scalable job queue.

---

## R-003 — Rate Limiting with slowapi

**Decision**: `slowapi` is already present in `pyproject.toml` (≥0.1.9). Use `Limiter(key_func=get_remote_address)`, attach to `app.state.limiter`, and decorate the `POST /v1/predict` route with `@limiter.limit("10/minute")`.

**Key finding**: `slowapi` requires the FastAPI handler to accept `request: Request` as the first positional parameter (before Pydantic body models). This is a breaking change to the existing handler signature and must be applied carefully.

**Key finding**: `RateLimitExceeded` must be registered as a global exception handler via `app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)`. Without this, exceeded limit exceptions bubble up as 500 errors.

**Alternatives considered**:
- Nginx rate limiting: Out of scope (infrastructure-only constraint).
- Custom middleware rate limiting: More code than needed; slowapi already handles this.

---

## R-004 — Structured JSON Logging with python-json-logger

**Decision**: Replace the plaintext `Formatter` in `get_logger()` with `jsonlogger.JsonFormatter`. Remove the `RotatingFileHandler` — containers write to stdout; the container runtime handles log collection and rotation.

**Import name**: The PyPI package is `python-json-logger` but the import is `from pythonjsonlogger import jsonlogger`. This counter-intuitive name is a known footgun.

**Key finding**: All modules already call `get_logger(__name__)` with no arguments. Making the formatter change in the factory is sufficient — zero call-site changes needed (constitution principle `[P3-LOG]` preserved).

**Rationale for removing file handler**: Docker containers should not write to files inside the container filesystem — they are ephemeral. Docker's log driver captures stdout. Persisting logs to a file inside a container is an anti-pattern.

**Alternatives considered**:
- Keeping file handler in addition to JSON stdout: Adds complexity with no benefit in containerized deployments. The constitution says "rotating file handler" but also says "container stdout preferred"; stdout JSON wins for Phase 8.

---

## R-005 — Request ID Middleware

**Decision**: Use `starlette.middleware.base.BaseHTTPMiddleware`. This is already a dependency of FastAPI (zero new packages). The middleware reads `X-Request-ID` from inbound headers or generates a `uuid4`. The ID is stored on `request.state.request_id` and echoed in the response header.

**Key finding**: Storing the request ID in `request.state` makes it accessible to route handlers and background tasks invoked within the same request lifecycle. Log integration (injecting request_id into every log line during a request) is a follow-up task — it requires either a ContextVar or a custom log filter; not in scope for this implementation phase.

**Alternatives considered**:
- Starlette `RequestIdMiddleware` third-party library: Unnecessary given the straightforward implementation.
- Storing in a thread-local: Incompatible with async FastAPI. `request.state` is the canonical async-safe location.

---

## R-006 — Tenacity Retry Strategy

**Decision**: Wrap `get_rag_explanation()` with `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)`. Extract the `gemini_client.models.generate_content()` call into a private helper `_call_gemini()` that carries the same decorator.

**Rationale**: `tenacity` is already in `pyproject.toml` (≥8.2.0 present). `reraise=True` means after 3 failed attempts the original exception propagates to the caller (the Celery task error boundary) rather than raising a `RetryError` — which matches constitution principle `[P4-EXC]`.

**Key finding**: The `@retry` decorator on `get_rag_explanation` makes the entire function retriable. If the ChromaDB query (`retrieve_docs`) also fails, it will be retried too. Isolating the Gemini call into `_call_gemini` allows independent retry control if needed later.

**Alternatives considered**:
- Manual try/except retry loop: More verbose; tenacity is already present.
- `retry_on_exception` with specific exception types: Good practice but the Gemini SDK throws varied exception types; catching `Exception` at this boundary is justified per constitution.

---

## R-007 — Gunicorn Multi-Worker Production Pattern

**Decision**: Use `gunicorn backend.main:app --worker-class uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8000 --timeout 120` as the production CMD.

**Rationale**: `UvicornWorker` allows Gunicorn to manage worker lifecycle (crash recovery, graceful reload) while Uvicorn's async event loop handles individual requests. `--workers 4` matches a 2-CPU container. `--timeout 120` gives inference tasks time to complete if they somehow reach the HTTP layer synchronously.

**Key finding**: `gunicorn` is NOT in `pyproject.toml` yet. It must be added to `[project.dependencies]` (not just dev) because it runs in production.

---

## R-008 — SQLAlchemy Pool Settings

**Decision**: Add `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`, `pool_recycle=3600` to the `create_engine()` call.

**Key finding (IMPORTANT)**: The current `db_engine()` function creates a **new engine instance on every call**. Connection pooling is meaningless if the engine is recreated on each request — each new engine has its own pool. This is a latent bug. The immediate fix (as required by spec) is to add the pool parameters; the correct fix (follow-up) is to make `_engine` a module-level singleton via `functools.lru_cache` or lazy initialization.

**Pool sizing rationale**: `pool_size=10` + `max_overflow=20` = max 30 connections. With 4 Gunicorn workers and 50 concurrent requests, 30 connections is sufficient for typical synchronous DB calls.

**Alternatives considered**:
- `NullPool` for stateless HTTP handlers: Would break job status polling patterns that expect connection reuse.
- `pool_size=5`: Too small for 4 workers under moderate load.

---

## R-009 — Pre-commit + MyPy Configuration

**Decision**: Pre-commit config uses `ruff-pre-commit` (official Astral repo) for linting and formatting, and `mirrors-mypy` for type checking. MyPy runs in `strict=false` mode with `check_untyped_defs=true` to catch obvious errors without requiring full annotation coverage.

**Key finding**: `mypy>=1.10` and `pre-commit>=3.7` are already in `[project.optional-dependencies.dev]`. No version changes needed. Only `.pre-commit-config.yaml` (new file) and `[tool.mypy]` section in `pyproject.toml` are missing.

**Key finding**: `ruff-pre-commit` hook version (`v0.4.0`) must match the `ruff>=0.4.0` constraint already in dev dependencies to avoid conflicts.

---

## R-010 — Health Check Response Shape

**Decision**: Change health check response keys from `{"postgresql":"ok","chromadb":"ok"}` to `{"db":"ok","chroma":"ok"}` to match the spec contract and the existing `ui/utils/api_client.py` `health()` method expectation.

**Key finding**: The existing `health.py` already performs real PostgreSQL `SELECT 1` and ChromaDB heartbeat probes — the probe logic is correct. Only the JSON response key names need changing. The HTTP status logic (200/503) is also already correct.

**Docker healthcheck**: `curl` must be available in the `python:3.12-slim` image. Slim images do not include `curl` by default. Options: (a) install `curl` via `apt-get` in the Dockerfile, or (b) use a Python-based healthcheck script. Recommendation: use `CMD-SHELL` with `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"` to avoid adding a system dependency. Alternatively, add `RUN apt-get update && apt-get install -y curl --no-install-recommends && rm -rf /var/lib/apt/lists/*` to the Dockerfile.
