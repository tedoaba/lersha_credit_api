# Tasks: Application-Level Security & Reliability Hardening

**Input**: Design documents from `/specs/004-harden-app-security/`
**Branch**: `004-harden-app-security`
**Date**: 2026-03-29

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no blocking dependency on a sibling task)
- **[Story]**: Which user story this task belongs to (US1–US10, maps to spec.md)
- Exact file paths are included in every task description

---

## Phase 1: Setup — Dependency Updates

**Purpose**: Add all missing runtime packages to `pyproject.toml` and sync the lock file. Every subsequent phase depends on this.

- [X] T001 Add `alembic>=1.13`, `celery>=5.3`, `redis>=5.0`, `python-json-logger>=2.0`, `gunicorn>=21.2` to `[project.dependencies]` in `pyproject.toml`
- [X] T002 Run `uv sync` from repo root to install new packages and update `uv.lock` (verify exit code 0)
- [X] T003 Verify imports after sync: `uv run python -c "import alembic, celery, redis, pythonjsonlogger, gunicorn; print('OK')"`

**Checkpoint**: All new packages importable; `uv.lock` committed with new entries.

---

## Phase 2: Foundational — Shared Infrastructure Changes

**Purpose**: Changes that underpin multiple user stories and MUST complete before US1/US2 work begins. All tasks in this phase can be worked in parallel once Phase 1 is done.

- [X] T004 [P] Add `self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")` to `backend/config/config.py` inside `Config.__init__` (after the MLflow section)
- [X] T005 [P] Add `REDIS_URL=redis://redis:6379/0` entry with descriptive comment to `.env.example`
- [X] T006 [P] Rewrite `get_logger()` in `backend/logger/logger.py`: replace `logging.Formatter` with `jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")`; remove `RotatingFileHandler`; keep `StreamHandler` only; preserve function signature for backward compatibility
- [X] T007 [P] Add `[tool.mypy]` section to `pyproject.toml` with: `python_version = "3.12"`, `strict = false`, `ignore_missing_imports = true`, `check_untyped_defs = true`, `exclude = ["backend/tests", "backend/scripts"]`
- [X] T008 [P] Create `.pre-commit-config.yaml` at repo root with `ruff-pre-commit` hooks (`ruff --fix` + `ruff-format`) and `mirrors-mypy` hook (with `--ignore-missing-imports` arg and `types-PyYAML`, `types-requests` additional dependencies)
- [X] T009 Add `pre-commit` and `typecheck` targets to `Makefile`; update `.PHONY` list and help text block
    ```makefile
    pre-commit:
        uv run pre-commit run --all-files

    typecheck:
        uv run mypy backend/
    ```

**Checkpoint**: Config has `redis_url`; logger emits JSON to stdout; mypy config present; pre-commit config present; Makefile has new targets.

---

## Phase 3: User Story 1 — DB Schema Applied Before Traffic (P1) 🎯 MVP

**Goal**: Alembic owns all DDL; schema is applied automatically at container startup; `db_init.py` is data-only.

**Independent Test**: Start app from a fresh empty PostgreSQL database → both `candidate_result` and `inference_jobs` tables exist without any manual SQL.

- [X] T010 [US1] Initialise Alembic: run `cd backend && uv run alembic init alembic` (creates `backend/alembic/env.py`, `backend/alembic/versions/`, `backend/alembic/script.py.mako`, `backend/alembic.ini`)
- [X] T011 [US1] Configure `backend/alembic/env.py`: import `from backend.config.config import config` and `from backend.services.db_model import Base`; call `context.config.set_main_option("sqlalchemy.url", config.db_uri)`; set `target_metadata = Base.metadata` in both online and offline run contexts
- [X] T012 [US1] Generate initial migration: `uv run alembic revision --autogenerate -m "initial_schema"` — review generated file in `backend/alembic/versions/` to confirm `candidate_result` table DDL is present
- [X] T013 [US1] Generate second migration: `uv run alembic revision --autogenerate -m "add_inference_jobs"` — review generated file to confirm `inference_jobs` table DDL is present
- [X] T014 [US1] Remove the `create_candidate_result_table()` function entirely from `backend/services/db_utils.py` (DDL is now owned by Alembic)
- [X] T015 [US1] Remove any call to `create_candidate_result_table()` from `backend/scripts/db_init.py`; ensure `db_init.py` only performs CSV → PostgreSQL data loading via `load_data_to_database()`
- [X] T016 [US1] Update `backend/Dockerfile` CMD to: `CMD ["sh", "-c", "uv run alembic upgrade head && uv run gunicorn backend.main:app --worker-class uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8000 --timeout 120"]`
- [X] T017 [US1] Add a Dockerfile comment above CMD documenting the development alternative: `# Dev: uv run uvicorn backend.main:app --reload --port 8000`
- [X] T018 [US1] Verify end-to-end: `uv run alembic upgrade head` against a test PostgreSQL instance → confirm output ends with `Done.` and no errors

**Checkpoint**: Fresh DB → `alembic upgrade head` → both tables exist → app starts and serves `/health`.

---

## Phase 4: User Story 2 — Async Inference via Celery (P1)

**Goal**: `POST /v1/predict` enqueues a Celery task and returns `202` immediately; a separate worker process executes inference and updates job status.

**Independent Test**: Submit prediction → receive `202 + job_id` in < 500 ms → start Celery worker → poll `GET /v1/predict/{job_id}` until status = `completed`.

- [X] T019 [US2] Create `backend/worker.py`: define `celery_app = Celery("lersha", broker=config.redis_url, backend=config.redis_url)`; define `run_inference_task(job_id: str, payload: dict) -> None` task with `try/except` wrapping `db_utils.update_job_status("processing")` → `match_inputs` → `run_inferences` loop → `db_utils.update_job_result`; catch `Exception` → `db_utils.update_job_error` + `logger.error(exc_info=True)`; use `logger = get_logger(__name__)` at module top
- [X] T020 [US2] Update `backend/api/routers/predict.py`: remove `BackgroundTasks` import and parameter from `submit_prediction`; add `from fastapi import Request` and `from backend.worker import run_inference_task`; replace `background_tasks.add_task(...)` with `run_inference_task.delay(job_id, item.dict())`; add `request: Request` as first positional parameter (required by slowapi in next phase)
- [X] T021 [US2] Remove `_run_prediction_background()` private function from `backend/api/routers/predict.py` (logic now lives in `backend/worker.py`)
- [X] T022 [US2] Add `redis` service to `docker-compose.yml`: `image: redis:7-alpine`, `restart: unless-stopped`, `ports: ["6379:6379"]`
- [X] T023 [US2] Add `depends_on: [redis]` to the `backend` service in `docker-compose.yml`
- [X] T024 [US2] Verify Celery task registration: `uv run celery -A backend.worker inspect registered` → confirm `run_inference_task` appears in the list

**Checkpoint**: Redis running → POST /v1/predict returns 202 immediately → worker processes job → GET /v1/predict/{job_id} returns `completed`.

---

## Phase 5: User Story 3 — Rate Limiting on Predict Endpoint (P2)

**Goal**: A single IP can make at most 10 `POST /v1/predict` requests per minute; the 11th returns `429`.

**Independent Test**: Use curl loop to send 11 requests from same IP in < 1 minute → first 10 return `202`, 11th returns `429`.

- [X] T025 [US3] Add `Limiter` instantiation to `backend/api/dependencies.py`: `from slowapi import Limiter; from slowapi.util import get_remote_address; limiter = Limiter(key_func=get_remote_address)`
- [X] T026 [US3] Update `backend/main.py` `create_app()`: import `limiter` from `backend.api.dependencies`; import `RateLimitExceeded` from `slowapi.errors`; import `_rate_limit_exceeded_handler` from `slowapi`; add `app.state.limiter = limiter`; add `app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)`
- [X] T027 [US3] Apply rate-limit decorator to `submit_prediction` in `backend/api/routers/predict.py`: add `from backend.api.dependencies import limiter` import; add `@limiter.limit("10/minute")` decorator immediately above `@router.post("/")`
- [X] T028 [US3] Verify rate limiting: send 11 requests in a loop → confirm 10 × `202` and 1 × `429` with JSON error body

**Checkpoint**: 11th predict request in a minute from same IP → `429 Too Many Requests`.

---

## Phase 6: User Story 4 — Structured JSON Logging (P2)

**Goal**: Every log line from the `backend/` package is valid JSON with `asctime`, `levelname`, `name`, `message` fields.

**Independent Test**: `make api 2>&1 | python -m json.tool | head -5` produces no parse errors.

*Note*: T006 (Phase 2) already rewrites `get_logger()`. This phase validates and ensures no modules bypass the factory.

- [X] T029 [US4] Audit all Python files in `backend/` for any `print()` calls outside `backend/scripts/` — remove or replace with `logger.*` calls per constitution `[P3-LOG]`
- [X] T030 [US4] Audit all Python files in `backend/` for direct `logging.basicConfig()` or standalone `logging.getLogger()` calls that bypass `get_logger` — refactor to use `get_logger(__name__)`
- [X] T031 [US4] Start the application locally (`make api`) and confirm first log line is valid JSON: `make api 2>&1 | python -m json.tool | head -1`

**Checkpoint**: Application startup log lines parse as JSON; no `print()` or raw logging calls exist in `backend/`.

---

## Phase 7: User Story 5 — Request ID Middleware (P2)

**Goal**: Every API response carries `X-Request-ID` header; inbound ID is echoed, absent ID is auto-generated.

**Independent Test**: `curl -H "X-Request-ID: test-123" http://localhost:8000/health` → response headers include `X-Request-ID: test-123`. `curl http://localhost:8000/health` → response headers include `X-Request-ID: <uuid>`.

- [X] T032 [US5] Create `backend/api/middleware.py`: define `class RequestIDMiddleware(BaseHTTPMiddleware)` with `dispatch()` that reads `X-Request-ID` header or generates `str(uuid.uuid4())`; sets `request.state.request_id`; calls `await call_next(request)`; sets `response.headers["X-Request-ID"] = request_id`; returns response; use `from starlette.middleware.base import BaseHTTPMiddleware`
- [X] T033 [US5] Register `RequestIDMiddleware` in `backend/main.py` `create_app()`: add `from backend.api.middleware import RequestIDMiddleware` import; add `app.add_middleware(RequestIDMiddleware)` call (after limiter setup)
- [X] T034 [US5] Verify: `curl -v -H "X-Request-ID: my-trace-abc" http://localhost:8000/health 2>&1 | grep -i x-request` → confirms `X-Request-ID: my-trace-abc` in response headers

**Checkpoint**: Every response carries `X-Request-ID`; custom header is echoed; missing header generates UUID.

---

## Phase 8: User Story 6 — Live Health Check (P2)

**Goal**: `GET /health` probes real DB and ChromaDB; returns `{"db":"ok","chroma":"ok"}` on success and `503` with failing key on dependency failure.

**Independent Test**: Stop postgres container → `curl http://localhost:8000/health` → `503` with `db` key showing error. Restart → `200` with both keys `"ok"`.

- [X] T035 [US6] Update `backend/api/routers/health.py` `health_check()`: change response key `"postgresql"` → `"db"`; change key `"chromadb"` → `"chroma"`; change success response body to `{"db": "ok", "chroma": "ok"}`; keep `503` on any failure; keep probe logic (`SELECT 1` + `heartbeat()`) unchanged
- [X] T036 [US6] Add Docker healthcheck to `backend` service in `docker-compose.yml`: add `healthcheck:` block with `test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\""]`, `interval: 30s`, `timeout: 10s`, `retries: 3` (use Python urllib to avoid needing curl in slim image)
- [X] T037 [US6] Verify: `curl http://localhost:8000/health` → `{"db":"ok","chroma":"ok"}` with HTTP 200; then stop postgres and re-check for `503`

**Checkpoint**: `/health` returns `{"db":"ok","chroma":"ok"}` exactly; `503` when any dependency is down.

---

## Phase 9: User Story 7 — Tenacity Retries on AI Calls (P3)

**Goal**: Transient failures in `get_rag_explanation()` and `generate_content()` are automatically retried up to 3 times with exponential backoff.

**Independent Test**: Monkey-patch `generate_content` to raise `Exception` twice then succeed → verify `get_rag_explanation` returns the success result without propagating the first two failures.

- [X] T038 [US7] Extract Gemini `generate_content` call from `get_rag_explanation()` into a private helper `_call_gemini(prompt: str) -> str` in `backend/chat/rag_engine.py`; move the `hasattr(response, "text")` extraction logic into this helper
- [X] T039 [US7] Apply retry decorator to `_call_gemini()` in `backend/chat/rag_engine.py`: `from tenacity import retry, stop_after_attempt, wait_exponential`; add `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)` above `def _call_gemini`
- [X] T040 [US7] Apply retry decorator to `get_rag_explanation()` in `backend/chat/rag_engine.py` with same parameters: `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)`
- [X] T041 [US7] Update all HTTP calls in `ui/utils/api_client.py` to use `timeout=(5, 60)` tuple: update `self._session.get(...)` and `self._session.post(...)` calls throughout the class; remove the `self.timeout` int fallback from those calls

**Checkpoint**: `get_rag_explanation` retries up to 3× on transient failure; `ui/utils/api_client.py` has (5, 60) timeout on all requests.

---

## Phase 10: User Story 8 — Gunicorn Multi-Worker (P3)

**Goal**: Production CMD in Dockerfile launches 4 Uvicorn workers via Gunicorn; development make target unchanged.

**Independent Test**: Build Docker image and start container → `docker exec <id> ps aux | grep gunicorn` shows master + 4 worker processes.

*Note*: T016 (Phase 3/US1) already updates the Dockerfile CMD. This phase adds documentation and validation.

- [X] T042 [US8] Verify `backend/Dockerfile` CMD contains the full gunicorn command with `--worker-class uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8000 --timeout 120` (confirm T016 was applied correctly)
- [X] T043 [US8] Confirm `make api` target in `Makefile` still uses `uvicorn backend.main:app --reload` (development mode — single process)
- [X] T044 [US8] Add `worker` service entry comment block to `docker-compose.yml` (as an inline comment template) showing how to add a Celery worker container with `command: uv run celery -A backend.worker worker --loglevel=info --concurrency=4`

**Checkpoint**: Dockerfile CMD = gunicorn multi-worker; `make api` = single-process uvicorn dev server.

---

## Phase 11: User Story 9 — SQLAlchemy Connection Pool (P3)

**Goal**: `db_engine()` creates engines with explicit pool settings; no default pool size in production.

**Independent Test**: Under 50-concurrent-request load, `SELECT count(*) FROM pg_stat_activity WHERE datname = 'lersha'` returns ≤ 30.

- [X] T045 [US9] Update `db_engine()` in `backend/services/db_utils.py`: add keyword arguments `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`, `pool_recycle=3600` to the `create_engine(config.db_uri, ...)` call
- [X] T046 [US9] Add an inline code comment above `db_engine()` documenting the tech-debt: `# TODO: Refactor to module-level singleton via lru_cache to ensure pool is shared across calls`

**Checkpoint**: `create_engine` call includes all 4 pool parameters; tech-debt comment present.

---

## Phase 12: User Story 10 — Code Quality Automation (P3)

**Goal**: Pre-commit hooks block non-compliant commits; mypy reports 0 errors on backend/.

**Independent Test**: `uv run pre-commit run --all-files` exits 0 on a clean codebase; `uv run mypy backend/` exits 0.

*Note*: T007 (Phase 2) adds `[tool.mypy]`; T008 adds `.pre-commit-config.yaml`; T009 adds Makefile targets. This phase validates, runs, and fixes any findings.

- [X] T047 [US10] Run `uv run mypy backend/` → fix any type errors reported on files modified in this feature (primarily `backend/worker.py`, `backend/api/middleware.py`, `backend/api/dependencies.py`, `backend/api/routers/predict.py`)
- [X] T048 [US10] Run `uv run pre-commit run --all-files` → fix any ruff lint or format violations found across the codebase; re-run until exit code is 0
- [X] T049 [US10] Run `uv run pre-commit install` to register hooks (for the developer machine; document this in `quickstart.md` if not already present)
- [X] T050 [US10] Verify CI pipeline file (`.github/workflows/ci.yml`) includes `uv run mypy backend/` step in the `lint` job, after the existing `ruff check` step

**Checkpoint**: `make typecheck` exits 0; `make pre-commit` exits 0; CI lint job includes mypy step.

---

## Phase 13: Polish & Cross-Cutting Concerns

**Purpose**: Final wiring, `.env` documentation, and end-to-end verification across all user stories.

- [X] T051 [P] Update `Makefile` `.PHONY` line to include `pre-commit` and `typecheck` targets, and update the `help` target echo block with descriptions for both new targets
- [X] T052 [P] Update `backend/main.py` docstring/usage comment to reflect new production CMD (gunicorn) and that `RequestIDMiddleware` + rate limiter are registered in `create_app()`
- [X] T053 [P] Ensure `backend/worker.py` module-level docstring describes: purpose, how to start the worker, which task is registered, and what env vars are required
- [X] T054 Run full end-to-end verification using `quickstart.md` steps: health check → rate limit → JSON log parse → X-Request-ID header → alembic upgrade
- [X] T055 Run `make ci-quality` (lint + format check) to confirm the entire codebase passes before PR
- [X] T056 Run `make test` to confirm existing test suite still passes with all changes applied

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup / Deps)
    └─► Phase 2 (Foundational)
            ├─► Phase 3  (US1 — Migrations)       P1 🎯 MVP
            ├─► Phase 4  (US2 — Celery Queue)      P1
            ├─► Phase 5  (US3 — Rate Limiting)     P2  ← depends on US2 handler signature
            ├─► Phase 6  (US4 — JSON Logging)      P2  ← depends on Phase 2 T006
            ├─► Phase 7  (US5 — Request ID)        P2
            ├─► Phase 8  (US6 — Health Check)      P2
            ├─► Phase 9  (US7 — Retries)           P3
            ├─► Phase 10 (US8 — Gunicorn)          P3  ← depends on US1 Dockerfile
            ├─► Phase 11 (US9 — Connection Pool)   P3
            └─► Phase 12 (US10 — Code Quality)     P3
                    └─► Phase 13 (Polish)
```

### Inter-Story Dependencies

| Story | Depends On | Notes |
|---|---|---|
| US1 | Phase 2 complete | Independent |
| US2 | Phase 2 complete | Independent |
| US3 | US2 (handler signature with `request: Request` set in T020) | Minor coupling |
| US4 | Phase 2 T006 | Already done in foundational |
| US5 | Phase 2 complete | Independent |
| US6 | Phase 2 complete | Independent |
| US7 | Phase 2 complete | Independent |
| US8 | US1 T016 (Dockerfile CMD) | Minor coupling |
| US9 | Phase 2 complete | Independent |
| US10 | Phase 2 T007+T008 | Already done in foundational |

### Parallel Opportunities Within Phases

**Phase 2** (all 6 tasks are independent files — run all in parallel):
- T004 `config.py` | T005 `.env.example` | T006 `logger.py` | T007 `pyproject.toml [mypy]` | T008 `.pre-commit-config.yaml` | T009 `Makefile`

**Phase 3** (US1 — sequential; each migration depends on the previous):
- T010 → T011 → T012 → T013 → T014 ‖ T015 ‖ T016 ‖ T017 → T018

**Phase 4** (US2):
- T019 `worker.py` [P] | T022 `docker-compose.yml` [P] can run parallel to T020 `predict.py`
- T021 → after T020 | T023 → after T022 | T024 → after all

**Phase 9** (US7):
- T038 → T039 (sequential, same file) | T041 `api_client.py` [P] (independent file)

---

## Parallel Example: Phase 2 (Foundational)

All 6 foundational tasks touch different files with no inter-dependencies:

```
Parallel batch 1 (all simultaneously):
  Task T004: backend/config/config.py         → add redis_url
  Task T005: .env.example                     → add REDIS_URL entry
  Task T006: backend/logger/logger.py         → swap to JSON formatter
  Task T007: pyproject.toml [tool.mypy]       → add mypy config section
  Task T008: .pre-commit-config.yaml          → create hooks file
  Task T009: Makefile                         → add pre-commit + typecheck targets
```

---

## Implementation Strategy

### MVP First (US1 + US2 — P1 stories only)

1. Complete **Phase 1**: Install dependencies → `uv sync`
2. Complete **Phase 2**: All 6 foundational tasks (JSON logging, config, pre-commit, mypy)
3. Complete **Phase 3** (US1): Alembic migrations + Dockerfile CMD
4. Complete **Phase 4** (US2): Celery worker + predict router update
5. **STOP and VALIDATE**: `alembic upgrade head` → POST /predict → job completes via Celery
6. Deploy / demo P1 stories

### Full Incremental Delivery

- After MVP: add US3 (rate limiting) → validate 429
- Add US4 (JSON logging) → already done in Phase 2, just validate
- Add US5 (request ID) → validate X-Request-ID header
- Add US6 (health check) → validate `{"db":"ok","chroma":"ok"}`
- Add US7 (retries) → already testable via unit test mocking
- Add US8+US9 (gunicorn + pool) → validate via Docker
- Add US10 (code quality) → `make pre-commit` exits 0

---

## Notes

- All `[P]` tasks use different files and have no blocking sibling dependency — safe to run concurrently
- `[Story]` labels map to spec.md user stories for full traceability
- No test tasks generated — tests are not requested in the spec for this feature (existing `make test` suite validates)
- Commit after each phase checkpoint to maintain a clean git history
- The `db_engine()` singleton tech-debt (T046 comment) is tracked but deferred — pool settings are functionally correct even without the singleton
- Pre-commit hooks should be installed (`uv run pre-commit install`) once per developer clone — include in team onboarding
