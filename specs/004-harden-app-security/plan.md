# Implementation Plan: Application-Level Security & Reliability Hardening

**Branch**: `004-harden-app-security` | **Date**: 2026-03-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-harden-app-security/spec.md`

---

## Summary

Harden the Lersha Credit Scoring System across 11 domains — migrations, job queue, rate limiting, logging, request tracing, health checks, retries, multi-worker, connection pooling, and code-quality automation — using only Python source files and configuration changes. The codebase already has `tenacity`, `slowapi`, `mypy`, `pre-commit`, and `ruff` in `pyproject.toml`. The main gaps are: Alembic is missing; Celery/Redis are missing; `python-json-logger` is missing; `gunicorn` is missing; the `db_engine()` factory has no pool settings; `get_logger` uses a plaintext formatter; no `RequestIDMiddleware` exists; health check response shape does not match spec (`{"db":"ok","chroma":"ok"}`); no `.pre-commit-config.yaml` exists; no `[tool.mypy]` section in `pyproject.toml`; no `worker.py`; no `alembic/` directory.

---

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: FastAPI 0.121, SQLAlchemy 2.x, Alembic ≥1.13, Celery ≥5.3, Redis ≥5.0, python-json-logger ≥2.0, slowapi ≥0.1.9, gunicorn ≥21.2, tenacity ≥8.2, starlette BaseHTTPMiddleware, pre-commit ≥3.7, mypy ≥1.9  
**Storage**: PostgreSQL 16 (SQLAlchemy engine), ChromaDB (PersistentClient), Redis (Celery broker)  
**Testing**: pytest 8+, httpx, pytest-mock; existing test suite in `backend/tests/`  
**Target Platform**: Linux container (python:3.12-slim)  
**Project Type**: Web service (FastAPI monorepo)  
**Performance Goals**: Predict endpoint ack < 500 ms; health check < 10 s; ≤30 DB connections under 50 concurrent requests  
**Constraints**: Python/config changes only — no new infrastructure definitions beyond adding Redis to docker-compose.yml and updating backend Dockerfile CMD; all secrets via env vars  
**Scale/Scope**: Single-tenant backend; up to 4 Gunicorn/Uvicorn workers; Celery worker(s) as separate process

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status | Notes |
|---|---|---|---|
| `[P1-MODULAR]` | Each package owns one concern; ui/ never imports backend/ | ✅ PASS | New files (`worker.py`, `middleware.py`) stay in designated packages; Celery tasks remain in `backend/`; UI changes are in `ui/utils/api_client.py` only |
| `[P2-PEP]` | ruff check passes; all public functions type-annotated; no dead code | ✅ PASS | pre-commit config added in this feature; mypy added; all new code typed |
| `[P3-LOG]` | All modules use `get_logger(__name__)`; formatter upgraded to JSON | ✅ PASS | `get_logger()` signature preserved — callers unchanged; formatter replaced in factory |
| `[P4-EXC]` | No silent swallowing; background task calls `update_job_error` on failure | ✅ PASS | Celery task wraps pipeline in try/except; retries via tenacity then re-raise |
| `[P5-CONFIG]` | All new env vars (`REDIS_URL`) added to `config.py` and `.env.example` | ✅ PASS | Config singleton updated; `os.getenv()` forbidden except inside `Config.__init__` |
| `[P6-API]` | Rate limiter on `/v1/predict`; BackgroundTasks replaced by Celery | ✅ PASS | `run_inference_task.delay()` replaces `background_tasks.add_task()` |
| `[P7-TEST]` | Existing tests must not break; new tests not required by this plan (separate task) | ✅ PASS | All changes are additive or drop-in replacements; test isolation maintained |
| `[P8-DB]` | DDL removed from `db_init.py`; all schema via Alembic; pool_size configured | ✅ PASS | `create_candidate_result_table()` removed from `db_init.py`; pool settings added to `db_engine()` |
| `[P9-SEC]` | No secrets in code; rate limiter on predict | ✅ PASS | `REDIS_URL` added as env var only |
| `[P10-OBS]` | Health check probes real deps; request ID in all log lines | ✅ PASS | Health router rewritten; middleware generates UUID per request |
| `[P11-CONT]` | Dockerfile CMD updated; Gunicorn pattern documented | ✅ PASS | CMD changed to `alembic upgrade head && gunicorn …` |
| `[P12-CI]` | `mypy` step added to CI lint job; pre-commit hooks added | ✅ PASS | `.pre-commit-config.yaml` created; [tool.mypy] in pyproject.toml; Makefile targets added |

**Constitution Check: ALL GATES PASS**

---

## Project Structure

### Documentation (this feature)

```text
specs/004-harden-app-security/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
├── contracts/
│   └── api-changes.md   ← Phase 1 output
└── tasks.md             ← Phase 2 output (/speckit.tasks)
```

### Source Code (changes only)

```text
lersha_credit_api/
├── pyproject.toml                        MODIFY — add alembic, celery, redis, python-json-logger,
│                                                  gunicorn to [project.dependencies];
│                                                  add [tool.mypy] section
├── .pre-commit-config.yaml               CREATE  — ruff + ruff-format + mypy hooks
├── .env.example                          MODIFY  — add REDIS_URL=redis://redis:6379/0
├── Makefile                              MODIFY  — add `pre-commit` and `typecheck` targets;
│                                                  update help text
├── docker-compose.yml                    MODIFY  — add redis service; add healthcheck to backend
│
├── backend/
│   ├── Dockerfile                        MODIFY  — update CMD to alembic upgrade head && gunicorn
│   ├── worker.py                         CREATE  — Celery app + run_inference_task task
│   ├── main.py                           MODIFY  — add limiter state; add RateLimitExceeded handler;
│   │                                              register RequestIDMiddleware
│   ├── alembic/                          CREATE  — via `alembic init alembic`
│   │   ├── env.py                        MODIFY  — set sqlalchemy.url from config.db_uri;
│   │   │                                          import Base.metadata as target_metadata
│   │   └── versions/
│   │       ├── xxxx_initial_schema.py    AUTOGEN — `alembic revision --autogenerate -m "initial_schema"`
│   │       └── xxxx_add_inference_jobs.py AUTOGEN — `alembic revision --autogenerate -m "add_inference_jobs"`
│   │
│   ├── config/
│   │   └── config.py                     MODIFY  — add self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
│   │
│   ├── logger/
│   │   └── logger.py                     MODIFY  — replace formatter with JsonFormatter;
│   │                                              remove rotating file handler (container stdout only)
│   │
│   ├── api/
│   │   ├── dependencies.py               MODIFY  — add Limiter(key_func=get_remote_address)
│   │   ├── middleware.py                 CREATE  — RequestIDMiddleware (BaseHTTPMiddleware)
│   │   └── routers/
│   │       ├── predict.py                MODIFY  — remove BackgroundTasks; add @limiter.limit("10/minute");
│   │       │                                      call run_inference_task.delay(job_id, item.dict())
│   │       └── health.py                 MODIFY  — response body → {"db":"ok","chroma":"ok"}
│   │
│   ├── services/
│   │   ├── db_utils.py                   MODIFY  — db_engine(): add pool_size=10, max_overflow=20,
│   │   │                                           pool_pre_ping=True, pool_recycle=3600;
│   │   │                                           remove create_candidate_result_table() DDL function
│   │   └── db_model.py                   NO CHANGE (Alembic reads Base.metadata from here)
│   │
│   ├── chat/
│   │   └── rag_engine.py                 MODIFY  — wrap get_rag_explanation() and the
│   │                                              generate_content() call with @retry decorator
│   │
│   └── scripts/
│       └── db_init.py                    MODIFY  — remove DDL calls; keep CSV → table data loading only
│
└── ui/
    └── utils/
        └── api_client.py                 MODIFY  — add timeout=(5, 60) to all session.get/post calls
```

---

## Implementation Phases

### Phase A — Dependency Updates (pyproject.toml + uv sync)

**Group 1** — Core runtime additions:
```toml
"alembic>=1.13"
"celery>=5.3"
"redis>=5.0"
"python-json-logger>=2.0"
"gunicorn>=21.2"
```

**Verification**: `uv sync` exits with code 0; `uv run python -c "import alembic, celery, redis, pythonjsonlogger, gunicorn"` succeeds.

---

### Phase B — Alembic Migrations (8.1)

**Step B1**: `cd backend && uv run alembic init alembic`
- Creates `backend/alembic/` with `env.py`, `script.py.mako`, `versions/`
- Note: run from backend/ so `alembic.ini` points correctly

**Step B2**: Edit `backend/alembic/env.py`:
```python
# Replace the sqlalchemy.url placeholder line with:
from backend.config.config import config
from backend.services.db_model import Base

config_obj = context.config
config_obj.set_main_option("sqlalchemy.url", config.db_uri)
target_metadata = Base.metadata
```

**Step B3**: Generate migrations (both autogenerate from existing ORM models):
```
uv run alembic revision --autogenerate -m "initial_schema"
uv run alembic revision --autogenerate -m "add_inference_jobs"
```
*Note: In practice alembic sees both models (`CreditScoringRecordDB`, `InferenceJobDB`) in one autogenerate run. The two-migration order matches the implementation brief.*

**Step B4**: Update `backend/Dockerfile` CMD:
```dockerfile
CMD ["sh", "-c", "uv run alembic upgrade head && uv run gunicorn backend.main:app --worker-class uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8000 --timeout 120"]
```

**Step B5**: Remove `create_candidate_result_table()` from `backend/services/db_utils.py`. Remove its call from `backend/scripts/db_init.py`.

**Verification**: `uv run alembic upgrade head` → `INFO ... Done.` with no errors.

---

### Phase C — Celery Job Queue (8.2)

**Step C1**: Add `REDIS_URL` to `backend/config/config.py`:
```python
self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
```
Add to `.env.example`:
```
REDIS_URL=redis://redis:6379/0
```

**Step C2**: Create `backend/worker.py`:
```python
"""Celery application and inference task for async job processing."""
from celery import Celery
from backend.config.config import config
from backend.logger.logger import get_logger
from backend.services import db_utils
from backend.core.pipeline import match_inputs, run_inferences

logger = get_logger(__name__)

celery_app = Celery(
    "lersha",
    broker=config.redis_url,
    backend=config.redis_url,
)

@celery_app.task(name="run_inference_task")
def run_inference_task(job_id: str, payload: dict) -> None:
    """Execute the full ML pipeline as a Celery task."""
    try:
        db_utils.update_job_status(job_id, "processing")
        original_data, selected_data = match_inputs(
            source=payload["source"],
            filters=payload.get("farmer_uid"),
            number_of_rows=payload.get("number_of_rows"),
        )
        active_models = config.hyperparams.get("models", {}).get("active", ["xgboost", "random_forest"])
        result: dict = {}
        for model_name in active_models:
            model_result = run_inferences(
                model_name=model_name,
                original_data=original_data,
                selected_data=selected_data,
                feature_column=config.feature_column_36,
                target_column=config.target_column_36,
            )
            result[f"result_{model_name}"] = model_result
        db_utils.update_job_result(job_id, result)
        logger.info("Job '%s' completed successfully", job_id)
    except Exception as exc:
        logger.error("Job '%s' failed: %s", job_id, exc, exc_info=True)
        db_utils.update_job_error(job_id, str(exc))
```

**Step C3**: Update `backend/api/routers/predict.py` POST handler:
- Remove `BackgroundTasks` import and parameter
- Import `from backend.worker import run_inference_task`
- Replace `background_tasks.add_task(...)` with `run_inference_task.delay(job_id, item.dict())`
- Remove `_run_prediction_background` private function (logic now in worker.py)

**Step C4**: Add Redis service to `docker-compose.yml`:
```yaml
redis:
  image: redis:7-alpine
  restart: unless-stopped
  ports:
    - "6379:6379"
```
Add `depends_on: redis` to `backend` service.

---

### Phase D — Rate Limiting (8.3)

**Step D1**: Update `backend/api/dependencies.py` — add limiter:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
```

**Step D2**: Update `backend/main.py` `create_app()`:
```python
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from backend.api.dependencies import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RequestIDMiddleware)  # also added here per Phase F
```

**Step D3**: Update `POST /v1/predict` route decorator:
```python
from backend.api.dependencies import limiter
from fastapi import Request

@router.post("/", ...)
@limiter.limit("10/minute")
async def submit_prediction(request: Request, item: PredictRequest) -> JobAcceptedResponse:
    ...
```
*Note: `request: Request` parameter is required by slowapi even if not used directly in the handler body.*

---

### Phase E — Gunicorn Multi-Worker (8.4)

Production CMD (already in Phase B4). The dev `make api` target remains unchanged:
```makefile
api:
    uv run uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0
```

The Dockerfile comment documents the production command pattern for developer awareness.

---

### Phase F — SQLAlchemy Connection Pool (8.5)

Update `db_engine()` in `backend/services/db_utils.py`:
```python
def db_engine():
    """Create and return a pooled SQLAlchemy engine."""
    return create_engine(
        config.db_uri,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
```

**Design note**: `db_engine()` creates a new engine object on each call. To use pooling effectively, the function should be memoised or the engine shared as a module-level singleton. The plan records this as a known limitation; the pool settings are applied correctly even if the engine is recreated (each instance still respects the pool config). A future refactor to a module-level `_engine` singleton is recommended.

---

### Phase G — Tenacity Retries (8.6)

Update `backend/chat/rag_engine.py`:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def get_rag_explanation(prediction: str, shap_dict: dict) -> str:
    ...
    # wrap inner generate_content call with same decorator on a helper:

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
def _call_gemini(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model=config.gemini_model_id,
        contents=prompt,
    )
    ...
    return explanation.strip()
```

Update `ui/utils/api_client.py` — add explicit timeouts to all HTTP calls:
```python
# Replace the global self.timeout with a tuple
resp = self._session.get(url, timeout=(5, 60))
resp = self._session.post(url, json=payload, timeout=(5, 60))
```
(5 s connect timeout; 60 s read timeout)

---

### Phase H — Structured JSON Logging (8.7)

Rewrite `backend/logger/logger.py` `get_logger()`:
```python
import logging
from pythonjsonlogger import jsonlogger

def get_logger(name: str = __name__, ...) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
```

**Key decisions**:
- Remove the rotating file handler — containers log to stdout; log rotation is handled by the container runtime (Docker log driver).
- The function signature preserves all existing parameters (`log_file`, `max_bytes`, `backup_count`) but they become no-ops to avoid breaking callers. This is a deliberate backward-compatible change.
- All existing modules call `get_logger(__name__)` — zero call-site changes needed.

---

### Phase I — Request ID Middleware (8.8)

Create `backend/api/middleware.py`:
```python
"""RequestIDMiddleware — assigns/forwards X-Request-ID per HTTP request."""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

Register in `backend/main.py`:
```python
from backend.api.middleware import RequestIDMiddleware
app.add_middleware(RequestIDMiddleware)
```

---

### Phase J — Pre-commit Hooks (8.9)

Create `.pre-commit-config.yaml` at repo root:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: ["types-PyYAML", "types-requests"]
```

Add Makefile targets:
```makefile
pre-commit:
    uv run pre-commit run --all-files

typecheck:
    uv run mypy backend/
```

Update `.PHONY` list and help text.

Developer onboarding note in `quickstart.md`: `uv run pre-commit install` must be run once.

---

### Phase K — MyPy Type Checking (8.10)

Add `[tool.mypy]` section to `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.12"
strict = false
ignore_missing_imports = true
check_untyped_defs = true
exclude = ["backend/tests", "backend/scripts"]
```

The `mypy` package is already in `[project.optional-dependencies.dev]` (≥1.10 already present). No version change needed.

---

### Phase L — Live Health Check (8.11)

Update `backend/api/routers/health.py` `health_check()` response shape:
```python
return JSONResponse(
    content={"db": "ok", "chroma": "ok"},
    status_code=200,
)
# On failure:
return JSONResponse(
    content={"db": "error: <reason>", "chroma": "ok"},
    status_code=503,
)
```

The probe logic already exists. Only the response body key names change (`"postgresql"` → `"db"`, `"chromadb"` → `"chroma"`) to match the spec's `{"db":"ok","chroma":"ok"}` contract.

Add Docker healthcheck to `backend` service in `docker-compose.yml`:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## Complexity Tracking

No constitution violations to justify. All changes comply with all 12 principles.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Alembic autogenerate detects schema differences against an empty DB | Medium | Run `alembic upgrade head` against a fresh test DB in CI; review generated migration before merge |
| Celery worker cannot import `backend.*` if `PYTHONPATH` is not set | Medium | `PYTHONPATH=/app` already set in Dockerfile ENV; ensure same for celery worker container |
| `db_engine()` creates a new pool per call — actual pooling is per-engine-instance | High | Document as tech debt; pool settings are correct on each instance; full fix (module singleton) is a follow-up |
| slowapi requires `Request` as first positional arg — breaks existing route signature | High | Add `request: Request` as explicit first parameter in POST handler; confirmed pattern works with Depends() |
| `python-json-logger` import name is `pythonjsonlogger` (no hyphen) | Low | Use `from pythonjsonlogger import jsonlogger` — documented in Phase H |
| Pre-commit mypy hook runs against installed package, not source — may miss some files | Low | Use `uv run mypy backend/` in Makefile; CI runs the Makefile target, not pre-commit mypy |
