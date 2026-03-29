# Implementation Plan: FastAPI Backend & Streamlit HTTP Integration

**Branch**: `002-fastapi-backend-http-wire` | **Date**: 2026-03-29 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/002-fastapi-backend-http-wire/spec.md`

---

## Summary

The backend is already 90%+ implemented from the prior architectural refactor. The remaining work is:
1. Two small gaps in existing backend modules (`update_job_status` in `db_utils.py`; strict `FileNotFoundError` guard in `config.py`)
2. A full rewrite of both Streamlit pages (`New_Prediction.py`, `Dashboard.py`) to remove all direct backend/config imports and replace them with `LershaAPIClient` HTTP calls

See `research.md` for the full component-by-component gap analysis.

---

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: FastAPI 0.115+, Pydantic v2, SQLAlchemy 2.x, Streamlit ~1.40, requests, PyYAML, python-dotenv  
**Storage**: PostgreSQL (via SQLAlchemy 2.x engine; `inference_jobs` + `candidate_result` tables)  
**Testing**: pytest + pytest-cov + pytest-mock (unit in `backend/tests/unit/`, integration in `backend/tests/integration/`)  
**Target Platform**: Linux containers (Python 3.12-slim); dev on Windows  
**Project Type**: Web service (FastAPI) + Web application (Streamlit)  
**Performance Goals**: 202 response to `POST /v1/predict` in < 500 ms; `GET /v1/results` in < 3 s for 500 records  
**Constraints**: UI layer must contain **zero** imports from `backend.*`; all communication exclusively via HTTP  
**Scale/Scope**: Single-threaded FastAPI dev server; Docker Compose for local full-stack; Gunicorn workers for production

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Rule | Status |
|-----------|------|--------|
| `[P1-MODULAR]` | `ui/` communicates via `ui/utils/api_client.py` over HTTP only — zero `backend.*` imports | ⚠ **VIOLATION** in `ui/pages/New_Prediction.py` and `ui/pages/Dashboard.py` — **must be fixed** |
| `[P2-PEP]` | ruff check + format enforced; type annotations on all public functions | ✅ Existing backend files comply; new code must comply |
| `[P3-LOG]` | `get_logger(__name__)` in every module; no `print()` | ✅ All backend modules use structured logging |
| `[P4-EXC]` | Background task catches all exceptions; calls `update_job_error`; logs with `exc_info=True` | ✅ Implemented in `predict.py::_run_prediction_background` |
| `[P5-CONFIG]` | Missing `hyperparams.yaml` MUST raise `FileNotFoundError`; no silent fallback | ⚠ **Gap**: `config.py` silently falls back to `{}` — **must be fixed** |
| `[P6-API]` | All routes versioned `/v1/`; router-level auth; 202 async pattern | ✅ Fully implemented |
| `[P7-TEST]` | New functions require unit tests; 80% coverage gate | 📋 Tests for new `update_job_status`, rewritten UI pages needed |
| `[P8-DB]` | All DB ops in `db_utils.py`; parameterised queries only | ✅ Compliant |
| `[P9-SEC]` | `X-API-Key` on all non-health routes; `API_KEY` raises `ValueError` at startup | ✅ Implemented |
| `[P10-OBS]` | `GET /health` checks real dependencies; logging correlated | ✅ Implemented |
| `[P11-CONT]` | `docker-compose.yml` exists; Python 3.12-slim | ✅ Present |
| `[P12-CI]` | Makefile targets; pre-commit hooks | ✅ In place |

**Violations requiring resolution** (blocking):
1. `[P1-MODULAR]` — UI pages import backend modules directly → **fix in Phase 2**
2. `[P5-CONFIG]` — Silent `hyperparams.yaml` fallback → **fix in Phase 2**

---

## Project Structure

### Documentation (this feature)

```text
specs/002-fastapi-backend-http-wire/
├── plan.md              ← This file
├── spec.md              ← Feature specification
├── research.md          ← Phase 0: gap analysis and decisions
├── data-model.md        ← Phase 1: entities and schemas
├── quickstart.md        ← Phase 1: local dev guide
├── contracts/
│   └── api-v1.md       ← Phase 1: HTTP API contracts
└── checklists/
    └── requirements.md  ← Spec quality checklist
```

### Source Code (repository root)

```text
backend/
├── main.py                       ← ✅ App factory (complete)
├── api/
│   ├── dependencies.py           ← ✅ require_api_key (complete)
│   ├── schemas.py                ← ✅ All Pydantic schemas (complete)
│   └── routers/
│       ├── health.py             ← ✅ GET /, GET /health (complete)
│       ├── predict.py            ← ✅ POST + GET /v1/predict (complete)
│       └── results.py            ← ✅ GET /v1/results (complete)
├── config/
│   ├── config.py                 ← ⚠ Needs FileNotFoundError guard
│   └── hyperparams.yaml          ← ✅ Complete
├── services/
│   ├── db_utils.py               ← ⚠ Needs update_job_status() added
│   └── db_model.py               ← ✅ InferenceJobDB + CreditScoringRecordDB
└── scripts/
    └── db_init.py                ← ✅ inference_jobs DDL (complete)

ui/
├── utils/
│   └── api_client.py             ← ✅ LershaAPIClient (complete)
└── pages/
    ├── New_Prediction.py         ← ❌ Must be rewritten (no backend imports)
    └── Dashboard.py              ← ❌ Must be rewritten (no backend imports)
```

**Structure Decision**: Monorepo with `backend/` (FastAPI) and `ui/` (Streamlit) as separate independent packages. Communication exclusively via HTTP through `ui/utils/api_client.py`.

---

## Phase 0: Research

✅ **Complete** — see [research.md](research.md)

Key findings:
- Backend is 90%+ implemented; 2 minor gaps identified
- UI pages are legacy code that must be fully rewritten
- No NEEDS CLARIFICATION items — all ambiguities resolved via codebase audit

---

## Phase 1: Design & Contracts

✅ **Complete** — see artifacts:
- [data-model.md](data-model.md) — Entity schemas, state transitions, Pydantic models
- [contracts/api-v1.md](contracts/api-v1.md) — Full HTTP contract for all routes
- [quickstart.md](quickstart.md) — Local dev startup guide

---

## Phase 2: Implementation Tasks

> **Note**: Tasks are ordered by dependency. Items marked ✅ require no code changes.

### Task 1 — Fix `config.py`: Hard-fail on missing `hyperparams.yaml` [P5-CONFIG]

**File**: `backend/config/config.py`  
**Change**: Replace the `if _hparams_path.exists()` block with a guard that raises `FileNotFoundError` when the file is absent.

**Before** (lines 125–130):
```python
_hparams_path = BASE_DIR / "backend" / "config" / "hyperparams.yaml"
if _hparams_path.exists():
    with open(_hparams_path, encoding="utf-8") as f:
        self.hyperparams: dict = yaml.safe_load(f) or {}
else:
    self.hyperparams = {}
```

**After**:
```python
_hparams_path = BASE_DIR / "backend" / "config" / "hyperparams.yaml"
if not _hparams_path.exists():
    raise FileNotFoundError(f"hyperparams.yaml not found at {_hparams_path}")
with open(_hparams_path, encoding="utf-8") as f:
    self.hyperparams: dict = yaml.safe_load(f) or {}
```

**Tests**: `backend/tests/unit/test_config.py` — add test that patching `_hparams_path.exists()` to return False raises `FileNotFoundError`.

---

### Task 2 — Add `update_job_status()` to `db_utils.py` [P8-DB]

**File**: `backend/services/db_utils.py`  
**Change**: Add a standalone `update_job_status(job_id, status)` function that sets only the `status` column. This enables `_run_prediction_background` to mark jobs as `"processing"` before results are ready.

```python
def update_job_status(job_id: str, status: str) -> None:
    """Update the status field of an inference job.

    Used to mark a job as 'processing' before the pipeline result is available.

    Args:
        job_id: UUID string of the job to update.
        status: New status value ('pending', 'processing', 'completed', 'failed').
    """
    engine = db_engine()
    with Session(engine) as session:
        job = session.get(InferenceJobDB, job_id)
        if job:
            job.status = status
            session.commit()
    logger.info("Job '%s' status updated to '%s'", job_id, status)
```

**Also update** `_run_prediction_background` in `backend/api/routers/predict.py` to call `db_utils.update_job_status(job_id, "processing")` as the first line inside the `try` block.

**Tests**: `backend/tests/unit/test_db_utils.py` — add `test_update_job_status_changes_status_field`.

---

### Task 3 — Rewrite `ui/pages/New_Prediction.py` [P1-MODULAR]

**File**: `ui/pages/New_Prediction.py`  
**Change**: Complete rewrite — remove `from config.config import config` and `from src.inference_pipeline import ...`. Replace prediction execution with `LershaAPIClient` calls and a polling loop.

**Design**:
1. Import only `streamlit`, `pandas`, and `from ui.utils.api_client import LershaAPIClient`.
2. Instantiate `client = LershaAPIClient()` at module level.
3. Source radio + UID/row-count inputs unchanged.
4. On "Run Prediction" click:
   - Wrap in `try/except requests.ConnectionError` → show `st.error("Backend unavailable. Is the API server running?")`
   - Call `client.submit_prediction(source, farmer_uid, number_of_rows)` → get `job_id`
   - Show `st.info(f"Job accepted: {job_id}. Waiting for result...")`
   - Call `client.poll_until_complete(job_id)` inside `st.spinner("Running inference…")`
   - On `status == "completed"`: render evaluation expanders for each model
   - On `status == "failed"`: `st.error(f"Inference failed: {result['error']}")`
5. Preserve footer HTML.

**Key constraint**: Zero imports from `backend.*` or `config.*` or `src.*`.

**Tests**: Manual verification (UI integration); no pytest unit test required for Streamlit page rendering.

---

### Task 4 — Rewrite `ui/pages/Dashboard.py` [P1-MODULAR]

**File**: `ui/pages/Dashboard.py`  
**Change**: Replace the data-loading section — remove `from config.config import config` and `from utils.eda import load_table`. Replace with `LershaAPIClient.get_results()`.

**Design**:
1. Import only `streamlit`, `pandas`, `io`, `re`, `requests`, and `from ui.utils.api_client import LershaAPIClient`.
2. Instantiate `client = LershaAPIClient()` at module level.
3. Replace:
   ```python
   df = load_table(config.candidate_result)
   ```
   With:
   ```python
   try:
       response = client.get_results(limit=500)
       df = pd.DataFrame(response["records"])
   except requests.ConnectionError:
       st.error("Backend unavailable. Is the API server running?")
       st.stop()
   ```
4. Preserve **all** existing UI logic: metrics, pagination, search, filter, styled table, download buttons, footer.

**Key constraint**: Zero imports from `backend.*`, `config.*`, or `utils.eda`.

---

### Task 5 — Constitution Re-check (Post-Implementation)

After Tasks 1–4 are complete, verify:

| Check | Pass Condition |
|-------|---------------|
| `[P1-MODULAR]` | `grep -r "from backend" ui/` returns 0 matches; `grep -r "from config" ui/` returns 0 matches; `grep -r "from src" ui/` returns 0 matches |
| `[P5-CONFIG]` | Starting the app without `hyperparams.yaml` present raises `FileNotFoundError` immediately |
| `[P2-PEP]` | `uv run ruff check backend/ ui/` passes with 0 errors |
| `[P7-TEST]` | `uv run pytest backend/tests/ --cov-fail-under=80` passes |
| All curl checks | See `contracts/api-v1.md` curl verification cheatsheet |

---

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| UI pages retain complex rendering logic (pagination, styles) | Business requirement — users rely on this dashboard functionality | Stripping the UI to a basic table would degrade operator experience |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Streamlit reexecution model causes polling loop to block UI thread | Medium | High | Use `st.spinner` context; `poll_until_complete` has `max_wait=300s` timeout; consider `st.rerun()` pattern if needed |
| Dashboard `get_results` serialisation fails for legacy `top_feature_contributions` format | Low | Medium | `ResultsRecord` model uses `list[dict]` — tolerant of varied JSONB shapes |
| `update_job_status("processing")` call fails at DB layer mid-pipeline | Low | Low | Non-fatal — job will remain `pending` until terminal state; `update_job_error` still called on exception |
