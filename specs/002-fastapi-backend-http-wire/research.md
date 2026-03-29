# Research: FastAPI Backend & Streamlit HTTP Integration

**Date**: 2026-03-29  
**Branch**: `002-fastapi-backend-http-wire`

---

## 1. Current Codebase State Audit

### Decision: Treat this as a completion/gap-filling task, not greenfield
**Rationale**: A thorough codebase audit reveals that the previous architectural refactor (conversation `9e16ecb9`) already implemented the majority of the required components. The implementation plan must focus on what is **missing or incomplete**, not re-implement existing work.

**Alternatives considered**: Rewriting all files from scratch — rejected because 80%+ of the required code is already correct and production-quality.

---

## 2. Component-by-Component Gap Analysis

### 2.1 `backend/main.py` — App Factory ✅ COMPLETE
- `create_app()` factory exists and is correct.
- `app = create_app()` module-level export is present.
- All three routers registered with correct prefixes and tags.
- **Gap**: None.

### 2.2 `backend/api/routers/health.py` ✅ COMPLETE
- `GET /` and `GET /health` both implemented.
- Health endpoint does live PostgreSQL `SELECT 1` and ChromaDB heartbeat checks — already exceeds the spec requirement.
- **Gap**: None.

### 2.3 `backend/api/routers/predict.py` ✅ COMPLETE
- `POST /v1/predict` → 202 + `job_id` implemented.
- `GET /v1/predict/{job_id}` → 404/JobStatus implemented.
- `_run_prediction_background` background task with `update_job_result`/`update_job_error` implemented.
- `router = APIRouter(dependencies=[Depends(require_api_key)])` present.
- **Gap**: None.

### 2.4 `backend/api/routers/results.py` ✅ COMPLETE
- `GET /v1/results?limit=500` implemented.
- Reads from `candidate_result` via `db_utils.get_all_results()`.
- Returns `ResultsResponse`.
- `router = APIRouter(dependencies=[Depends(require_api_key)])` present.
- **Gap**: None.

### 2.5 `backend/api/schemas.py` ✅ SUBSTANTIALLY COMPLETE (minor gap)
- `PredictRequest`, `JobAcceptedResponse`, `JobStatusResponse`, `ResultsResponse` all present.
- **Gap**: Spec requires schema named `JobAccepted` but file uses `JobAcceptedResponse`. This is a cosmetic naming difference; the functional contract is identical. The `predict.py` router already imports `JobAcceptedResponse` — renaming would be a breaking change with no benefit. **Decision**: Accept current naming; document the alias in contracts.
- Additional `ResultsRecord` Pydantic model exists (richer than spec's `list[dict]` requirement) — this is an improvement, not a gap.

### 2.6 `backend/api/dependencies.py` ✅ COMPLETE
- `require_api_key` implemented, reads `X-API-Key` header, raises 403 on mismatch.
- **Gap**: None.

### 2.7 `backend/config/config.py` ✅ COMPLETE
- `API_KEY` read from env; raises `ValueError` if absent.
- `GEMINI_API_KEY` and `GEMINI_MODEL` raise `ValueError` if absent.
- Hyperparams loaded via `yaml.safe_load` and exposed as `config.hyperparams`.
- **Minor gap**: Config silently sets `self.hyperparams = {}` if `hyperparams.yaml` is missing rather than raising `FileNotFoundError`. Spec says it should raise. Given the file exists in the repo, this is low risk. **Decision**: Add a `FileNotFoundError` guard in `config.py` to match spec intent.

### 2.8 `backend/config/hyperparams.yaml` ✅ COMPLETE
- All required keys present: `inference.default_batch_size`, `inference.max_batch_size`, `inference.shap_max_samples`, `inference.rag_top_k`, `models.active`.
- Additional `job_timeout_minutes` and `rate_limiting` keys are present — improvements beyond spec.
- **Gap**: None.

### 2.9 `backend/services/db_utils.py` ✅ COMPLETE
- All 5 CRUD functions present: `create_job`, `update_job_result`, `update_job_error`, `get_job`, `update_job_status` (implicitly covered — `update_job_result`/`update_job_error` both set status).
- **Gap**: `update_job_status` standalone function not explicitly defined (status is only changed as part of `update_job_result` and `update_job_error`). Spec calls for it. **Decision**: Add a bare `update_job_status(job_id, status)` function.

### 2.10 `backend/scripts/db_init.py` ✅ SUBSTANTIALLY COMPLETE
- `inference_jobs` DDL already present with all required columns.
- `job_id` is `VARCHAR(36)` rather than `UUID` native type — functionally equivalent for UUID-as-string storage. **Decision**: Accept; warrants a note in data model.
- **Gap**: None.

### 2.11 `backend/services/db_model.py` ✅ COMPLETE
- `InferenceJobDB` ORM model exists with all columns.
- `CreditScoringRecordDB` ORM model exists.
- **Gap**: None.

### 2.12 `ui/utils/api_client.py` ✅ COMPLETE
- `LershaAPIClient` with `requests.Session`, `X-API-Key` header, all 3 required methods.
- Bonus `poll_until_complete` helper and `health()` method present.
- **Gap**: None.

### 2.13 `.env.example` ✅ COMPLETE
- `API_KEY` present with comment.
- `API_BASE_URL` present.
- **Gap**: None.

### 2.14 `ui/pages/New_Prediction.py` ❌ NOT UPDATED
- Still imports `from config.config import config` and `from src.inference_pipeline import match_inputs, run_inferences` — direct backend Python imports.
- Does not use `LershaAPIClient` at all.
- **Gap**: Entire page must be rewritten to use `LershaAPIClient` with polling loop and `ConnectionError` handling.

### 2.15 `ui/pages/Dashboard.py` ❌ NOT UPDATED
- Still imports `from config.config import config` and `from utils.eda import load_table` — direct backend imports.
- Does not use `LershaAPIClient`.
- **Gap**: Data loading section must be rewritten to use `LershaAPIClient.get_results()`.
- The rich UI (pagination, styling, download buttons) should be preserved — only the data source changes.

---

## 3. Technology Decisions

### FastAPI BackgroundTasks vs. Celery
- **Decision**: FastAPI `BackgroundTasks` (already implemented)
- **Rationale**: Spec explicitly requires `BackgroundTasks`; Celery/Redis is out of scope for v1. BackgroundTasks is sufficient for single-worker deployments.
- **Alternatives considered**: Celery + Redis — rejected (out of scope for this feature per spec Assumptions section).

### Polling Strategy in UI
- **Decision**: Streamlit loop with `st.spinner`, configurable `poll_interval=2.0s`, `max_wait=300s` timeout
- **Rationale**: `LershaAPIClient.poll_until_complete()` already implements this; UI page should call it inside a spinner context.
- **Alternatives considered**: WebSockets — rejected (not applicable to the HTTP-first spec).

### `update_job_status` Implementation
- **Decision**: Add a standalone function to `db_utils.py` that updates only the `status` field
- **Rationale**: `_run_prediction_background` sets status to `"processing"` at the start of the pipeline; this requires a standalone status update separate from result/error writes.
- **Alternatives considered**: Folding into `update_job_result` — rejected because status must be updated independently before results are available.

### `config.py` `FileNotFoundError` Guard
- **Decision**: Convert the silent `{}` fallback to a `FileNotFoundError` raise when `hyperparams.yaml` is absent
- **Rationale**: Spec FR-021 and the constitution's `[P5-CONFIG]` principle both require hard-fail on missing config.
- **Alternatives considered**: Keep silent fallback — rejected because it creates invisible configuration failures.

---

## 4. Resolved NEEDS CLARIFICATION Items

None required — the feature description and codebase audit together resolved all ambiguities.

---

## 5. Summary of Implementation Delta

| Component | Status | Action Required |
|-----------|--------|-----------------|
| `backend/main.py` | ✅ Complete | None |
| `backend/api/routers/health.py` | ✅ Complete | None |
| `backend/api/routers/predict.py` | ✅ Complete | None |
| `backend/api/routers/results.py` | ✅ Complete | None |
| `backend/api/schemas.py` | ✅ Complete | None (naming difference is cosmetic) |
| `backend/api/dependencies.py` | ✅ Complete | None |
| `backend/config/config.py` | ⚠ Minor gap | Add `FileNotFoundError` for missing `hyperparams.yaml` |
| `backend/config/hyperparams.yaml` | ✅ Complete | None |
| `backend/services/db_utils.py` | ⚠ Minor gap | Add standalone `update_job_status(job_id, status)` function |
| `backend/services/db_model.py` | ✅ Complete | None |
| `backend/scripts/db_init.py` | ✅ Complete | None |
| `ui/utils/api_client.py` | ✅ Complete | None |
| `ui/pages/New_Prediction.py` | ❌ Not updated | Full rewrite to use `LershaAPIClient` + polling |
| `ui/pages/Dashboard.py` | ❌ Not updated | Rewrite data-loading section to use `LershaAPIClient` |
| `.env.example` | ✅ Complete | None |
