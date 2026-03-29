# Feature Specification: FastAPI Backend & Streamlit HTTP Integration

**Feature Branch**: `002-fastapi-backend-http-wire`  
**Created**: 2026-03-29  
**Status**: Draft  
**Input**: User description: "Implement the full FastAPI backend and wire the Streamlit UI to communicate with it exclusively via HTTP."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Submit a Credit Prediction Job (Priority: P1)

A data operator opens the Streamlit **New Prediction** page, selects a data source, optionally provides a farmer UID and row count, and submits the form. The UI immediately receives a job ID confirming the request was accepted, then polls for a result and displays it when ready — without ever importing Python backend modules directly.

**Why this priority**: This is the system's primary user-facing action. All other stories depend on jobs being created and tracked reliably.

**Independent Test**: Can be fully tested by submitting a prediction via the UI form and verifying that a job ID is returned and eventually a completed status with a result is displayed.

**Acceptance Scenarios**:

1. **Given** the UI is running and the backend service is reachable, **When** the user submits a prediction request without an API key in the client configuration, **Then** the UI displays a clear authentication error (403) rather than crashing.
2. **Given** a valid API key is configured, **When** the user submits a prediction with a source and optional parameters, **Then** the UI shows "Job accepted" with a job ID and begins polling.
3. **Given** a job is accepted and the background task completes successfully, **When** the UI polls the job status endpoint, **Then** status transitions from `pending` → `processing` → `completed` and the result is rendered in the UI.
4. **Given** the pipeline raises an exception during background processing, **When** the UI polls, **Then** status returns `failed` with a human-readable error message.
5. **Given** the backend is unreachable, **When** the user submits a prediction, **Then** the UI shows a graceful "Service unavailable" message instead of an unhandled stack trace.

---

### User Story 2 — View Historical Results on Dashboard (Priority: P2)

A business analyst opens the Streamlit **Dashboard** page and sees a tabular view of all historical prediction records, loaded exclusively from the backend API — no direct database or SQLAlchemy calls in the UI layer.

**Why this priority**: Reporting is a secondary consumer of prediction data; it has no dependencies on job submission but completes the end-to-end user journey.

**Independent Test**: Can be fully tested by loading the Dashboard page and verifying records are fetched from the API and rendered in a DataFrame.

**Acceptance Scenarios**:

1. **Given** the backend has records in the results table, **When** the Dashboard page loads, **Then** up to 500 records are displayed in a tabular format sourced from `GET /v1/results`.
2. **Given** no records exist yet, **When** the Dashboard loads, **Then** an empty table with column headers is shown and no error is raised.
3. **Given** the API key is invalid, **When** the Dashboard loads, **Then** a clear authentication error is displayed rather than a raw exception.

---

### User Story 3 — Secure All Inference Endpoints (Priority: P2)

Any caller (curl, UI, external integration) that omits or provides a wrong `X-API-Key` header is denied access to prediction and results endpoints with a 403 response, while the health endpoints remain public.

**Why this priority**: Security is non-negotiable before any external exposure; it must be layered at the router level, not the application level.

**Independent Test**: Can be fully tested with curl by hitting `/v1/predict` without a key (expect 403) and with the correct key (expect 202/200).

**Acceptance Scenarios**:

1. **Given** a `POST /v1/predict` request with no `X-API-Key` header, **When** the backend processes it, **Then** it responds with HTTP 403 Forbidden.
2. **Given** a `POST /v1/predict` request with an incorrect `X-API-Key` value, **When** the backend processes it, **Then** it responds with HTTP 403 Forbidden.
3. **Given** a `GET /health` request with no API key, **When** the backend processes it, **Then** it responds with HTTP 200 OK.
4. **Given** a valid API key, **When** results or predict endpoints are called, **Then** responses are served normally (202/200 as appropriate).

---

### User Story 4 — Async Job Lifecycle Management (Priority: P1)

When a prediction job is submitted, the system records the job in a durable store, executes the prediction pipeline asynchronously in the background, and updates the job record with either a result or an error — ensuring no data is lost if the UI disconnects mid-poll.

**Why this priority**: Without persistent job tracking the system cannot reliably serve results; polling would have no source of truth.

**Independent Test**: Can be fully tested by submitting a job, immediately querying its status (expect `pending`), waiting for completion, and querying again (expect `completed` with a result or `failed` with an error).

**Acceptance Scenarios**:

1. **Given** a valid `POST /v1/predict` call, **When** the request is processed, **Then** a new job record with status `pending` is created and `job_id` is returned in 202 response.
2. **Given** a job with a known `job_id`, **When** `GET /v1/predict/{job_id}` is called, **Then** the current status, optional result, and optional error are returned.
3. **Given** an unknown `job_id`, **When** `GET /v1/predict/{job_id}` is called, **Then** HTTP 404 is returned.
4. **Given** background pipeline succeeds, **When** the task completes, **Then** job status transitions to `completed` with the result stored in the job record.
5. **Given** background pipeline raises an exception, **When** the task completes, **Then** job status transitions to `failed` with the error message stored in the job record.

---

### Edge Cases

- What happens when `API_KEY` environment variable is missing at startup? → The application must raise a `ValueError` at boot and refuse to start, preventing silent security failures.
- What happens when the `inference_jobs` table does not exist? → `db_init.py` must be run first; missing table causes a clear database error, not a generic 500.
- What happens if `number_of_rows` exceeds `max_batch_size` (100)? → The backend should respect the hyperparameter limit and either truncate or reject with a validation error.
- What happens if the polling loop in the UI runs indefinitely? → A timeout or maximum retry count must exist to prevent the UI from hanging forever.
- What happens when `GET /v1/results` is called with no records? → Returns `{"records": [], "total": 0}` with HTTP 200.
- What happens if `hyperparams.yaml` is missing at startup? → `config.py` raises a `FileNotFoundError` on boot.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Application Factory
- **FR-001**: The backend MUST expose a `create_app()` factory function in `backend/main.py` that returns a fully configured FastAPI application instance, enabling test-time instantiation without side effects.
- **FR-002**: The module-level `app = create_app()` MUST be the sole entry point for all ASGI servers (uvicorn, gunicorn).

#### API Routers
- **FR-003**: The health router MUST serve `GET /` and `GET /health` at no prefix, returning HTTP 200 without requiring authentication.
- **FR-004**: The predict router MUST serve `POST /v1/predict` (202 + `job_id`) and `GET /v1/predict/{job_id}` under the prefix `/v1/predict` with the tag `v1 — Inference`.
- **FR-005**: The results router MUST serve `GET /v1/results?limit=500` under the prefix `/v1/results` with the tag `v1 — Results`, reading from the `candidate_result` table.

#### Pydantic Schemas
- **FR-006**: `PredictRequest` MUST contain: `source: str` (required), `farmer_uid: Optional[str]`, `number_of_rows: Optional[int]`.
- **FR-007**: `JobAccepted` MUST contain: `job_id: str`, `status: Literal["accepted"]`.
- **FR-008**: `JobStatus` MUST contain: `job_id: str`, `status: Literal["pending","processing","completed","failed"]`, `result: Optional[dict]`, `error: Optional[str]`.
- **FR-009**: `ResultsResponse` MUST contain: `records: list[dict]`, `total: int`.

#### API Key Authentication
- **FR-010**: The `require_api_key` dependency MUST read the `X-API-Key` request header and compare it to the value loaded from `config.api_key`.
- **FR-011**: A missing or mismatched API key MUST result in HTTP 403 Forbidden; under no circumstances shall a 401 be raised.
- **FR-012**: Both the predict router and the results router MUST declare `dependencies=[Depends(require_api_key)]` at the `APIRouter` level so every route inherits the check.
- **FR-013**: `config.py` MUST read `API_KEY` from environment and raise `ValueError` if the variable is absent or empty.
- **FR-014**: `.env.example` MUST include `API_KEY=<your-secret-api-key>`.

#### Async Job System
- **FR-015**: `backend/scripts/db_init.py` MUST include a `CREATE TABLE IF NOT EXISTS inference_jobs` DDL block with columns: `job_id UUID PRIMARY KEY`, `status VARCHAR(20) DEFAULT 'pending'`, `result JSONB`, `error TEXT`, `created_at TIMESTAMPTZ DEFAULT NOW()`, `completed_at TIMESTAMPTZ`.
- **FR-016**: `backend/services/db_utils.py` MUST expose: `create_job(job_id)`, `update_job_result(job_id, result)`, `update_job_error(job_id, error)`, `get_job(job_id)`, `update_job_status(job_id, status)`.
- **FR-017**: `POST /v1/predict` MUST create a job record, enqueue `_run_prediction_background` via FastAPI `BackgroundTasks`, and return HTTP 202 with the `job_id`.
- **FR-018**: `GET /v1/predict/{job_id}` MUST call `get_job()` and return HTTP 404 if the job does not exist, or a `JobStatus` response otherwise.
- **FR-019**: The background function MUST call `update_job_result` on success and `update_job_error` on exception, ensuring the job record always settles into a terminal state.

#### Hyperparameters
- **FR-020**: `backend/config/hyperparams.yaml` MUST define an `inference` block (`default_batch_size: 10`, `max_batch_size: 100`, `shap_max_samples: 100`, `rag_top_k: 5`) and a `models` block (`active: [xgboost, random_forest]`).
- **FR-021**: `config.py` MUST load `hyperparams.yaml` via `yaml.safe_load` and expose the result as `config.hyperparams`.

#### Streamlit HTTP Client
- **FR-022**: `ui/utils/api_client.py` MUST define `LershaAPIClient` with `__init__` reading `API_BASE_URL` (default `http://localhost:8000`) and `API_KEY` from environment, attaching `X-API-Key` to every outgoing request via a `requests.Session`.
- **FR-023**: `LershaAPIClient` MUST expose: `submit_prediction(source, farmer_uid, number_of_rows) → POST /v1/predict`, `get_prediction_result(job_id) → GET /v1/predict/{job_id}`, `get_results(limit=500) → GET /v1/results`.

#### Streamlit Page Updates
- **FR-024**: `ui/pages/New_Prediction.py` MUST remove all direct backend Python imports and instead use `LershaAPIClient.submit_prediction()` and `get_prediction_result()` with a polling loop; a graceful error message MUST be shown on `ConnectionError`.
- **FR-025**: `ui/pages/Dashboard.py` MUST remove all SQLAlchemy and config imports and instead use `LershaAPIClient.get_results()`, converting `response["records"]` to a Pandas DataFrame.

### Key Entities

- **InferenceJob**: Represents a submitted prediction job. Attributes: `job_id` (UUID), `status` (pending/processing/completed/failed), `result` (JSONB payload), `error` (text), `created_at`, `completed_at`.
- **PredictRequest**: Caller-supplied parameters for a prediction: `source`, optional `farmer_uid`, optional `number_of_rows`.
- **CandidateResult**: Existing table (`candidate_result`) storing historical prediction outputs, consumed by the results endpoint.
- **Hyperparameters**: External YAML configuration governing batch sizes and active model list; loaded once at application startup.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A prediction request submitted without an API key is rejected in under 100 ms with a clear 403 response, 100% of the time.
- **SC-002**: A valid prediction request (202 accepted) begins background processing within 500 ms; job status is queryable immediately after acceptance.
- **SC-003**: The Streamlit UI submits and polls a prediction job without importing any Python module from the `backend/` package — verified by the absence of `from backend` or `import backend` in any `ui/` file.
- **SC-004**: The Dashboard loads up to 500 historical records in under 3 seconds on a local deployment with a seeded database.
- **SC-005**: The application refuses to start if `API_KEY` is not set in the environment, raising a clear error at boot rather than failing silently at request time.
- **SC-006**: All five curl verification commands defined in the feature prompt produce the expected HTTP status codes and JSON shapes on a fresh local deployment.
- **SC-007**: No SQLAlchemy, config, or backend pipeline import statements remain in any `ui/pages/` file.

---

## Assumptions

- The `candidate_result` table already exists and is populated by the existing prediction pipeline; the results endpoint reads from it without schema changes.
- The existing `backend/services/db_utils.py` already has a database connection helper that the new job CRUD functions can reuse.
- `requests` is already a declared dependency of the UI layer (or will be added to `pyproject.toml`/`requirements.txt`).
- The Streamlit UI and the FastAPI backend run as separate processes (separate containers or separate terminal sessions) — no in-process communication is assumed.
- The polling loop in the UI will use a configurable timeout (e.g., 60 seconds with 2-second intervals) to prevent indefinite blocking.
- For v1, background tasks run in-process via FastAPI `BackgroundTasks`; a persistent queue (e.g., Celery/Redis) is out of scope for this feature.
- The `yaml` package (`PyYAML`) is already a declared dependency of the backend.
- `API_BASE_URL` defaults to `http://localhost:8000` for local development; production overrides via environment variable.
- Authentication uses a static shared secret (`X-API-Key`) for simplicity; OAuth2/JWT rotation is out of scope for this feature.
