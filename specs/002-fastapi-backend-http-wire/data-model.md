# Data Model: FastAPI Backend & Streamlit HTTP Integration

**Date**: 2026-03-29  
**Branch**: `002-fastapi-backend-http-wire`

---

## 1. Entity: InferenceJob

Tracks the lifecycle of an asynchronous prediction request.

**Table**: `inference_jobs`  
**ORM model**: `backend/services/db_model.py::InferenceJobDB`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| `job_id` | `VARCHAR(36)` | NO | — | UUID stored as string; PK |
| `status` | `VARCHAR(20)` | NO | `'pending'` | FSM: see state transitions below |
| `result` | `JSONB` | YES | NULL | Set on `completed`; keys: `result_xgboost`, `result_random_forest` |
| `error` | `TEXT` | YES | NULL | Set on `failed`; exception message |
| `created_at` | `TIMESTAMPTZ` | NO | `NOW()` | UTC timestamp of job creation |
| `completed_at` | `TIMESTAMPTZ` | YES | NULL | UTC timestamp of terminal state |

### State Transitions

```
pending → processing → completed
                    ↘ failed
```

- `pending`: Set on `create_job()`
- `processing`: Set by `update_job_status(job_id, "processing")` at the start of `_run_prediction_background`
- `completed`: Set by `update_job_result()` on pipeline success
- `failed`: Set by `update_job_error()` on exception

**Validation rules**:
- `status` MUST be one of: `pending`, `processing`, `completed`, `failed`
- `result` and `error` are mutually exclusive in terminal states (`completed` has `result`, `failed` has `error`)
- `completed_at` MUST be set when status transitions to `completed` or `failed`

---

## 2. Entity: CandidateResult

Stores the persistent audit trail of every individual farmer's credit scoring evaluation.

**Table**: `candidate_result` (existing, not modified by this feature)  
**ORM model**: `backend/services/db_model.py::CreditScoringRecordDB`

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| `id` | `SERIAL` | NO | Auto-increment PK |
| `farmer_uid` | `VARCHAR(100)` | NO | Farmer identifier |
| `first_name` | `VARCHAR(100)` | YES | |
| `middle_name` | `VARCHAR(100)` | YES | |
| `last_name` | `VARCHAR(100)` | YES | |
| `predicted_class_name` | `VARCHAR(100)` | NO | e.g., `"Eligible"`, `"Not Eligible"`, `"Review"` |
| `top_feature_contributions` | `JSONB` | NO | Array of `{feature, value}` objects |
| `rag_explanation` | `TEXT` | NO | LLM-generated explanation |
| `model_name` | `TEXT` | NO | `"xgboost"` or `"random_forest"` |
| `timestamp` | `TIMESTAMPTZ` | NO | UTC time of inference |

---

## 3. API I/O Schemas (Pydantic)

### 3.1 `PredictRequest` (request)
```python
source: Literal["Single Value", "Batch Prediction"]   # required
farmer_uid: str | None = None                          # required if source == "Single Value"
number_of_rows: int | None = Field(default=None, ge=1, le=100)  # required if source == "Batch Prediction"
```
Cross-field validation enforced by `@model_validator`.

### 3.2 `JobAcceptedResponse` (response to POST /v1/predict)
```python
job_id: str
status: Literal["accepted"] = "accepted"
```
Note: Spec names this `JobAccepted`; codebase uses `JobAcceptedResponse`. Functionally identical.

### 3.3 `JobStatusResponse` (response to GET /v1/predict/{job_id})
```python
job_id: str
status: Literal["pending", "processing", "completed", "failed"]
result: dict[str, Any] | None = None   # keys: result_xgboost, result_random_forest
error: str | None = None
```

### 3.4 `ResultsResponse` (response to GET /v1/results)
```python
total: int
records: list[ResultsRecord]   # richer than spec's list[dict]; backwards compatible
```

### 3.5 `ResultsRecord` (element within ResultsResponse)
```python
farmer_uid: str
first_name: str | None
middle_name: str | None
last_name: str | None
predicted_class_name: str
top_feature_contributions: list[dict]
rag_explanation: str
model_name: str
timestamp: datetime | None
```

---

## 4. Configuration Objects

### 4.1 `config.hyperparams` structure (from `hyperparams.yaml`)
```yaml
inference:
  default_batch_size: 10
  max_batch_size: 100
  shap_max_samples: 100
  rag_top_k: 5
  job_timeout_minutes: 30

models:
  active:
    - xgboost
    - random_forest

rate_limiting:
  requests_per_minute: 10
```

---

## 5. Relationship Diagram

```
PredictRequest ──POST──► InferenceJob (pending)
                              │
                         BackgroundTask
                              │
                    ┌─────────┴──────────┐
                    │                    │
               pipeline OK          exception
                    │                    │
              InferenceJob          InferenceJob
             (completed +          (failed +
              result JSONB)         error TEXT)
                    │
              writes N rows
                    │
              CandidateResult
              (candidate_result table)
                    │
                    └──GET /v1/results──► ResultsResponse
```
