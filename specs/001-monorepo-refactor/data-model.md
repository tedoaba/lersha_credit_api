# Data Model: Lersha Monorepo Refactor

**Phase**: Phase 1 — Design & Contracts  
**Branch**: `001-monorepo-refactor`  
**Date**: 2026-03-29

---

## Entities

### 1. `Config` (singleton — `backend/config/config.py`)

Central configuration object. Instantiated once at module load. All modules import `from backend.config.config import config`.

| Attribute | Type | Source | Notes |
|---|---|---|---|
| `output_dir` | `str \| Path` | `OUTPUT_DIR` env | Directory for SHAP PNGs and JSON |
| `model_dir` | `str \| Path` | `MODEL_DIR` env | Parent dir for `.pkl` model files |
| `log_dir` | `str \| Path` | `LOG_DIR` env | Log file parent dir |
| `log_file` | `str \| Path` | `LOG_FILE` env | Rotating log file path |
| `xgb_model_36` | `str \| Path` | `XGB_MODEL_36` env | XGBoost `.pkl` path |
| `rf_model_36` | `str \| Path` | `RF_MODEL_36` env | **FIXED**: was reading `XGB_MODEL_36` |
| `cab_model_36` | `str \| Path` | `CAB_MODEL_36` env | **FIXED**: was reading `XGB_MODEL_36` |
| `chroma_db_path` | `str \| Path` | `CHROMA_DB_PATH` env | **NEW**: ChromaDB persistent data dir |
| `feature_column_36` | `str \| Path` | `FEATURE_COLUMN_36` env | `36_feature_columns.pkl` path |
| `target_column_36` | `str \| Path` | `TARGET_COLUMN_36` env | `36_label_classes.pkl` path |
| `db_uri` | `str \| None` | `DB_URI` env | PostgreSQL connection string |
| `farmer_data_all` | `str \| None` | `FARMER_DATA_ALL` env | Primary farmer data table name |
| `candidate_result` | `str` | `CANDIDATE_RESULT` env | Result table name |
| `api_key` | `str` | `API_KEY` env | **NEW**: hard-fail if absent |
| `gemini_api_key` | `str` | `GEMINI_API_KEY` env | Hard-fail if absent |
| `gemini_model_id` | `str` | `GEMINI_MODEL` env | Hard-fail if absent |
| `embedder_model` | `str` | `EMBEDDER_MODEL` env | SentenceTransformer model name |
| `mlflow_tracking_uri` | `str` | `MLFLOW_TRACKING_URI` env | MLflow backend URI |
| `columns_36` | `list[str]` | hardcoded constant | 36 feature column names |
| `hyperparams` | `dict` | `hyperparams.yaml` | Externalized tuning knobs |
| `prompt_path` | `str \| Path` | `PROMPT_PATH` env | Path to `prompts.yaml` |

**Constraints**:
- `BASE_DIR = Path(__file__).resolve().parents[2]` — project root
- Startup assertion: `assert Path(config.xgb_model_36).exists()`
- `api_key`, `gemini_api_key`, `gemini_model_id` must all raise `ValueError` if absent

---

### 2. `CreditScoringRecord` (Pydantic — `backend/services/schema.py`)

Validation gate before any ORM write to `candidate_result`.

| Field | Type | Validation |
|---|---|---|
| `farmer_uid` | `str` | non-empty |
| `first_name` | `str` | non-empty |
| `middle_name` | `str` | non-empty |
| `last_name` | `str` | non-empty |
| `predicted_class_name` | `str` | one of `["Eligible", "Review", "Not Eligible"]` |
| `top_feature_contributions` | `list[FeatureContribution]` | ≥ 1 item |
| `rag_explanation` | `str` | non-empty |
| `model_name` | `str` | one of `["xgboost", "random_forest", "catboost"]` |
| `timestamp` | `datetime` | UTC |

```python
class FeatureContribution(BaseModel):
    feature: str
    value: float

class CreditScoringRecord(BaseModel):
    farmer_uid: str
    first_name: str
    middle_name: str
    last_name: str
    predicted_class_name: str
    top_feature_contributions: list[FeatureContribution]
    rag_explanation: str
    model_name: str
    timestamp: datetime
```

---

### 3. `CreditScoringRecordDB` (SQLAlchemy ORM — `backend/services/db_model.py`)

Maps to the `candidate_result` PostgreSQL table.

| Column | SQLAlchemy Type | PostgreSQL Type | Constraints |
|---|---|---|---|
| `id` | `Integer` | `SERIAL` | Primary Key, autoincrement |
| `farmer_uid` | `String(100)` | `VARCHAR(100)` | NOT NULL |
| `first_name` | `String(100)` | `VARCHAR(100)` | NOT NULL |
| `middle_name` | `String(100)` | `VARCHAR(100)` | NOT NULL |
| `last_name` | `String(100)` | `VARCHAR(100)` | NOT NULL |
| `predicted_class_name` | `String(100)` | `VARCHAR(100)` | NOT NULL |
| `top_feature_contributions` | `JSON` | `JSONB` | NOT NULL |
| `rag_explanation` | `Text` | `TEXT` | NOT NULL |
| `model_name` | `Text` | `TEXT` | NOT NULL |
| `timestamp` | `DateTime(timezone=True)` | `TIMESTAMPTZ` | NOT NULL |

---

### 4. `InferenceJobDB` (SQLAlchemy ORM — `backend/services/db_model.py`)

Maps to the `inference_jobs` PostgreSQL table. New in this refactor (async inference pattern).

| Column | SQLAlchemy Type | PostgreSQL Type | Constraints |
|---|---|---|---|
| `job_id` | `UUID` | `UUID` | Primary Key |
| `status` | `String(20)` | `VARCHAR(20)` | NOT NULL; default `'pending'` |
| `result` | `JSON` | `JSONB` | NULLABLE |
| `error` | `Text` | `TEXT` | NULLABLE |
| `created_at` | `DateTime(timezone=True)` | `TIMESTAMPTZ` | default `NOW()` |
| `completed_at` | `DateTime(timezone=True)` | `TIMESTAMPTZ` | NULLABLE |

**State transitions**: `pending → processing → completed | failed`

---

### 5. `PredictRequest` (Pydantic — `backend/api/schemas.py`)

API input model for `POST /v1/predict`.

```python
class PredictRequest(BaseModel):
    source: Literal["Single Value", "Batch Prediction"]
    farmer_uid: str | None = None
    number_of_rows: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_source_fields(self) -> "PredictRequest":
        if self.source == "Single Value" and not self.farmer_uid:
            raise ValueError("farmer_uid is required for Single Value prediction")
        if self.source == "Batch Prediction" and not self.number_of_rows:
            raise ValueError("number_of_rows is required for Batch Prediction")
        return self
```

---

### 6. `JobAcceptedResponse` (Pydantic — `backend/api/schemas.py`)

API output model for `POST /v1/predict` (202 Accepted).

```python
class JobAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["accepted"]
```

---

### 7. `JobStatusResponse` (Pydantic — `backend/api/schemas.py`)

API output model for `GET /v1/predict/{job_id}`.

```python
class EvaluationRecord(BaseModel):
    farmer_uid: str
    first_name: str
    predicted_class_name: str
    top_feature_contributions: list[FeatureContribution]
    rag_explanation: str
    model_name: str
    timestamp: datetime

class ModelResult(BaseModel):
    status: str
    records_processed: int
    evaluations: list[EvaluationRecord]

class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    result: dict[str, ModelResult] | None = None  # keys: result_xgboost, result_random_forest
    error: str | None = None
```

---

### 8. `Hyperparams` (YAML — `backend/config/hyperparams.yaml`)

Externalized tuning knobs, loaded into `config.hyperparams` at startup.

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

## Key Relationships

```
PredictRequest ──► InferenceJobDB (created at POST /predict)
InferenceJobDB ──► CreditScoringRecord (one job creates N records, one per model per farmer)
CreditScoringRecord ──► CreditScoringRecordDB (Pydantic → ORM mapping before INSERT)
Config singleton ◄── all backend modules (single import point)
```

## State Machine: Inference Job

```
          ┌─────────────┐
          │   pending   │  ← job created by POST /predict
          └──────┬──────┘
                 │ background task starts
          ┌──────▼──────┐
          │  processing │
          └──────┬──────┘
                 │
       ┌─────────┴──────────┐
       │                    │
┌──────▼──────┐    ┌────────▼────────┐
│  completed  │    │     failed      │
│ result:JSON │    │  error: string  │
└─────────────┘    └─────────────────┘
```

## PostgreSQL Table DDL Summary

```sql
-- candidate_result (existing, confirmed from db_utils.py)
CREATE TABLE IF NOT EXISTS candidate_result (
    id SERIAL PRIMARY KEY,
    farmer_uid VARCHAR(100) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    middle_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    predicted_class_name VARCHAR(100) NOT NULL,
    top_feature_contributions JSONB NOT NULL,
    rag_explanation TEXT NOT NULL,
    model_name TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL
);

-- inference_jobs (new — added in Phase 5)
CREATE TABLE IF NOT EXISTS inference_jobs (
    job_id UUID PRIMARY KEY,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
```
