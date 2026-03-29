# Data Model: Test Suite, Docker Build System & CI/CD Pipeline

**Branch**: `003-tests-docker-cicd` | **Phase**: 1 — Design

---

## Entities

### 1. `TestDatabaseEngine` (Fixture Entity)

Owns the lifecycle of the `test_lersha` PostgreSQL schema for the entire test session.

| Attribute | Type | Description |
|-----------|------|-------------|
| `engine` | `sqlalchemy.Engine` | SQLAlchemy engine connected to `test_lersha` |
| `scope` | `"session"` | Shared across all tests in a single pytest run |

**Lifecycle**:
```
CREATE TABLE candidate_result   →   yield engine   →   DROP ALL TABLES
CREATE TABLE inference_jobs
```

**Tables Created**:

*`candidate_result`*:
| Column | Type | Notes |
|--------|------|-------|
| `id` | `SERIAL PRIMARY KEY` | Auto-increment surrogate key |
| `farmer_uid` | `VARCHAR(100) NOT NULL` | Business key |
| `first_name` | `VARCHAR(100)` | Nullable |
| `middle_name` | `VARCHAR(100)` | Nullable |
| `last_name` | `VARCHAR(100)` | Nullable |
| `predicted_class_name` | `VARCHAR(100) NOT NULL` | e.g. "Eligible" |
| `top_feature_contributions` | `JSONB NOT NULL` | List of `{feature, shap_value}` dicts |
| `rag_explanation` | `TEXT NOT NULL` | LLM-generated explanation text |
| `model_name` | `TEXT NOT NULL` | "xgboost" or "random_forest" |
| `timestamp` | `TIMESTAMPTZ NOT NULL` | UTC time of evaluation |

*`inference_jobs`*:
| Column | Type | Notes |
|--------|------|-------|
| `job_id` | `UUID PRIMARY KEY` | Job identifier |
| `status` | `VARCHAR(20) DEFAULT 'pending'` | pending / processing / completed / failed |
| `result` | `JSONB` | Nullable; populated on completion |
| `error` | `TEXT` | Nullable; populated on failure |
| `created_at` | `TIMESTAMPTZ DEFAULT NOW()` | Immutable after creation |
| `completed_at` | `TIMESTAMPTZ` | Nullable; set on terminal state |

---

### 2. `SampleFarmerDataFrame` (Fixture Entity)

The canonical 5-row synthetic input for all unit and integration tests.

**Schema** (all columns of the raw farmer table):

| Column | Type | Sample Values |
|--------|------|---------------|
| `farmer_uid` | `str` | F-001 … F-005 |
| `first_name` | `str` | Abebe, Bekele, Chaltu, Dawit, Eleni |
| `middle_name` | `str` | Hailu, Girma, Tigist, Amare, Lemma |
| `last_name` | `str` | Woldemariam, Abebe, Teshome, Kebede, Haile |
| `gender` | `str` | Male / Female |
| `age` | `int` | 28–55 |
| `family_size` | `int` | 3–8 |
| `estimated_income` | `float` | 8000–20000 |
| `estimated_income_another_farm` | `float` | 500–4000 |
| `estimated_expenses` | `float` | 2000–7000 |
| `estimated_cost` | `float` | 1000–3500 |
| `agricultureexperience` | `int` | 3–20 |
| `hasmemberofmicrofinance` | `int (0/1)` | binary |
| `hascooperativeassociation` | `int (0/1)` | binary |
| `agriculturalcertificate` | `int (0/1)` | binary |
| `hascommunityhealthinsurance` | `int (0/1)` | binary |
| `farmsizehectares` | `float` | 1.0–5.0 |
| `expectedyieldquintals` | `float` | 12–60 |
| `seedquintals` | `float` | 1.5–6.0 |
| `ureafertilizerquintals` | `float` | 0.5–3.0 |
| `dapnpsfertilizerquintals` | `float` | 0.5–2.5 |
| `value_chain` | `str` | maize / wheat / teff |
| `total_farmland_size` | `float` | 1.5–6.0 |
| `land_size` | `float` | 1.0–5.0 |
| `childrenunder12` | `int` | 0–4 |
| `elderlymembersover60` | `int` | 0–2 |
| `maincrops` | `str` | maize / wheat / teff |
| `lastyearaverageprice` | `float` | 400–700 |
| `decision` | `str` | Eligible / Review / Ineligible |

---

### 3. `CIWorkflow` (Pipeline Entity)

The three-job GitHub Actions pipeline.

**State machine**:
```
push / pull_request
        │
        ▼
   ┌─────────┐   fails   ┌──────┐
   │  lint   ├──────────►│  ✗   │
   └────┬────┘           └──────┘
        │ passes
        ▼
   ┌─────────┐   fails   ┌──────┐
   │  test   ├──────────►│  ✗   │
   └────┬────┘           └──────┘
        │ passes
        ▼
   ┌─────────┐   fails   ┌──────┐
   │  build  ├──────────►│  ✗   │
   └────┬────┘           └──────┘
        │ passes
        ▼
       ✓
```

**Job definitions**:

| Job | Runner | Services | Key Steps |
|-----|--------|----------|-----------|
| `lint` | `ubuntu-latest` | None | setup-uv, ruff check, ruff format --check, mypy |
| `test` | `ubuntu-latest` | `postgres:16` | setup-uv, uv sync --extra dev, pytest --cov-fail-under=80 |
| `build` | `ubuntu-latest` | None | setup-uv (for context), docker build backend, docker build ui |

---

### 4. `DockerService` (Infrastructure Entity)

The four services defined in `docker-compose.yml`.

| Service | Image | Port | Key Env Vars | Volumes |
|---------|-------|------|--------------|---------|
| `postgres` | `postgres:16` | 5432 | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | `postgres_data:/var/lib/postgresql/data` |
| `backend` | Built from `backend/Dockerfile` | 8000 | `DB_URI`, `API_KEY`, `GEMINI_API_KEY`, `MLFLOW_TRACKING_URI` | `backend/models:ro`, `chroma_db`, `mlruns`, `output`, `logs` |
| `ui` | Built from `ui/Dockerfile` | 8501 | `API_BASE_URL=http://backend:8000`, `API_KEY` | None |
| `mlflow` | `ghcr.io/mlflow/mlflow` | 5000 | None | `mlruns_data:/mlruns` |

---

## State Transitions

### InferenceJob Status FSM

```
                 create_job()
                      │
                      ▼
                  ┌─────────┐
                  │ pending │
                  └────┬────┘
                       │  update_job_status("processing")
                       ▼
                ┌────────────┐
                │ processing │
                └─────┬──────┘
              success │     │ exception
      ┌───────────────┘     └─────────────────┐
      ▼                                        ▼
┌───────────┐                          ┌──────────┐
│ completed │                          │  failed  │
└───────────┘                          └──────────┘
     (terminal)                           (terminal)
```

### CI Job Dependencies

```
lint ──┬──► test ──► build
       │
       └──► (if lint fails, test and build are skipped)
```
(Expressed as `needs: [lint, test]` on the `build` job)

---

## Validation Rules

| Entity | Rule |
|--------|------|
| `SampleFarmerDataFrame` | All numeric columns must be non-null positive values |
| `SampleFarmerDataFrame` | Binary columns (`hasmemberof*`, `has*`, `agricultural*`) must be 0 or 1 |
| `SampleFarmerDataFrame` | `farmsizehectares` must be > 0 to avoid division-by-zero in `yield_per_hectare` and `input_intensity` |
| `test_db_engine` | Must drop all tables AFTER yield, even if a test raises an exception |
| `CIWorkflow.test` | `--cov-fail-under=80` flag required; pipeline MUST fail if coverage < 80% |
| `build_contribution_table` | `len(feature_names) == len(class_shap_values) == len(feature_values)` — raises `ValueError` if violated |
