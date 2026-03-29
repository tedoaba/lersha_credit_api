# Test Interface Contracts: Feature 003

**Branch**: `003-tests-docker-cicd` | **Phase**: 1 — Design

This document defines the contracts — inputs, outputs, and invariants — for every
test file and shared fixture in this feature. It serves as the source of truth for
implementation and is used to verify spec completeness.

---

## Shared Fixtures (`conftest.py`)

### `test_db_engine`

| Property | Value |
|----------|-------|
| Scope | `session` |
| Type | `sqlalchemy.Engine` |
| Precondition | `DB_URI` env var points to a reachable PostgreSQL 16 instance |
| Setup | Creates `candidate_result` + `inference_jobs` tables using raw `CREATE TABLE IF NOT EXISTS` DDL |
| Teardown | Drops all tables (`DROP TABLE IF EXISTS candidate_result, inference_jobs CASCADE`) |
| Invariant | Tables exist for the entire session; cleaned up even if a test raises |

**Contract**:
```python
@pytest.fixture(scope="session")
def test_db_engine() -> Generator[Engine, None, None]:
    ...  # yields Engine; drops tables in finally block
```

---

### `sample_farmer_df`

| Property | Value |
|----------|-------|
| Scope | `function` |
| Type | `pd.DataFrame` |
| Rows | 5 |
| Columns | 29 (full raw farmer schema) |

**Invariants**:
- `farmsizehectares` > 0 for all rows
- `hasmemberofmicrofinance`, `hascooperativeassociation`, `agriculturalcertificate`, `hascommunityhealthinsurance` ∈ {0, 1}
- `farmer_uid` values are unique: `["F-001", "F-002", "F-003", "F-004", "F-005"]`
- `decision` column is present (used by pipeline; dropped before model input)

---

### `api_client`

| Property | Value |
|----------|-------|
| Scope | `function` |
| Type | `httpx.AsyncClient` |
| Transport | `ASGITransport(app=create_app())` |
| Base URL | `http://test` |
| Precondition | `API_KEY=ci-test-key`, `DB_URI`, `GEMINI_API_KEY`, `GEMINI_MODEL` set in `os.environ` before `create_app()` is called |

**Contract**:
```python
@pytest.fixture
async def api_client() -> AsyncGenerator[AsyncClient, None]:
    ...  # yields AsyncClient with ASGITransport
```

---

### `mock_gemini`

| Property | Value |
|----------|-------|
| Scope | `function` |
| Patches | The `generate_content` method of the Gemini client used in `backend/` |
| Return value | `"Fixed explanation for testing."` |
| Invariant | No real API calls are made; deterministic for all tests that request this fixture |

---

## Unit Tests — Feature Engineering

**File**: `backend/tests/unit/test_feature_engineering.py`

| Test | Input | Expected Output | Invariant |
|------|-------|-----------------|-----------|
| `test_net_income_formula` | `estimated_income=12000, income_other=2000, expenses=4000, cost=2000` | `net_income = 8000.0` | `round((12000+2000)-(4000+2000), 3)` |
| `test_institutional_support_score` | flags `[1,0,1,1]` | `3` | exact integer sum |
| `test_dropped_columns_absent` | any valid row | all 17 source columns absent from output | `col not in result.columns` for all |
| `test_age_group_fallback_single_row` | 1-row DataFrame | `age_group` column present, value ∈ labels | no ValueError raised |
| `test_age_group_binning_two_row_edge_case` | 2-row DataFrame | `age_group` column present, no NaN | `pd.qcut` fallback to `pd.cut` |
| `test_yield_per_hectare` | `yield=30, farmsize=2.5` | `12.0` | `round(30/2.5, 3)` |
| `test_input_intensity` | `seeds=3, urea=1.5, dap=1.0, farmsize=2.5` | `2.2` | `round((3+1.5+1)/2.5, 3)` |

---

## Unit Tests — Preprocessing

**File**: `backend/tests/unit/test_preprocessing.py`

| Test | Input | Expected Output | Invariant |
|------|-------|-----------------|-----------|
| `test_preprocessing_output_columns_match_pkl` | DataFrame matching pkl columns | output.columns == canonical list | exact set equality |
| `test_missing_columns_filled_with_zero` | DataFrame missing some columns | absent columns present with value `0` | `fill_value=0` from `reindex` |
| `test_output_shape_equals_feature_list` | any DataFrame | `output.shape[1] == len(feature_columns)` | no extra or missing columns |

---

## Unit Tests — Contribution Table

**File**: `backend/tests/unit/test_contribution_table.py`

| Test | `shap_values` format | Expected | Invariant |
|------|---------------------|----------|-----------|
| `test_catboost_list_input` | `list` of 2 × `np.ndarray (1,3)` | 3-row DataFrame | `["Feature","SHAP Value","Feature Value"]` present |
| `test_xgb_3d_multiclass` | `Explanation.values` shape `(1,3,3)` | 3-row DataFrame | correct class slice extracted |
| `test_2d_binary_ndarray` | `Explanation.values` shape `(1,3)` | 3-row DataFrame | single class extracted |
| `test_length_mismatch_raises_value_error` | 3 SHAP values for 2-feature DataFrame | `ValueError("Length mismatch")` | raised before any DataFrame creation |
| `test_sorted_descending_by_abs_shap` | `[-0.9, 0.1, 0.5]` | first row abs = 0.9 | `abs_vals == sorted(abs_vals, reverse=True)` |

---

## Integration Tests — Predict Endpoint

**File**: `backend/tests/integration/test_predict_endpoint.py`

| Test | Method + Path | Headers | Body | Expected Status | Expected Body |
|------|---------------|---------|------|-----------------|---------------|
| `test_no_api_key_returns_403` | POST `/v1/predict` | None | `{"source":"Batch Prediction","number_of_rows":2}` | 403 | `{"detail": ...}` |
| `test_wrong_api_key_returns_403` | POST `/v1/predict` | `X-API-Key: wrong-key` | same | 403 | `{"detail": ...}` |
| `test_valid_predict_returns_202_with_job_id` | POST `/v1/predict` | `X-API-Key: ci-test-key` | same | 202 | `{"job_id": "<UUID>", "status": "accepted"}` |
| `test_get_job_status_returns_200` | GET `/v1/predict/{job_id}` | `X-API-Key: ci-test-key` | None | 200 | body contains `"status"` key |
| `test_nonexistent_job_returns_404` | GET `/v1/predict/00000000-0000-0000-0000-000000000000` | `X-API-Key: ci-test-key` | None | 404 | `{"detail": "Job not found"}` |

**Fixtures required**: `api_client`, `test_db_engine`  
**Note**: UUID validity check — `job_id` from 202 response must match `uuid.UUID(job_id)` without raising.

---

## Integration Tests — Database Utilities

**File**: `backend/tests/integration/test_db_utils.py`

| Test | Function Under Test | Setup | Assertion |
|------|---------------------|-------|-----------|
| `test_fetch_raw_data_returns_matching_row` | `fetch_raw_data(table, uid)` | Insert 3 rows with distinct UIDs into farmer table | `len(result) == 1` and `result["farmer_uid"].iloc[0] == target_uid` |
| `test_fetch_multiple_raw_data_returns_n_rows` | `fetch_multiple_raw_data(table, n=3)` | Insert ≥ 3 rows | `len(result) == 3` |
| `test_save_batch_evaluations_inserts_correct_count` | `save_batch_evaluations(df, results)` | Empty `candidate_result` table | Row count in `candidate_result` == `len(results)` |
| `test_create_and_get_job_round_trip` | `create_job` + `get_job` | None | `get_job(job_id)["status"] == "pending"` |
| `test_update_job_result_sets_completed` | `update_job_result` | Job exists with status "pending" | `get_job(job_id)["status"] == "completed"` and `result` field populated |

**Fixtures required**: `test_db_engine`  
**Note**: Raw farmer data inserts for `fetch_raw_data` tests must use actual raw table — tests insert synthetic rows directly via the engine.

---

## CI Workflow Contract

**File**: `.github/workflows/ci.yml`

| Contract | Value |
|----------|-------|
| Trigger | `push` on any branch; `pull_request` to `main` |
| `lint` → `test` | `needs: lint` on test job |
| `test` → `build` | `needs: [lint, test]` on build job |
| Coverage gate | `--cov-fail-under=80` causes non-zero exit if coverage below threshold |
| Postgres service | `postgres:16` with health check; exposed on `localhost:5432` within job |
| Secrets injected | `GEMINI_API_KEY` from `secrets.GEMINI_API_KEY`; `API_KEY=ci-test-key` hardcoded |
| `uv` action | `astral-sh/setup-uv@v5` in every job |
| Docker build | `docker build -f backend/Dockerfile -t lersha-backend:ci .` and ui equivalent |
