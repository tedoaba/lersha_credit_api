# API Contract: Lersha Credit Scoring API v1

**Service**: `backend/main.py` (FastAPI)  
**Base URL** (local): `http://localhost:8000`  
**Base URL** (Docker): `http://backend:8000` (from within compose network)  
**Auth**: All routes except `/` and `/health` require header `X-API-Key: <value>` matching `API_KEY` env var  
**Content-Type**: `application/json` for all request/response bodies  
**OpenAPI docs**: `GET /docs` (Swagger UI), `GET /redoc`

---

## Routes

| Method | Path | Auth | Status Codes | Description |
|---|---|---|---|---|
| `GET` | `/` | No | 200 | Root health check |
| `GET` | `/health` | No | 200, 503 | Dependency health check (real DB + ChromaDB ping) |
| `POST` | `/v1/predict` | Yes | 202, 400, 403, 422 | Submit inference job |
| `GET` | `/v1/predict/{job_id}` | Yes | 200, 202, 403, 404 | Poll job status / retrieve result |
| `GET` | `/v1/results` | Yes | 200, 403 | Paginated evaluation history |

---

## `GET /health`

**Auth**: None

**Response 200** — all dependencies healthy:
```json
{
  "status": "ok",
  "dependencies": {
    "postgresql": "ok",
    "chromadb": "ok"
  }
}
```

**Response 503** — one or more dependencies unreachable:
```json
{
  "status": "degraded",
  "dependencies": {
    "postgresql": "ok",
    "chromadb": "error: connection refused"
  }
}
```

> A static `{"status": "ok"}` response is forbidden. The handler must perform live dependency pings.

---

## `POST /v1/predict`

**Auth**: `X-API-Key` header required

**Request body**:
```json
{
  "source": "Batch Prediction",
  "farmer_uid": null,
  "number_of_rows": 10
}
```

| Field | Type | Required | Validation |
|---|---|---|---|
| `source` | `"Single Value" \| "Batch Prediction"` | Yes | Must be one of the two literals |
| `farmer_uid` | `string \| null` | Conditional | Required when `source = "Single Value"` |
| `number_of_rows` | `integer \| null` | Conditional | Required when `source = "Batch Prediction"`; range 1–100 |

**Response 202 Accepted** — job created, inference running in background:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted"
}
```

**Response 400 Bad Request** — semantic validation error (cross-field):
```json
{
  "detail": "farmer_uid is required for Single Value prediction",
  "type": "validation_error"
}
```

**Response 403 Forbidden** — missing or invalid API key:
```json
{
  "detail": "Invalid or missing API key"
}
```

**Response 422 Unprocessable Entity** — Pydantic validation failure (malformed JSON, wrong types):
```json
{
  "detail": [
    {
      "loc": ["body", "source"],
      "msg": "value is not a valid enumeration member",
      "type": "type_error.enum"
    }
  ]
}
```

---

## `GET /v1/predict/{job_id}`

**Auth**: `X-API-Key` header required

**Path parameter**: `job_id` — UUID string of a previously submitted job

**Response 202 Accepted** — job still running:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "result": null,
  "error": null
}
```

**Response 200 OK — `status: completed`**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "result_xgboost": {
      "status": "batch_evaluation_completed",
      "records_processed": 10,
      "evaluations": [
        {
          "farmer_uid": "F-001",
          "first_name": "Abebe",
          "predicted_class_name": "Eligible",
          "top_feature_contributions": [
            { "feature": "net_income", "value": 0.412 },
            { "feature": "yield_per_hectare", "value": 0.287 }
          ],
          "rag_explanation": "The farmer was classified as Eligible primarily because...",
          "model_name": "xgboost",
          "timestamp": "2026-03-29T02:44:13+00:00"
        }
      ]
    },
    "result_random_forest": {
      "status": "batch_evaluation_completed",
      "records_processed": 10,
      "evaluations": [ ... ]
    }
  },
  "error": null
}
```

> **Critical constraint**: Fields `result_18`, `result_44`, `result_featured` MUST NOT appear in any response. Only `result_xgboost` and `result_random_forest` are valid result keys.

**Response 200 OK — `status: failed`**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "result": null,
  "error": "Model file not found at backend/models/xgboost_36_credit_score.pkl"
}
```

**Response 404 Not Found** — unknown job_id:
```json
{
  "detail": "Job not found",
  "type": "not_found"
}
```

---

## `GET /v1/results`

**Auth**: `X-API-Key` header required

**Query parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | `integer` | `500` | Max records to return; range 1–1000 |
| `model_name` | `string \| null` | `null` | Filter by model (`xgboost`, `random_forest`) |

**Response 200 OK**:
```json
{
  "total": 47,
  "records": [
    {
      "farmer_uid": "F-001",
      "first_name": "Abebe",
      "middle_name": "Bekele",
      "last_name": "Chala",
      "predicted_class_name": "Eligible",
      "top_feature_contributions": [
        { "feature": "net_income", "value": 0.412 }
      ],
      "rag_explanation": "The farmer was classified as Eligible...",
      "model_name": "xgboost",
      "timestamp": "2026-03-29T02:44:13+00:00"
    }
  ]
}
```

---

## Error Response Shape (all routes)

All error responses from application-level handlers follow the same envelope:
```json
{
  "detail": "<human-readable message>",
  "type": "<snake_case_error_type>"
}
```

FastAPI's native 422 Unprocessable Entity (Pydantic validation) retains FastAPI's standard `detail` array format — this is acceptable and not overridden.

---

## Authentication Details

- Header name: `X-API-Key`
- Value: Must exactly match `config.api_key` (env var `API_KEY`)
- Applied via `Depends(require_api_key)` on the `APIRouter` for `predict` and `results` routers
- Returns `HTTP 403 Forbidden` on mismatch — not 401 (intentional; no WWW-Authenticate challenge needed for API key auth)
- Health routes (`/`, `/health`) are explicitly excluded from the auth dependency

---

## Rate Limiting

- Route: `POST /v1/predict`
- Default limit: 10 requests per minute per client IP
- Configurable via `hyperparams.yaml` → `rate_limiting.requests_per_minute`
- Implemented via `slowapi` middleware
- Response on limit exceeded: `HTTP 429 Too Many Requests`

---

## Versioning Policy

- Current version: `v1` (prefix: `/v1/`)
- All breaking changes must introduce a new version prefix (`/v2/`)
- Old version kept alive for ≥ 30 days after a new version is released
- Deprecated routes return a `Deprecation` response header with the sunset date
