# API Contracts: FastAPI Backend v1

**Date**: 2026-03-29  
**Branch**: `002-fastapi-backend-http-wire`  
**Base URL**: `http://localhost:8000` (dev) / env `API_BASE_URL` (prod)

---

## Authentication

All routes under `/v1/` require:

```
X-API-Key: <value of API_KEY env var>
```

Missing or incorrect key → `403 Forbidden`.  
Health routes (`GET /`, `GET /health`) are **public** — no key required.

---

## Route Contracts

### `GET /`
**Purpose**: Root liveness probe  
**Auth**: None  
**Response 200**:
```json
{
  "message": "Lersha Credit Scoring API is running",
  "version": "1.0.0"
}
```

---

### `GET /health`
**Purpose**: Live dependency health check  
**Auth**: None  
**Response 200** (all deps OK):
```json
{
  "status": "ok",
  "dependencies": {
    "postgresql": "ok",
    "chromadb": "ok"
  }
}
```
**Response 503** (any dep degraded):
```json
{
  "status": "degraded",
  "dependencies": {
    "postgresql": "ok",
    "chromadb": "error: <message>"
  }
}
```

---

### `POST /v1/predict/`
**Purpose**: Submit an asynchronous inference job  
**Auth**: `X-API-Key` required  
**Content-Type**: `application/json`

**Request body**:
```json
{
  "source": "Batch Prediction",
  "farmer_uid": null,
  "number_of_rows": 5
}
```
| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `source` | `"Single Value" \| "Batch Prediction"` | YES | Literal enum |
| `farmer_uid` | `string \| null` | Conditional | Required when `source == "Single Value"` |
| `number_of_rows` | `integer \| null` | Conditional | Required when `source == "Batch Prediction"`; `1 ≤ n ≤ 100` |

**Response 202 Accepted**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted"
}
```

**Response 403 Forbidden** (missing/invalid API key):
```json
{"detail": "Invalid or missing API key"}
```

**Response 422 Unprocessable Entity** (schema violation):
```json
{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}
```

---

### `GET /v1/predict/{job_id}`
**Purpose**: Poll inference job status  
**Auth**: `X-API-Key` required  
**Path param**: `job_id` — UUID string from `POST /v1/predict`

**Response 200** (job found):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "result_xgboost": {
      "status": "success",
      "records_processed": 5,
      "evaluations": [...]
    },
    "result_random_forest": {
      "status": "success",
      "records_processed": 5,
      "evaluations": [...]
    }
  },
  "error": null
}
```

**Status field values**:
| Status | Meaning |
|--------|---------|
| `pending` | Job queued, not yet started |
| `processing` | Pipeline running |
| `completed` | Pipeline succeeded; `result` populated |
| `failed` | Pipeline raised exception; `error` populated |

**Response 404 Not Found**:
```json
{"detail": "Job not found"}
```

---

### `GET /v1/results/`
**Purpose**: Retrieve historical evaluation records  
**Auth**: `X-API-Key` required  
**Query params**:
| Param | Type | Default | Range |
|-------|------|---------|-------|
| `limit` | integer | 500 | 1–1000 |
| `model_name` | string | null | e.g., `"xgboost"` |

**Response 200**:
```json
{
  "total": 42,
  "records": [
    {
      "farmer_uid": "F001",
      "first_name": "Abebe",
      "middle_name": null,
      "last_name": "Bekele",
      "predicted_class_name": "Eligible",
      "top_feature_contributions": [
        {"feature": "net_income", "value": 0.23}
      ],
      "rag_explanation": "The farmer is eligible because...",
      "model_name": "xgboost",
      "timestamp": "2026-03-29T04:00:00+00:00"
    }
  ]
}
```

**Response 200** (no records):
```json
{"total": 0, "records": []}
```

---

## Error Envelope

All error responses follow FastAPI's default envelope:
```json
{"detail": "<human-readable message>"}
```

For validation errors (422), the `detail` field is an array of error objects per Pydantic/FastAPI convention.

---

## curl Verification Cheatsheet

```bash
# 1. Reject unauthenticated request → 403
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/v1/predict/

# 2. Submit prediction job → 202 + job_id
curl -s -X POST http://localhost:8000/v1/predict/ \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"source":"Batch Prediction","number_of_rows":2}'

# 3. Poll job status
curl -s http://localhost:8000/v1/predict/{JOB_ID} \
  -H "X-API-Key: your-secret-api-key-here"

# 4. Fetch results
curl -s "http://localhost:8000/v1/results/?limit=10" \
  -H "X-API-Key: your-secret-api-key-here"

# 5. Health check (no auth)
curl -s http://localhost:8000/health
```
