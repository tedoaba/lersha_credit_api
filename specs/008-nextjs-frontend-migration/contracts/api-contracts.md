# API Contracts: Frontend → Backend

**Feature**: 008-nextjs-frontend-migration  
**Phase**: 1 — Design & Contracts  
**Format**: REST endpoint contract (matches FastAPI backend in `backend/api/`)  
**Base URL (via proxy)**: `/v1/`  
**Authentication**: `X-API-Key: <stored_key>` header on all requests

---

## Contract 1 — Submit Prediction Job

**Endpoint**: `POST /v1/predict/`  
**Auth required**: Yes  
**Rate limit**: 10 requests / minute / IP (enforced by backend)  
**Response code**: `202 Accepted`

### Request Body

```json
{
  "source": "Batch Prediction",
  "number_of_rows": 10
}
```

```json
{
  "source": "Single Value",
  "farmer_uid": "F-001234"
}
```

### Response Body (202)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "accepted"
}
```

### Error Cases

| HTTP | Condition |
|------|-----------|
| `400 Bad Request` | Missing `farmer_uid` for Single Value, or missing `number_of_rows` for Batch |
| `401 Unauthorized` | Missing or invalid `X-API-Key` |
| `429 Too Many Requests` | Rate limit exceeded |
| `503 Service Unavailable` | Backend inference worker unreachable |

---

## Contract 2 — Poll Job Status

**Endpoint**: `GET /v1/predict/{job_id}`  
**Auth required**: Yes  
**Polling strategy**: Every 2 000ms while `status` is `pending` or `processing`; stop when `completed` or `failed`  
**Response code**: `200 OK`

### Response Body

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "result_xgboost": {
      "status": "ok",
      "records_processed": 10,
      "evaluations": [
        {
          "predicted_class_name": "Eligible",
          "top_feature_contributions": [
            { "feature": "loan_amount", "value": 0.42 },
            { "feature": "farm_size_ha", "value": -0.18 }
          ],
          "rag_explanation": "The farmer demonstrates adequate repayment signals...",
          "model_name": "xgboost"
        }
      ]
    }
  },
  "error": null
}
```

**While pending/processing**, `result` and `error` are `null`:
```json
{
  "job_id": "550e8400-...",
  "status": "pending",
  "result": null,
  "error": null
}
```

**On failure**, `result` is `null`, `error` is populated:
```json
{
  "job_id": "550e8400-...",
  "status": "failed",
  "result": null,
  "error": "Inference pipeline error: model file not found"
}
```

### Error Cases

| HTTP | Condition |
|------|-----------|
| `401 Unauthorized` | Missing or invalid `X-API-Key` |
| `404 Not Found` | `job_id` does not exist |

---

## Contract 3 — Retrieve All Results

**Endpoint**: `GET /v1/results/`  
**Auth required**: Yes  
**Query parameters**:
- `limit` — integer, 1–1000, default 500
- `model_name` — optional string filter (e.g. `"xgboost"`)  

**Response code**: `200 OK`

### Response Body

```json
{
  "total": 47,
  "records": [
    {
      "farmer_uid": "F-001234",
      "first_name": "Abebe",
      "middle_name": null,
      "last_name": "Girma",
      "predicted_class_name": "Eligible",
      "top_feature_contributions": [
        { "feature": "loan_amount", "value": 0.42 }
      ],
      "rag_explanation": "The farmer has demonstrated...",
      "model_name": "xgboost",
      "timestamp": "2026-04-03T08:00:00Z"
    }
  ]
}
```

### Error Cases

| HTTP | Condition |
|------|-----------|
| `401 Unauthorized` | Missing or invalid `X-API-Key` |
| `422 Unprocessable Entity` | `limit` outside allowed range |

---

## Contract 4 — Generate AI Explanation

**Endpoint**: `POST /v1/explain/`  
**Auth required**: Yes  
**Response code**: `200 OK`  
**Note**: Response may be served from Redis cache — check `cache_hit` field

### Request Body

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "record_index": 0,
  "model_name": "xgboost"
}
```

### Response Body

```json
{
  "farmer_uid": "F-001234",
  "prediction": "Eligible",
  "explanation": "The farmer demonstrates adequate repayment capacity...",
  "retrieved_doc_ids": [12, 45, 67],
  "cache_hit": false,
  "prompt_version": "v1",
  "latency_ms": 1240
}
```

### Error Cases

| HTTP | Condition |
|------|-----------|
| `401 Unauthorized` | Missing or invalid `X-API-Key` |
| `404 Not Found` | `job_id` not found, `record_index` out of range, or model key absent |
| `503 Service Unavailable` | Gemini/pgvector retrieval failed after retries; retry after 10s |

---

## Contract 5 — Health Check

**Endpoint**: `GET /health`  
**Auth required**: No  
**Response code**: `200 OK` (healthy) or `503 Service Unavailable` (degraded)

### Response Body (healthy)

```json
{
  "status": "ok",
  "dependencies": {
    "postgres": "ok",
    "redis": "ok"
  }
}
```

### Response Body (degraded)

```json
{
  "status": "degraded",
  "dependencies": {
    "postgres": "ok",
    "redis": "error: Connection refused"
  }
}
```

---

## Frontend Component → Contract Mapping

| Component / Page | Contracts Used |
|-----------------|----------------|
| `app/predict/page.tsx` — PredictionForm | Contract 1 (submit), Contract 2 (poll) |
| `app/results/page.tsx` — ResultsTable | Contract 3 (list) |
| `app/results/[id]/page.tsx` — EvaluationCard | Contract 2 (job detail), Contract 4 (explain) |
| `app/page.tsx` — Dashboard | Contract 3 (summary counts via ISR) |
| `app/settings/page.tsx` — Settings | No network calls (localStorage only) |
| `components/JobStatusBadge.tsx` | Contract 2 (status field) |
| `components/FeatureContribChart.tsx` | Local data from Contract 2 / 3 |
| `components/ExplanationPanel.tsx` | Contract 4 (explanation text + doc IDs) |
