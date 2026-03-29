# API Contract Changes: Application-Level Hardening

**Phase 1 — Interface Contracts**
**Date**: 2026-03-29

---

## Changed Endpoints

### POST /v1/predict

| Attribute | Before | After |
|---|---|---|
| Async mechanism | FastAPI BackgroundTasks (in-process) | Celery task via Redis queue |
| Rate limiting | None | 10 requests / minute / IP |
| Required header | `X-API-Key` | `X-API-Key` (unchanged) |
| Response on limit exceeded | N/A | `429 Too Many Requests` |
| Request signature | `(item: PredictRequest, background_tasks: BackgroundTasks)` | `(request: Request, item: PredictRequest)` |

**Rate limit exceeded response (HTTP 429)**:
```json
{
  "error": "Rate limit exceeded: 10 per 1 minute"
}
```

**No change to**:
- Response on success: `{"job_id": "<uuid>", "status": "accepted"}` (HTTP 202)
- Request body schema (`PredictRequest`)
- Authentication mechanism

---

### GET /health

| Attribute | Before | After |
|---|---|---|
| Response body (success) | `{"status":"ok","dependencies":{"postgresql":"ok","chromadb":"ok"}}` | `{"db":"ok","chroma":"ok"}` |
| Response body (failure) | `{"status":"degraded","dependencies":{"postgresql":"ok","chromadb":"error: ..."}}` | `{"db":"ok","chroma":"error: <reason>"}` |
| Status code (success) | `200` | `200` |
| Status code (failure) | `503` | `503` |

**Breaking change**: The response body shape changes. The `ui/utils/api_client.py` `health()` method must be updated to reflect the new key names (`db`, `chroma`) if it inspects individual dependency keys.

---

## New Response Headers (All Endpoints)

| Header | Value | Description |
|---|---|---|
| `X-Request-ID` | UUID v4 string | Echoes inbound `X-Request-ID` or newly generated UUID |

All API responses now carry this header, injected by `RequestIDMiddleware`.

---

## Unchanged Contracts

| Endpoint | Status |
|---|---|
| `GET /v1/predict/{job_id}` | Unchanged — job polling contract preserved |
| `GET /v1/results/` | Unchanged |
| `GET /` | Unchanged |
| Authentication (`X-API-Key`) | Unchanged for all protected routes |
| `PredictRequest` schema | Unchanged |
| `JobAcceptedResponse` schema | Unchanged |
| `JobStatusResponse` schema | Unchanged |

---

## New Internal Interfaces

### Celery Task: `run_inference_task`

**Task name**: `run_inference_task`
**Broker**: Redis (`REDIS_URL` env var)
**Backend**: Redis (`REDIS_URL` env var)

**Call signature**:
```python
run_inference_task.delay(job_id: str, payload: dict) -> AsyncResult
```

**Payload shape** (same as `PredictRequest.dict()`):
```json
{
  "source": "Single Value" | "Batch Prediction",
  "farmer_uid": "<string>" | null,
  "number_of_rows": <int> | null
}
```

**Side effects on success**: Updates `inference_jobs.status = "completed"`, `inference_jobs.result = {...}`, `inference_jobs.completed_at`

**Side effects on failure**: Updates `inference_jobs.status = "failed"`, `inference_jobs.error = "<message>"`, `inference_jobs.completed_at`

---

### Worker Process Launch Command

**Development** (single worker, no task queue):
```bash
uv run celery -A backend.worker worker --loglevel=info
```

**Production** (controlled by docker-compose / process manager):
```bash
uv run celery -A backend.worker worker --loglevel=info --concurrency=4
```
