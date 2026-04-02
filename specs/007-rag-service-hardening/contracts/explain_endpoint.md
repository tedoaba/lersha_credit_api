# API Contracts: RAG System Hardening (007-rag-service-hardening)

**Phase**: 1 — Design & Contracts
**Date**: 2026-04-02
**Branch**: `007-rag-service-hardening`

---

## New Endpoint: POST /v1/explain

### Overview

| Property | Value |
|----------|-------|
| **Method** | `POST` |
| **Path** | `/v1/explain` |
| **Authentication** | `X-API-Key` header (same as all other v1 routes) |
| **OpenAPI Tag** | `v1 — Explain` |
| **Router file** | `backend/api/routers/explain.py` |
| **Registration** | `backend/main.py` → `app.include_router(explain.router, prefix="/v1/explain", tags=["v1 — Explain"])` |

---

### Request Body — `ExplainRequest`

```json
{
  "job_id": "a3f7c1d2-4e8b-4f2a-9b1c-3d2e5f6a7b8c",
  "record_index": 0,
  "model_name": "xgboost"
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `job_id` | `string` | ✅ | UUID format | ID of the completed inference job |
| `record_index` | `integer` | ✅ | `>= 0` | 0-based index of the farmer record within the job's evaluation list |
| `model_name` | `string` | ✅ | non-empty | Name of the ML model (e.g. `"xgboost"`, `"random_forest"`) |

---

### Response Body — `ExplainResponse` (HTTP 200)

```json
{
  "farmer_uid": "ETH-2024-001234",
  "prediction": "Eligible",
  "explanation": "The farmer's strong yield per hectare and consistent repayment history are the primary drivers behind the Eligible prediction. The large farm size and mechanized inputs further reduce the perceived risk, aligning with the policy rule that rewards documented productivity. However, the moderate family size slightly increases the income-per-member burden, which had a minor negative influence on the overall score.",
  "retrieved_doc_ids": [12, 7, 23, 4, 18],
  "cache_hit": false,
  "prompt_version": "v1",
  "latency_ms": 1842
}
```

| Field | Type | Description |
|-------|------|-------------|
| `farmer_uid` | `string` | Farmer unique identifier from the prediction record |
| `prediction` | `string` | Predicted class label (e.g. `"Eligible"`, `"Not Eligible"`) |
| `explanation` | `string` | AI-generated natural-language explanation (2–3 sentences) |
| `retrieved_doc_ids` | `integer[]` | Ordered list of `rag_documents.id` values that formed the retrieval context |
| `cache_hit` | `boolean` | `true` if response was served from cache without calling the AI service |
| `prompt_version` | `string` | Active prompt version that was used (e.g. `"v1"`) |
| `latency_ms` | `integer` | End-to-end latency from request receipt to response in milliseconds |

---

### Error Responses

| HTTP Status | Condition | Response Body Shape |
|-------------|-----------|---------------------|
| `400 Bad Request` | Malformed `job_id` (not UUID format) | `{"detail": "...", "type": "validation_error"}` |
| `403 Forbidden` | Missing or invalid `X-API-Key` | `{"detail": "Invalid or missing API key"}` |
| `404 Not Found` | `job_id` does not exist or `record_index` out of range | `{"detail": "Job a3f7... not found or record_index 5 out of range"}` |
| `422 Unprocessable Entity` | Pydantic validation failure (e.g. `record_index < 0`) | Standard FastAPI 422 body |
| `503 Service Unavailable` | pgvector retrieval or Gemini generation failed after retries | `{"detail": "Explanation service temporarily unavailable. Retry after 10 seconds.", "type": "upstream_error"}` |

---

### Internal Service Interface

`RagService` (in `backend/chat/rag_service.py`) exposes:

```python
class RagService:
    def retrieve(self, query: str) -> list[RetrievedDoc]:
        """Retrieve top-k similar knowledge documents from pgvector.

        Writes an audit row after every call (hit or empty).
        """
        ...

    def explain(
        self,
        prediction: str,
        shap_dict: dict,
        farmer_uid: str,
        job_id: str | None = None,
        model_name: str | None = None,
    ) -> ExplainResult:
        """Generate (or return cached) explanation for a credit prediction.

        On cache miss:  retrieve → prompt assembly → Gemini → cache store → audit write.
        On cache hit:   return cached text → audit write (cache_hit=True).
        On cache error: log WARNING, proceed as cache miss.
        """
        ...
```

The router delegates entirely to `RagService.explain()` and maps the `ExplainResult` dataclass to the `ExplainResponse` Pydantic model. Zero business logic in the router.

---

## Modified Internal Interfaces

### `RagAuditLogDB` — two new columns (Alembic `004_add_audit_cache_fields`)

```python
cache_hit: Column = Column(Boolean, nullable=False, default=False)
prompt_version: Column = Column(String(20), nullable=True)
```

### `Config` — two new properties

```python
self.prompt_version: str = os.getenv("PROMPT_VERSION", "v1")
self.prompt_dir: Path = BASE_DIR / "backend" / "prompts"
```

---

## Existing Endpoints — Unchanged

| Method | Path | Contract Status |
|--------|------|----------------|
| `POST` | `/v1/predict` | ✅ Unchanged |
| `GET` | `/v1/predict/{job_id}` | ✅ Unchanged |
| `GET` | `/v1/results` | ✅ Unchanged |
| `GET` | `/health` | ✅ Unchanged |
