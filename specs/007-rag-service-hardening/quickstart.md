# Quickstart: RAG System Hardening (007-rag-service-hardening)

**Date**: 2026-04-02 | **Branch**: `007-rag-service-hardening`

---

## What This Feature Adds

This feature hardens the existing RAG explanation system with:

1. **`RagService` class** — encapsulates retrieval + explain in `backend/chat/rag_service.py`
2. **Prompt versioning** — `backend/prompts/v1.yaml` (rename from `prompts.yaml`) + `PROMPT_VERSION` env var
3. **Redis explanation cache** — 24-hour TTL, SHA-256 keyed, graceful degradation on connection failure
4. **`POST /v1/explain` endpoint** — dedicated HTTP interface in `backend/api/routers/explain.py`
5. **Audit log enhancements** — `cache_hit` + `prompt_version` columns via Alembic migration `004`
6. **Unit + integration tests** — `test_rag_service.py` + `test_explain_endpoint.py`

---

## Prerequisites

Ensure the following are already working (from feature `006-migrate-chroma-pgvector`):
- PostgreSQL running with `rag_documents` and `rag_audit_log` tables
- `pgvector` extension enabled
- At least some `rag_documents` rows ingested
- Redis running and `REDIS_URL` set in `.env`

---

## Environment Setup

Add to `.env` (and document in `.env.example`):

```bash
# Prompt versioning — controls which prompt YAML is loaded
PROMPT_VERSION=v1

# Redis URL (likely already set for Celery — same value)
REDIS_URL=redis://localhost:6379/0
```

---

## Apply the Alembic Migration

```bash
# From repo root
make migrate
# or directly:
uv run alembic upgrade head
```

Verify:
```sql
-- In psql / pgAdmin
\d rag_audit_log
-- Should show columns: cache_hit (boolean), prompt_version (varchar)
```

---

## Rename the Existing Prompt File

```bash
# Windows PowerShell
Move-Item backend/prompts/prompts.yaml backend/prompts/v1.yaml
```

The `config.prompt_path` legacy key will no longer be used by `RagService`; it is retained for the existing `get_rag_explanation()` function until that is fully replaced.

---

## Run the Tests

```bash
# Unit tests only (no DB/Redis required — all mocked)
uv run pytest backend/tests/unit/test_rag_service.py -v

# Integration tests (requires running PostgreSQL test DB + Redis)
uv run pytest backend/tests/integration/test_explain_endpoint.py -v

# Full suite with coverage
uv run pytest --cov=backend --cov-report=term-missing
```

---

## Call the Explain Endpoint

```bash
# POST /v1/explain — first call (cache miss, calls Gemini)
curl -X POST http://localhost:8000/v1/explain \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "job_id": "<completed-job-uuid>",
    "record_index": 0,
    "model_name": "xgboost"
  }'

# Response
{
  "farmer_uid": "ETH-2024-001234",
  "prediction": "Eligible",
  "explanation": "...",
  "retrieved_doc_ids": [12, 7, 23],
  "cache_hit": false,
  "prompt_version": "v1",
  "latency_ms": 1842
}

# Second identical call — same inputs within 24h → cache hit
# Response will have "cache_hit": true, latency_ms < 100
```

---

## Add a New Prompt Version (v2)

1. Copy and edit `backend/prompts/v1.yaml` → `backend/prompts/v2.yaml`
2. Update the `version:` field at the top of the YAML
3. Set `PROMPT_VERSION=v2` in `.env` (or the container environment)
4. Restart the API server (or next request cycle picks up the new env value)

> No code changes, no redeployment of the container image required.

---

## Verify Audit Log

```sql
SELECT id, query_text, array_length(retrieved_ids, 1) AS docs_retrieved,
       cache_hit, prompt_version, latency_ms, created_at
FROM rag_audit_log
ORDER BY created_at DESC
LIMIT 10;
```

---

## Key Files

| File | Role |
|------|------|
| `backend/chat/rag_service.py` | New — `RagService` class (retrieve + explain + cache + audit) |
| `backend/prompts/v1.yaml` | Renamed from `prompts.yaml` — versioned prompt template |
| `backend/api/routers/explain.py` | New — `POST /v1/explain` router |
| `backend/api/schemas.py` | Extended — `ExplainRequest`, `ExplainResponse` models |
| `backend/main.py` | Extended — register explain router |
| `backend/config/config.py` | Extended — `prompt_version`, `prompt_dir` properties |
| `backend/services/db_model.py` | Extended — `cache_hit`, `prompt_version` on `RagAuditLogDB` |
| `backend/alembic/versions/004_add_audit_cache_fields.py` | New — DB migration |
| `backend/tests/unit/test_rag_service.py` | New — unit tests for `RagService` |
| `backend/tests/integration/test_explain_endpoint.py` | New — integration tests for `/v1/explain` |
