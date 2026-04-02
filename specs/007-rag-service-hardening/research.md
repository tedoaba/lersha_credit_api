# Research: RAG System Hardening (007-rag-service-hardening)

**Phase**: 0 — Research & Unknowns Resolution
**Date**: 2026-04-02
**Branch**: `007-rag-service-hardening`

---

## 1. Current RAG Implementation Audit

### Decision
Existing logic lives exclusively in `backend/chat/rag_engine.py` as a module of top-level functions. There is no service class. The single prompt is read from `backend/prompts/prompts.yaml` via a hardcoded path in `config.prompt_path`. Redis is already in `pyproject.toml` (`redis>=5.0`) and `config.redis_url` already exists in `Config`. The audit log ORM model (`RagAuditLogDB`) exists in `backend/services/db_model.py` but does **not** yet store `cache_hit` or `prompt_version` columns.

### Rationale
This confirms the feature is an evolution, not a greenfield build. We refactor `rag_engine.py` into a `RagService` class and extend schemas/DB without breaking the existing pipeline that calls `get_rag_explanation()`.

### Alternatives Considered
- Keep functional API, add wrappers: rejected — violates `[P1-MODULAR]`; a class with explicit dependencies (DB, Redis, prompt loader) is testable in isolation.

---

## 2. Redis Caching Pattern

### Decision
Use the `redis.Redis` client (synchronous, matching existing Celery/Redis usage). Cache key = `SHA-256(canonical_prediction + canonical_json(shap_dict, sorted keys, 6 dp precision) + prompt_version)`. TTL = 86400 s (24 h). On connection error, log WARNING and proceed without cache (graceful degradation per SC-007).

```python
import hashlib, json

def _build_cache_key(prediction: str, shap_dict: dict, prompt_version: str) -> str:
    canonical = json.dumps(
        {"prediction": prediction, "shap": shap_dict, "version": prompt_version},
        sort_keys=True,
        separators=(",", ":"),
        # float precision handled by rounding SHAP values to 6dp before serialisation
    )
    return "rag:explain:" + hashlib.sha256(canonical.encode()).hexdigest()
```

### Rationale
SHA-256 hash eliminates key length concerns. Sorting keys on the canonical JSON prevents key divergence from dict ordering. 6 dp float rounding kills floating-point variance across environments (Python `round(v, 6)`).

### Alternatives Considered
- MD5: faster but collision-prone and not acceptable in a regulated system.
- Redis HSET per field: adds complexity with no benefit here.

---

## 3. Prompt Versioning Strategy

### Decision
Rename `backend/prompts/prompts.yaml` → `backend/prompts/v1.yaml` (keep `prompts.yaml` as a symlink/copy for backward compat during transition). Future versions: `v2.yaml`, etc. Active version is controlled by `PROMPT_VERSION` env var, defaulting to `"v1"`. The `Config` class gains `prompt_version: str` and `prompt_dir: Path`. `RagService` resolves the active prompt at instantiation time, raising `FileNotFoundError` with a clear message if the file is absent.

Prompt YAML structure (versioned):
```yaml
version: v1
system: |   # analyst role
  ...
context_header: |   # prefix for retrieved docs
  ...
task: |   # 2-3 sentence instruction
  ...
input_template: |   # {prediction}, {shap_json}, {farmer_uid}
  ...
response_directive: |   # output instruction
  ...
```

Template variables available for substitution: `{prediction}`, `{shap_json}`, `{retrieved_context}`, `{farmer_uid}`.

### Rationale
File-per-version is the simplest auditable approach. Switching versions requires only an env var change and no restart (the service reads the file on each instantiation or call, not at module import). Hot-reload out of scope (spec Assumption §8).

### Alternatives Considered
- DB-stored prompts: adds a table, complicates rollback; rejected for v1.
- Single file with version key: requires code changes to add versions; rejected.

---

## 4. Explain Endpoint Architecture

### Decision
New router: `backend/api/routers/explain.py`. Request/response Pydantic models added to `backend/api/schemas.py`. Router registered in `backend/main.py` with `prefix="/v1/explain"`, `tags=["v1 — Explain"]`. Auth via `Depends(require_api_key)` (consistent with existing routers). The endpoint:

1. Fetches prediction record from DB by `job_id` + `record_index`.
2. Instantiates (or receives via DI) `RagService`.
3. Calls `rag_service.explain(prediction, shap_dict, farmer_uid)` → `ExplainResult`.
4. Returns typed `ExplainResponse`.

**Request schema** (`ExplainRequest`):
```python
job_id: str        # UUID of the completed inference job
record_index: int  # 0-based index into the job's evaluation list
model_name: str    # e.g. "xgboost", "random_forest"
```

**Response schema** (`ExplainResponse`):
```python
farmer_uid: str
prediction: str
explanation: str
retrieved_doc_ids: list[int]
cache_hit: bool
prompt_version: str
latency_ms: int
```

### Rationale
Keeping the endpoint thin (all logic in `RagService`) satisfies `[P6-API]`. Pydantic models satisfy `[P6-API]` boundary requirements. Leveraging existing `require_api_key` satisfies `[P9-SEC]`.

---

## 5. RagAuditLog Schema Gap

### Decision
Two new columns are required on `rag_audit_log`:
- `cache_hit: bool` — default `False` — was the result served from cache?
- `prompt_version: varchar(20)` — which prompt version was active?

Add via a new Alembic migration: `004_add_audit_cache_fields.py`. The `RagAuditLogDB` ORM model gains these two columns.

### Rationale
Spec FR-007 requires `cache_hit` in audit entries. Prompt version traceability is mandatory for compliance (User Story 3).

### Alternatives Considered
- Store in existing `generated_text` field as metadata: unqueryable; rejected.

---

## 6. Retrieval Query Construction

### Decision
```python
shap_json = json.dumps(
    {k: round(v, 6) for k, v in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]},
    separators=(",", ":"),
)
query = f"Model predicted: {prediction}\nTop features: {shap_json}"
```
Top 10 features by absolute SHAP value, sorted descending, ensures a stable and concise query regardless of the raw dict size.

### Rationale
Including all SHAP features creates excessively long query strings. Limiting to top 10 by magnitude focuses the embedding on the most influential factors (spec FR-012).

---

## 7. Testing Strategy

### Decision
- **Unit tests** (`backend/tests/unit/test_rag_service.py`): mock `sqlalchemy.orm.Session`, `redis.Redis`, and `google.genai.Client`. Assert: (a) cache hit returns immediately without Gemini call; (b) cache miss calls Gemini and stores result; (c) audit row is inserted in both paths; (d) Redis connection error triggers WARNING log and does not raise.
- **Integration tests** (`backend/tests/integration/test_explain_endpoint.py`): use `TestClient(create_app())` + real `test_lersha` PostgreSQL DB (seeded in conftest). Mock Gemini (`mocker.patch`). Assert 200, non-empty explanation, audit log entry present.

### Rationale
Matches `[P7-TEST]` pattern: pure unit tests with no external connections; integration tests with real DB but mocked LLM.

---

## 8. Dependency Additions

### Decision
No new `pyproject.toml` dependencies required — `redis>=5.0` is already listed (line 43). No other new packages needed.

### Rationale
`redis`, `tenacity`, `pyyaml`, `sqlalchemy`, `pgvector`, `google-genai` are all present. Cross-encoder re-ranking (optional per spec) would require `sentence-transformers` cross-encoder model — deferred.

---

## Summary of Resolved Unknowns

| Unknown | Resolution |
|---------|-----------|
| Cache key construction | SHA-256 of canonical JSON (sorted keys, 6 dp floats) |
| Prompt versioning format | `backend/prompts/v{N}.yaml` per version, `PROMPT_VERSION` env var |
| Audit schema gaps | New Alembic migration adds `cache_hit` + `prompt_version` columns |
| New dependencies | None — `redis>=5.0` already in `pyproject.toml` |
| RagService placement | `backend/chat/rag_service.py` (constitution §P1 mandates `backend/chat/` for RAG) |
| Explain router placement | `backend/api/routers/explain.py` |
| Prompt template variables | `{prediction}`, `{shap_json}`, `{retrieved_context}`, `{farmer_uid}` |
| Float determinism | `round(v, 6)` applied before serialisation |
| Graceful cache degradation | `try/except redis.RedisError` → `logger.warning`, proceed without cache |
