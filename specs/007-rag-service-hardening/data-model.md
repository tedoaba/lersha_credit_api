# Data Model: RAG System Hardening (007-rag-service-hardening)

**Phase**: 1 — Design & Contracts
**Date**: 2026-04-02
**Branch**: `007-rag-service-hardening`

---

## Existing Tables (unchanged)

### `rag_documents`
Stores the semantic knowledge corpus (feature definitions, policy rules).

| Column | Type | Notes |
|--------|------|-------|
| `id` | `INTEGER PK` | Auto-increment surrogate key |
| `doc_id` | `UUID UNIQUE` | Stable identifier for upsert deduplication |
| `category` | `VARCHAR(100)` | `'feature_definition'` or `'policy_rule'` |
| `title` | `VARCHAR(255)` | Human-readable title |
| `content` | `TEXT` | Full document text for RAG context assembly |
| `embedding` | `VECTOR(384)` | `all-MiniLM-L6-v2` sentence-transformer embedding |
| `metadata` | `JSONB` | Extensible attribute bag |
| `created_at` | `TIMESTAMPTZ` | UTC ingestion timestamp |
| `updated_at` | `TIMESTAMPTZ` | UTC last-update timestamp |

---

## Modified Table

### `rag_audit_log` ← extended by migration `004_add_audit_cache_fields`

Append-only audit trail for every retrieval and explanation event.

| Column | Type | Notes |
|--------|------|-------|
| `id` | `INTEGER PK` | Auto-increment surrogate key |
| `query_text` | `TEXT NOT NULL` | Raw query string used for retrieval |
| `retrieved_ids` | `INTEGER[]` | Array of `rag_documents.id` values returned |
| `prediction` | `VARCHAR(100)` | ML model prediction label |
| `model_name` | `VARCHAR(100)` | ML model name |
| `job_id` | `UUID` | FK link to `inference_jobs.job_id` |
| `generated_text` | `TEXT` | Gemini-generated explanation (null on retrieval-only) |
| `latency_ms` | `INTEGER` | End-to-end latency in milliseconds |
| `created_at` | `TIMESTAMPTZ` | UTC event timestamp |
| **`cache_hit`** | **`BOOLEAN DEFAULT FALSE`** | **NEW — was result served from cache?** |
| **`prompt_version`** | **`VARCHAR(20)`** | **NEW — active prompt version (e.g. 'v1')** |

**Alembic migration**: `backend/alembic/versions/004_add_audit_cache_fields.py`

---

## New Python Value Objects

These are pure dataclasses / typed dicts — no new DB tables.

### `RetrievedDoc`
```python
@dataclass
class RetrievedDoc:
    doc_id: int           # rag_documents.id
    content: str          # full document text
    similarity: float     # cosine similarity score (0.0 – 1.0)
```

### `ExplainResult`
```python
@dataclass
class ExplainResult:
    farmer_uid: str
    prediction: str
    explanation: str
    retrieved_doc_ids: list[int]
    cache_hit: bool
    prompt_version: str
    latency_ms: int
```

---

## Prompt Version File Schema

Each versioned prompt lives at `backend/prompts/v{N}.yaml`:

```yaml
version: "v1"                    # must match filename prefix
system: |
  You are an agricultural credit scoring analyst …
context_header: |
  The following knowledge documents are relevant …
task: |
  Explain the prediction in 2–3 sentences …
input_template: |
  Prediction: {prediction}
  SHAP contributions (top features): {shap_json}
  Farmer UID: {farmer_uid}
response_directive: |
  Provide 2–3 concise sentences only. No lists, headings, or JSON.
```

Available substitution variables: `{prediction}`, `{shap_json}`, `{retrieved_context}`, `{farmer_uid}`.

---

## Cache Entry (Redis)

Not a DB row — lives in Redis with 24-hour TTL.

| Field | Value |
|-------|-------|
| **Key** | `rag:explain:<sha256_hex>` |
| **Value** | Plain-text UTF-8 explanation string |
| **TTL** | `86400` seconds (24 hours) |
| **Key construction** | `SHA-256(canonical_json({prediction, shap_dict (sorted keys, 6 dp), prompt_version}))` |

---

## Entity Relationships

```
inference_jobs ─────────────── rag_audit_log
  (job_id)         FK job_id →   (job_id, UUID nullable)

rag_documents ──────── rag_audit_log
  (id)         ids in →  (retrieved_ids, INTEGER[])

Redis ────────────────────────────────────────
  key: rag:explain:<sha256>   value: explanation text
```

---

## State Transitions: Explain Request

```
Request arrives at POST /v1/explain
  │
  ▼
Fetch job record (job_id + record_index) from candidate_result
  │
  ├─ Not found → 404
  │
  ▼
Build cache key (prediction + shap_dict + prompt_version)
  │
  ├─ Redis hit  → return cached explanation (cache_hit=True)
  │               write audit log (cache_hit=True)
  │
  └─ Redis miss → retrieve docs from rag_documents (pgvector)
                  ├─ SQLAlchemyError → 503
                  │
                  └─ OK → assemble versioned prompt
                           → call Gemini
                           ├─ All retries fail → 503
                           │
                           └─ OK → store in Redis (TTL 24h)
                                    write audit log (cache_hit=False)
                                    return ExplainResponse (200)
```
