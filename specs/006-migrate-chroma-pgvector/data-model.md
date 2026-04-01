# Data Model: Migrate Vector Store — ChromaDB → PostgreSQL pgvector

**Branch**: `006-migrate-chroma-pgvector`  
**Phase**: 1 — Data model design  
**Date**: 2026-04-01

---

## New Entities

### 1. `rag_documents` Table

Stores all knowledge documents (feature definitions, policy rules) with their semantic vector embeddings.

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | `SERIAL` | `PRIMARY KEY` | Auto-incrementing surrogate key |
| `doc_id` | `UUID` | `UNIQUE NOT NULL DEFAULT gen_random_uuid()` | Stable external identifier |
| `category` | `VARCHAR(100)` | `NOT NULL` | Document category (e.g. `feature_definition`, `policy_rule`) |
| `title` | `VARCHAR(255)` | `NOT NULL` | Human-readable document title |
| `content` | `TEXT` | `NOT NULL` | Full document text |
| `embedding` | `VECTOR(384)` | `NOT NULL` | 384-dim sentence-transformer embedding |
| `metadata` | `JSONB` | `DEFAULT '{}'` | Extensible metadata bag |
| `created_at` | `TIMESTAMPTZ` | `DEFAULT NOW()` | Creation timestamp (UTC) |
| `updated_at` | `TIMESTAMPTZ` | `DEFAULT NOW()` | Last update timestamp (UTC) |

**Indexes**:
- `idx_rag_documents_embedding` — `USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)` — ANN retrieval
- `idx_rag_documents_category` — `(category)` — category filter pre-scan

**ORM Model**: `RagDocumentDB` in `backend/services/db_model.py`

---

### 2. `rag_audit_log` Table

Records every RAG retrieval event for compliance and debugging.

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | `SERIAL` | `PRIMARY KEY` | Auto-incrementing key |
| `query_text` | `TEXT` | `NOT NULL` | Raw query string (prediction + SHAP JSON) |
| `retrieved_ids` | `INTEGER[]` | `NULLABLE` | Array of `rag_documents.id` values returned |
| `prediction` | `VARCHAR(100)` | `NULLABLE` | Model prediction label (e.g. `"Eligible"`) |
| `model_name` | `VARCHAR(100)` | `NULLABLE` | ML model name that generated the prediction |
| `job_id` | `UUID` | `NULLABLE` | Links to `inference_jobs.job_id` |
| `generated_text` | `TEXT` | `NULLABLE` | Gemini-generated explanation text |
| `latency_ms` | `INTEGER` | `NULLABLE` | End-to-end RAG retrieval latency in ms |
| `created_at` | `TIMESTAMPTZ` | `DEFAULT NOW()` | Event timestamp (UTC) |

**ORM Model**: `RagAuditLogDB` in `backend/services/db_model.py`

---

## Modified Entities

### `Config` (backend/config/config.py)

| Change | Direction | Details |
|--------|-----------|---------|
| Remove `chroma_db_path` | Remove | Was `str`, env `CHROMA_DB_PATH` |
| Remove `CHROMA_DB_PATH` env var | Remove | From `.env.example` and `config.py` |

### `hyperparams.yaml` (backend/config/hyperparams.yaml)

| Change | Direction | Default |
|--------|-----------|---------|
| `inference.rag_similarity_threshold` | Add | `0.75` |
| `inference.rag_top_k` | Existing | `5` (unchanged) |

---

## Entity Relationships

```
inference_jobs ──────────────┐
  job_id (UUID PK)           │
                             │ (soft ref via UUID)
rag_audit_log ───────────────┤
  id (SERIAL PK)             │
  job_id (UUID, nullable) ───┘
  retrieved_ids → rag_documents.id[]

rag_documents
  id (SERIAL PK)
  doc_id (UUID, unique)
```

**Note**: `retrieved_ids` is a denormalized integer array (not a foreign key) to keep audit writes fast and non-blocking. This avoids cascading delete constraints on the audit log.

---

## Migration Chain

```
001_initial_schema.py   (candidate_result table)
       ↓
002_add_inference_jobs.py  (inference_jobs table)
       ↓
003_add_pgvector.py     ← NEW (rag_documents, rag_audit_log, indexes, extension)
```

**Migration `003_add_pgvector` steps (in order)**:
1. `op.execute("CREATE EXTENSION IF NOT EXISTS vector")` — enables pgvector
2. `op.create_table("rag_documents", ...)` — main document store
3. `op.create_table("rag_audit_log", ...)` — retrieval audit log
4. `op.execute("CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding ...")` — ANN index (must be after table creation)
5. `op.execute("CREATE INDEX IF NOT EXISTS idx_rag_documents_category ...")` — category filter index

**Rollback `downgrade` steps (in reverse order)**:
1. Drop `idx_rag_documents_category`
2. Drop `idx_rag_documents_embedding`
3. Drop `rag_audit_log`
4. Drop `rag_documents`
5. `DROP EXTENSION IF EXISTS vector` — only if no other tables use it

---

## Retrieval Query Contract

The core retrieval query executed by `rag_engine.py` → `retrieve_docs()`:

```sql
SELECT id, content, 1 - (embedding <=> :query_vec) AS similarity
FROM rag_documents
WHERE category = ANY(:categories)
  AND 1 - (embedding <=> :query_vec) > :threshold
ORDER BY embedding <=> :query_vec
LIMIT :top_k
```

**Bound parameters**:
- `:query_vec` — 384-float list cast to `::vector` at bind time
- `:categories` — `['feature_definition', 'policy_rule']` (list of allowed categories)
- `:threshold` — `config.hyperparams.inference.rag_similarity_threshold` (default `0.75`)
- `:top_k` — `config.hyperparams.inference.rag_top_k` (default `5`)

**Return type**: `List[Tuple[int, str, float]]` — `(id, content, similarity_score)`

---

## Validation Data Requirements

For the migration integration test (`backend/tests/integration/test_rag_pgvector.py`):

- Insert exactly 10 documents across 2 categories
- Query with a clearly similar string
- Assert: top-3 results all have `similarity > 0.75`
- Assert: `rag_audit_log` has exactly 1 new row after the query
- Assert: retrieved `id` values appear in `rag_audit_log.retrieved_ids`
- Assert: round-trip latency (query + audit write) < 50 ms
