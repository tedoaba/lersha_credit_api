# Implementation Plan: Migrate Vector Store from ChromaDB to PostgreSQL pgvector

**Branch**: `006-migrate-chroma-pgvector` | **Date**: 2026-04-01 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/006-migrate-chroma-pgvector/spec.md`

---

## Summary

Replace the ChromaDB persistent vector store with PostgreSQL pgvector to achieve transactional consistency and a unified data layer. The migration involves: enabling the `vector` extension via a new Alembic migration (003), adding two new ORM models (`RagDocumentDB`, `RagAuditLogDB`), rewriting `rag_engine.py` to query PostgreSQL via parameterized SQL, creating a one-shot ingestion script, and cleaning up ChromaDB infrastructure references. The ChromaDB dependency is removed in a final post-validation task.

---

## Technical Context

**Language/Version**: Python 3.12 (`python:3.12-slim` base image per `[P11-CONT]`)  
**Primary Dependencies**: FastAPI, SQLAlchemy 2.x, Alembic, `pgvector>=0.3.0`, `sentence-transformers>=2.7.0`, Celery, tenacity  
**Storage**: PostgreSQL 16 via `pgvector/pgvector:pg16` Docker image + `pgvector.sqlalchemy.Vector` ORM column  
**Testing**: pytest 8.x; unit tests in `backend/tests/unit/`, integration tests in `backend/tests/integration/` against `test_lersha` DB; `pytest-mock` for Gemini mocking; 80% coverage gate  
**Target Platform**: Linux (Docker container) — `backend` and `worker` services  
**Project Type**: Web service (FastAPI) + async worker (Celery)  
**Performance Goals**: RAG retrieval < 50 ms p95 after IVFFlat index is applied (SC-002)  
**Constraints**: No public API contract changes; parameterized SQL only (`[P8-DB]`); no `os.getenv()` outside `config.py` (`[P5-CONFIG]`); ruff line-length 120 (`[P2-PEP]`)  
**Scale/Scope**: Initial corpus is tens-of-thousands of documents; IVFFlat `lists=100` is appropriate; no re-indexing expected at this scale

---

## Constitution Check

*GATE: Must pass before implementation begins. Re-checked after Phase 1 design.*

| Principle | Check | Status |
|-----------|-------|--------|
| `[P1-MODULAR]` | New ORM models added to `db_model.py` (services layer). RAG logic stays in `backend/chat/`. Config changes only in `backend/config/`. Ingestion script in `backend/scripts/` (one-shot, not imported by app). | ✅ PASS |
| `[P2-PEP]` | All new code: PEP 8, type annotations on all public functions, PEP 257 docstrings, ruff-compliant. | ✅ PASS (enforced by CI) |
| `[P3-LOG]` | `rag_engine.py` rewrite uses `logger = get_logger(__name__)`. Ingestion script uses `print()` (script allowance). Audit log events at `INFO`. | ✅ PASS |
| `[P4-EXC]` | All DB calls wrapped in try/except. PostgreSQL errors caught as `sqlalchemy.exc.SQLAlchemyError`. Audit log write failure is logged + re-raised, not silenced. | ✅ PASS |
| `[P5-CONFIG]` | `rag_similarity_threshold` added to `hyperparams.yaml`. `chroma_db_path` removed from `config.py`. No new `os.getenv()` calls in business logic. `.env.example` updated. | ✅ PASS |
| `[P6-API]` | No API route changes. `rag_engine.py` is internal to the chat layer. `GET /health` updated to remove ChromaDB heartbeat. | ✅ PASS |
| `[P7-TEST]` | Integration test: insert 10 docs, query top-3, assert similarity > 0.75, assert audit log populated. Unit tests for `retrieve_docs()` with mocked session. 80% coverage gate enforced. | ✅ PASS |
| `[P8-DB]` | New SQL query uses `sqlalchemy.text()` with `:param` named bindings. ORM models follow `db_model.py` pattern. All writes via SQLAlchemy Session. | ✅ PASS |
| `[P9-SEC]` | No new auth surfaces. Audit log must not record PII in `query_text` beyond prediction label + feature names. | ✅ PASS |
| `[P10-OBS]` | `GET /health` updated: replaces ChromaDB heartbeat with pgvector extension existence check. Retrieval timing logged at `DEBUG`. | ✅ PASS |
| `[P11-CONT]` | `docker-compose.yml` postgres image changed to `pgvector/pgvector:pg16`. Same change applied to `docker-compose.prod.yml`. `chroma_data` volume and mounts removed from backend + worker. | ✅ PASS |
| `[P12-CI]` | No CI pipeline changes needed. Existing `test`, `lint`, `mypy` jobs cover all new code. Migration test added to integration suite. | ✅ PASS |

**Violations requiring justification**: None.

---

## Project Structure

### Documentation (this feature)

```text
specs/006-migrate-chroma-pgvector/
├── plan.md              ← this file
├── research.md          ← Phase 0 output (complete)
├── data-model.md        ← Phase 1 output (complete)
├── quickstart.md        ← Phase 1 output (complete)
├── checklists/
│   └── requirements.md  ← spec quality checklist
└── tasks.md             ← Phase 2 output (via /speckit.tasks — NOT YET)
```

### Source Code — Files Modified or Created

```text
backend/
├── alembic/
│   └── versions/
│       └── 003_add_pgvector.py              ← NEW — migration: extension + tables + indexes
│
├── config/
│   ├── config.py                            ← MODIFY — remove chroma_db_path; expose rag params via hyperparams
│   └── hyperparams.yaml                     ← MODIFY — add rag_similarity_threshold: 0.75
│
├── services/
│   └── db_model.py                          ← MODIFY — add RagDocumentDB + RagAuditLogDB models
│
├── chat/
│   └── rag_engine.py                        ← REWRITE — replace ChromaDB calls with pgvector SQL queries
│
├── scripts/
│   └── populate_pgvector.py                 ← NEW — one-shot ingestion: embed + batch-insert rag_documents
│
├── tests/
│   ├── integration/
│   │   └── test_rag_pgvector.py             ← NEW — integration test (insert, query, assert, benchmark)
│   └── unit/
│       └── test_rag_engine.py               ← NEW/MODIFY — unit tests for retrieve_docs() with mocked DB
│
└── main.py                                  ← MODIFY — update /health endpoint (remove ChromaDB probe)

docker-compose.yml                           ← MODIFY — postgres image → pgvector/pgvector:pg16; remove chroma_data
docker-compose.prod.yml                      ← MODIFY — same changes as docker-compose.yml
docker-compose.override.yml                  ← MODIFY — remove chroma_data volume mount
pyproject.toml                               ← MODIFY — add pgvector>=0.3.0; remove chromadb>=0.5.0 (post-validation)
.env.example                                 ← MODIFY — remove CHROMA_DB_PATH; document rag params
```

---

## Implementation Phases

### Phase 1 — Infrastructure & Schema Foundation

**Goal**: Postgres can serve vector queries. No application code changed yet.

1. **Update Docker image** in `docker-compose.yml`, `docker-compose.prod.yml`, `docker-compose.override.yml`:
   - `postgres:16` → `pgvector/pgvector:pg16`
   - Remove `chroma_data` volume declaration
   - Remove `chroma_data:/app/chroma_db` mounts from `backend` and `worker` services

2. **Add `pgvector>=0.3.0`** to `pyproject.toml` dependencies via `uv add "pgvector>=0.3.0"`.

3. **Create Alembic migration `003_add_pgvector.py`**:
   - `revision = "003"`, `down_revision = "002"`
   - `upgrade()`: enable vector extension → create `rag_documents` → create `rag_audit_log` → create IVFFlat index → create category index
   - `downgrade()`: drop indexes → drop tables → (optionally) drop extension

4. **Validate migration**: `uv run alembic upgrade head` against local postgres, verify `\d rag_documents` shows `embedding vector(384)`.

---

### Phase 2 — ORM Models

**Goal**: SQLAlchemy can read/write both new tables.

5. **Extend `backend/services/db_model.py`** with:
   - `RagDocumentDB` class (all columns from data-model.md)
   - `RagAuditLogDB` class (all columns from data-model.md)
   - Import: `from pgvector.sqlalchemy import Vector`
   - Import: `from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB`
   - Both models use the shared `Base = declarative_base()` already in the file

---

### Phase 3 — Configuration Cleanup

**Goal**: Config reflects the new dependency surface.

6. **Modify `backend/config/config.py`**:
   - Remove `self.chroma_db_path` and its `os.getenv("CHROMA_DB_PATH", ...)` line
   - No new config attributes needed (RAG params read from `hyperparams.yaml` which already exists)

7. **Modify `backend/config/hyperparams.yaml`**:
   - Add `rag_similarity_threshold: 0.75` under the `inference` key

8. **Modify `.env.example`**:
   - Remove `CHROMA_DB_PATH=` line
   - Add comment block documenting that `rag_top_k` and `rag_similarity_threshold` are now in `hyperparams.yaml`

---

### Phase 4 — RAG Engine Rewrite

**Goal**: `rag_engine.py` is fully decoupled from ChromaDB.

9. **Rewrite `backend/chat/rag_engine.py`**:

   **Remove**:
   - `import chromadb`
   - `from chromadb.utils import embedding_functions`
   - `_chroma_client = chromadb.PersistentClient(...)`
   - `collection = _chroma_client.get_or_create_collection(...)`
   - All `collection.query(...)` calls

   **Add**:
   - `from sentence_transformers import SentenceTransformer`
   - `from sqlalchemy import text`
   - `from backend.services.db_utils import get_session` (or equivalent session factory)
   - `from backend.services.db_model import RagAuditLogDB`

   **New `retrieve_docs()` signature**:
   ```python
   def retrieve_docs(query: str, k: int | None = None) -> list[tuple[int, str, float]]:
       """Returns List[Tuple[doc_id, content, similarity_score]]"""
   ```

   **New flow**:
   1. Embed `query` with `_embedder.encode(query)` → 384-float list
   2. Execute parameterized SQL (see data-model.md retrieval query contract)
   3. Return `[(row.id, row.content, row.similarity), ...]`
   4. After results returned, insert one row into `rag_audit_log` (query_text, retrieved_ids, latency_ms)
   5. If no results above threshold, return `[]` and still write audit log row

   **Updated `get_rag_explanation()`**:
   - Calls updated `retrieve_docs()` → receives `List[Tuple[int, str, float]]`
   - Joins `content` strings as before
   - Passes `prediction`, `model_name`, `job_id` to audit log write
   - `@retry` decorator preserved on both public functions

   **Health check update in `backend/main.py`**:
   - Remove `chromadb_heartbeat()` call (or equivalent)
   - Add SQL probe: `SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'`

---

### Phase 5 — Ingestion Script

**Goal**: All existing documents loaded into `rag_documents` before go-live.

10. **Create `backend/scripts/populate_pgvector.py`**:
    - Scan `backend/data/` for YAML/JSON feature definition files (same source as `populate_chroma.py`)
    - For each document: compute 384-dim embedding via `SentenceTransformer("all-MiniLM-L6-v2")`
    - Batch-insert in groups of 1,000 using `Session.bulk_save_objects()` with `insert().on_conflict_do_update()` on `doc_id` for upsert semantics
    - Log batch success/failure counts (use `print()` per script allowance in `[P3-LOG]`)
    - Exit with non-zero code if any batch fails completely

---

### Phase 6 — Testing

**Goal**: SC-001 through SC-006 verified by automated tests.

11. **Create `backend/tests/integration/test_rag_pgvector.py`**:
    - Fixture: insert 10 documents (5 per category) into a clean `test_lersha` DB
    - `test_top3_similarity_above_threshold`: query, assert top-3 similarities > 0.75
    - `test_audit_log_populated`: assert 1 new audit log row after retrieval, with populated `retrieved_ids`
    - `test_latency_under_50ms`: assert round-trip retrieval + audit write < 50 ms
    - `test_empty_retrieval_no_crash`: query with nonsense string, assert `[]` returned and no exception
    - `test_alembic_migration_idempotent`: run `alembic upgrade head` twice, assert no error

12. **Create/update `backend/tests/unit/test_rag_engine.py`**:
    - `test_retrieve_docs_returns_list_of_tuples`: mock Session, assert return type `list[tuple[int, str, float]]`
    - `test_retrieve_docs_empty_when_no_results`: mock DB returning zero rows, assert `[]` (no crash)
    - `test_audit_log_write_called`: mock Session, assert `RagAuditLogDB` inserted exactly once per call

---

### Phase 7 — ChromaDB Cleanup (Post-Validation)

**Goal**: Remove all ChromaDB references after integration tests pass.

13. **Remove `chromadb>=0.5.0`** from `pyproject.toml` via `uv remove chromadb`.

14. **Delete or archive `backend/scripts/populate_chroma.py`** — move to `backup/` with deprecation note, not deleted (audit trail).

15. **Remove `chroma_db/` host directory** from `.gitignore` or document it as a legacy artifact directory.

16. **Final CI run**: confirm all lint, type check, test, and build jobs pass on the feature branch.

---

## Complexity Tracking

> No constitution violations to justify. All changes comply with existing principles.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| IVFFlat index requires rows to build | Medium | Build index after ingestion (migration creates table only; index added post-ingest or via separate migration step) |
| `psycopg2` vs `psycopg3` pgvector compatibility | Low | pgvector-python supports both; project uses `psycopg2>=2.9.11` (already in `pyproject.toml`) |
| Embedding model version drift (query vs ingest) | Low | Pinned via `sentence-transformers>=2.7.0`; model name locked in `config.embedder_model` |
| Similarity threshold too strict for existing corpus | Medium | Threshold starts at 0.75 in `hyperparams.yaml`; discoverable and adjustable without code change |
| Docker image change breaks existing data volume | Low | `pgvector/pgvector:pg16` is fully API-compatible with `postgres:16`; existing `postgres_data` volume is unaffected |
