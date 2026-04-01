# Research: Migrate Vector Store — ChromaDB → PostgreSQL pgvector

**Branch**: `006-migrate-chroma-pgvector`  
**Phase**: 0 — Pre-design research  
**Date**: 2026-04-01

---

## 1. pgvector Extension Capability

**Decision**: Use `pgvector>=0.3.0` with `pgvector.sqlalchemy.Vector` column type.

**Rationale**: pgvector ships a native SQLAlchemy dialect (`pgvector.sqlalchemy`) that exposes a `Vector(n)` column type, cosine-distance operator (`<=>`) and L2 operator (`<->`). This integrates cleanly with the project's existing SQLAlchemy ORM pattern in `backend/services/db_model.py`, adding zero new abstraction layers.

**Alternatives considered**:
- **LanceDB**: Better support for very large datasets but no SQL query interface; would break the parameterized-query requirement in `[P8-DB]`.
- **Weaviate / Qdrant**: Standalone external services; same availability concern as ChromaDB and adds infrastructure complexity.

---

## 2. Embedding Model & Vector Dimension

**Decision**: `sentence-transformers/all-MiniLM-L6-v2` → 384-dimensional `VECTOR(384)` columns.

**Rationale**: Already used in `rag_engine.py` via ChromaDB's `SentenceTransformerEmbeddingFunction`. Using the same model at ingestion time and query time guarantees embedding-space consistency. Swapping to a higher-dimension model (e.g. 768-dim) would require re-ingesting all documents and an Alembic migration to change the column type.

**Alternatives considered**:
- `all-mpnet-base-v2` (768-dim): Higher recall but 2× memory; overkill for the current document volume.
- OpenAI `text-embedding-3-small`: Requires external API dependency; unsuitable for air-gapped deployments.

---

## 3. ANN Index Strategy

**Decision**: `IVFFlat` index using cosine ops (`vector_cosine_ops`), `lists = 100`.

**Rationale**: IVFFlat is the simplest pgvector index type and is well-supported across pgvector versions ≥ 0.3.0. For a collection expected to stay in the tens-of-thousands range, `lists = 100` offers a good recall/latency trade-off (pgvector documentation recommends `sqrt(n_rows)` for up to ~1M rows). `HNSW` would give better recall but requires pgvector ≥ 0.5.0 and significantly more build time. `IVFFlat` is sufficient for the SC-002 target of <50 ms p95.

**Critical note**: IVFFlat requires at least `lists` rows present to build the index. The migration must insert all documents **before** creating the index, or use `CREATE INDEX IF NOT EXISTS` in a post-ingestion step (recommended approach in the migration script).

**Alternatives considered**:
- `HNSW`: Superior recall, but requires pgvector ≥ 0.5.0; version lock risk.
- No index (exact scan): Acceptable for <5,000 rows but violates SC-002 at scale.

---

## 4. pgvector-Enabled Postgres Image

**Decision**: Replace `postgres:16` in `docker-compose.yml` with `pgvector/pgvector:pg16`.

**Rationale**: The `pgvector/pgvector:pg16` image is the official, maintained image that bundles the pgvector extension alongside PostgreSQL 16. The alternative — installing pgvector from source inside the existing Dockerfile — is fragile and depends on build toolchain availability in the container. The official image is a single character change in `docker-compose.yml` and is maintained in step with pgvector releases.

**Impact on constitution**: `[P11-CONT]` — Docker image change must be reflected in all compose files (base, override, prod). No code path changes; purely infrastructure.

---

## 5. ORM Pattern for pgvector Column

**Decision**: Extend `backend/services/db_model.py` with `RagDocumentDB` and `RagAuditLogDB` models using `pgvector.sqlalchemy.Vector`.

**Rationale**: Both new tables follow the exact same ORM pattern as `CreditScoringRecordDB` and `InferenceJobDB` (same file, same `Base`). Keeping them in `db_model.py` satisfies `[P1-MODULAR]` (services layer owns all ORM definitions) without introducing new files. The `RagAuditLogDB.retrieved_ids` field maps to `ARRAY(Integer)` via `postgresql.ARRAY`.

**Key implementation note**: The `pgvector.sqlalchemy` module register the `Vector` type through a SQLAlchemy extension mechanism. At import time, `from pgvector.sqlalchemy import Vector` triggers the registration. No engine-level setup is required beyond having the extension loaded in Postgres.

---

## 6. Raw SQL Parameterisation for Retrieval

**Decision**: Use SQLAlchemy `text()` with `:param` named bindings, NOT f-string interpolation.

**Rationale**: `[P8-DB]` explicitly forbids f-string SQL construction. pgvector cosine distance is expressed as `embedding <=> :query_vec`. SQLAlchemy's `text("""...""").bindparams(query_vec=...)` is the correct parameterized form. The embedding must be cast to `::vector` in the SQL literal for psycopg2 compatibility.

**Pattern**:
```sql
SELECT id, content, 1 - (embedding <=> :query_vec) AS similarity
FROM rag_documents
WHERE category = ANY(:categories)
  AND 1 - (embedding <=> :query_vec) > :threshold
ORDER BY embedding <=> :query_vec
LIMIT :top_k
```

---

## 7. Config Changes — RAG Parameters

**Decision**: Add `rag_top_k` (default 5) and `rag_similarity_threshold` (default 0.75) to `hyperparams.yaml` under the `inference` key. Remove `chroma_db_path` / `CHROMA_DB_PATH` from `config.py`.

**Rationale**: `rag_top_k` already exists in `hyperparams.yaml` (`inference.rag_top_k: 5`). Only `rag_similarity_threshold` is new. `chroma_db_path` must be removed per FR-006, and the corresponding `CHROMA_DB_PATH` env var must be removed from `.env.example`. Per `[P5-CONFIG]`, all tuning knobs live in `hyperparams.yaml`.

---

## 8. Health Check Update

**Decision**: In `GET /health`, replace the ChromaDB heartbeat probe with a `pg_catalog.pg_extension` existence check for the `vector` extension.

**Rationale**: Constitution Principle 10 (`[P10-OBS]`) requires the health endpoint to check real dependencies. After migration, ChromaDB is no longer a dependency. The check becomes:
```sql
SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'
```
This validates both PostgreSQL connectivity and the vector extension presence in one query.

---

## 9. Ingestion Script Pattern

**Decision**: Model `populate_pgvector.py` after the existing `populate_chroma.py` (same `backend/scripts/` location, same `print()`-for-operator-feedback allowance per `[P3-LOG]`). Read source from `backend/data/` directory YAML/JSON files.

**Rationale**: The existing `populate_chroma.py` (4,518 bytes) shows the established pattern: open data files from `backend/data/`, embed with sentence-transformers, batch-insert, log counts. The new script follows the same structure but writes to PostgreSQL via SQLAlchemy rather than to ChromaDB. Using SQLAlchemy `Session.bulk_save_objects()` or `insert().on_conflict_do_update()` for upsert semantics satisfies the duplicate-handling edge case.

---

## 10. Dependency Cleanup Timeline

**Decision**: `chromadb>=0.5.0` is removed from `pyproject.toml` in a **separate final task** after integration validation passes, not upfront.

**Rationale**: Running both stores in parallel during the migration task allows for rollback without re-ingesting. Once the integration test suite (SC-001 through SC-007) all pass, the chromadb import and dependency are removed. This matches the two-phase approach documented in the spec Assumptions section.

---

## Summary — All NEEDS CLARIFICATION Resolved

| Question | Decision |
|---------|---------|
| pgvector version | `>=0.3.0` (IVFFlat available, widely deployed) |
| Vector dimension | 384 (all-MiniLM-L6-v2, already deployed) |
| Index type | IVFFlat, lists=100, cosine ops |
| Postgres image | `pgvector/pgvector:pg16` (official managed image) |
| RAG SQL pattern | SQLAlchemy `text()` with `:param` binds |
| Config location | `hyperparams.yaml` inference section |
| ChromaDB removal timing | Post-validation (Phase 2 cleanup task) |
| Health check update | pg_extension check replaces ChromaDB heartbeat |
