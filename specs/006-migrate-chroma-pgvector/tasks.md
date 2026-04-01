# Tasks: Migrate Vector Store from ChromaDB to PostgreSQL pgvector

**Input**: Design documents from `/specs/006-migrate-chroma-pgvector/`  
**Prerequisites**: plan.md ✅ | spec.md ✅ | research.md ✅ | data-model.md ✅ | quickstart.md ✅  
**Branch**: `006-migrate-chroma-pgvector`

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies on each other)
- **[Story]**: Which user story this task belongs to (US1–US4)
- Exact file paths included in every task description

---

## Phase 1: Setup — Infrastructure Foundation

**Purpose**: Switch the Postgres image to one that supports vector operations and add the Python library. No application code changes yet.

**⚠️ CRITICAL**: These tasks unblock everything. Validate Docker compose starts cleanly before proceeding.

- [ ] T001 Add `pgvector>=0.3.0` to `pyproject.toml` dependencies (under the `dependencies` list, after `psycopg2`) and run `uv add "pgvector>=0.3.0"` to update `uv.lock`
- [ ] T002 [P] In `docker-compose.yml`: change `postgres:16` → `pgvector/pgvector:pg16`; remove the `chroma_data` named volume declaration from the `volumes:` section; remove `- chroma_data:/app/chroma_db` volume mount from both the `backend:` and `worker:` service definitions
- [ ] T003 [P] In `docker-compose.prod.yml`: apply the same three changes as T002 (pgvector image, remove chroma_data volume declaration, remove chroma_data mounts from backend and worker)
- [ ] T004 [P] In `docker-compose.override.yml`: remove any `chroma_data` volume mount or declaration if present; verify no remaining ChromaDB references

**Checkpoint**: Run `docker compose up -d postgres` — container must start healthy with `pgvector/pgvector:pg16` image before any further work.

---

## Phase 2: Foundational — Schema & ORM

**Purpose**: The database can store and query vector embeddings; SQLAlchemy ORM models exist for both new tables. These are hard prerequisites for all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T005 Create `backend/alembic/versions/003_add_pgvector.py` — Alembic migration with `revision="003"`, `down_revision="002"`. The `upgrade()` function must execute in this exact order: (1) `op.execute("CREATE EXTENSION IF NOT EXISTS vector")`, (2) `op.create_table("rag_documents", ...)` with all columns from `data-model.md` — `id SERIAL PK`, `doc_id UUID UNIQUE DEFAULT gen_random_uuid()`, `category VARCHAR(100) NOT NULL`, `title VARCHAR(255) NOT NULL`, `content TEXT NOT NULL`, `embedding VECTOR(384) NOT NULL`, `metadata JSONB DEFAULT '{}'`, `created_at TIMESTAMPTZ DEFAULT NOW()`, `updated_at TIMESTAMPTZ DEFAULT NOW()`, (3) `op.create_table("rag_audit_log", ...)` with columns: `id SERIAL PK`, `query_text TEXT NOT NULL`, `retrieved_ids INTEGER[] NULLABLE`, `prediction VARCHAR(100) NULLABLE`, `model_name VARCHAR(100) NULLABLE`, `job_id UUID NULLABLE`, `generated_text TEXT NULLABLE`, `latency_ms INTEGER NULLABLE`, `created_at TIMESTAMPTZ DEFAULT NOW()`, (4) `op.execute("CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding ON rag_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")`, (5) `op.execute("CREATE INDEX IF NOT EXISTS idx_rag_documents_category ON rag_documents (category)")`. The `downgrade()` must reverse those five steps in order.
- [ ] T006 Extend `backend/services/db_model.py` — add two new ORM classes after the existing `InferenceJobDB` class. Add imports: `from pgvector.sqlalchemy import Vector` and `from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY, JSONB`. Class `RagDocumentDB(Base)` maps `__tablename__ = "rag_documents"` with all nine columns. Class `RagAuditLogDB(Base)` maps `__tablename__ = "rag_audit_log"` with all nine columns. Both classes must have PEP 257 docstrings and full type annotations on all column definitions.
- [ ] T007 [P] Update `backend/config/config.py` — remove the `self.chroma_db_path` attribute and its `os.getenv("CHROMA_DB_PATH", ...)` assignment (lines ~134–135). No other config changes are needed; RAG tuning params are read from `hyperparams.yaml`.
- [ ] T008 [P] Update `backend/config/hyperparams.yaml` — add `rag_similarity_threshold: 0.75` as a new key under the `inference:` section, directly below `rag_top_k: 5`.
- [ ] T009 [P] Update `.env.example` — remove the `CHROMA_DB_PATH=` line and its associated comment. Add a new comment block in the `# ── LLM / Embeddings ──` section noting that `rag_top_k` and `rag_similarity_threshold` are configured in `backend/config/hyperparams.yaml` rather than as environment variables.

**Checkpoint**: Run `uv run alembic upgrade head` against a local postgres container. Then verify with `docker compose exec postgres psql -U lersha -d lersha -c "\d rag_documents"` — the output must show an `embedding` column of type `vector(384)`.

---

## Phase 3: User Story 1 — Uninterrupted AI-Powered Credit Assessments (Priority: P1) 🎯 MVP

**Goal**: The RAG engine retrieves relevant documents from PostgreSQL and returns grounded credit explanations — with no ChromaDB dependency remaining in the query path.

**Independent Test**: Submit a credit assessment query (prediction + SHAP dict) to `get_rag_explanation()` and verify a non-empty explanation string is returned. Confirm no `chromadb` import is triggered. Run `uv run pytest backend/tests/unit/test_rag_engine.py -v` to confirm all unit tests pass.

### Implementation for User Story 1

- [ ] T010 [US1] Rewrite `backend/chat/rag_engine.py` — remove all ChromaDB imports and client setup: delete `import chromadb`, `from chromadb.utils import embedding_functions`, `_chroma_client = chromadb.PersistentClient(...)`, and `collection = _chroma_client.get_or_create_collection(...)`. Add new imports: `from sentence_transformers import SentenceTransformer`, `from sqlalchemy import text`, `from sqlalchemy.orm import Session`, `from backend.services.db_utils import get_db_session` (use the project's existing session factory pattern). Module-level: `_embedder = SentenceTransformer(config.embedder_model)`. Keep `gemini_client` and all `@retry` decorators unchanged.
- [ ] T011 [US1] In `backend/chat/rag_engine.py` — implement the new `retrieve_docs(query: str, k: int | None = None) -> list[tuple[int, str, float]]` function. It must: (1) read `k` from `config.hyperparams["inference"]["rag_top_k"]` if None, (2) read `threshold` from `config.hyperparams["inference"]["rag_similarity_threshold"]`, (3) encode `query` with `_embedder.encode(query).tolist()` → 384-float list, (4) execute the parameterized SQL query from `data-model.md` using `sqlalchemy.text()` with `:query_vec`, `:categories`, `:threshold`, `:top_k` bound parameters — `categories` must be `['feature_definition', 'policy_rule']`, (5) return `[(row.id, row.content, row.similarity), ...]`. Cast query vector to `::vector` in the SQL literal. Use `with session.execute(stmt, {...}) as result`.
- [ ] T012 [US1] In `backend/chat/rag_engine.py` — implement the audit log write within `retrieve_docs()`. After the SELECT result is collected (but before returning), open a new session write, instantiate `RagAuditLogDB(query_text=query, retrieved_ids=[r[0] for r in results], latency_ms=elapsed_ms)`, and call `session.add()` + `session.commit()`. Wrap in `try/except SQLAlchemyError` — log at ERROR with `exc_info=True` and re-raise so it propagates to the job error boundary. Emit `logger.debug("RAG retrieval latency: %d ms, %d docs returned", elapsed_ms, len(results))`.
- [ ] T013 [US1] In `backend/chat/rag_engine.py` — update `get_rag_explanation(prediction: str, shap_dict: dict) -> str` to consume the new `retrieve_docs()` return type `list[tuple[int, str, float]]`. Change the context assembly from `"\n".join(retrieved_docs)` to `"\n".join(doc for _, doc, _ in retrieved_docs)`. Pass `prediction=prediction` to the audit log write in T012. Keep all YAML prompt loading, `_call_gemini()` delegation, and `@retry` decorators exactly as they are. Add `model_name` parameter (optional, `str | None = None`) and forward it to the `RagAuditLogDB` write.
- [ ] T014 [US1] Create `backend/tests/unit/test_rag_engine.py` — write three unit tests using `pytest-mock` to mock the SQLAlchemy session (do not connect to a real DB): `test_retrieve_docs_returns_list_of_tuples` (mock execute returns two rows, assert return type is `list[tuple[int, str, float]]`), `test_retrieve_docs_empty_when_no_results` (mock returns zero rows, assert `[]` returned with no exception raised), `test_audit_log_write_called_once` (mock session, call `retrieve_docs()`, assert `RagAuditLogDB` was instantiated exactly once and `session.add()` was called). All tests must use `from unittest.mock import MagicMock, patch` and must NOT import `chromadb`.

**Checkpoint**: `uv run pytest backend/tests/unit/test_rag_engine.py -v` — all 3 tests pass. `python -c "from backend.chat.rag_engine import retrieve_docs; print('OK')"` — no chromadb ImportError.

---

## Phase 4: User Story 2 — Operators Can Ingest & Index Knowledge Documents (Priority: P2)

**Goal**: A one-shot ingestion script loads all documents from `backend/data/` into `rag_documents` with correct embeddings. Duplicates are handled via upsert; batch failures are logged.

**Independent Test**: Run `uv run python -m backend.scripts.populate_pgvector` against a database that already has the migration applied. Verify documents appear in `rag_documents` with `SELECT count(*) FROM rag_documents`, and that re-running the script produces no duplicate rows (upsert idempotency).

### Implementation for User Story 2

- [ ] T015 [US2] Create `backend/scripts/populate_pgvector.py` — top-level script (not imported by the app). Structure: (1) imports: `SentenceTransformer`, `sqlalchemy` session factory, `RagDocumentDB`, `Path`, `yaml`/`json`, `sys`; (2) load all `.yaml` and `.json` files from `backend/data/` that contain feature definitions (use the same source files as the existing `populate_chroma.py` for source consistency); (3) embed each document's content using `SentenceTransformer("all-MiniLM-L6-v2").encode(content).tolist()`; (4) batch in groups of 1,000 using `session.execute(insert(RagDocumentDB).values([...]).on_conflict_do_update(index_elements=["doc_id"], set_={"content": ..., "embedding": ..., "updated_at": ...}))` for upsert semantics; (5) print per-batch success/failure counts to stdout (script `print()` allowance per `[P3-LOG]`); (6) call `sys.exit(1)` if any batch raises an unhandled exception.
- [ ] T016 [US2] In `backend/scripts/populate_pgvector.py` — add complete error handling: wrap each batch insert in `try/except SQLAlchemyError as e:` → `print(f"[ERROR] Batch {batch_num} failed: {e}", file=sys.stderr)`; keep a running `failures` counter; after all batches, print summary `[OK] Ingested {success} documents in {n} batch(es). Failures: {failures}.`; exit 0 on clean run, exit 1 if `failures > 0`. Ensure model loading (SentenceTransformer) happens once at script start outside any loop to avoid repeated weight loading.
- [ ] T017 [P] [US2] Archive `backend/scripts/populate_chroma.py` — move the file to `backup/scripts/populate_chroma.py` (create the `backup/scripts/` directory if it doesn't exist) and add a deprecation header comment at line 1: `# DEPRECATED: Replaced by populate_pgvector.py as of 006-migrate-chroma-pgvector (2026-04-01).` Do not delete it — it serves as an audit trail.

**Checkpoint**: Run `uv run python -m backend.scripts.populate_pgvector`. Print output must show `[OK] Ingested N documents`. Re-run — count must not increase (upsert). Query `SELECT count(*), category FROM rag_documents GROUP BY category` to confirm both categories present.

---

## Phase 5: User Story 3 — Retrieval Activity Is Auditable (Priority: P3)

**Goal**: Every RAG retrieval call produces a row in `rag_audit_log` with all required fields populated. The audit log is queryable by timestamp and job ID.

**Independent Test**: Call `retrieve_docs()` once with a known query string, then query `SELECT * FROM rag_audit_log ORDER BY id DESC LIMIT 1` and assert that `query_text` matches, `retrieved_ids` is a non-null array, and `latency_ms` is a positive integer. Verify the log row is present even when retrieval returns zero results (below-threshold case).

### Implementation for User Story 3

- [ ] T018 [US3] Update `get_rag_explanation()` in `backend/chat/rag_engine.py` — add `job_id: str | None = None` parameter to the function signature. Forward `job_id` to the `RagAuditLogDB` write call inside `retrieve_docs()` or via a separate audit update after the explanation is generated. Also add `generated_text=explanation` to the `RagAuditLogDB` write so the complete audit record is populated (prediction, model_name, job_id, generated_text). Update all callers of `get_rag_explanation()` in `backend/core/` or `backend/worker.py` to pass ``job_id`` where available.
- [ ] T019 [US3] Create `backend/tests/integration/test_rag_pgvector.py` — write the full integration test suite using a `test_lersha` PostgreSQL database (per constitution `[P7-TEST]`). Required tests: (1) `test_top3_similarity_above_threshold` — insert 10 docs (5 per category), query with a similar string, assert all top-3 similarity scores > 0.75; (2) `test_audit_log_populated_after_retrieval` — call `retrieve_docs()`, assert exactly 1 new row in `rag_audit_log`, assert `retrieved_ids` is a non-empty integer list; (3) `test_audit_log_populated_on_empty_retrieval` — query with a nonsense string that returns no results, assert audit log row still created with `retrieved_ids = []`; (4) `test_latency_under_50ms` — measure round-trip retrieval time, assert < 50 ms. Use `conftest.py` fixtures for DB setup/teardown. Mock `GeminiClient.models.generate_content` using `pytest-mock` in any test that calls `get_rag_explanation()`.
- [ ] T020 [US3] Add migration idempotency test to `backend/tests/integration/test_rag_pgvector.py` — `test_alembic_upgrade_head_idempotent`: programmatically invoke `alembic upgrade head` twice via `alembic.command.upgrade(alembic_cfg, "head")` and assert no exception is raised on either call. Use the `test_lersha` DB connection from the shared conftest fixture.

**Checkpoint**: `uv run pytest backend/tests/integration/test_rag_pgvector.py -v` — all 5 tests pass. Verify `rag_audit_log` table has rows with populated `retrieved_ids` and `latency_ms`.

---

## Phase 6: User Story 4 — Database Schema Is Version-Controlled and Repeatable (Priority: P4)

**Goal**: The schema migration (003) applies cleanly on a fresh database and rolls back cleanly. Every developer environment reaches the correct state with `alembic upgrade head`.

**Independent Test**: Run `uv run alembic downgrade 002` followed by `uv run alembic upgrade head` on a running PostgreSQL container. Assert the migration completes without errors and `\d rag_documents` shows the correct column set including `embedding vector(384)`.

### Implementation for User Story 4

- [ ] T021 [US4] Verify reversibility of `backend/alembic/versions/003_add_pgvector.py` — run the full down/up cycle: `uv run alembic downgrade 002` (assert `rag_documents` and `rag_audit_log` tables are dropped, `vector` extension may optionally be retained), then `uv run alembic upgrade head` (assert both tables and all indexes are recreated correctly). Document any caveats about extension removal in a comment inside the `downgrade()` function.
- [ ] T022 [US4] Update `backend/main.py` — locate the `GET /health` endpoint handler. Remove any call to a ChromaDB heartbeat or `chromadb` import. Add a pgvector extension check using a raw SQL probe via SQLAlchemy: `session.execute(text("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'"))`. If the query returns no row, include `"pgvector": "unavailable"` in the `503` dependency breakdown response. If it succeeds, include `"pgvector": "ok"`. This satisfies `[P10-OBS]` for the new dependency surface.
- [ ] T023 [P] [US4] Update `README.md` or `docs/` — add a "pgvector Migration" section documenting: (1) the requirement for `pgvector/pgvector:pg16` Docker image, (2) how to run `alembic upgrade head` for the new migration, (3) how to run the ingestion script, (4) reference to `specs/006-migrate-chroma-pgvector/quickstart.md` for the full developer guide.

**Checkpoint**: On a completely fresh database, run `alembic upgrade head` — must complete in a single command with no manual steps. Run `alembic downgrade 002` — tables must be cleanly removed.

---

## Phase 7: Polish & ChromaDB Cleanup (Post-Validation)

**Purpose**: Remove all ChromaDB dependencies and references only after all phases above validate cleanly.

**⚠️ GATE**: Only proceed after `uv run pytest backend/tests/ --cov=backend --cov-fail-under=80` passes in full.

- [ ] T024 Remove `chromadb>=0.5.0` from `pyproject.toml` dependencies list. Run `uv remove chromadb` to update `uv.lock`. Verify `uv run python -c "from backend.chat.rag_engine import get_rag_explanation; print('OK')"` still passes (no chromadb import).
- [ ] T025 [P] Search the entire codebase for any remaining `chromadb` string references: `grep -r "chromadb" backend/ --include="*.py"`. Resolve any remaining imports or references in non-script files. Update `backend/tests/conftest.py` if it contains ChromaDB fixtures.
- [ ] T026 [P] Update `.gitignore` — remove or comment out the `chroma_db/` directory entry (or annotate it as a legacy artifact location). Remove `CHROMA_DB_PATH` from any remaining `.env` or `.env.example` references not caught by T009.
- [ ] T027 Run the complete validation suite per `specs/006-migrate-chroma-pgvector/quickstart.md`: (1) `docker compose up -d postgres redis mlflow`, (2) `uv run alembic upgrade head`, (3) `uv run python -m backend.scripts.populate_pgvector`, (4) `uv run pytest backend/tests/ --cov=backend --cov-fail-under=80`, (5) latency smoke test from `quickstart.md` Step 6. All five steps must complete without error.
- [ ] T028 [P] Run the full CI quality gate locally: `uv run ruff check backend/ ui/`, `uv run ruff format --check backend/ ui/`, `uv run mypy backend/`. Resolve any lint, format, or type errors introduced by this feature branch before opening a PR.

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)       → No dependencies — start immediately
       ↓
Phase 2 (Foundation)  → Depends on Phase 1 completion — BLOCKS all user stories
       ↓
Phase 3 (US1 — RAG Engine)   ┐
Phase 4 (US2 — Ingestion)    ├── All depend on Phase 2; can run in parallel
Phase 5 (US3 — Audit Log)    │   (US3 depends on US1's retrieve_docs() signature)
Phase 6 (US4 — Schema CI)    ┘
       ↓
Phase 7 (Cleanup)     → Depends on ALL prior phases + full test suite passing
```

### User Story Dependencies

| Story | Depends On | Can Start After |
|-------|-----------|-----------------|
| US1 (RAG Engine) | Phase 2 (T005, T006, T007, T008) | Phase 2 checkpoint |
| US2 (Ingestion) | Phase 2 (T005, T006) | Phase 2 checkpoint |
| US3 (Audit Log) | US1 `retrieve_docs()` signature (T011, T012) | T012 complete |
| US4 (Schema CI) | Phase 2 (T005 migration file) | T005 complete |

### Within Each Phase

- T010 → T011 → T012 → T013 (sequential — same file `rag_engine.py`)
- T015 → T016 (sequential — same file `populate_pgvector.py`)
- T017 (parallel with T015/T016 — different file)
- T018 → T019 → T020 (sequential — depends on T012 audit logic)
- T021 → T022 (T022 depends on T005 migration being validated by T021)

---

## Parallel Opportunities

### Setup Phase (Phase 1)

```
T001 (pyproject.toml)
T002 (docker-compose.yml)     ← can run simultaneously
T003 (docker-compose.prod.yml) ← can run simultaneously
T004 (docker-compose.override.yml) ← can run simultaneously
```

### Foundational Phase (Phase 2)

```
T005 (migration file)         ← must complete first
T006 (db_model.py)            ← must complete first
        ↓
T007 (config.py)   ┐
T008 (hyperparams) ├── can run in parallel after T005+T006
T009 (.env.example)┘
```

### Once Foundation is Complete

```
Developer A: Phase 3 (US1) — T010 → T011 → T012 → T013 → T014
Developer B: Phase 4 (US2) — T015 → T016, T017 in parallel
Developer C: Phase 6 (US4) — T021 → T022, T023 in parallel
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T004)
2. Complete Phase 2: Foundational (T005–T009) — **CRITICAL GATE**
3. Complete Phase 3: US1 RAG Engine (T010–T014)
4. **STOP and VALIDATE**: Run unit tests, confirm no chromadb import in query path
5. Demo: call `get_rag_explanation()` end-to-end with pgvector

### Incremental Delivery

1. Setup + Foundation → database serves vector queries
2. Add US1 → RAG engine reads from PostgreSQL → validate → **first working increment**
3. Add US2 → documents ingested via script → validate → **knowledge base populated**
4. Add US3 → audit log populated → validate → **compliance-ready**
5. Add US4 → migration CI validated → **production-safe**
6. Phase 7 cleanup → chromadb removed → **fully migrated**

### Suggested Task Ordering for Solo Developer

```
T001 → T002 → T003 → T004           # Phase 1 (~30 min)
→ T005 → T006 → T007 → T008 → T009 # Phase 2 (~60 min)
→ T010 → T011 → T012 → T013 → T014 # Phase 3 US1 (~90 min)
→ T019 → T020                       # Phase 5 (integration tests, written now)
→ T015 → T016 → T017                # Phase 4 US2 (~45 min)
→ T018 → T021 → T022 → T023        # Phase 5+6 audit + schema CI (~30 min)
→ T024 → T025 → T026 → T027 → T028 # Phase 7 cleanup (~30 min)
```

---

## Notes

- **[P]** tasks operate on different files and have no shared in-progress dependencies
- **[Story]** labels map each task to the user story it directly satisfies (for traceability back to spec.md)
- Run `uv run alembic upgrade head` before starting any US1/US2 implementation
- Run `uv run python -m backend.scripts.populate_pgvector` before running integration tests (data must exist)
- IVFFlat index requires minimum `lists` (100) rows to be effective — ensure ingestion runs before latency benchmark
- Commit after each phase checkpoint, not after individual tasks
- The `chromadb` package must remain in `pyproject.toml` until T024 — do not remove it earlier as `populate_chroma.py` may still be referenced
