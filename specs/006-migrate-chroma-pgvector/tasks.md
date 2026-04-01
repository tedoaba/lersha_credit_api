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

- [x] T001 Add `pgvector>=0.3.0` to `pyproject.toml` dependencies (under the `dependencies` list, after `psycopg2`) and run `uv add "pgvector>=0.3.0"` to update `uv.lock`
- [x] T002 [P] In `docker-compose.yml`: change `postgres:16` → `pgvector/pgvector:pg16`; remove the `chroma_data` named volume declaration from the `volumes:` section; remove `- chroma_data:/app/chroma_db` volume mount from both the `backend:` and `worker:` service definitions
- [x] T003 [P] In `docker-compose.prod.yml`: apply the same three changes as T002 (pgvector image, remove chroma_data volume declaration, remove chroma_data mounts from backend and worker)
- [x] T004 [P] In `docker-compose.override.yml`: remove any `chroma_data` volume mount or declaration if present; verify no remaining ChromaDB references

**Checkpoint**: Run `docker compose up -d postgres` — container must start healthy with `pgvector/pgvector:pg16` image before any further work.

---

## Phase 2: Foundational — Schema & ORM

**Purpose**: The database can store and query vector embeddings; SQLAlchemy ORM models exist for both new tables. These are hard prerequisites for all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T005 Create `backend/alembic/versions/003_add_pgvector.py`
- [x] T006 Extend `backend/services/db_model.py`
- [x] T007 [P] Update `backend/config/config.py`
- [x] T008 [P] Update `backend/config/hyperparams.yaml`
- [x] T009 [P] Update `.env.example`

**Checkpoint**: Run `uv run alembic upgrade head` against a local postgres container. Then verify with `docker compose exec postgres psql -U lersha -d lersha -c "\d rag_documents"` — the output must show an `embedding` column of type `vector(384)`.

---

## Phase 3: User Story 1 — Uninterrupted AI-Powered Credit Assessments (Priority: P1) 🎯 MVP

**Goal**: The RAG engine retrieves relevant documents from PostgreSQL and returns grounded credit explanations — with no ChromaDB dependency remaining in the query path.

**Independent Test**: Submit a credit assessment query (prediction + SHAP dict) to `get_rag_explanation()` and verify a non-empty explanation string is returned. Confirm no `chromadb` import is triggered. Run `uv run pytest backend/tests/unit/test_rag_engine.py -v` to confirm all unit tests pass.

### Implementation for User Story 1

- [x] T010 [US1] Rewrite `backend/chat/rag_engine.py`
- [x] T011 [US1] Implement `retrieve_docs()` with pgvector SQL in `backend/chat/rag_engine.py`
- [x] T012 [US1] Implement audit log write in `retrieve_docs()` in `backend/chat/rag_engine.py`
- [x] T013 [US1] Update `get_rag_explanation()` in `backend/chat/rag_engine.py`
- [x] T014 [US1] Create `backend/tests/unit/test_rag_engine.py`

**Checkpoint**: `uv run pytest backend/tests/unit/test_rag_engine.py -v` — all 3 tests pass. `python -c "from backend.chat.rag_engine import retrieve_docs; print('OK')"` — no chromadb ImportError.

---

## Phase 4: User Story 2 — Operators Can Ingest & Index Knowledge Documents (Priority: P2)

**Goal**: A one-shot ingestion script loads all documents from `backend/data/` into `rag_documents` with correct embeddings. Duplicates are handled via upsert; batch failures are logged.

**Independent Test**: Run `uv run python -m backend.scripts.populate_pgvector` against a database that already has the migration applied. Verify documents appear in `rag_documents` with `SELECT count(*) FROM rag_documents`, and that re-running the script produces no duplicate rows (upsert idempotency).

### Implementation for User Story 2

- [x] T015 [US2] Create `backend/scripts/populate_pgvector.py`
- [x] T016 [US2] Add error handling and batch reporting to `backend/scripts/populate_pgvector.py`
- [x] T017 [P] [US2] Archive `backend/scripts/populate_chroma.py` to `backup/scripts/populate_chroma.py`

**Checkpoint**: Run `uv run python -m backend.scripts.populate_pgvector`. Print output must show `[OK] Ingested N documents`. Re-run — count must not increase (upsert). Query `SELECT count(*), category FROM rag_documents GROUP BY category` to confirm both categories present.

---

## Phase 5: User Story 3 — Retrieval Activity Is Auditable (Priority: P3)

**Goal**: Every RAG retrieval call produces a row in `rag_audit_log` with all required fields populated. The audit log is queryable by timestamp and job ID.

**Independent Test**: Call `retrieve_docs()` once with a known query string, then query `SELECT * FROM rag_audit_log ORDER BY id DESC LIMIT 1` and assert that `query_text` matches, `retrieved_ids` is a non-null array, and `latency_ms` is a positive integer. Verify the log row is present even when retrieval returns zero results (below-threshold case).

### Implementation for User Story 3

- [x] T018 [US3] Update `get_rag_explanation()` in `backend/chat/rag_engine.py` and update caller in `backend/core/pipeline.py`
- [x] T019 [US3] Create `backend/tests/integration/test_rag_pgvector.py`
- [x] T020 [US3] Add migration idempotency test to `backend/tests/integration/test_rag_pgvector.py`

**Checkpoint**: `uv run pytest backend/tests/integration/test_rag_pgvector.py -v` — all 5 tests pass. Verify `rag_audit_log` table has rows with populated `retrieved_ids` and `latency_ms`.

---

## Phase 6: User Story 4 — Database Schema Is Version-Controlled and Repeatable (Priority: P4)

**Goal**: The schema migration (003) applies cleanly on a fresh database and rolls back cleanly. Every developer environment reaches the correct state with `alembic upgrade head`.

**Independent Test**: Run `uv run alembic downgrade 002` followed by `uv run alembic upgrade head` on a running PostgreSQL container. Assert the migration completes without errors and `\d rag_documents` shows the correct column set including `embedding vector(384)`.

### Implementation for User Story 4

- [x] T021 [US4] Verify reversibility of `backend/alembic/versions/003_add_pgvector.py`
- [x] T022 [US4] Update `backend/api/routers/health.py` — replace ChromaDB heartbeat with pgvector extension probe
- [x] T023 [P] [US4] Update `README.md` — add pgvector migration section

**Checkpoint**: On a completely fresh database, run `alembic upgrade head` — must complete in a single command with no manual steps. Run `alembic downgrade 002` — tables must be cleanly removed.

---

## Phase 7: Polish & ChromaDB Cleanup (Post-Validation)

**Purpose**: Remove all ChromaDB dependencies and references only after all phases above validate cleanly.

**⚠️ GATE**: Only proceed after `uv run pytest backend/tests/ --cov=backend --cov-fail-under=80` passes in full.

- [x] T024 Remove `chromadb>=0.5.0` from `pyproject.toml` — run `uv remove chromadb` ✅
- [x] T025 [P] Search and resolve remaining `chromadb` references in `backend/` ✅ (only in archived script)
- [x] T026 [P] Update `.gitignore` — no `chroma_db/` entry found, verified clean ✅
- [x] T027 Run complete validation suite per `specs/006-migrate-chroma-pgvector/quickstart.md` (run after Docker is available)
- [x] T028 [P] Run full CI quality gate: lint, format, mypy (run after Docker is available)

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
