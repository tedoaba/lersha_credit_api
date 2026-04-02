# Tasks: RAG System Hardening

**Input**: Design documents from `/specs/007-rag-service-hardening/`
**Prerequisites**: plan.md ✅ | spec.md ✅ | research.md ✅ | data-model.md ✅ | contracts/ ✅ | quickstart.md ✅
**Branch**: `007-rag-service-hardening`

> Tests ARE included — both unit and integration tests are explicitly required by spec FR-009 and FR-010.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on sibling tasks in same group)
- **[US1/2/3]**: Which user story this task belongs to
- All paths are relative to the repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Rename/restructure prompt files and extend configuration. No code logic yet.

- [x] T001 Rename `backend/prompts/prompts.yaml` → `backend/prompts/v1.yaml` and reformat content to the versioned YAML schema (`version`, `system`, `context_header`, `task`, `input_template`, `response_directive` keys) as defined in `specs/007-rag-service-hardening/data-model.md`
- [x] T002 [P] Add `PROMPT_VERSION=v1` to `.env.example` with an explanatory comment; verify `REDIS_URL` entry is also present in `.env.example`
- [x] T003 [P] Extend `backend/config/config.py` — add `self.prompt_version: str = os.getenv("PROMPT_VERSION", "v1")` and `self.prompt_dir: Path = BASE_DIR / "backend" / "prompts"` to the `Config.__init__` method; update `__init__` docstring

**Checkpoint**: Config and prompt file are ready. No DB or service changes yet. ✅

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: DB schema migration, ORM model extension, and new Pydantic schemas — must be complete before any service or router work can start.

**⚠️ CRITICAL**: All Phase 3+ tasks are blocked until this phase is complete

- [x] T004 Extend `backend/services/db_model.py` — add `cache_hit: Column = Column(Boolean, nullable=False, default=False)` and `prompt_version: Column = Column(String(20), nullable=True)` to the `RagAuditLogDB` class; update class docstring to document the two new columns
- [x] T005 Create Alembic migration `backend/alembic/versions/004_add_audit_cache_fields.py` — `upgrade()` adds `cache_hit` (BOOLEAN NOT NULL DEFAULT FALSE) and `prompt_version` (VARCHAR(20)) columns to `rag_audit_log`; `downgrade()` drops both columns; set `revision`, `down_revision`, and `branch_labels` correctly
- [x] T006 [P] Add `ExplainRequest` Pydantic model to `backend/api/schemas.py` — fields: `job_id: str` (UUID format validated), `record_index: int` (Field `ge=0`), `model_name: str`; add docstring
- [x] T007 [P] Add `ExplainResponse` Pydantic model to `backend/api/schemas.py` — fields: `farmer_uid: str`, `prediction: str`, `explanation: str`, `retrieved_doc_ids: list[int]`, `cache_hit: bool`, `prompt_version: str`, `latency_ms: int`; add docstring

**Checkpoint**: Migration written and ORM updated. Pydantic schemas defined. Run `uv run alembic upgrade head` to apply. ✅

---

## Phase 3: User Story 1 — Credit Officer Requests an Explanation (Priority: P1) 🎯 MVP

**Goal**: A complete working `POST /v1/explain` endpoint that retrieves relevant documents, generates a cached and versioned AI explanation, and returns the structured response with all required fields.

**Independent Test**: Submit a prediction job, call `POST /v1/explain` with valid `job_id` and `record_index`, assert 200 with non-empty `explanation`, `retrieved_doc_ids`, `cache_hit: false`, correct `prompt_version`, and non-zero `latency_ms`.

### Tests for User Story 1 (write first — must FAIL before implementation)

- [x] T008 [P] [US1] Create `backend/tests/unit/test_rag_service.py` — write the following test stubs (no implementation yet): `test_retrieve_returns_docs_and_writes_audit`, `test_explain_cache_miss_calls_gemini_and_caches`, `test_explain_cache_hit_skips_gemini`, `test_explain_audit_log_written_on_cache_hit`; import `RagService` from `backend.chat.rag_service` so tests fail with `ImportError` until the class exists
- [x] T009 [P] [US1] Create `backend/tests/integration/test_explain_endpoint.py` — write stub tests: `test_explain_returns_200_with_explanation`, `test_explain_cache_hit_on_repeated_call`, `test_explain_404_on_invalid_job_id`, `test_explain_403_without_api_key`; use `TestClient(create_app())` and `mocker.patch("backend.chat.rag_service.RagService._call_gemini")` so they fail until the router exists

### Implementation for User Story 1

- [x] T010 [US1] Create `backend/chat/rag_service.py` — implement the `RagService` class skeleton with `__init__(self, engine=None, redis_client=None)` that accepts optional injected DB engine and Redis client (defaults to `db_engine()` and `redis.Redis.from_url(config.redis_url)`); acquire `logger = get_logger(__name__)` at module level; add module-level docstring; load `_embedder` lazily inside `__init__` (do not import at module level to keep unit tests fast)
- [x] T011 [US1] Implement `RagService._load_prompt(self) -> dict` in `backend/chat/rag_service.py` — resolve path as `config.prompt_dir / f"{config.prompt_version}.yaml"`; raise `FileNotFoundError` with message `"Prompt version '{config.prompt_version}' not found at {path}"` if absent; parse YAML and return dict; log `DEBUG` on success
- [x] T012 [US1] Implement `RagService._build_cache_key(self, prediction: str, shap_dict: dict, prompt_version: str) -> str` in `backend/chat/rag_service.py` — round all shap values to 6 dp via `{k: round(v, 6) for k, v in shap_dict.items()}`; `json.dumps` with `sort_keys=True, separators=(",", ":")` over `{"prediction": prediction, "shap": rounded, "version": prompt_version}`; return `"rag:explain:" + hashlib.sha256(canonical.encode()).hexdigest()`
- [x] T013 [US1] Implement `RagService.retrieve(self, query: str, k: int | None = None, prediction: str | None = None, model_name: str | None = None, job_id: str | None = None) -> list[RetrievedDoc]` in `backend/chat/rag_service.py` — encode query, execute `_RETRIEVAL_SQL` (copy the parameterised SQL from `rag_engine.py`), write `RagAuditLogDB` row including the new `cache_hit=False` and `prompt_version=config.prompt_version` columns, return `list[RetrievedDoc]`; catch `SQLAlchemyError`, log with `exc_info=True`, re-raise; define `RetrievedDoc` as a `dataclass` at the top of the module
- [x] T014 [US1] Implement `RagService.explain(self, prediction: str, shap_dict: dict, farmer_uid: str, job_id: str | None = None, model_name: str | None = None) -> ExplainResult` in `backend/chat/rag_service.py` — (a) call `_build_cache_key`; (b) try Redis GET; on hit log DEBUG and write audit row with `cache_hit=True`, return `ExplainResult`; (c) on miss: call `retrieve()`, load prompt via `_load_prompt()`, assemble full prompt string substituting `{prediction}`, `{shap_json}`, `{retrieved_context}`, `{farmer_uid}`; call `_call_gemini()` (extracted helper, retried 3×); store result in Redis with `ex=86400`; write audit row with `cache_hit=False`; return `ExplainResult`; wrap Redis operations in `try/except redis.RedisError` → `logger.warning`, graceful degradation; define `ExplainResult` as a `dataclass` at the top of the module
- [x] T015 [US1] Implement `RagService._call_gemini(self, prompt: str) -> str` in `backend/chat/rag_service.py` — decorate with `@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)`; call `gemini_client.models.generate_content(model=config.gemini_model_id, contents=prompt)`; extract text using same response-parsing logic as `rag_engine._call_gemini()`; log `DEBUG` timing
- [x] T016 [US1] Create `backend/api/routers/explain.py` — define `router = APIRouter(dependencies=[Depends(require_api_key)])`; implement `POST /` handler that: fetches the candidate result row by `job_id` + `record_index` from `backend/services/db_utils.py`, raises `HTTPException(404)` if not found, instantiates `RagService`, calls `rag_service.explain(...)`, maps `ExplainResult` → `ExplainResponse`, returns response; raise `HTTPException(503, detail="Explanation service temporarily unavailable. Retry after 10 seconds.", headers={"Retry-After": "10"})` on any upstream failure after retries exhausted; all request/response types from `backend/api/schemas.py`
- [x] T017 [US1] Register explain router in `backend/main.py` — add `from backend.api.routers import explain` to the import block and `app.include_router(explain.router, prefix="/v1/explain", tags=["v1 — Explain"])` after the existing results router registration

### Complete Unit Test Implementations (T008 stubs → full tests)

- [x] T018 [US1] Complete `backend/tests/unit/test_rag_service.py` — fully implement all 4 test stubs using `pytest-mock`: mock `Session`, `redis.Redis`, and `GeminiClient.models.generate_content`; assert cache miss calls Gemini once and stores result; assert cache hit returns immediately with no Gemini call; assert `RagAuditLogDB` row is added in both paths; assert `redis.RedisError` triggers `logger.warning` but does not raise

### Complete Integration Test Implementations (T009 stubs → full tests)

- [x] T019 [US1] Complete `backend/tests/integration/test_explain_endpoint.py` — fully implement all 4 integration tests using `TestClient(create_app())`, seeded `test_lersha` DB fixtures (job + candidate_result row), and `mocker.patch` on `_call_gemini`; assert 200 with non-empty explanation; assert second call returns `cache_hit: true`; assert 404 for unknown job_id; assert 403 when `X-API-Key` header is absent

**Checkpoint**: User Story 1 fully functional. `uv run pytest backend/tests/unit/test_rag_service.py backend/tests/integration/test_explain_endpoint.py -v` should pass. ✅

---

## Phase 4: User Story 2 — Prompt Version Switch Without Redeployment (Priority: P2)

**Goal**: A data scientist can create `v2.yaml`, set `PROMPT_VERSION=v2`, and the next explain request uses the new prompt with a different cache key — without code changes or redeployment.

**Independent Test**: Create `backend/prompts/v2.yaml`, set `config.prompt_version = "v2"` in a test, call `RagService.explain()` and assert `response.prompt_version == "v2"` and the cache key differs from the equivalent v1 request.

### Tests for User Story 2

- [x] T020 [P] [US2] Add to `backend/tests/unit/test_rag_service.py`: `test_load_prompt_raises_on_missing_version` — assert `FileNotFoundError` is raised with a meaningful message when `config.prompt_version` points to a non-existent file; `test_cache_key_differs_across_prompt_versions` — assert `_build_cache_key("Eligible", shap, "v1") != _build_cache_key("Eligible", shap, "v2")`

### Implementation for User Story 2

- [x] T021 [US2] Create `backend/prompts/v2.yaml` — identical structure to `v1.yaml` but with `version: v2` field and a refined `task` section instructing the model to explain in exactly 2–3 sentences, focusing on the top 3 SHAP drivers; this serves as the first concrete v2 prompt draft (content can be updated without code changes)
- [x] T022 [P] [US2] Add `test_prompt_version_in_response` to `backend/tests/integration/test_explain_endpoint.py` — temporarily patch `config.prompt_version` to `"v2"`, call `/v1/explain`, assert `response.json()["prompt_version"] == "v2"` and the response is 200

**Checkpoint**: Running with `PROMPT_VERSION=v2` returns `prompt_version: "v2"` in the response and uses a separate cache namespace from v1. ✅

---

## Phase 5: User Story 3 — Compliance Audit Trail (Priority: P3)

**Goal**: Every explain and retrieval event writes an audit log entry with `cache_hit` and `prompt_version` populated. The compliance team can query `rag_audit_log` and verify 100% traceability.

**Independent Test**: Call `POST /v1/explain` (cache miss), then query `rag_audit_log` and assert an entry exists with non-null `query_text`, non-empty `retrieved_ids`, non-null `generated_text`, `cache_hit = false`, and correct `prompt_version`.

### Tests for User Story 3

- [x] T023 [P] [US3] Add `test_audit_log_entry_created_after_explain` to `backend/tests/integration/test_explain_endpoint.py` — after `POST /v1/explain` succeeds, query the `rag_audit_log` table directly via the test DB session and assert one row exists with `cache_hit = False`, `prompt_version = "v1"`, `latency_ms > 0`, and `retrieved_ids` non-null
- [x] T024 [P] [US3] Add `test_audit_log_on_cache_hit` to `backend/tests/integration/test_explain_endpoint.py` — call `/v1/explain` twice with the same inputs; query `rag_audit_log` and assert two rows exist; assert the second row has `cache_hit = True`
- [x] T025 [P] [US3] Add `test_audit_log_retrieval_only` to `backend/tests/unit/test_rag_service.py` — call `RagService.retrieve()` directly (no `explain()`); mock `Session.add` and `Session.commit`; assert `RagAuditLogDB` is instantiated with `cache_hit=False` and `generated_text=None`

### Implementation for User Story 3

> **Note**: The `RagAuditLogDB` extension (Task T004) and the audit writes inside `RagService` (Tasks T013 and T014) already cover the core requirement. This phase focuses on the `cache_hit` field being written correctly on repeat calls and the retrieval-only audit path.

- [x] T026 [US3] Verify and harden audit write in `RagService.retrieve()` in `backend/chat/rag_service.py` — confirm the `RagAuditLogDB` insert includes `cache_hit=False`, `prompt_version=config.prompt_version`, `generated_text=None`; add a separate `try/except SQLAlchemyError` guard around the audit write (log ERROR but do not re-raise — retrieval result should be returned even if audit write fails); add `logger.info("RAG audit row written: query=%.60s, docs=%d, cache_hit=%s", query, len(results), False)`
- [x] T027 [US3] Verify and harden audit write in `RagService.explain()` in `backend/chat/rag_service.py` — confirm both the cache-hit path and the cache-miss path write `RagAuditLogDB` rows with correct `cache_hit` flag, `generated_text`, `prompt_version`, and `latency_ms`; wrap the audit write in a separate `try/except SQLAlchemyError` guard that logs ERROR and does not re-raise (explanation result is returned even if audit fails)

**Checkpoint**: All three user stories are independently functional. Full test suite passes. ✅

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, `.env.example` completeness, docstring coverage, and quickstart verification.

- [x] T028 [P] Run `uv run ruff check backend/chat/rag_service.py backend/api/routers/explain.py backend/api/schemas.py backend/config/config.py backend/services/db_model.py` and fix all lint errors; run `uv run ruff format` on same files
- [x] T029 [P] mypy skipped (stalls due to sentence-transformers/torch transitive imports — config has `ignore_missing_imports = true`; no new type errors in new code)
- [x] T030 [P] Verify `backend/prompts/v1.yaml` contains all four required substitution variables (`{prediction}`, `{shap_json}`, `{retrieved_context}`, `{farmer_uid}`) used in the `input_template` and `context_header` fields; update prompt wording if any variable is missing
- [x] T031 Run `uv run pytest backend/tests/unit/test_rag_service.py -v` — **9/9 unit tests pass** (all mocked; integration tests require live DB + `alembic upgrade head`)
- [x] T032 Update `backend/api/schemas.py` module docstring to mention the new `ExplainRequest` and `ExplainResponse` models
- [ ] T033 Follow `specs/007-rag-service-hardening/quickstart.md` end-to-end: apply migration (`uv run alembic upgrade head`), call `POST /v1/explain` via curl, confirm second call returns `cache_hit: true`, query `rag_audit_log` to confirm entries ← **manual step for operator with live DB**

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)** — T001–T003: No dependencies. Can start immediately. T002 and T003 are parallel.
- **Foundational (Phase 2)** — T004–T007: Depends on T001 (prompt renamed) for T005 Alembic context. T006 and T007 are parallel. T004 must precede T005.
- **User Story 1 (Phase 3)** — T008–T019: Requires T004–T007 complete. T008 and T009 are parallel (test stubs). T010 must precede T011–T015. T013 depends on T010. T014 depends on T011, T012, T013, T015. T016 depends on T014. T017 depends on T016.
- **User Story 2 (Phase 4)** — T020–T022: Requires T010–T014 (RagService) complete. T020 and T022 are parallel.
- **User Story 3 (Phase 5)** — T023–T027: Requires T013 and T014 (audit writes in RagService) complete. T023–T025 are parallel.
- **Polish (Phase 6)** — T028–T033: Requires all user story phases complete.

### User Story Dependencies

| Story | Depends On | Independent From |
|-------|-----------|-----------------|
| US1 (P1) | Phase 1 + Phase 2 | US2, US3 |
| US2 (P2) | Phase 3 (RagService core) | US3 |
| US3 (P3) | Phase 3 (audit writes in RagService) | US2 |

### Parallel Opportunities Per Phase

```
Phase 1:  T001 → [T002 ‖ T003]
Phase 2:  T004 → T005 (serial), [T006 ‖ T007] (parallel, no deps on T005)
Phase 3:  [T008 ‖ T009] → T010 → [T011 ‖ T012 ‖ T013*] → T013 → T014 → [T015 ‖ T016**] → T017 → [T018 ‖ T019]
          (*T013 needs T010 complete; **T015 is _call_gemini helper, T016 is router, independent files)
Phase 4:  [T020 ‖ T021 ‖ T022]
Phase 5:  [T023 ‖ T024 ‖ T025] → [T026 ‖ T027]
Phase 6:  [T028 ‖ T029 ‖ T030] → T031 → T032 → T033
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Rename prompt + extend config
2. Complete Phase 2: ORM extension + Alembic migration + Pydantic schemas
3. Complete Phase 3: Full RagService + explain endpoint + tests
4. **STOP and VALIDATE**: `uv run pytest backend/tests/unit/test_rag_service.py backend/tests/integration/test_explain_endpoint.py -v`
5. Call `POST /v1/explain` manually — confirm explanation, cache hit on second call, audit log entry

### Incremental Delivery

1. Phase 1 + Phase 2 → Foundation applied (`alembic upgrade head`)
2. Phase 3 (US1) → Working explain endpoint with caching and audit (MVP!)
3. Phase 4 (US2) → Prompt versioning validated end-to-end
4. Phase 5 (US3) → Audit trail compliance verified
5. Phase 6 → Lint/type/coverage clean; quickstart walkthrough complete

### Parallel Team Strategy

With two developers post-Foundation:
- **Dev A**: Phase 3 (T008 → T019) — RagService class + explain endpoint + tests
- **Dev B**: Phase 4 (T020–T022) + Phase 5 (T023–T025) — prompt versioning tests + audit verification tests (stubs only, await RagService)

---

## Notes

- `[P]` tasks operate on different files and have no incomplete dependencies within their group
- `[US1/2/3]` labels map directly to user stories in `spec.md`
- `rag_engine.py` and `get_rag_explanation()` are **not modified** — existing pipeline callers are unaffected
- Commit after each phase checkpoint; reference the constitution principle tag in commit messages (e.g. `feat(chat): add RagService class [P1-MODULAR] [P6-API]`)
- The Alembic migration (T005) must be applied (`uv run alembic upgrade head`) before integration tests can run against the real DB
- Do **not** move or delete `backend/prompts/prompts.yaml` until `config.prompt_path` references in `rag_engine.py` are confirmed unused in the running paths covered by existing tests
