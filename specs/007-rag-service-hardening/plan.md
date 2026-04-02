# Implementation Plan: RAG System Hardening

**Branch**: `007-rag-service-hardening` | **Date**: 2026-04-02 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/007-rag-service-hardening/spec.md`

---

## Summary

Harden the existing RAG explanation system from a module of top-level functions into a well-structured, production-grade service. The three pillars of the change are: (1) a `RagService` class that encapsulates retrieval, Redis-cached explanation generation, and audit logging; (2) a prompt versioning system that allows iteration on explanation quality through YAML files controlled by an environment variable; and (3) a dedicated `POST /v1/explain` HTTP endpoint that exposes the service to API consumers.

All existing `get_rag_explanation()` callers in the pipeline continue to work during transition — the new class is additive, not a breaking replacement.

---

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: FastAPI 0.121, SQLAlchemy 2.x, pgvector 0.3+, redis 5.0+, google-genai 1.0+, sentence-transformers 2.7+, tenacity 8.2+, PyYAML 6.0+, Alembic 1.13+
**Storage**: PostgreSQL 15+ with pgvector extension; Redis (existing, already in `config.redis_url`)
**Testing**: pytest 8+, pytest-mock 3.14+, pytest-cov 5+, httpx 0.27+
**Target Platform**: Linux container (`python:3.12-slim`), Docker Compose
**Project Type**: Web service (FastAPI backend behind Caddy reverse proxy)
**Performance Goals**: ≤ 3 s cache-miss latency, ≤ 200 ms cache-hit latency (SC-001)
**Constraints**: Zero new `pyproject.toml` dependencies — `redis>=5.0` already present. Audit 100% coverage (SC-003). Graceful cache degradation — no 5xx on Redis failure (SC-007).
**Scale/Scope**: Same scale as existing `/v1/predict` (10 req/min/IP rate limit applies). Single-tenant production deployment.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-evaluated post-design below.*

| Principle | Gate Evaluation | Status |
|-----------|----------------|--------|
| `[P1-MODULAR]` | `RagService` placed in `backend/chat/` (RAG ownership). Router in `backend/api/routers/`. Zero business logic in router. Schemas in `backend/api/schemas.py`. | ✅ PASS |
| `[P2-PEP]` | All new files use `ruff` conventions: type annotations on all public methods, PEP 257 docstrings, `snake_case` throughout, no dead code. | ✅ PASS |
| `[P3-LOG]` | All modules acquire `logger = get_logger(__name__)`. Cache miss, hit, Redis errors, Gemini calls logged at correct levels. No `print()`. | ✅ PASS |
| `[P4-EXC]` | `SQLAlchemyError` caught in retrieval; `redis.RedisError` caught for graceful degradation (WARNING, not re-raise); Gemini failures re-raised to `tenacity` boundary; all caught with `exc_info=True`. | ✅ PASS |
| `[P5-CONFIG]` | `PROMPT_VERSION` and `prompt_dir` added to `Config` singleton. `os.getenv()` only in `config.py`. `.env.example` updated. | ✅ PASS |
| `[P6-API]` | Router delegates to `RagService` only. Pydantic `ExplainRequest`/`ExplainResponse` defined in `schemas.py`. `require_api_key` applied to router. Path prefixed `/v1/`. | ✅ PASS |
| `[P7-TEST]` | Unit tests mock DB/Redis/Gemini. Integration tests use `test_lersha` DB + mocked Gemini. Coverage gates apply: 80% minimum on `backend/chat/`. | ✅ PASS |
| `[P8-DB]` | New `cache_hit`/`prompt_version` columns via Alembic migration `004`. All SQL in `db_utils.py` or ORM. No DDL in application code. Parameterised queries only. | ✅ PASS |
| `[P9-SEC]` | `require_api_key` on explain router. No PII in logs. `PROMPT_VERSION` documented in `.env.example` — not a secret. | ✅ PASS |
| `[P10-OBS]` | Every retrieval and generation event writes an audit row. Cache hit/miss logged at DEBUG. Redis errors at WARNING. | ✅ PASS |
| `[P11-CONT]` | No new containers or volumes needed. Redis already in `docker-compose.yml`. | ✅ PASS |
| `[P12-CI]` | New test files trigger existing CI `test` job automatically. Coverage gate already at 80%. | ✅ PASS |

**Constitution Check: ALL GATES PASS — No violations.**

---

## Project Structure

### Documentation (this feature)

```text
specs/007-rag-service-hardening/
├── plan.md                              # This file
├── research.md                          # Phase 0 output
├── data-model.md                        # Phase 1 output
├── quickstart.md                        # Phase 1 output
├── contracts/
│   └── explain_endpoint.md              # Phase 1 output
└── checklists/
    └── requirements.md                  # Spec quality checklist
```

### Source Code Layout (changes only)

```text
backend/
├── chat/
│   ├── __init__.py                      # unchanged
│   ├── rag_engine.py                    # unchanged (legacy callers preserved)
│   └── rag_service.py                   # NEW — RagService class
│
├── prompts/
│   ├── v1.yaml                          # RENAMED from prompts.yaml
│   └── v2.yaml                          # (example future version, not in scope)
│
├── api/
│   ├── schemas.py                       # EXTENDED — ExplainRequest, ExplainResponse
│   └── routers/
│       ├── explain.py                   # NEW — POST /v1/explain
│       └── [health, predict, results]   # unchanged
│
├── config/
│   └── config.py                        # EXTENDED — prompt_version, prompt_dir
│
├── services/
│   └── db_model.py                      # EXTENDED — RagAuditLogDB: cache_hit, prompt_version
│
├── alembic/
│   └── versions/
│       └── 004_add_audit_cache_fields.py   # NEW — migration
│
├── main.py                              # EXTENDED — register explain router
│
└── tests/
    ├── unit/
    │   └── test_rag_service.py          # NEW — unit tests for RagService
    └── integration/
        └── test_explain_endpoint.py     # NEW — integration tests for /v1/explain
```

**Structure Decision**: Option 2 (web application backend layout) applies — this is an additive feature within the existing `backend/` monorepo. No new top-level directories.

---

## Implementation Phases

### Phase A — Foundation

1. **Alembic migration `004`** — add `cache_hit` (boolean, default false) and `prompt_version` (varchar 20) to `rag_audit_log`. Update `RagAuditLogDB` ORM model.
2. **Rename prompt file** — `backend/prompts/prompts.yaml` → `backend/prompts/v1.yaml`. Update its YAML structure to the versioned schema defined in `data-model.md`.
3. **Extend `Config`** — add `prompt_version` and `prompt_dir`. Document `PROMPT_VERSION` in `.env.example`.
4. **Extend `backend/api/schemas.py`** — add `ExplainRequest` and `ExplainResponse` Pydantic models.

### Phase B — Service

5. **Create `backend/chat/rag_service.py`** — `RagService` class with:
   - `__init__`: accept optional injected DB engine + Redis client for testability.
   - `_load_prompt()`: load versioned YAML from `config.prompt_dir / f"{config.prompt_version}.yaml"`, raise on missing.
   - `_build_cache_key()`: SHA-256 canonical JSON key.
   - `retrieve()`: embed query → pgvector SQL → audit write → return `list[RetrievedDoc]`.
   - `explain()`: cache check → retrieve → prompt assemble → Gemini → cache store → audit write → return `ExplainResult`.
   - Redis graceful degradation: `try/except redis.RedisError` → `logger.warning`, treat as cache miss.

### Phase C — Endpoint

6. **Create `backend/api/routers/explain.py`** — thin router:
   - `APIRouter(dependencies=[Depends(require_api_key)])`.
   - `POST /` handler: validate request → fetch record from DB → call `RagService.explain()` → map to `ExplainResponse`.
   - Return `ExplainResponse` on 200; raise `HTTPException(404)` for missing job/record; raise `HTTPException(503)` on upstream failures.
7. **Register router in `backend/main.py`** — `app.include_router(explain.router, prefix="/v1/explain", tags=["v1 — Explain"])`.

### Phase D — Tests

8. **`backend/tests/unit/test_rag_service.py`** — mock all I/O:
   - `test_retrieve_returns_docs_and_writes_audit()`
   - `test_explain_cache_miss_calls_gemini_and_caches()`
   - `test_explain_cache_hit_skips_gemini()`
   - `test_explain_redis_error_degrades_gracefully()` — Redis raises `RedisError`, explain still returns result.
   - `test_explain_audit_log_written_on_cache_hit()`
   - `test_load_prompt_raises_on_missing_version()`
9. **`backend/tests/integration/test_explain_endpoint.py`** — `TestClient` + real `test_lersha` DB + mocked Gemini:
   - `test_explain_returns_200_with_explanation()`
   - `test_explain_cache_hit_on_repeated_call()`
   - `test_explain_404_on_invalid_job_id()`
   - `test_explain_403_without_api_key()`
   - `test_audit_log_entry_created_after_explain()`

---

## Complexity Tracking

No constitution violations were identified. No exceptions to document.
