# Lersha Credit Scoring System — Production Upgrade Plan

**Version:** 1.0.0
**Date:** 2026-03-29
**Author:** Architecture Review
**Status:** Approved for Implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Inventory](#2-current-state-inventory)
3. [Target Architecture](#3-target-architecture)
4. [Phase 1 — pgvector Migration (ChromaDB → PostgreSQL)](#4-phase-1--pgvector-migration)
5. [Phase 2 — RAG System Hardening](#5-phase-2--rag-system-hardening)
6. [Phase 3 — Next.js Frontend Migration](#6-phase-3--nextjs-frontend-migration)
7. [Phase 4 — Multi-Interface Standardisation](#7-phase-4--multi-interface-standardisation)
8. [Phase 5 — Production System Design](#8-phase-5--production-system-design)
9. [Dependency Map & Risk Register](#9-dependency-map--risk-register)
10. [Migration Sequencing & Milestones](#10-migration-sequencing--milestones)

---

## 1. Executive Summary

The Lersha Credit Scoring System has completed its initial production-hardening phase (Phases 1–8 per `docs/REFACTOR_PLAN.md`). The system operates a **FastAPI backend** with Celery async workers, PostgreSQL persistence, ChromaDB-backed RAG, and a Streamlit frontend. This document defines the next evolutionary cycle:

| Upgrade | Priority | Complexity | Risk |
|---|---|---|---|
| pgvector (replaces ChromaDB) | P0 | Medium | Low |
| RAG system hardening | P0 | High | Medium |
| Next.js frontend | P1 | High | Low |
| Multi-interface standardisation | P1 | Medium | Low |
| Production system design | P0 | High | Medium |

**Guiding constraint:** The FastAPI backend is the single source of truth. All interfaces (CLI, Next.js, Streamlit, direct API) consume the same versioned service layer. No interface is permitted to import `backend.*` modules directly.

---

## 2. Current State Inventory

### 2.1 Backend

| Component | Technology | File(s) | State |
|---|---|---|---|
| API Framework | FastAPI 0.121 | `backend/main.py` | ✅ Stable |
| Inference Pipeline | XGBoost / RF / CatBoost + SHAP | `backend/core/pipeline.py` | ✅ Stable |
| Async Task Queue | Celery 5.3 + Redis | `backend/worker.py` | ✅ Stable |
| ORM | SQLAlchemy 2.x | `backend/services/db_model.py` | ✅ Stable |
| DB Migrations | Alembic | `backend/alembic/versions/` | ✅ 2 migrations |
| RAG Engine | ChromaDB + Gemini | `backend/chat/rag_engine.py` | ⚠ Needs upgrade |
| Vector Store | ChromaDB (file-based) | `chroma_db/` | ❌ Replace |
| Configuration | env vars + YAML | `backend/config/config.py` | ✅ Stable |
| Rate Limiting | slowapi | `backend/api/dependencies.py` | ✅ Stable |
| Observability | python-json-logger | `backend/logger/` | ⚠ Partial |
| Reverse Proxy | Caddy 2 (HTTPS) | `Caddyfile` | ✅ Prod-ready |

### 2.2 Frontend (Streamlit)

| Component | File | API Coupling |
|---|---|---|
| Entry page | `ui/Introduction.py` | None (static) |
| Prediction page | `ui/pages/New_Prediction.py` | `LershaAPIClient` (HTTP only) |
| Dashboard | `ui/pages/Dashboard.py` | `LershaAPIClient` (HTTP only) |
| API client | `ui/utils/api_client.py` | HTTP → FastAPI |

**Gap:** Streamlit is single-threaded, has no built-in WebSocket support, cannot deliver real-time polling without hacks, and is not suitable for a production-facing B2B product. No state management, no routing, no SSR.

### 2.3 Infrastructure

```
docker-compose.yml         ← service skeletons
docker-compose.override.yml← dev overrides
docker-compose.prod.yml    ← production (Caddy, Gunicorn, secrets, backup)
```

| Service | Image | Role |
|---|---|---|
| postgres | postgres:16 | Primary DB + job storage |
| redis | redis:7-alpine | Celery broker/backend |
| backend | custom | FastAPI (Gunicorn 4 workers) |
| worker | custom | Celery (4 concurrency) |
| ui | custom | Streamlit |
| mlflow | ghcr.io/mlflow/mlflow | Experiment tracking |
| caddy | caddy:2-alpine | TLS / reverse proxy |
| backup | postgres-backup-local | Daily pg_dump |

### 2.4 Known Technical Debt

1. `db_engine()` creates a new engine per call — pool sharing is broken.
2. ChromaDB is a standalone file-based store disconnected from PostgreSQL transactions.
3. No OpenTelemetry / structured trace propagation across services.
4. Streamlit cannot serve WebSocket-based real-time updates without Tornado hacks.
5. RAG prompt is currently YAML-file-based with no versioning or A/B test capability.
6. No API versioning beyond `/v1/` prefix.
7. No authentication on the Next.js/UI layer (only API key on backend).

---

## 3. Target Architecture

### 3.1 High-Level Service Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Caddy (TLS Termination)                     │
│   /       → Next.js UI (port 3000)                                  │
│   /api/*  → FastAPI Backend (port 8000)                             │
│   /mlflow → MLflow (port 5000)  [internal only]                     │
└─────────────────────────────────────────────────────────────────────┘
         │                          │                   │
┌────────▼──────────┐   ┌──────────▼────────┐   ┌──────▼─────────┐
│  Next.js UI        │   │  FastAPI Backend   │   │  Celery Worker │
│  (Next 14, App     │   │  (Gunicorn 4w)     │   │  (4 procs)     │
│   Router, React 18)│   │  /v1/predict       │   │  ML Pipeline   │
│  State: Zustand    │   │  /v1/results       │   │  SHAP + RAG    │
│  Fetch: TanStack   │   │  /v1/explain       │   └──────┬─────────┘
│  Query             │   │  /v2/* (future)    │          │
└────────────────────┘   └────────┬───────────┘          │
                                  │                        │
                         ┌────────▼────────────────────────▼──────┐
                         │         PostgreSQL (single instance)     │
                         │                                          │
                         │  candidate_raw_data_table (farmer data) │
                         │  candidate_result (predictions)         │
                         │  inference_jobs (job queue state)       │
                         │  rag_documents (pgvector embeddings)    │
                         │  rag_audit_log (RAG decision trace)     │
                         └────────────────────────────────────────┘
                                  │
                         ┌────────▼────────────────────────────────┐
                         │  Redis (broker + result backend)         │
                         └────────────────────────────────────────┘
```

### 3.2 Module Boundaries

```
lersha_credit_api/
├── backend/                   ← Python backend (unchanged root)
│   ├── api/                   ← HTTP layer (routers, schemas, middleware)
│   │   └── routers/           ← health, predict, results, explain (new)
│   ├── chat/                  ← RAG engine (pgvector-backed)
│   ├── core/                  ← ML pipeline (no HTTP concern)
│   ├── services/              ← DB utilities (shared by all interfaces)
│   ├── cli/                   ← CLI entrypoints (thin wrappers)
│   └── config/                ← Config singleton + hyperparams
├── ui/                        ← Streamlit (deprecated; retained ≤ Phase 3)
├── frontend/                  ← Next.js 14 (new)
│   ├── app/                   ← App Router pages
│   ├── components/            ← UI components
│   ├── lib/                   ← API client, type definitions
│   └── store/                 ← Zustand global state
└── docs/
```

---

## 4. Phase 1 — pgvector Migration

### 4.1 As-Is

- Vector embeddings stored in `chroma_db/` (file-based ChromaDB `PersistentClient`).
- ChromaDB operates outside PostgreSQL — no transactional consistency with prediction records.
- The `chroma_data` Docker volume is separate from `postgres_data`.
- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- Collection: `credit_features` — feature definition documents for RAG retrieval.

### 4.2 To-Be

- Embeddings stored in PostgreSQL via the `pgvector` extension.
- Two new tables: `rag_documents` (document store) and `rag_audit_log` (trace).
- All RAG queries use parameterised SQL through SQLAlchemy — no separate client.
- Single connection pool shared between relational and vector queries.
- Cosine similarity search via `vector <=> query_embedding` operator.

### 4.3 Gap Analysis

| Gap | Impact | Resolution |
|---|---|---|
| ChromaDB API vs pg client | High | Rewrite `rag_engine.py` retrieve layer |
| `pgvector` extension must be enabled | Medium | Add `CREATE EXTENSION IF NOT EXISTS vector` to migration |
| Embedding dimension must be fixed at table creation | Medium | Lock to 384 (all-MiniLM-L6-v2) |
| Existing documents need re-embedding | Low | One-time migration script |
| `chroma_data` Docker volume becomes obsolete | Low | Remove after migration |

### 4.4 Schema Design

```sql
-- Migration 003: Enable pgvector and create RAG tables

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE rag_documents (
    id           SERIAL PRIMARY KEY,
    doc_id       UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    category     VARCHAR(100) NOT NULL,     -- e.g. 'feature_definition', 'policy'
    title        VARCHAR(255) NOT NULL,
    content      TEXT NOT NULL,
    embedding    VECTOR(384) NOT NULL,      -- all-MiniLM-L6-v2 output dimension
    metadata     JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- IVFFlat index — cosine similarity, 100 lists (suitable for < 1M vectors)
-- Rebuild as HNSW if document count exceeds 500k
CREATE INDEX idx_rag_documents_embedding
    ON rag_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX idx_rag_documents_category ON rag_documents (category);

CREATE TABLE rag_audit_log (
    id              SERIAL PRIMARY KEY,
    query_text      TEXT NOT NULL,
    retrieved_ids   INTEGER[] NOT NULL,     -- rag_documents.id references
    prediction      VARCHAR(100),
    model_name      VARCHAR(100),
    job_id          UUID,
    generated_text  TEXT,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 4.5 Indexing Strategy

- **IVFFlat** (`lists = 100`) for initial deployment. Suitable up to ~500k documents with acceptable recall.
- **HNSW** (`m = 16, ef_construction = 64`) for sustained production load exceeding 500k vectors — lower latency at query time, higher build-time memory cost.
- The migration uses IVFFlat; a follow-up migration converts to HNSW once the corpus exceeds the threshold.
- Partial index by `category` for filtered retrieval.

### 4.6 Implementation Steps

1. **Migration 003** (`backend/alembic/versions/003_add_pgvector.py`):
   - Enable the `vector` extension.
   - Create `rag_documents` and `rag_audit_log` tables with above schema.
   - Add IVFFlat index on `embedding`.

2. **ORM models** (`backend/services/db_model.py`):
   - Add `RagDocumentDB` and `RagAuditLogDB` SQLAlchemy models using `pgvector.sqlalchemy.Vector` column type.

3. **Document ingestion script** (`backend/scripts/populate_pgvector.py`):
   - Read the existing YAML/JSON feature definition corpus from `backend/data/` or the current ChromaDB collection.
   - Compute embeddings using the same `SentenceTransformer` model.
   - Batch-insert into `rag_documents` via SQLAlchemy (1000 docs/batch).
   - Log count and any failures.

4. **Rewrite RAG retrieval** (`backend/chat/rag_engine.py`, `retrieve_docs` function):
   - Replace `collection.query()` with a parameterised SQL nearest-neighbour query.
   - Return `List[Tuple[str, float]]` — text + cosine distance — for downstream ranking.
   - Emit an `INSERT INTO rag_audit_log` after every retrieval for traceability.

5. **Configuration updates** (`backend/config/config.py`):
   - Remove `chroma_db_path` and `CHROMA_DB_PATH`.
   - Add `rag_top_k` (default 5), `rag_similarity_threshold` (default 0.75).

6. **Docker Compose** (`docker-compose.yml`):
   - Remove `chroma_data` volume from all services.
   - Remove `chroma_data:/app/chroma_db` volume mount from `backend` and `worker` services.
   - Add `POSTGRES_EXTENSIONS: vector` env var to `postgres` service if using a pgvector image.

7. **Dependency update** (`pyproject.toml`):
   - Add `pgvector>=0.3.0`.
   - Remove `chromadb>=0.5.0` after migration is validated.

8. **Validation**:
   - Integration test: insert 10 documents, query for top-3, assert cosine scores > 0.75.
   - Confirm `rag_audit_log` records are created for each retrieval.
   - Benchm emark: single retrieval latency < 50ms with full IVFFlat index.

---

## 5. Phase 2 — RAG System Hardening

### 5.1 As-Is

- RAG explanation is generated per-row inside `run_inferences()` loop.
- Single prompt template loaded from `backend/prompts/prompts.yaml` (no versioning).
- Gemini API called synchronously inside the Celery worker.
- No retrieval quality metrics or auditability.
- Explanation stored as raw text in `candidate_result.rag_explanation`.

### 5.2 To-Be

- RAG pipeline is a standalone, testable service class: `RagService`.
- Prompts are versioned and stored in the DB alongside their outputs.
- Retrieval quality is measured (MRR, recall@k) on a held-out eval set.
- Explanations are deterministic for the same `(prediction, shap_dict)` input via a content-addressed cache.
- Audit log captures every retrieval + generation event.
- A dedicated `/v1/explain` API endpoint surfaces explanations on-demand.

### 5.3 Data Sources for Embeddings

| Source | Content | Format |
|---|---|---|
| Feature definitions | Plain-language definitions of the 36 model features | YAML → text |
| Credit policy rules | Eligibility thresholds, risk tier logic | Markdown |
| SHAP interpretation guides | Directional explanations ("high age_group increases eligibility") | Text |
| Historical RAG audit log | Past queries and their generated explanations | Structured text |

### 5.4 Retrieval Strategy

```
Query construction:
  query = f"Model predicted: {prediction}\nTop features: {shap_json}"

Retrieval:
  1. Embed query → vector q (384-dim, all-MiniLM-L6-v2)
  2. SELECT content, 1 - (embedding <=> q) AS similarity
     FROM rag_documents
     WHERE category IN ('feature_definition', 'policy_rule')
       AND 1 - (embedding <=> q) > 0.75
     ORDER BY embedding <=> q
     LIMIT 5

Post-retrieval filtering:
  3. Cross-encoder re-ranking (optional, Phase 2b): use a lightweight
     cross-encoder (e.g. ms-marco-MiniLM-L-6-v2) to re-rank the top-5
     results by relevance to the query before passing to LLM.
```

### 5.5 Prompt Architecture

- Prompts are stored in `backend/prompts/` as versioned YAML files (`v1.yaml`, `v2.yaml`).
- Config references `PROMPT_VERSION` env var (default: `v1`).
- Prompt template variables: `{prediction}`, `{shap_json}`, `{retrieved_context}`, `{farmer_uid}`.
- The `rag_audit_log` captures `prompt_version`, enabling before/after quality comparisons.

**Prompt structure (target):**

```
SYSTEM: You are a credit analyst for agricultural lending in Ethiopia. 
        You explain model decisions in clear, non-technical language.

CONTEXT (retrieved):
{retrieved_context}

TASK:
Given the credit model predicted {prediction} for this farmer, explain
the decision in 2–3 sentences focused on {top_feature_names}.

Rules:
- Never invent information not present in CONTEXT or INPUT.
- Do not reveal raw SHAP values; translate them to plain language.
- Be empathetic and actionable.
- Output must be ≤ 150 words.

INPUT:
prediction = {prediction}
shap_contributions = {shap_json}

RESPONSE:
```

### 5.6 Determinism & Caching

- Cache key: `sha256(prediction + canonical_json(shap_dict) + prompt_version)`.
- Cache store: Redis with 24h TTL.
- On cache hit: skip Gemini call, return cached text.
- Cache miss: call Gemini, store result in Redis, write to `rag_audit_log`.
- This guarantees the same farmer/model combination returns identical explanations within a 24h window, which is required for audit reproducibility.

### 5.7 `/v1/explain` Endpoint

**Request:**
```
POST /v1/explain
{
  "job_id": "uuid",
  "record_index": 0,
  "model_name": "xgboost"
}
```

**Response:**
```
{
  "farmer_uid": "ETH-2024-001",
  "prediction": "Eligible",
  "explanation": "...",
  "retrieved_doc_ids": [1, 4, 7],
  "cache_hit": false,
  "prompt_version": "v1",
  "latency_ms": 312
}
```

### 5.8 Implementation Steps

1. Create `backend/chat/rag_service.py` — a class `RagService` with:
   - `retrieve(query: str) -> List[RetrievedDoc]`
   - `explain(prediction: str, shap_dict: dict, farmer_uid: str) -> ExplainResult`
   - Internal Redis cache check/write.
   - Audit log write after every call.

2. Migrate `rag_engine.py` to be a thin compatibility shim that delegates to `RagService`.

3. Add `backend/api/routers/explain.py` — `POST /v1/explain`.

4. Register the explain router in `backend/main.py`.

5. Add `pgvector>=0.3.0` and `redis>=5.0` to non-optional deps (already present).

6. Write unit tests for `RagService` using mocked DB and Redis.

7. Write integration test: submit a prediction job, then call `/v1/explain`, verify explanation is non-empty and audit log row exists.

---

## 6. Phase 3 — Next.js Frontend Migration

### 6.1 As-Is

| Feature | Streamlit State |
|---|---|
| Routing | Single-page with `pages/` directory |
| State management | Session state (`st.session_state`) |
| API calls | `requests` HTTP via `LershaAPIClient` |
| Real-time polling | Blocking `while` loop in Streamlit |
| Auth UI | None |
| SSR / SEO | None |
| Component library | Streamlit built-ins |
| Deployment | Docker, port 8501 |

### 6.2 To-Be

| Feature | Next.js Target |
|---|---|
| Routing | App Router (Next 14) with `app/` directory |
| State management | Zustand for global state + React Query (TanStack) for server state |
| API calls | `fetch()` wrapper in `frontend/lib/api.ts` (typed, with retry) |
| Real-time polling | TanStack Query `refetchInterval` with WebSocket upgrade path |
| Auth UI | API key input with localStorage persistence (Phase 3); OAuth2 (Phase 5) |
| SSR / SEO | Next.js Server Components for static content |
| Component library | shadcn/ui (Radix primitives + Tailwind) |
| Deployment | Docker, port 3000 |

### 6.3 UI Architecture

#### 6.3.1 App Router Structure

```
frontend/
├── app/
│   ├── layout.tsx              ← Root layout (nav, theme)
│   ├── page.tsx                ← Landing / dashboard
│   ├── predict/
│   │   └── page.tsx            ← Prediction submission form
│   ├── results/
│   │   └── page.tsx            ← Historical results table
│   ├── results/[id]/
│   │   └── page.tsx            ← Single result detail + explanation
│   └── settings/
│       └── page.tsx            ← API key configuration
├── components/
│   ├── PredictionForm.tsx      ← Source selector + submit
│   ├── JobStatusBadge.tsx      ← Pending / Processing / Done badge
│   ├── FeatureContribChart.tsx ← Recharts bar chart for SHAP values
│   ├── EvaluationCard.tsx      ← Single farmer result card
│   └── ExplanationPanel.tsx   ← RAG explanation text + metadata
├── lib/
│   ├── api.ts                  ← Typed API client (wraps fetch)
│   ├── types.ts                ← TypeScript types matching API schemas
│   └── utils.ts                ← Helpers (polling, formatting)
└── store/
    ├── useApiKeyStore.ts       ← Zustand: API key state
    └── useJobStore.ts          ← Zustand: active job state
```

#### 6.3.2 Routing Strategy

| Route | Render Mode | Description |
|---|---|---|
| `/` | ISR (60s) | Dashboard — summary counts from `/v1/results` |
| `/predict` | CSR | Interactive prediction form |
| `/results` | CSR + SWR | Paginated results table |
| `/results/[id]` | CSR | Job detail + per-farmer explainability |
| `/settings` | CSR | API key management |

#### 6.3.3 State Management

- **Zustand** for global app state: `apiKey`, `activeJobId`, `theme`.
- **TanStack Query** (`@tanstack/react-query`) for all server state: caching, background refetch, polling.
- Prediction polling: `useQuery` with `refetchInterval: 2000` while `status` is `pending` or `processing`. Stops on `completed` or `failed`.
- No Redux. No Context for data fetching.

#### 6.3.4 API Integration Layer

All backend calls flow through `frontend/lib/api.ts`:

```
Class: LershaClient
  constructor(baseUrl: string, apiKey: string)
  submitPrediction(req: PredictRequest): Promise<JobAcceptedResponse>
  getJobStatus(jobId: string): Promise<JobStatusResponse>
  getResults(params?: ResultsParams): Promise<ResultsResponse>
  getExplanation(req: ExplainRequest): Promise<ExplainResponse>
```

- All methods throw `ApiError` on non-2xx responses.
- Retry with exponential backoff (max 3 attempts) on 5xx and network errors.
- API key injected via `X-API-Key` header on every request.
- Base URL read from `NEXT_PUBLIC_API_URL` environment variable.

#### 6.3.5 SHAP Visualisation

- Use **Recharts** `BarChart` for horizontal feature contribution charts.
- Positive SHAP → green bars; negative → red bars.
- Sorted by absolute contribution descending.
- Tooltip shows raw SHAP value + feature description (from RAG document metadata).

### 6.4 Streamlit Retention Strategy

Streamlit (`ui/`) is **retained through Phase 3** as an internal-only tool for field agents who are trained on it. It is deprecated at Phase 3 GA and removed at Phase 4 GA.

**Temporal coexistence:**
- Both `ui` (Streamlit, port 8501) and `frontend` (Next.js, port 3000) run as separate Docker services.
- Caddy routes `/legacy/*` to Streamlit for the deprecation window.
- `LershaAPIClient` in `ui/utils/api_client.py` requires no changes — it already calls the same FastAPI endpoints.

### 6.5 Next.js Docker Configuration

**`frontend/Dockerfile`:**
- Multi-stage build: `node:20-alpine` for builder, `node:20-alpine` for runner.
- Output: standalone (`next.config.js` → `output: 'standalone'`).
- Non-root user (`nextjs:nodejs`).
- Port 3000.

**`docker-compose.yml` addition:**
```yaml
frontend:
  build:
    context: .
    dockerfile: frontend/Dockerfile
  restart: unless-stopped
  depends_on:
    - backend
  environment:
    NEXT_PUBLIC_API_URL: http://backend:8000
  ports:  # override.yml only
    - "3000:3000"
```

**Caddyfile update:**
- Route `/` and `/api/next/*` to `frontend:3000`.
- Route `/api/v1/*` to `backend:8000`.
- Keep `/api/v1/*` secured at the Caddy level with rate limiting headers.

### 6.6 Implementation Steps

1. Scaffold Next.js app: `npx create-next-app@latest frontend --typescript --app --tailwind --src-dir=false --import-alias="@/*"`.
2. Install deps: `shadcn/ui`, `@tanstack/react-query`, `zustand`, `recharts`, `lucide-react`.
3. Implement `frontend/lib/api.ts` — typed client with retry logic.
4. Implement `frontend/lib/types.ts` — mirror of `backend/api/schemas.py`.
5. Implement Zustand stores: `useApiKeyStore`, `useJobStore`.
6. Build pages in order: `settings` → `predict` → `results` → `results/[id]` → `/`.
7. Build shared components: `PredictionForm` → `JobStatusBadge` → `FeatureContribChart` → `EvaluationCard` → `ExplanationPanel`.
8. Write `frontend/Dockerfile` (multi-stage, standalone output).
9. Add `frontend` service to `docker-compose.yml` and `docker-compose.override.yml`.
10. Update `Caddyfile` to route frontend traffic.
11. End-to-end test: full prediction flow through Next.js UI.

---

## 7. Phase 4 — Multi-Interface Standardisation

### 7.1 As-Is

All non-HTTP interfaces (CLI scripts) import directly from `backend.*` modules. There is no unified CLI contract, and the Streamlit UI is the only GUI. There are no service-layer abstractions shared between interfaces.

### 7.2 To-Be

A **Service Layer** (`backend/services/inference_service.py`) becomes the single entrypoint for all non-HTTP callers. The HTTP layer (FastAPI routers) becomes a thin adapter over this service layer. All interfaces share the same business logic.

```
┌─────────────┐  ┌───────────┐  ┌──────────────┐  ┌─────────────┐
│  FastAPI     │  │  CLI      │  │  Next.js UI  │  │  Streamlit  │
│  Routers     │  │  (Typer)  │  │  (fetch)     │  │  (requests) │
└──────┬───────┘  └─────┬─────┘  └──────┬───────┘  └──────┬──────┘
       │                │               │ HTTP              │ HTTP
       │         ┌──────▼─────┐         ▼                  ▼
       └────────►│ Service    │    FastAPI /v1/*       FastAPI /v1/*
                 │ Layer      │
                 └──────┬─────┘
                        │
               ┌────────▼───────────────────────────┐
               │  Core Pipeline  |  RAG Service      │
               │  DB Utils       |  Config           │
               └────────────────────────────────────┘
```

### 7.3 Service Layer Design

**`backend/services/inference_service.py`:**

```
Class: InferenceService
  submit_job(source, farmer_uid, number_of_rows) -> job_id: str
    - Creates DB job record
    - Dispatches Celery task OR BackgroundTask (dev mode)
    - Returns job_id

  get_job_status(job_id) -> JobStatus
    - Reads inference_jobs table
    - Returns typed JobStatus dataclass

  get_results(limit, model_name) -> List[ResultRecord]
    - Reads candidate_result table
    - Returns typed list
```

**FastAPI routers** become thin adapters:
```python
# predict.py (after refactor)
service = InferenceService()

@router.post("/")
async def submit_prediction(item: PredictRequest, ...) -> JobAcceptedResponse:
    job_id = service.submit_job(item.source, item.farmer_uid, item.number_of_rows)
    return JobAcceptedResponse(job_id=job_id)
```

### 7.4 CLI Interface

Implement `backend/cli/main.py` using **Typer**:

```
lersha predict --source single --farmer-uid ETH-2024-001
lersha predict --source batch --rows 10
lersha results --limit 50 --model xgboost
lersha explain --job-id <uuid> --record 0 --model xgboost
lersha db init
lersha db populate-vectors
```

**CLI calls `InferenceService` directly** — no HTTP overhead for local use.

`Makefile` targets map to CLI commands:
```makefile
cli-predict-single:
    uv run python -m backend.cli.main predict --source single --farmer-uid $(UID)
```

### 7.5 Implementation Steps

1. Create `backend/services/inference_service.py` with `InferenceService` class.
2. Refactor `backend/api/routers/predict.py` to delegate to `InferenceService`.
3. Refactor `backend/api/routers/results.py` to delegate to `InferenceService`.
4. Install Typer: `typer[all]>=0.12.0`.
5. Create `backend/cli/main.py` with subcommands: `predict`, `results`, `explain`, `db`.
6. Add `[project.scripts]` entry in `pyproject.toml`: `lersha = "backend.cli.main:app"`.
7. Add Makefile targets for all CLI commands.
8. Ensure Streamlit pages continue to call via HTTP (no changes needed — already clean).

---

## 8. Phase 5 — Production System Design

### 8.1 Backend Architecture

#### 8.1.1 Engine Singleton (Critical Fix)

**Current problem:** `db_engine()` creates a new `Engine` instance on every call, bypassing connection pool sharing.

**Target:** Module-level singleton via `functools.lru_cache`:

```python
# backend/services/db_utils.py
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_engine() -> Engine:
    uri = config.db_uri
    if uri.startswith("postgresql"):
        return create_engine(uri, pool_size=10, max_overflow=20,
                             pool_pre_ping=True, pool_recycle=3600)
    return create_engine(uri)

def db_engine() -> Engine:
    return _get_engine()
```

This change is a critical P0 fix that must land before any load testing.

#### 8.1.2 API Versioning Strategy

- Current: `/v1/` prefix.
- Target: Dual-version coexistence (`/v1/`, `/v2/`) for ≥ 6 months during any breaking change.
- Version in response headers: `X-API-Version: 1`.
- Breaking change definition: any removal or type change of a required response field.
- Non-breaking additions are released in-place within the same version.

#### 8.1.3 Request Validation & Error Handling

**Current gaps:**
- 422 Unprocessable Entity responses are not structured consistently.
- No correlation between request ID and error response.

**Target error envelope:**
```json
{
  "error": "VALIDATION_ERROR",
  "message": "farmer_uid is required for Single Value prediction",
  "request_id": "req-uuid",
  "timestamp": "2026-03-29T21:00:00Z",
  "details": [...]
}
```

- Override FastAPI's default `RequestValidationError` handler in `main.py` to emit this envelope.
- `RequestIDMiddleware` (already present) propagates `X-Request-ID` header.
- All `HTTPException` responses must include `request_id` from middleware context.

### 8.2 Authentication & Authorization

#### 8.2.1 Current State

- Single API key (`X-API-Key` header) for all endpoints.
- No user identity, no scopes, no role separation.

#### 8.2.2 Target (Phase 5a — API Key Scopes)

- Multiple API keys with scopes: `predict:write`, `results:read`, `admin`.
- API keys stored in `api_keys` PostgreSQL table (hashed with bcrypt).
- `require_api_key` dependency upgraded to look up key from DB, validate scope.
- Key rotation: old key remains valid for 7 days after rotation, then invalidated.

**Migration 004:** `api_keys` table:
```sql
CREATE TABLE api_keys (
    id           SERIAL PRIMARY KEY,
    key_hash     VARCHAR(255) NOT NULL UNIQUE,
    name         VARCHAR(100) NOT NULL,       -- e.g. 'frontend-prod', 'cli-dev'
    scopes       TEXT[] NOT NULL,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ
);
```

#### 8.2.3 Target (Phase 5b — OAuth2 / OIDC, optional)

- If multi-user access is required: integrate Keycloak or Auth0.
- FastAPI `OAuth2PasswordBearer` dependency for JWT validation.
- Next.js: NextAuth.js with the same OIDC provider.
- Streamlit: not supported (deprecation timeline moves forward).

### 8.3 Logging, Monitoring & Observability

#### 8.3.1 Structured Logging

- Already implemented: `python-json-logger` in `backend/logger/`.
- **Gap:** No correlation of `request_id` into Celery task logs.
- **Fix:** Pass `job_id` through all log calls in `worker.py` and `pipeline.py`. Use `logging.LoggerAdapter` to inject `job_id` as a structured field.

#### 8.3.2 Metrics (Prometheus)

Add `prometheus-fastapi-instrumentator>=6.1.0`:
- Auto-instruments all HTTP routes: latency histogram, request counter, error rate.
- Expose `/metrics` endpoint (internal only, not routed through Caddy to public).
- Celery metrics: `celery-prometheus-exporter` or `flower` (for UI).

Key metrics to define and alert on:
| Metric | Alert Threshold |
|---|---|
| `http_request_duration_p99` | > 2s on `/v1/predict` |
| `celery_task_failure_rate` | > 5% over 5min |
| `rag_retrieval_latency_p95` | > 100ms |
| `db_pool_exhausted_count` | > 0 in 1min window |
| `gemini_api_error_rate` | > 10% in 5min |

#### 8.3.3 Distributed Tracing

- Add `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-instrumentation-sqlalchemy`.
- Export traces to Jaeger (dev) or Google Cloud Trace (prod).
- Celery task span: manually create a child span from the HTTP request's trace context (pass `traceparent` via job metadata in PostgreSQL).
- Every RAG call logged as a named span: `rag.retrieve`, `rag.generate`.

#### 8.3.4 Health Checks

Upgrade `/health` to return structured health:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "checks": {
    "database": "healthy",
    "redis": "healthy",
    "rag_store": "healthy"
  },
  "timestamp": "2026-03-29T21:00:00Z"
}
```

- Each check probes its dependency with a 1-second timeout.
- Returns `HTTP 503` if any critical check fails (`database`, `redis`).
- RAG check failure returns `HTTP 200` with `"rag_store": "degraded"` (non-critical).

### 8.4 Caching Strategy

| Layer | Mechanism | TTL | Key |
|---|---|---|---|
| RAG explanation | Redis | 24h | `sha256(prediction+shap+prompt_version)` |
| Results page | TanStack Query (client) | 60s | `results-{page}-{model}` |
| `/health` | In-process dict | 10s | N/A |
| Feature definitions | Application startup | App lifetime | N/A |

### 8.5 Asynchronous Processing

**Current state:** Celery + Redis — already implemented.

**Enhancements:**

1. **Dead Letter Queue (DLQ):** Configure Celery `task_reject_on_worker_lost=True` and a dedicated `DLQ` queue for tasks that fail after max retries. Route them to `celery_dlq` for manual inspection.

2. **Task prioritisation:** Define two queues — `inference.high` (single predictions, interactive) and `inference.low` (batch, background). Route `number_of_rows > 10` to the low-priority queue.

3. **Result TTL:** Set `result_expires = 86400` (24h) on Celery app config to auto-clean Redis result backend.

4. **Celery Beat:** Schedule a nightly job (`celery beat`) to:
   - Purge `inference_jobs` rows older than 30 days.
   - Re-index the pgvector IVFFlat index if document count changed by > 10%.

### 8.6 Deployment Architecture

#### 8.6.1 Environment Separation

| Environment | Compose files | Secrets | Workers |
|---|---|---|---|
| Dev | `base + override` | `.env` file | 1 (eager mode) |
| Staging | `base + prod` | Docker secrets | 2 |
| Production | `base + prod` | Docker secrets | 4 |

#### 8.6.2 Container Strategy

**No changes to Dockerfiles** for Phase 1–4. Phase 5 enhancements:

- Pin all base images to SHA256 digest in `docker-compose.prod.yml`.
- Add `HEALTHCHECK` instruction to `backend/Dockerfile` and `frontend/Dockerfile`.
- Separate `Dockerfile.worker` from `Dockerfile.backend` to enable independent ML stack vs. API stack updates.
- Use Docker BuildKit `--mount=type=cache` for `uv` and `npm` caches in CI.

#### 8.6.3 CI/CD Pipeline (GitHub Actions)

Existing `.github/workflows/` — **augment** with:

**On PR:**
1. `ruff check` + `mypy` (backend)
2. `pytest --cov=backend --cov-fail-under=80`
3. `eslint` + `tsc --noEmit` (frontend, Phase 3+)
4. `docker build` smoke test (both services)

**On merge to `main`:**
1. All PR checks above.
2. `docker buildx build --push` to registry for both images.
3. `docker compose -f docker-compose.yml -f docker-compose.prod.yml pull && ... up -d --no-deps backend worker` (rolling update — backend and worker only).
4. `alembic upgrade head` via init container or `backend` startup command.
5. Smoke test: `curl -f https://api.domain.com/health`.

**Rollback:** Tag every release image with `git rev-parse --short HEAD`. Rollback = re-deploy previous tag. Migrations are additive-only — no `downgrade` in production.

#### 8.6.4 Environment Variables Policy

All new environment variables must be:
1. Documented in `.env.example` with description and default.
2. Read via `config.py` — no raw `os.getenv()` outside config.
3. Marked as `# REQUIRED` or `# OPTIONAL (default: X)`.
4. Added to CI secrets if used in test runs.

New variables for this upgrade:
| Variable | Required | Default | Purpose |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | Yes | `http://localhost:8000` | Frontend → Backend URL |
| `PROMPT_VERSION` | No | `v1` | RAG prompt version |
| `RAG_SIMILARITY_THRESHOLD` | No | `0.75` | Min cosine similarity |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | — | OpenTelemetry endpoint |
| `CELERY_HIGH_CONCURRENCY` | No | `4` | Workers for high-priority queue |
| `CELERY_LOW_CONCURRENCY` | No | `2` | Workers for low-priority queue |

---

## 9. Dependency Map & Risk Register

### 9.1 Phase Dependencies

```
Phase 1 (pgvector)
  └── Phase 2 (RAG hardening) — requires pgvector schema
        └── Phase 3 (Next.js) — requires /v1/explain endpoint from Phase 2
              └── Phase 4 (CLI) — can run in parallel with Phase 3

Phase 5 (Production design) runs continuously alongside all phases.
P0 fix: db_engine singleton must land in Phase 1.
```

### 9.2 Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| pgvector extension not available on managed PostgreSQL | Low | High | Use `ankane/pgvector` Docker image or verify cloud support (AWS RDS, Supabase, Neon all support it) |
| IVFFlat index recall degrades with corpus growth | Medium | Medium | Monitor recall@5 in `rag_audit_log`; migrate to HNSW via a new migration at 500k docs |
| Gemini API quota exhaustion under batch load | Medium | High | Redis cache for explanations; implement circuit breaker pattern (already using Tenacity) |
| Next.js SSR and FastAPI CORS misconfiguration | Low | Medium | Configure `fastapi.middleware.cors` explicitly; scope `CORS_ORIGINS` env var |
| Model `.pkl` files use legacy `logic.*` module paths | Known | High | Already mitigated via `sys.modules` shim in `main.py`; remove after retraining |
| Celery worker OOM under large batch prediction | Medium | Medium | Set `--max-tasks-per-child=10` to recycle workers; add memory limit in compose |
| ChromaDB → pgvector embedding mismatch | Low | High | Use identical `SentenceTransformer` model and version; validate embeddings match in test |

### 9.3 Backward Compatibility

- **API contracts:** `/v1/predict`, `/v1/results` signatures must not change during Phases 1–4.
- **Streamlit UI:** `LershaAPIClient` requires no changes — interfaces only with `/v1/` HTTP endpoints.
- **Database:** All migrations are additive (new tables, new columns with defaults). No `DROP COLUMN` or type changes.
- **Model files:** `.pkl` compatibility shim in `main.py` remains until artifacts are retrained.
- **Config keys:** Removed keys (`CHROMA_DB_PATH`) must be kept as deprecated no-ops in `config.py` for one release cycle.

---

## 10. Migration Sequencing & Milestones

### 10.1 Recommended Execution Order

```
Week 1:  Phase 1 — pgvector migration
         ├── Day 1-2: Migration 003, ORM models, populate script
         ├── Day 3-4: Rewrite rag_engine.py retrieve layer
         └── Day 5:   Integration tests, remove ChromaDB, validate

Week 2:  Phase 2 — RAG hardening
         ├── Day 1-2: RagService class, Redis cache, audit log
         ├── Day 3:   /v1/explain endpoint
         └── Day 4-5: Prompt versioning, unit + integration tests

Week 3-4: Phase 3 — Next.js frontend
         ├── Day 1:   Scaffold, configure, Dockerfile
         ├── Day 2-3: API client, Zustand stores, types
         ├── Day 4-6: Pages: settings → predict → results → detail
         └── Day 7-8: Tests, Caddy routing, compose integration

Week 5:  Phase 4 — Multi-interface standardisation
         ├── Day 1-2: InferenceService refactor
         └── Day 3-4: Typer CLI, Makefile targets

Ongoing: Phase 5 — Production hardening
         ├── Sprint 1: db_engine singleton fix, error envelope, health check upgrade
         ├── Sprint 2: Prometheus metrics, structured health
         ├── Sprint 3: OpenTelemetry tracing
         └── Sprint 4: API key scopes, DLQ, Celery Beat
```

### 10.2 Definition of Done per Phase

| Phase | Done When |
|---|---|
| Phase 1 | `rag_documents` populated, retrieval latency < 50ms, ChromaDB volume removed, all tests pass |
| Phase 2 | `/v1/explain` returns non-empty text, `rag_audit_log` has entries, cache hit rate > 80% on repeat calls |
| Phase 3 | Full prediction round-trip works in Next.js, Streamlit still functional, Caddy routes both |
| Phase 4 | `lersha predict --source single --farmer-uid X` completes, all routers delegate to `InferenceService` |
| Phase 5 | `/metrics` endpoint live, P99 latency < 2s on load test, API key scopes enforced |

### 10.3 Deprecation Timeline

| Component | Deprecation Announced | Removed |
|---|---|---|
| ChromaDB | Phase 1 start | Phase 1 end |
| `chroma_data` Docker volume | Phase 1 end | Phase 2 start |
| `CHROMA_DB_PATH` env var | Phase 1 | Phase 3 |
| Streamlit UI (`ui/`) | Phase 3 start | Phase 4 end |
| `ui` Docker service | Phase 3 GA | Phase 4 GA |
| `chroma_data` volume reference in compose | Phase 1 end | Phase 1 end |

---

*Document maintained by the architecture team. All pull requests implementing work from this plan must reference the corresponding phase and step number in their description.*
