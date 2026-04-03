# Implementation Plan: Migrate Streamlit UI to Next.js 14 Frontend

**Branch**: `008-nextjs-frontend-migration` | **Date**: 2026-04-03 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `specs/008-nextjs-frontend-migration/spec.md`

---

## Summary

Replace the existing Streamlit UI (`ui/`) with a production-grade Next.js 14 frontend (`frontend/`) using the App Router, Zustand for persistent client state, TanStack Query for server state and live job polling, and shadcn/ui + Tailwind CSS for components. The frontend connects exclusively to the existing FastAPI backend via the `/v1/` REST API layer, mirrors all backend Pydantic schemas as TypeScript interfaces, and is containerised and integrated into the existing Docker Compose + Caddy reverse proxy stack.

---

## Technical Context

**Language/Version**: TypeScript 5 / Node.js 18 LTS  
**Framework**: Next.js 14 (App Router, React 18)  
**Primary Dependencies**: `@tanstack/react-query@^5`, `zustand@^4`, `recharts@^2`, `shadcn/ui`, `tailwindcss@^3`  
**Storage**: `localStorage` (API key persistence via Zustand persist middleware); no new database tables  
**Testing**: Vitest + React Testing Library (unit); Playwright (E2E smoke); `npm run build` as primary integration gate  
**Target Platform**: Browser (desktop/tablet ≥ 768px), containerised on Linux (Docker, Node 18 Alpine)  
**Project Type**: Web application (new frontend service within existing monorepo)  
**Performance Goals**: Page interactive < 3s (LCP), status update visible ≤ 5s of state change, container cold start ≤ 60s  
**Constraints**: No backend schema changes; API key in localStorage only (no server-side sessions); no mobile-first layout required for v1  
**Scale/Scope**: Single-tenant internal tool; small batch sizes (≤ 100 farmers per job); 1–5 concurrent users

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Rule | Status | Justification / Action |
|-----------|------|--------|------------------------|
| **P1 — MODULAR** | `ui/` and `frontend/` MUST communicate with backend exclusively via HTTP API | ✅ PASS | `LershaClient` in `frontend/lib/api.ts` is the sole HTTP layer; no imports from `backend/` |
| **P1 — MODULAR** | Monorepo layout: new `frontend/` MUST be a peer of `backend/` | ✅ PASS | `frontend/` added at repo root, no coupling to `backend/` internals |
| **P6 — API-FIRST** | Frontend uses `/v1/` versioned routes exclusively | ✅ PASS | All `LershaClient` methods target `/v1/predict`, `/v1/results`, `/v1/explain` |
| **P9 — SECURITY** | API key passed in `X-API-Key` header; never in source code or logs | ✅ PASS | Key stored in `localStorage` via Zustand persist; injected per-request in `LershaClient` |
| **P9 — SECURITY** | Production traffic terminated at reverse proxy (Caddy); no direct port exposure | ✅ PASS | Caddy routes `/*` → `frontend:3000` and `/v1/*` → `backend:8000` |
| **P11 — CONTAINER** | Frontend defined as a Docker service in `docker-compose.yml` | ✅ PASS | `frontend/Dockerfile` + `docker-compose.yml` `frontend` service added |
| **P11 — CONTAINER** | No manual setup steps outside Makefile / Dockerfile | ✅ PASS | `make frontend-dev`, `make frontend-build` targets added |
| **P12 — CI** | `npm run build` must pass without errors as a CI gate | **⚠ NEW GATE** | CI job `frontend-build` added to `.github/workflows/ci.yml` |
| **P2 — PEP** | N/A (TypeScript, not Python) | ➖ N/A | ESLint + `tsc --noEmit` enforce equivalent standards for the frontend |
| **P7 — TEST** | 80% coverage gate applies to `backend/` only | ➖ N/A for coverage % | Vitest unit tests required for `lib/api.ts`, `lib/stores.ts`, `lib/queries.ts`; Playwright smoke for critical path |

**Constitution Check Result**: ✅ PASS — No violations. All integration points align with existing architectural principles.

---

## Project Structure

### Documentation (this feature)

```text
specs/008-nextjs-frontend-migration/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 — technology decisions
├── data-model.md        # Phase 1 — TypeScript interfaces + entity model
├── quickstart.md        # Phase 1 — developer onboarding guide
├── contracts/
│   └── api-contracts.md # Phase 1 — REST endpoint contracts
└── tasks.md             # Phase 2 output (created by /speckit-tasks)
```

### Source Code (repository root)

```text
frontend/                              # NEW — Next.js 14 frontend
├── app/
│   ├── layout.tsx                     # Root layout: QueryClientProvider, NavBar
│   ├── page.tsx                       # Dashboard — ISR, revalidate: 60
│   ├── predict/
│   │   └── page.tsx                   # Prediction form — "use client"
│   ├── results/
│   │   ├── page.tsx                   # Results table — "use client"
│   │   └── [id]/
│   │       └── page.tsx               # Farmer detail view — "use client"
│   └── settings/
│       └── page.tsx                   # API key settings — "use client"
├── components/
│   ├── PredictionForm.tsx
│   ├── JobStatusBadge.tsx
│   ├── FeatureContribChart.tsx        # Recharts bar chart for SHAP values
│   ├── ExplanationPanel.tsx           # RAG text + doc metadata
│   └── EvaluationCard.tsx             # Single farmer result card
├── lib/
│   ├── api.ts                         # LershaClient class (typed fetch wrapper)
│   ├── stores.ts                      # Zustand: useApiKeyStore, useJobStore
│   ├── queries.ts                     # TanStack Query hooks
│   └── types.ts                       # TypeScript interfaces (mirrors schemas.py)
├── Dockerfile                         # Multi-stage, node:18-alpine, output: standalone
├── next.config.js                     # output: 'standalone'
├── tailwind.config.ts
├── tsconfig.json
└── package.json

# MODIFIED files in existing layout:
docker-compose.yml                     # Add `frontend` service block
docker-compose.override.yml            # Add bind mount + port 3000 for dev
Caddyfile                              # Route /* → frontend:3000
Makefile                               # Add frontend targets
```

**Structure Decision**: Web application layout (Option 2). `frontend/` is a peer of `backend/` and `ui/` at the repo root. The Streamlit `ui/` remains unchanged during the migration phase and is decommissioned once the new frontend is validated in production. No files are moved or deleted from `ui/` in this feature.

---

## Complexity Tracking

> No Constitution Check violations requiring justification.

---

## Implementation Phases

### Phase 1 — Foundation (Scaffolding & Core Infrastructure)

1. Create `frontend/` directory; bootstrap with `create-next-app@14` (TypeScript, Tailwind, App Router, no Git)
2. Install: `@tanstack/react-query@^5`, `zustand@^4`, `recharts@^2`, shadcn/ui (`button`, `badge`, `card`, `table`, `select`, `input`)
3. Write `frontend/lib/types.ts` — all TypeScript interfaces derived from `backend/api/schemas.py`
4. Write `frontend/lib/api.ts` — `LershaClient` class with methods: `submitPrediction`, `getJobStatus`, `getResults`, `getExplanation`, `health`
5. Write `frontend/lib/stores.ts` — `useApiKeyStore` (localStorage persist) + `useJobStore`
6. Write `frontend/lib/queries.ts` — TanStack Query hooks: `useJobStatus` (with polling), `useResults`, `useExplanation`
7. Write `frontend/app/layout.tsx` — root layout with `QueryClientProvider`, persistent `NavBar` component

**Phase 1 output**: Runnable `npm run dev` with no page content yet; types/client/stores verified by TypeScript compilation.

---

### Phase 2 — Pages & Components

1. `app/settings/page.tsx` + Settings form component — API key input, save to `useApiKeyStore`
2. `app/predict/page.tsx` + `PredictionForm.tsx` — source selector (Single Value / Batch), farmer_uid or number_of_rows conditional fields, submit button, `JobStatusBadge` live status display
3. `components/JobStatusBadge.tsx` — colour-coded badge for 4 status values
4. `app/results/page.tsx` — results table using `useResults`, sortable columns, result count badge
5. `components/EvaluationCard.tsx` — farmer result summary card (name, score, tier)
6. `components/FeatureContribChart.tsx` — Recharts `BarChart` from `top_feature_contributions` — horizontal, sorted by absolute value, colour-coded positive/negative
7. `components/ExplanationPanel.tsx` — explanation text block + retrieved doc IDs + cache badge + prompt version
8. `app/results/[id]/page.tsx` — farmer detail page combining `EvaluationCard`, `FeatureContribChart`, `ExplanationPanel`
9. `app/page.tsx` (dashboard) — job state summary tiles (pending / completed / failed counts), recent activity table, ISR `revalidate: 60`

**Phase 2 output**: All pages navigable via nav bar; full prediction → polling → results → detail workflow functional.

---

### Phase 3 — Error Handling & Edge Cases

1. Global error boundaries per route segment (`error.tsx` files) — user-facing error messages (FR-009)
2. Loading skeletons (`loading.tsx` files) for results table and detail page
3. Empty state components — zero results, job not found, explanation unavailable
4. 404 handling for unknown `job_id` in `results/[id]/page.tsx`
5. Missing / invalid API key guard — redirect to Settings with toast message on 401 responses
6. Network error toast on submission failure — form state preserved (FR spec acceptance scenario 3)

---

### Phase 4 — Containerisation & Infrastructure

1. `frontend/Dockerfile` — multi-stage: `node:18-alpine` builder + runner; `output: standalone` for minimal image
2. `frontend/next.config.js` — `output: 'standalone'`; `env.NEXT_PUBLIC_API_BASE_URL`
3. `docker-compose.yml` — add `frontend` service: `build: frontend/`, `depends_on: backend`, port binding in override
4. `docker-compose.override.yml` — add `frontend` bind mount (`./frontend:/app`) and `port: 3000`
5. `Caddyfile` — update catch-all: `reverse_proxy /* frontend:3000` (keep `/v1/*` → `backend:8000` first)
6. `Makefile` — add targets: `frontend-dev`, `frontend-build`, `frontend-up`
7. CI: `.github/workflows/ci.yml` — add `frontend-build` job (`npm ci && npm run build`) after lint/test

---

### Phase 5 — Testing & Validation

1. Unit tests — Vitest + React Testing Library:
   - `lib/api.ts`: mock `fetch`, verify all 5 methods, `X-API-Key` header presence
   - `lib/stores.ts`: `useApiKeyStore` localStorage persistence, `useJobStore` set/clear
   - `components/JobStatusBadge.tsx`: renders correct colour for each status
   - `components/FeatureContribChart.tsx`: renders bars from fixture data
2. Playwright E2E smoke (against running stack):
   - Settings → enter key → reload → key present
   - Predict → Batch submit → status badge transitions to `processing` → eventually `completed`
   - Results table → click row → detail page loads with chart and explanation
3. `npm run build` — zero TypeScript + ESLint errors (CI gate)
4. Manual validation checklist from `quickstart.md §10`

---

## Architecture Diagram

```
Browser
  │  (HTTP via Caddy)
  │
  ├── /*           → frontend:3000 (Next.js)
  │     ├── app/page.tsx          (ISR, 60s)
  │     ├── app/predict/page.tsx  (CSR)
  │     ├── app/results/**        (CSR)
  │     └── app/settings/page.tsx (CSR)
  │
  └── /v1/*        → backend:8000 (FastAPI)
        ├── POST /v1/predict/      (submit job)
        ├── GET  /v1/predict/{id}  (poll status)
        ├── GET  /v1/results/      (list all)
        └── POST /v1/explain/      (AI explanation)

Client State (browser only):
  Zustand useApiKeyStore → localStorage "lersha-api-key"
  Zustand useJobStore    → session memory (active job ID)

Server State (TanStack Query):
  useJobStatus(jobId) → polling every 2s → stops on terminal
  useResults()        → cached, manual invalidate on new job
  useExplanation()    → cached, keyed by (job_id, index, model)
```
