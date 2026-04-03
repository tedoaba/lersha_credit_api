# Tasks: Migrate Streamlit UI to Next.js 14 Frontend

**Input**: Design documents from `specs/008-nextjs-frontend-migration/`  
**Branch**: `008-nextjs-frontend-migration`  
**Prerequisites**: plan.md ✅ spec.md ✅ research.md ✅ data-model.md ✅ contracts/ ✅ quickstart.md ✅

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.  
**Format**: `- [ ] [TaskID] [P?] [StoryLabel?] Description — file/path`

---

## Phase 1: Setup (Project Scaffolding)

**Purpose**: Bootstrap the `frontend/` project, install all dependencies, and establish the baseline configuration so every subsequent phase can proceed.

- [ ] T001 Bootstrap Next.js 14 App Router project in `frontend/` using `npx create-next-app@14` with TypeScript, Tailwind CSS, ESLint, and `--import-alias "@/*"` — `frontend/`
- [ ] T002 [P] Install server-state and client-state dependencies: `@tanstack/react-query@^5`, `zustand@^4` — `frontend/package.json`
- [ ] T003 [P] Install chart and UI dependencies: `recharts@^2`, then run `npx shadcn-ui@latest init` (New York style, slate base) and add components `button badge card table select input label separator` — `frontend/package.json` + `frontend/components/ui/`
- [ ] T004 Configure `frontend/next.config.js` — set `output: 'standalone'` and expose `NEXT_PUBLIC_API_BASE_URL` env — `frontend/next.config.js`
- [ ] T005 [P] Create `frontend/.env.local` with `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` and document in `frontend/.env.local.example` — `frontend/.env.local.example`
- [ ] T006 [P] Verify `npm run build` and `npm run dev` succeed on the empty scaffold (zero custom code pages yet) — `frontend/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core TypeScript types, HTTP client, Zustand stores, and TanStack Query hooks — every user story page depends on these. Must be complete before Phase 3+.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T007 Write TypeScript interfaces mirroring all backend Pydantic models in `data-model.md`: `FeatureContribution`, `PredictRequest`, `PredictSource`, `JobAcceptedResponse`, `JobStatus`, `TERMINAL_STATUSES`, `isTerminalStatus()`, `EvaluationRecord`, `ModelResult`, `JobStatusResponse`, `ResultsRecord`, `ResultsResponse`, `ExplainRequest`, `ExplainResponse` — `frontend/lib/types.ts`
- [ ] T008 Implement `LershaClient` class with typed `fetch`-based methods: `submitPrediction`, `getJobStatus`, `getResults`, `getExplanation`, `health`; reads `apiKey` from constructor parameter; sets `X-API-Key` header on every request; throws typed errors on non-2xx responses — `frontend/lib/api.ts`
- [ ] T009 [P] Implement `useApiKeyStore` Zustand store (persisted to `localStorage` key `"lersha-api-key"` via `zustand/middleware` `persist`) and `useJobStore` (session memory, no persistence) — `frontend/lib/stores.ts`
- [ ] T010 [P] Implement TanStack Query hooks: `useJobStatus(jobId)` with `refetchInterval` callback stopping on terminal status; `useResults(limit?, modelName?)` with 60s stale time; `useExplanation(req)` keyed by `(job_id, record_index, model_name)`; singleton `queryClient` export — `frontend/lib/queries.ts`
- [ ] T011 Create root layout with `QueryClientProvider` wrapping, HTML shell (`lang="en"`), global `NavBar` component with links to Dashboard `/`, Predict `/predict`, Results `/results`, Settings `/settings`, and Tailwind `globals.css` import — `frontend/app/layout.tsx` + `frontend/app/globals.css`

**Checkpoint**: TypeScript compiles (`tsc --noEmit`), `npm run dev` starts, NavBar renders on all routes. All stores and hooks are importable.

---

## Phase 3: User Story 4 — Configure API Key (Priority: P4) 🛠 First Build

> **Why implement US4 first?** Settings / API key storage is a zero-dependency leaf feature and the prerequisite for all other pages to authenticate. Completing it first means every subsequent test can supply a real key.

**Goal**: User enters and persists their API key via the Settings page; all subsequent API calls include it automatically.

**Independent Test**: Navigate to `/settings`, enter a key, refresh the browser, confirm the key still appears and a test `GET /health` call includes `X-API-Key`.

- [ ] T012 [US4] Build `app/settings/page.tsx` — client component (`"use client"`); reads `apiKey` from `useApiKeyStore`; renders an `<Input>` bound to local state, a `<Button>` that calls `setApiKey()`, and a confirmation toast on save; shows masked key if already stored — `frontend/app/settings/page.tsx`
- [ ] T013 [P] [US4] Build health-check utility that calls `lershaClient.health()` using the current stored key and displays backend status on the Settings page — `frontend/app/settings/page.tsx` (add to same file as T012)
- [ ] T014 [P] [US4] Verify `LershaClient` is instantiated with `apiKeyStore.apiKey` at call-time (not at module init) so key changes take effect without page reload — `frontend/lib/api.ts` (pattern validation, no new file)

**Checkpoint**: API key survives browser refresh (localStorage); health check returns `ok` with stored key; health check returns `401` error state with empty key.

---

## Phase 4: User Story 1 — Submit a Credit Prediction (Priority: P1) 🎯 MVP

**Goal**: User selects a data source, submits a prediction job, receives a job ID, and sees a live status badge updating without page refresh.

**Independent Test**: Open `/predict`, select "Batch Prediction" + set rows to 5, click Submit, verify job ID returned and status badge transitions from `pending` → `processing` → `completed` (or `failed`) automatically.

### Implementation for User Story 1

- [ ] T015 [US1] Build `JobStatusBadge` component — renders a colour-coded badge for each of `pending` (grey), `processing` (blue/spinner), `completed` (green), `failed` (red); accepts `status: JobStatus` prop — `frontend/components/JobStatusBadge.tsx`
- [ ] T016 [US1] Build `PredictionForm` component — controlled form with a `<Select>` for source (`"Single Value"` / `"Batch Prediction"`), conditional `<Input>` for `farmer_uid` (Single Value) or `<Input type="number">` for `number_of_rows` (Batch), and a `<Button>` that calls `submitPrediction`; on success sets `useJobStore.activeJobId`; on error shows inline error message and preserves form state — `frontend/components/PredictionForm.tsx`
- [ ] T017 [US1] Build `app/predict/page.tsx` — client component; renders `PredictionForm`; after job submission renders a status section with `JobStatusBadge` driven by `useJobStatus(activeJobId)` polling hook; shows spinner while pending/processing; shows "View Results" button link to `/results` when completed — `frontend/app/predict/page.tsx`
- [ ] T018 [US1] Add network error handling to `PredictionForm` — catch fetch errors and display a user-facing error card below the form; form fields remain populated on error (FR-009, spec acceptance scenario 3) — `frontend/components/PredictionForm.tsx`

**Checkpoint**: End-to-end submit flow works. Status badge counts down to `completed` or `failed` with no manual refresh. Polling stops automatically. Network tab confirms poll requests stop after terminal state.

---

## Phase 5: User Story 2 — Browse and Inspect Prediction Results (Priority: P2)

**Goal**: After a job completes, user browses a results table and clicks into an individual farmer's detail view showing score, SHAP chart, and AI explanation.

**Independent Test**: Navigate to `/results` with a completed job in the DB; table renders all rows; click one row; detail page shows score, feature contribution chart, and explanation panel rendered from live API data.

### Implementation for User Story 2

- [ ] T019 [P] [US2] Build `EvaluationCard` component — displays farmer name (or UID if name absent), `predicted_class_name` as a prominent badge, `model_name`, and formatted `timestamp`; accepts a `ResultsRecord` prop — `frontend/components/EvaluationCard.tsx`
- [ ] T020 [P] [US2] Build `FeatureContribChart` component — Recharts `BarChart` (horizontal) from a `FeatureContribution[]` prop; bars sorted by absolute value descending; positive values coloured amber (risk) and negative values coloured teal (protective); responsive container — `frontend/components/FeatureContribChart.tsx`
- [ ] T021 [P] [US2] Build `ExplanationPanel` component — displays AI explanation text in a card; shows `retrieved_doc_ids` as a badge list; shows `cache_hit` indicator; shows `prompt_version` and `latency_ms` in metadata row — `frontend/components/ExplanationPanel.tsx`
- [ ] T022 [US2] Build `app/results/page.tsx` — client component; calls `useResults()` hook; renders a `<Table>` (shadcn) with columns: farmer UID/name, model, predicted class, timestamp; each row links to `/results/[id]` where `id` encodes `job_id` + `record_index` + `model_name` as a URL-safe slug or query params; shows total count badge; shows empty-state card when `total === 0` — `frontend/app/results/page.tsx`
- [ ] T023 [US2] Build `app/results/[id]/page.tsx` — client component; parses `[id]` param to extract `job_id`, `record_index`, `model_name`; calls `useJobStatus(job_id)` to get evaluation record; calls `useExplanation()` for AI explanation; renders `EvaluationCard` + `FeatureContribChart` + `ExplanationPanel` stacked; shows skeleton loaders while fetching — `frontend/app/results/[id]/page.tsx`
- [ ] T024 [US2] Add 404 handling in `app/results/[id]/page.tsx` — if job not found or `record_index` out of range, show a friendly "Result not found" card with a Back link; handle `503` explanation errors with a "Explanation unavailable" fallback panel — `frontend/app/results/[id]/page.tsx`

**Checkpoint**: Full P2 flow works: results table → row click → detail page with SHAP chart and explanation text. All three components render with real data.

---

## Phase 6: User Story 3 — Monitor the Dashboard (Priority: P3)

**Goal**: Team lead opens the dashboard and sees job state totals and a recent activity feed without any manual submission.

**Independent Test**: Navigate to `/`; verify summary tiles show numeric counts for pending/completed/failed jobs; counts reflect current DB state without page manual refresh (ISR within 60s).

### Implementation for User Story 3

- [ ] T025 [US3] Build `app/page.tsx` (dashboard) — server component with `revalidate = 60` (ISR); fetches results via `LershaClient.getResults()` server-side; derives counts (`pending`, `completed`, `failed`) from the `records` array; renders three summary `<Card>` tiles for each count; renders a recent-activity table (last 10 results by timestamp) — `frontend/app/page.tsx`
- [ ] T026 [P] [US3] Add a server-side `getServerSideJobSummary()` helper that calls the backend results endpoint from the Next.js server layer (not the browser), ensuring ISR cache is used and no `X-API-Key` exposure in browser network logs for this unauthenticated-friendly dashboard view — `frontend/lib/server-api.ts`

**Checkpoint**: Dashboard loads at `/`; summary tiles show real counts; refreshing within 60s serves cached data (no backend hit visible in logs); after 60s a fresh request is sent.

---

## Phase 7: Error Handling & Polish (Cross-Cutting)

**Purpose**: Error boundaries, loading states, empty states, and infrastructure files that affect all user stories.

- [ ] T027 [P] Add per-route `loading.tsx` skeleton screens for the results list and farmer detail page — `frontend/app/results/loading.tsx`, `frontend/app/results/[id]/loading.tsx`
- [ ] T028 [P] Add per-route `error.tsx` boundaries for Predict, Results, and Detail pages — each shows a friendly error card with a Retry button — `frontend/app/predict/error.tsx`, `frontend/app/results/error.tsx`, `frontend/app/results/[id]/error.tsx`
- [ ] T029 Add global 401 intercept in `LershaClient` — on any `401 Unauthorized` response, set an error flag in `useApiKeyStore`; NavBar shows a warning badge prompting user to check Settings — `frontend/lib/api.ts` + `frontend/lib/stores.ts`
- [ ] T030 [P] Add `not-found.tsx` for unknown routes — renders a "Page not found" card with a link to the Dashboard — `frontend/app/not-found.tsx`
- [ ] T031 [P] Add empty-state components for the results table (no records yet), dashboard (no jobs), and explanation panel (explanation not available) — inline in respective page files

---

## Phase 8: Containerisation & Infrastructure

**Purpose**: Docker, docker-compose, Caddyfile, and Makefile changes needed to deploy the frontend as a service.

- [ ] T032 Write `frontend/Dockerfile` — multi-stage build: Stage 1 `node:18-alpine` (`deps`) installs production deps; Stage 2 (`builder`) runs `npm run build`; Stage 3 (`runner`) copies `.next/standalone` only; sets `NODE_ENV=production`; `CMD ["node", "server.js"]`; exposes port 3000 — `frontend/Dockerfile`
- [ ] T033 Add `frontend` service to `docker-compose.yml` — `build: { context: ., dockerfile: frontend/Dockerfile }`, `restart: unless-stopped`, `depends_on: [backend]`; no hardcoded ports in base file — `docker-compose.yml`
- [ ] T034 [P] Add `frontend` override to `docker-compose.override.yml` — bind mount `./frontend:/app` (exclude `node_modules`), expose `3000:3000` for dev — `docker-compose.override.yml`
- [ ] T035 Update `Caddyfile` — replace `reverse_proxy /* ui:8501` with `reverse_proxy /* frontend:3000`; keep `reverse_proxy /v1/* backend:8000` as the first (priority) directive — `Caddyfile`
- [ ] T036 [P] Add Makefile targets: `frontend-dev` (`npm run dev` in `frontend/`), `frontend-build` (`npm run build` in `frontend/`), `frontend-up` (`docker compose up frontend`) — `Makefile`
- [ ] T037 [P] Add `frontend-build` CI job to `.github/workflows/ci.yml` — step: `cd frontend && npm ci && npm run build`; runs after `lint` and `test` jobs pass — `.github/workflows/ci.yml`

---

## Phase 9: Validation

**Purpose**: Verify the complete frontend against the spec acceptance criteria and quickstart checklist before marking the feature done.

- [ ] T038 Run `npm run build` inside `frontend/` and confirm zero TypeScript errors and zero ESLint errors (SC-008) — `frontend/`
- [ ] T039 [P] Manual validation: open `/settings`, enter a valid API key, refresh browser, confirm key persists (SC-004) — browser
- [ ] T040 [P] Manual validation: open `/predict`, submit Batch Prediction (5 rows), confirm job ID returned within 3s and status badge updates to `processing` then `completed` without page reload (SC-001, SC-002) — browser + Network tab
- [ ] T041 [P] Manual validation: open Network tab while polling — confirm poll requests stop after `completed`/`failed` status (FR-011) — browser DevTools
- [ ] T042 [P] Manual validation: open `/results`, confirm table renders; click a row; confirm detail page loads within 3s with SHAP chart and explanation text (SC-003) — browser
- [ ] T043 [P] Manual validation: docker compose build + up frontend; confirm container starts within 60s and `/` is reachable via Caddy (SC-005) — terminal
- [ ] T044 [P] Manual validation: resize browser to 768px viewport width; confirm all pages remain usable (SC-006) — browser DevTools device toolbar

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **blocks all user story phases**
- **Phase 3 (US4 — Settings)**: Depends on Phase 2 — delivers API key storage
- **Phase 4 (US1 — Predict)**: Depends on Phase 2 + Phase 3 (needs stored key for test) — MVP
- **Phase 5 (US2 — Results)**: Depends on Phase 2 + Phase 4 (needs a completed job to test with)
- **Phase 6 (US3 — Dashboard)**: Depends on Phase 2 — can run in parallel with US1/US2
- **Phase 7 (Polish)**: Depends on Phases 3–6 all complete
- **Phase 8 (Infrastructure)**: Can begin after Phase 1 — Dockerfile/compose independent of page code
- **Phase 9 (Validation)**: Depends on Phases 1–8 all complete

### User Story Dependencies

| Story | Phase | Depends On |
|-------|-------|------------|
| US4 — Settings (P4) | Phase 3 | Foundational only |
| US1 — Predict (P1) | Phase 4 | Foundational + US4 (key needed) |
| US2 — Results (P2) | Phase 5 | Foundational + US1 (job needed for test) |
| US3 — Dashboard (P3) | Phase 6 | Foundational only (parallel with US1) |

### Within Each Phase

- `types.ts` (T007) before `api.ts` (T008) — types are imported by client
- `api.ts` (T008) before `queries.ts` (T010) — queries use `LershaClient`
- `stores.ts` (T009) before any page — pages read from stores
- Components (T015, T019–T021) before pages that use them (T016–T017, T022–T023)
- `loading.tsx` and `error.tsx` (T027–T028) are independent of page logic

---

## Parallel Opportunities

### Phase 1 — can be parallelised

```
T001 (bootstrap)
  ├── T002 [P] install query/state deps
  ├── T003 [P] install UI/chart deps
  ├── T004 next.config.js
  ├── T005 [P] .env.local
  └── T006 [P] verify build
```

### Phase 2 — partial parallelism

```
T007 (types.ts — first, no deps)
  └── T008 (api.ts — after T007)
      ├── T009 [P] stores.ts
      └── T010 [P] queries.ts
T011 (layout.tsx — after T007, T009)
```

### Phase 5 (US2) — heavy parallelism

```
T019 [P] EvaluationCard.tsx
T020 [P] FeatureContribChart.tsx       ← all three can run in parallel
T021 [P] ExplanationPanel.tsx
  └── T022 results/page.tsx (after T019)
  └── T023 results/[id]/page.tsx (after T019, T020, T021)
  └── T024 404 handling (after T023)
```

### Phase 8 — fully parallelisable after Phase 1

```
T032 Dockerfile
T033 docker-compose.yml
T034 [P] docker-compose.override.yml
T035 [P] Caddyfile
T036 [P] Makefile targets
T037 [P] CI job
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 — Setup
2. Complete Phase 2 — Foundational (types, client, stores, queries, layout)
3. Complete Phase 3 — US4 Settings (API key — needed for testing US1)
4. Complete Phase 4 — US1 Predict form + status polling
5. **STOP and VALIDATE** — run T038 + T039 + T040 + T041
6. Deploy or demo this increment — core value delivered

### Incremental Delivery

| Increment | Phases | Value Delivered |
|-----------|--------|-----------------|
| 1 — MVP | 1 + 2 + 3 + 4 | Submit prediction, live status polling |
| 2 — Results | + 5 | Browse + inspect results with SHAP chart |
| 3 — Dashboard | + 6 | Operational monitoring |
| 4 — Production | + 7 + 8 + 9 | Error resilience, Docker, Caddy, CI |

### Parallel Team Strategy (2 developers)

After Phase 2 completes:
- **Dev A**: Phase 4 (US1 — Predict) → Phase 5 (US2 — Results) — sequential
- **Dev B**: Phase 3 (US4 — Settings) → Phase 6 (US3 — Dashboard) → Phase 8 (Infrastructure)

---

## Task Summary

| Phase | Tasks | Parallelisable | User Story |
|-------|-------|----------------|------------|
| Phase 1 — Setup | T001–T006 | T002, T003, T005, T006 | — |
| Phase 2 — Foundational | T007–T011 | T009, T010 | — |
| Phase 3 — US4 Settings | T012–T014 | T013, T014 | US4 |
| Phase 4 — US1 Predict | T015–T018 | — | US1 |
| Phase 5 — US2 Results | T019–T024 | T019, T020, T021 | US2 |
| Phase 6 — US3 Dashboard | T025–T026 | T026 | US3 |
| Phase 7 — Polish | T027–T031 | T027, T028, T030, T031 | — |
| Phase 8 — Infrastructure | T032–T037 | T034, T035, T036, T037 | — |
| Phase 9 — Validation | T038–T044 | T039–T044 | — |
| **Total** | **44 tasks** | **23 parallelisable** | |

---

## Notes

- `[P]` tasks operate on different files with no blocking dependencies — safe to run simultaneously
- `[US_]` label traces each task to its spec.md user story for full traceability
- Phase 8 (Infrastructure) can be started by a second developer as soon as Phase 1 finishes — Dockerfile and docker-compose do not depend on any TypeScript code
- `npm run build` (T038) is the primary integration gate — if it passes, all TypeScript types and imports are valid
- The Streamlit `ui/` directory is **not touched** by any task in this list — it remains operational throughout the migration
- Run `quickstart.md §10` validation checklist after T044 before closing the branch
