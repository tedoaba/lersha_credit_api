# Tasks: Migrate Streamlit UI to Next.js 16 Frontend

**Input**: Design documents from `specs/008-nextjs-frontend-migration/`  
**Branch**: `008-nextjs-frontend-migration`  
**Prerequisites**: plan.md ✅ spec.md ✅ research.md ✅ data-model.md ✅ contracts/ ✅ quickstart.md ✅

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.  
**Format**: `- [x] done  - [ ] pending`

---

## Phase 1: Setup ✅

- [x] T001 Bootstrap Next.js 16.2.2 with TypeScript, Tailwind v4, ESLint, App Router — `frontend/`
- [x] T002 [P] Install `@tanstack/react-query@^5`, `zustand@^4`, `recharts@^2` — `frontend/package.json`
- [x] T003 [P] Init shadcn@4.1.2 (Tailwind v4 mode) + add button badge card table select input label separator — `frontend/components/ui/`
- [x] T004 Configure `next.config.ts` — `output: 'standalone'`, `NEXT_PUBLIC_API_BASE_URL` — `frontend/next.config.ts`
- [x] T005 [P] Create `.env.local` + `.env.local.example` — `frontend/.env.local`
- [x] T006 [P] Verify scaffold builds — `frontend/`

---

## Phase 2: Foundational ✅

- [x] T007 Write all TypeScript interfaces from `data-model.md` — `frontend/lib/types.ts`
- [x] T008 Implement `LershaClient` class (fetch, X-API-Key, typed methods) — `frontend/lib/api.ts`
- [x] T009 [P] Implement `useApiKeyStore` (localStorage persist) + `useJobStore` — `frontend/lib/stores.ts`
- [x] T010 [P] Implement TanStack Query hooks: `useJobStatus`, `useResults`, `useExplanation` — `frontend/lib/queries.ts`
- [x] T011 Root layout: `Providers` (QueryClientProvider) + `NavBar` + metadata — `frontend/app/layout.tsx`

**Checkpoint** ✅ — `npm run build` passes, all 7 routes compiled.

---

## Phase 3: US4 — Configure API Key ✅

- [x] T012 [US4] Settings page (API key input + localStorage + masked display) — `frontend/app/settings/page.tsx`
- [x] T013 [P] [US4] Backend health check widget on settings page — `frontend/app/settings/page.tsx`
- [x] T014 [P] [US4] Key passed per-call in `LershaClient` (no stale key issue) — `frontend/lib/api.ts`

---

## Phase 4: US1 — Submit Prediction (MVP) 🎯 ✅

- [x] T015 [US1] `JobStatusBadge` — 4-status colour-coded badge with pulse on processing — `frontend/components/JobStatusBadge.tsx`
- [x] T016 [US1] `PredictionForm` — source selector, conditional fields, 401 detection, error preserved on failure — `frontend/components/PredictionForm.tsx`
- [x] T017 [US1] Predict page — form + live polling badge + "View Results" link on complete — `frontend/app/predict/page.tsx`
- [x] T018 [US1] Network error handling inline in PredictionForm — `frontend/components/PredictionForm.tsx`

---

## Phase 5: US2 — Browse Results ✅

- [x] T019 [P] [US2] `EvaluationCard` — farmer identity, class badge, model, timestamp — `frontend/components/EvaluationCard.tsx`
- [x] T020 [P] [US2] `FeatureContribChart` — Recharts horizontal bar, amber/teal SHAP colouring — `frontend/components/FeatureContribChart.tsx`
- [x] T021 [P] [US2] `ExplanationPanel` — RAG text, doc IDs, cache badge, metadata — `frontend/components/ExplanationPanel.tsx`
- [x] T022 [US2] Results table page — shadcn Table, row links, empty-state — `frontend/app/results/page.tsx`
- [x] T023 [US2] Farmer detail page — EvaluationCard + SHAP chart + inline explanation — `frontend/app/results/[id]/page.tsx`
- [x] T024 [US2] 404 handling + skeleton loader — `frontend/app/results/[id]/page.tsx`

---

## Phase 6: US3 — Dashboard ✅

- [x] T025 [US3] ISR dashboard (`revalidate=60`) — summary tiles + recent activity table — `frontend/app/page.tsx`
- [x] T026 [P] [US3] Server-side `getServerSideJobSummary()` — `frontend/lib/server-api.ts`

---

## Phase 7: Error Handling & Polish ✅

- [x] T027 [P] `loading.tsx` skeletons for results list + detail — `frontend/app/results/loading.tsx`, `frontend/app/results/[id]/loading.tsx`
- [x] T028 [P] `error.tsx` boundaries for predict, results, detail — `frontend/app/predict/error.tsx`, `frontend/app/results/error.tsx`, `frontend/app/results/[id]/error.tsx`
- [x] T029 Global 401 intercept + `authError` flag in `useApiKeyStore` — `frontend/lib/api.ts` + `frontend/lib/stores.ts`
- [x] T030 [P] `not-found.tsx` — `frontend/app/not-found.tsx`
- [x] T031 [P] Empty states inline in results page and dashboard

---

## Phase 8: Infrastructure ✅

- [x] T032 `frontend/Dockerfile` — multi-stage Node 18 Alpine, standalone output, non-root user
- [x] T033 `docker-compose.yml` — `frontend` service added (depends_on backend healthy)
- [x] T034 [P] `docker-compose.override.yml` — frontend port 3000 + bind-mount
- [x] T035 [P] `Caddyfile` — `reverse_proxy /* frontend:3000` (Streamlit ui:8501 replaced)
- [x] T036 [P] `Makefile` — `frontend-dev`, `frontend-build`, `frontend-up`, `dev-next` targets added
- [x] T037 [P] `frontend-build` CI job added to `.github/workflows/ci.yml` — runs after lint; Docker `build` job now also builds `lersha-frontend:ci` image

---

## Phase 9: Validation

- [x] T038 `npm run build` — ✅ PASSED (Next.js 16.2.2, zero TS/ESLint errors, 7 routes)
- [ ] T039 [P] Manual: `/settings` → enter key → refresh → key persists (SC-004)
- [ ] T040 [P] Manual: `/predict` → Batch submit → status badge updates (SC-001, SC-002)
- [ ] T041 [P] Manual: Network tab — poll requests stop on terminal status (FR-011)
- [ ] T042 [P] Manual: `/results` → row click → detail page with SHAP chart (SC-003)
- [ ] T043 [P] Manual: `docker compose build` frontend → starts within 60s (SC-005)
- [ ] T044 [P] Manual: 768px viewport — all pages usable (SC-006)

---

## Dependencies & Execution Order

*(unchanged from original — see plan.md for full dependency graph)*

### Summary

| Phase | Status | Tasks |
|-------|--------|-------|
| 1 — Setup | ✅ Done | T001–T006 |
| 2 — Foundational | ✅ Done | T007–T011 |
| 3 — US4 Settings | ✅ Done | T012–T014 |
| 4 — US1 Predict | ✅ Done | T015–T018 |
| 5 — US2 Results | ✅ Done | T019–T024 |
| 6 — US3 Dashboard | ✅ Done | T025–T026 |
| 7 — Polish | ✅ Done | T027–T031 |
| 8 — Infrastructure | ✅ (T037 pending) | T032–T036 done |
| 9 — Validation | 🔲 Manual steps remain | T038 ✅, T039–T044 pending |
