# Research: Next.js 14 Frontend Migration

**Feature**: 008-nextjs-frontend-migration  
**Phase**: 0 — Outline & Research  
**Date**: 2026-04-03

---

## R-001: Next.js 14 App Router vs Pages Router

**Decision**: App Router (Next.js 14)  
**Rationale**: App Router enables React Server Components (RSC) for ISR/SSG pages (dashboard), co-located loading/error boundaries per route, and a clean separation of server-rendered and client-interactive segments. Pages Router is the legacy model and lacks RSC and nested layouts.  
**Alternatives considered**:
- Pages Router — rejected because it does not support RSC or nested layouts, which are needed for the navigation shell + per-route content pattern.
- Pure SPA (Vite + React) — rejected because Next.js provides ISR, file-based routing, and a production Node server in one package, reducing infrastructure burden.

---

## R-002: State Management — Zustand

**Decision**: Zustand for client-global persistent state (API key, active job ID)  
**Rationale**: The two cross-cutting state slices (API key from localStorage, active polling job ID) are session-scoped singletons that multiple components need to read. Zustand provides a near-zero-boilerplate, hook-based API, persists with `zustand/middleware` `persist` plugin (localStorage), and has no Provider wrapping required. React Context alone would require prop drilling or a custom hook; Redux is disproportionately heavy for two stores.  
**Alternatives considered**:
- React Context — rejected for persistent storage (no built-in localStorage serialisation) and verbose boilerplate.
- Redux Toolkit — rejected as over-engineered for two small stores.
- Jotai — viable alternative but Zustand has wider Next.js App Router community adoption and proven middleware ecosystem.

---

## R-003: Data Fetching & Server State — TanStack Query v5

**Decision**: TanStack Query (React Query) v5 for all async server state (job status polling, results, explanations)  
**Rationale**: TanStack Query provides built-in request deduplication, background refetching, configurable polling (`refetchInterval`), and automatic stale-while-revalidate behaviour. The status polling requirement (stop when terminal state reached) maps directly to `refetchInterval: (data) => isTerminal(data) ? false : 2000`. It also integrates cleanly with Next.js App Router client components.  
**Alternatives considered**:
- `useEffect` + `setInterval` — rejected: error handling, cancellation, and caching must be hand-rolled; high maintenance surface.
- SWR — viable but has weaker TypeScript generics and fewer features (no query invalidation tree, no dependent queries).

---

## R-004: UI Component Library — shadcn/ui + Tailwind CSS

**Decision**: shadcn/ui component collection with Tailwind CSS utility layer  
**Rationale**: shadcn/ui components (from Radix UI primitives) are copy-paste-owned — no version-locked black box, full accessibility, and they compose with Tailwind. This gives the team full control over styling while avoiding building every form element, badge, and table from scratch. Tailwind is the required pairing.  
**Alternatives considered**:
- Chakra UI — requires Provider, heavier runtime, less idiomatic with Next.js App Router RSC.
- MUI (Material UI) — extensive but opinionated theming system conflicts with custom brand identity; significant bundle weight.
- Vanilla CSS — viable for simple apps but table, modal, and dropdown accessibility is non-trivial to hand-implement correctly.

---

## R-005: Chart Library — Recharts

**Decision**: Recharts for SHAP feature contribution bar charts  
**Rationale**: Recharts is React-native (no D3 imperative DOM manipulation), supports responsive containers, and is the most commonly paired charting library with shadcn/ui projects. The SHAP output is a `{feature, value}[]` array which maps directly to a Recharts `BarChart` with positive/negative value representation.  
**Alternatives considered**:
- Chart.js (via react-chartjs-2) — viable but requires a canvas element and has a larger API surface for simple bar charts.
- D3 directly — rejected: requires imperative DOM manipulation incompatible with React declarative model.
- Visx — lower-level than needed for a simple ranked bar chart.

---

## R-006: API Contract — Typed HTTP Client

**Decision**: Thin class-based `LershaClient` in `frontend/lib/api.ts` using native `fetch`  
**Rationale**: The existing backend exposes a well-defined JSON REST API. A typed class with typed return interfaces mirrors the existing Python `LershaAPIClient` pattern, providing a clear seam for mocking in tests. Native `fetch` has first-class support in Node.js 18+ (used in the frontend container) and Next.js server actions. No runtime HTTP library (axios, got) is needed.  
**Alternatives considered**:
- Axios — adds ~40 KB to the client bundle with no advantage over fetch given the simple request patterns needed.
- tRPC — requires a corresponding tRPC server which would require wrapping the FastAPI backend; significant overhead for a read/write REST client.
- OpenAPI generated client — ideal long-term; rejected for now because the backend does not yet export an OpenAPI client generation artifact. Manual TypeScript interfaces derived from `schemas.py` are equivalent for current scale.

---

## R-007: Polling Strategy — Conditional `refetchInterval`

**Decision**: `refetchInterval: (query) => isTerminal(query.state.data?.status) ? false : 2000`  
**Rationale**: TanStack Query v5 accepts a callback for `refetchInterval` that receives the current query state. Returning `false` stops polling cleanly without side effects. This satisfies FR-011 (cease polling on terminal state) and SC-002 (update within 5 seconds — 2s interval is well within that).  
**Alternatives considered**:
- `setInterval` in `useEffect` with `clearInterval` on completion — rejected: manual cleanup prone to stale closure bugs and memory leaks.
- WebSocket / SSE — rejected: backend does not expose a WebSocket endpoint; adding one is out of scope for this migration sprint.

---

## R-008: Containerisation — Node.js 18 Alpine, Standalone Output

**Decision**: `FROM node:18-alpine` multi-stage Dockerfile with `output: 'standalone'` in `next.config.js`  
**Rationale**: `output: 'standalone'` copies only the files needed to run the production server into `.next/standalone`, resulting in a minimal image (typically 80–150 MB vs. 400+ MB with full `node_modules`). Node 18 LTS is the minimum version for native fetch support and is kept consistent with the constitution's LTS pinning principle.  
**Alternatives considered**:
- `node:20-alpine` — viable but Node 18 is LTS and sufficient; avoids upgrading during the migration sprint.
- `node:18` (non-alpine) — larger base image with no benefit.
- Bun — immature production story for Next.js 14 App Router; rejected.

---

## R-009: Routing Strategy — ISR Dashboard, CSR Interactive Pages

**Decision**: Dashboard (`app/page.tsx`) uses ISR `revalidate = 60`; all other routes are client components (`"use client"`)  
**Rationale**: The dashboard summary data (job counts) changes slowly and benefits from ISR to reduce backend load. All other pages (predict form, results table, detail, settings) require interactive state (query params, polling, form input) and must be client-side rendered via TanStack Query hooks.  
**Alternatives considered**:
- Full SSR on every page — rejected: job status polling cannot be initiated server-side; SSR for the results table adds no value over CSR + TanStack Query stale-while-revalidate.
- Full SPA/CSR for all pages — viable, but misses the ISR caching benefit on the dashboard where it genuinely applies.

---

## R-010: Reverse Proxy — Caddy Route Priority

**Decision**: Update Caddyfile to route `/api/*` → `backend:8000`, `/*` → `frontend:3000`  
**Rationale**: The existing Caddyfile routes `/v1/*` to the backend and `/*` to the Streamlit UI. The new frontend replaces the Streamlit catch-all. The `/v1/*` pattern matches all current versioned API routes. Adding `/api/*` as an alias improves discoverability but the existing `/v1/*` is sufficient because the `LershaClient` will call `/v1/` paths directly.  
**Alternatives considered**:
- Nginx — rejected: Caddy is already the production reverse proxy; switching adds operational complexity.
- Direct port exposure (frontend on :3000, backend on :8000) — rejected: violates the constitution's P9-SEC requirement that direct access to service ports from outside the Docker network is forbidden in production.

---

## Summary of All Decisions

| ID | Topic | Decision |
|----|-------|----------|
| R-001 | Framework | Next.js 14 App Router |
| R-002 | Client global state | Zustand + localStorage middleware |
| R-003 | Server/async state | TanStack Query v5 |
| R-004 | UI components | shadcn/ui + Tailwind CSS |
| R-005 | Charts | Recharts |
| R-006 | HTTP client | Native fetch, typed `LershaClient` class |
| R-007 | Polling | `refetchInterval` callback, stop on terminal |
| R-008 | Containerisation | Node 18 Alpine, `output: standalone` |
| R-009 | Rendering strategy | ISR (dashboard), CSR (interactive pages) |
| R-010 | Reverse proxy | Caddy `/v1/*` → backend, `/*` → frontend |
