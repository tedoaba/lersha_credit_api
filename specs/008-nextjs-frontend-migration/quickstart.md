# Quickstart: 008 — Next.js Frontend Migration

**Feature**: 008-nextjs-frontend-migration  
**Target**: Developer setting up the new Next.js frontend for local development  
**Prerequisite**: Backend stack is already running via `make docker-up` or `docker compose up`

---

## 1. Prerequisites

| Requirement | Version | Check command |
|-------------|---------|---------------|
| Node.js | 18 LTS or 20 LTS | `node -v` |
| npm | ≥9 | `npm -v` |
| Docker Compose | ≥2.20 | `docker compose version` |
| A running backend | — | `curl http://localhost:8000/health` |

---

## 2. Create the Frontend Project

```bash
# From repo root
mkdir frontend
cd frontend

# Bootstrap Next.js 14 with App Router
npx create-next-app@14 . \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --src-dir=false \
  --import-alias "@/*" \
  --no-git
```

---

## 3. Install Additional Dependencies

```bash
# Server state / data fetching
npm install @tanstack/react-query@^5

# Client state
npm install zustand@^4

# Charts
npm install recharts@^2

# shadcn/ui (interactive installer — choose "New York" style, "slate" base color)
npx shadcn-ui@latest init
# Then add components as needed:
npx shadcn-ui@latest add button badge card table select input label separator
```

---

## 4. Configure Environment Variables

Create `frontend/.env.local`:

```env
# Base URL of the backend API, as seen from the browser (proxied via Caddy)
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# NOTE: The API key is NOT stored here — it is stored in localStorage via
# the useApiKeyStore Zustand store (set via the Settings page at runtime).
```

---

## 5. Run in Development Mode

```bash
# In a separate terminal from the backend
cd frontend
npm run dev
# → http://localhost:3000
```

---

## 6. Project Structure (after setup)

```text
frontend/
├── app/
│   ├── layout.tsx            # Root layout: QueryClientProvider, nav bar, theme
│   ├── page.tsx              # Dashboard — ISR (revalidate: 60)
│   ├── predict/
│   │   └── page.tsx          # Prediction form — CSR
│   ├── results/
│   │   ├── page.tsx          # Results table — CSR
│   │   └── [id]/
│   │       └── page.tsx      # Farmer detail — CSR
│   └── settings/
│       └── page.tsx          # API key config — CSR
├── components/
│   ├── PredictionForm.tsx    # Source selector + submit
│   ├── JobStatusBadge.tsx    # Status badge (pending/processing/completed/failed)
│   ├── FeatureContribChart.tsx  # Recharts SHAP bar chart
│   ├── ExplanationPanel.tsx  # RAG text + doc metadata
│   └── EvaluationCard.tsx    # Single farmer result card
├── lib/
│   ├── api.ts                # LershaClient class (typed fetch wrapper)
│   ├── stores.ts             # Zustand stores (useApiKeyStore, useJobStore)
│   ├── queries.ts            # TanStack Query hooks (useJobStatus, useResults, etc.)
│   └── types.ts              # TypeScript interfaces (mirrors backend schemas.py)
├── Dockerfile                # Multi-stage Node 18 Alpine build
├── next.config.js            # output: 'standalone'
├── tailwind.config.ts
├── tsconfig.json
└── package.json
```

---

## 7. Key Implementation Patterns

### API Key — Settings Page

```typescript
// lib/stores.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useApiKeyStore = create(
  persist(
    (set) => ({
      apiKey: '',
      setApiKey: (key: string) => set({ apiKey: key }),
      clearApiKey: () => set({ apiKey: '' }),
    }),
    { name: 'lersha-api-key' }   // localStorage key
  )
)
```

### Job Status Polling

```typescript
// lib/queries.ts
import { useQuery } from '@tanstack/react-query'
import { isTerminalStatus } from './types'

export function useJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => lershaClient.getJobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) =>
      query.state.data && isTerminalStatus(query.state.data.status)
        ? false
        : 2000,
  })
}
```

### TanStack Query Provider (Root Layout)

```typescript
// app/layout.tsx (server component wrapper)
'use client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient()

export default function RootLayout({ children }) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}
```

---

## 8. Docker Build

```bash
# From repo root (after frontend/ Dockerfile is created)
docker compose up --build frontend

# Verify the frontend is healthy:
curl http://localhost:3000
```

---

## 9. Full Stack via Docker Compose

After adding the `frontend` service to `docker-compose.yml` and updating `Caddyfile`:

```bash
# Dev (all services including frontend)
docker compose up --build

# Verify routing:
curl http://localhost/           # → Next.js frontend
curl http://localhost/v1/predict  # → FastAPI backend (via Caddy /v1/*)
```

---

## 10. Validation Checklist

Before marking the feature complete, verify:

- [ ] `npm run build` inside `frontend/` completes with no TypeScript or ESLint errors
- [ ] Frontend loads at `:3000` (or via Caddy at `/`)
- [ ] Settings page: enter API key → refresh → key still present
- [ ] Predict page: submit Batch Prediction (5 rows) → job ID returned → status badge updates
- [ ] Status polling stops when status is `completed` or `failed` (check Network tab)
- [ ] Results page: table renders all returned rows
- [ ] Result detail page: SHAP chart renders, explanation panel shows text + doc IDs
- [ ] All pages display error state (toast/message) when backend is unreachable
- [ ] Docker image builds and container starts within 60 seconds
- [ ] Caddy correctly proxies `/v1/*` to backend and `/*` to frontend
