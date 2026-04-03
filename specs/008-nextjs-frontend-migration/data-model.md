# Data Model: Next.js Frontend

**Feature**: 008-nextjs-frontend-migration  
**Phase**: 1 — Design & Contracts  
**Date**: 2026-04-03  
**Source**: Derived from `backend/api/schemas.py` and spec entities

---

## TypeScript Interface Definitions

These interfaces live in `frontend/lib/types.ts` and are the canonical client-side data model. They mirror the Pydantic schemas in `backend/api/schemas.py`.

---

### FeatureContribution

Maps to `backend/api/schemas.py::FeatureContribution`.

```typescript
export interface FeatureContribution {
  feature: string;  // Feature name (e.g. "loan_amount")
  value: number;    // SHAP contribution value (positive = risky, negative = safe)
}
```

---

### PredictRequest

Maps to `backend/api/schemas.py::PredictRequest`.

```typescript
export type PredictSource = "Single Value" | "Batch Prediction";

export interface PredictRequest {
  source: PredictSource;
  farmer_uid?: string;       // Required when source === "Single Value"
  number_of_rows?: number;   // Required when source === "Batch Prediction"; 1–100
}
```

---

### JobAcceptedResponse

Maps to `backend/api/schemas.py::JobAcceptedResponse`. Returned on `POST /v1/predict/` (HTTP 202).

```typescript
export interface JobAcceptedResponse {
  job_id: string;
  status: "accepted";
}
```

---

### JobStatus

Union type derived from the `status` field across responses.

```typescript
export type JobStatus = "pending" | "processing" | "completed" | "failed";

export const TERMINAL_STATUSES: JobStatus[] = ["completed", "failed"];

export function isTerminalStatus(status: JobStatus): boolean {
  return TERMINAL_STATUSES.includes(status);
}
```

---

### EvaluationRecord

Maps to `backend/api/schemas.py::EvaluationRecord`. One farmer's model output within a job.

```typescript
export interface EvaluationRecord {
  predicted_class_name: string;           // e.g. "Eligible", "Not Eligible"
  top_feature_contributions: FeatureContribution[];
  rag_explanation: string;                // Inline explanation stored at inference time
  model_name: string;                     // e.g. "xgboost", "random_forest"
}
```

---

### ModelResult

Maps to `backend/api/schemas.py::ModelResult`. Wraps all evaluation records for one model.

```typescript
export interface ModelResult {
  status: string;
  records_processed: number;
  evaluations: EvaluationRecord[];
}
```

---

### JobStatusResponse

Maps to `backend/api/schemas.py::JobStatusResponse`. Returned by `GET /v1/predict/{job_id}`.

```typescript
export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  result: {
    result_xgboost?: ModelResult;
    result_random_forest?: ModelResult;
  } | null;
  error: string | null;
}
```

---

### ResultsRecord

Maps to `backend/api/schemas.py::ResultsRecord`. One row from the `candidate_result` table.

```typescript
export interface ResultsRecord {
  farmer_uid: string;
  first_name: string | null;
  middle_name: string | null;
  last_name: string | null;
  predicted_class_name: string;
  top_feature_contributions: FeatureContribution[];
  rag_explanation: string;
  model_name: string;
  timestamp: string | null;   // ISO 8601 datetime string
}
```

---

### ResultsResponse

Maps to `backend/api/schemas.py::ResultsResponse`. Returned by `GET /v1/results/`.

```typescript
export interface ResultsResponse {
  total: number;
  records: ResultsRecord[];
}
```

---

### ExplainRequest

Maps to `backend/api/schemas.py::ExplainRequest`. Sent to `POST /v1/explain/`.

```typescript
export interface ExplainRequest {
  job_id: string;        // UUID of the completed inference job
  record_index: number;  // 0-based index within the job's evaluations list
  model_name: string;    // e.g. "xgboost"
}
```

---

### ExplainResponse

Maps to `backend/api/schemas.py::ExplainResponse`. Returned by `POST /v1/explain/`.

```typescript
export interface ExplainResponse {
  farmer_uid: string;
  prediction: string;            // e.g. "Eligible"
  explanation: string;           // AI-generated narrative
  retrieved_doc_ids: number[];   // RAG document IDs used
  cache_hit: boolean;            // True if served from Redis cache
  prompt_version: string;        // e.g. "v1"
  latency_ms: number;
}
```

---

## Zustand Store Shapes

### useApiKeyStore

Persisted to `localStorage` via Zustand `persist` middleware.

```typescript
interface ApiKeyStore {
  apiKey: string;
  setApiKey: (key: string) => void;
  clearApiKey: () => void;
}
```

### useJobStore

Session-only (no persistence). Tracks the currently active polling job.

```typescript
interface JobStore {
  activeJobId: string | null;
  setActiveJobId: (id: string | null) => void;
  clearActiveJobId: () => void;
}
```

---

## Entity Relationships

```
PredictRequest
    │ POST /v1/predict/
    ▼
JobAcceptedResponse (job_id)
    │ GET /v1/predict/{job_id}  [polling]
    ▼
JobStatusResponse
    └── result
        ├── result_xgboost: ModelResult
        │       └── evaluations[]: EvaluationRecord
        └── result_random_forest: ModelResult
                └── evaluations[]: EvaluationRecord

GET /v1/results/
    ▼
ResultsResponse
    └── records[]: ResultsRecord

POST /v1/explain/  (job_id + record_index + model_name)
    ▼
ExplainResponse
```

---

## State Transition Diagram — Job Status

```
         ┌─────────┐
  POST   │         │
 ──────► │ pending │
         │         │
         └────┬────┘
              │  worker picks up job
              ▼
         ┌────────────┐
         │            │
         │ processing │  ← polling active (2s interval)
         │            │
         └──────┬─────┘
                │
        ┌───────┴────────┐
        ▼                ▼
   ┌─────────┐      ┌────────┐
   │completed│      │ failed │  ← polling stops
   └─────────┘      └────────┘
```

**FR-011 implementation**: TanStack Query `refetchInterval` callback returns `false` when `isTerminalStatus(data?.status)` is `true`.

---

## Validation Rules

| Field | Rule |
|-------|------|
| `PredictRequest.farmer_uid` | Required when `source === "Single Value"`, non-empty string |
| `PredictRequest.number_of_rows` | Required when `source === "Batch Prediction"`, integer 1–100 |
| `ExplainRequest.record_index` | Non-negative integer |
| `ExplainRequest.model_name` | Non-empty string |
| `ApiKeyStore.apiKey` | Non-empty string before any API call; enforced in `LershaClient` |
