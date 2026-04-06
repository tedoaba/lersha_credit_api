/**
 * TypeScript interfaces mirroring backend/api/schemas.py Pydantic models.
 * These are the canonical client-side data model for the Lersha frontend.
 */

// ── Feature contributions (SHAP) ───────────────────────────────────────────

export interface FeatureContribution {
  feature: string;
  value: number;
}

// ── Prediction request ──────────────────────────────────────────────────────

export type PredictSource = "Single Value" | "Batch Prediction";

export interface PredictRequest {
  source: PredictSource;
  farmer_uid?: string;      // required when source === "Single Value"
  number_of_rows?: number;  // required when source === "Batch Prediction"; 1–100
  gender?: string;           // optional batch filter
  age_min?: number;          // optional batch filter
  age_max?: number;          // optional batch filter
}

// ── Farmer search (autocomplete) ───────────────────────────────────────────

export interface FarmerSearchResult {
  farmer_uid: string;
  first_name: string | null;
  middle_name: string | null;
  last_name: string | null;
}

export interface FarmerSearchResponse {
  results: FarmerSearchResult[];
}

// ── Job lifecycle ───────────────────────────────────────────────────────────

export interface JobAcceptedResponse {
  job_id: string;
  status: "accepted";
}

export type JobStatus = "pending" | "processing" | "completed" | "failed";

export const TERMINAL_STATUSES: readonly JobStatus[] = ["completed", "failed"] as const;

export function isTerminalStatus(status: JobStatus | undefined | null): boolean {
  if (!status) return false;
  return (TERMINAL_STATUSES as readonly string[]).includes(status);
}

// ── Job result payloads ─────────────────────────────────────────────────────

export interface EvaluationRecord {
  predicted_class_name: string;
  top_feature_contributions: FeatureContribution[];
  rag_explanation: string;
  model_name: string;
  class_probabilities: Record<string, number> | null;
  confidence_score: number | null;
}

export interface ModelResult {
  status: string;
  records_processed: number;
  evaluations: EvaluationRecord[];
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  result: {
    result_xgboost?: ModelResult;
    result_random_forest?: ModelResult;
  } | null;
  error: string | null;
}

// ── Results table ───────────────────────────────────────────────────────────

export interface ResultsRecord {
  farmer_uid: string;
  first_name: string | null;
  middle_name: string | null;
  last_name: string | null;
  predicted_class_name: string;
  top_feature_contributions: FeatureContribution[];
  rag_explanation: string;
  model_name: string;
  class_probabilities: Record<string, number> | null;
  confidence_score: number | null;
  timestamp: string | null; // ISO 8601
  gender: string | null;
}

// ── Grouped farmer (deduped across models) ─────────────────────────────────

export interface GroupedFarmer {
  farmer_uid: string;
  first_name: string | null;
  middle_name: string | null;
  last_name: string | null;
  gender: string | null;
  timestamp: string | null;
  models: ResultsRecord[];
  consensus: "agree" | "mixed";
  primaryDecision: string;
}

export function groupByFarmer(records: ResultsRecord[]): GroupedFarmer[] {
  const map = new Map<string, ResultsRecord[]>();
  for (const r of records) {
    const existing = map.get(r.farmer_uid) ?? [];
    existing.push(r);
    map.set(r.farmer_uid, existing);
  }

  const groups: GroupedFarmer[] = [];
  for (const [farmer_uid, models] of map) {
    const first = models[0];
    const decisions = new Set(models.map((m) => m.predicted_class_name));
    groups.push({
      farmer_uid,
      first_name: first.first_name,
      middle_name: first.middle_name,
      last_name: first.last_name,
      gender: first.gender,
      timestamp: first.timestamp,
      models,
      consensus: decisions.size === 1 ? "agree" : "mixed",
      primaryDecision: decisions.size === 1
        ? models[0].predicted_class_name
        : "Mixed",
    });
  }
  return groups;
}

export interface ResultsResponse {
  total: number;
  records: ResultsRecord[];
}

export interface PaginatedResultsResponse {
  total: number;
  page: number;
  per_page: number;
  records: ResultsRecord[];
}

// ── Analytics ───────────────────────────────────────────────────────────────

export interface AnalyticsSummaryResponse {
  total: number;
  total_farmers: number;
  by_decision: Record<string, number>;
  by_consensus: Record<string, number>;
  by_gender: Record<string, Record<string, number>>;
  by_model: Record<string, Record<string, number>>;
}

// ── Jobs list ───────────────────────────────────────────────────────────────

export interface JobRecord {
  job_id: string;
  status: string;
  error: string | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface JobsListResponse {
  jobs: JobRecord[];
}

// ── Explain endpoint ────────────────────────────────────────────────────────

export interface ExplainRequest {
  job_id: string;
  record_index: number;
  model_name: string;
}

export interface ExplainResponse {
  farmer_uid: string;
  prediction: string;
  explanation: string;
  retrieved_doc_ids: number[];
  cache_hit: boolean;
  prompt_version: string;
  latency_ms: number;
}

// ── API error ───────────────────────────────────────────────────────────────

export interface ApiError {
  status: number;
  message: string;
}

// ── Decision types ──────────────────────────────────────────────────────────

export type Decision = "Eligible" | "Review" | "Not Eligible";

export const DECISIONS: Decision[] = ["Eligible", "Review", "Not Eligible"];
