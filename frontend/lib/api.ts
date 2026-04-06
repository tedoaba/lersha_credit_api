/**
 * LershaClient — typed HTTP client for the Lersha Credit Scoring API.
 *
 * All requests go through Next.js API route proxies (/api/*).
 * The API key is injected server-side in those routes — it never
 * reaches the browser.
 */

import type {
  AnalyticsSummaryResponse,
  ExplainRequest,
  ExplainResponse,
  FarmerSearchResponse,
  JobAcceptedResponse,
  JobsListResponse,
  JobStatusResponse,
  PaginatedResultsResponse,
  PredictRequest,
  ResultsResponse,
} from "./types";

export class ApiRequestError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

export class LershaClient {
  private async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const res = await fetch(path, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      throw new ApiRequestError(res.status, text);
    }

    return res.json() as Promise<T>;
  }

  /** POST /api/predict → 202 Accepted */
  async submitPrediction(payload: PredictRequest): Promise<JobAcceptedResponse> {
    return this.request<JobAcceptedResponse>("/api/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  /** GET /api/predict/{job_id} */
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    return this.request<JobStatusResponse>(`/api/predict/${jobId}`);
  }

  /** GET /api/results (legacy flat mode) */
  async getResults(
    options: { limit?: number; model_name?: string } = {},
  ): Promise<ResultsResponse> {
    const params = new URLSearchParams();
    if (options.limit) params.set("limit", String(options.limit));
    if (options.model_name) params.set("model_name", options.model_name);
    const qs = params.toString();
    return this.request<ResultsResponse>(`/api/results${qs ? `?${qs}` : ""}`);
  }

  /** GET /api/results with pagination and filters */
  async getResultsPaginated(
    options: {
      page?: number;
      per_page?: number;
      search?: string;
      decision?: string;
      gender?: string;
      model_name?: string;
      job_id?: string;
    } = {},
  ): Promise<PaginatedResultsResponse> {
    const params = new URLSearchParams();
    params.set("page", String(options.page ?? 1));
    params.set("per_page", String(options.per_page ?? 20));
    if (options.search) params.set("search", options.search);
    if (options.decision) params.set("decision", options.decision);
    if (options.gender) params.set("gender", options.gender);
    if (options.model_name) params.set("model_name", options.model_name);
    if (options.job_id) params.set("job_id", options.job_id);
    return this.request<PaginatedResultsResponse>(`/api/results?${params}`);
  }

  /** GET /api/analytics */
  async getAnalytics(): Promise<AnalyticsSummaryResponse> {
    return this.request<AnalyticsSummaryResponse>("/api/analytics");
  }

  /** GET /api/jobs */
  async getJobs(limit?: number): Promise<JobsListResponse> {
    const params = limit ? `?limit=${limit}` : "";
    return this.request<JobsListResponse>(`/api/jobs${params}`);
  }

  /** GET /api/farmers/search?q=... */
  async searchFarmers(q: string, limit = 10): Promise<FarmerSearchResponse> {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return this.request<FarmerSearchResponse>(`/api/farmers/search?${params}`);
  }

  /** POST /api/explain */
  async getExplanation(payload: ExplainRequest): Promise<ExplainResponse> {
    return this.request<ExplainResponse>("/api/explain", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
}

/** Singleton client instance */
export const lershaClient = new LershaClient();
