/**
 * LershaClient — typed HTTP client for the Lersha Credit Scoring API.
 *
 * All requests go through Next.js API route proxies (/api/*).
 * The API key is injected server-side in those routes — it never
 * reaches the browser.
 */

import type {
  ExplainRequest,
  ExplainResponse,
  JobAcceptedResponse,
  JobStatusResponse,
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

  /** GET /api/health */
  async health(): Promise<{ status: string }> {
    return this.request("/api/health");
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

  /** GET /api/results */
  async getResults(
    options: { limit?: number; model_name?: string } = {},
  ): Promise<ResultsResponse> {
    const params = new URLSearchParams();
    if (options.limit) params.set("limit", String(options.limit));
    if (options.model_name) params.set("model_name", options.model_name);
    const qs = params.toString();
    return this.request<ResultsResponse>(`/api/results${qs ? `?${qs}` : ""}`);
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
