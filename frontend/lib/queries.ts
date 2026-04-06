"use client";

import { QueryClient, useQuery } from "@tanstack/react-query";
import { lershaClient } from "./api";
import { isTerminalStatus } from "./types";
import type { ExplainRequest } from "./types";

// ── Singleton QueryClient ───────────────────────────────────────────────────

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// ── Job status polling ──────────────────────────────────────────────────────

export function useJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => lershaClient.getJobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return isTerminalStatus(status) ? false : 2000;
    },
  });
}

// ── Results list (legacy) ──────────────────────────────────────────────────

export function useResults(options: { limit?: number; model_name?: string } = {}) {
  return useQuery({
    queryKey: ["results", options],
    queryFn: () => lershaClient.getResults(options),
    staleTime: 60_000,
  });
}

// ── Results paginated ──────────────────────────────────────────────────────

export function useResultsPaginated(options: {
  page?: number;
  per_page?: number;
  search?: string;
  decision?: string;
  gender?: string;
  model_name?: string;
} = {}) {
  return useQuery({
    queryKey: ["results-paginated", options],
    queryFn: () => lershaClient.getResultsPaginated(options),
    staleTime: 30_000,
  });
}

// ── Analytics summary ──────────────────────────────────────────────────────

export function useAnalytics() {
  return useQuery({
    queryKey: ["analytics"],
    queryFn: () => lershaClient.getAnalytics(),
    staleTime: 60_000,
  });
}

// ── Jobs list ──────────────────────────────────────────────────────────────

export function useJobs(limit?: number) {
  return useQuery({
    queryKey: ["jobs", limit],
    queryFn: () => lershaClient.getJobs(limit),
    staleTime: 15_000,
  });
}

// ── AI explanation ──────────────────────────────────────────────────────────

export function useExplanation(req: ExplainRequest | null) {
  return useQuery({
    queryKey: ["explanation", req?.job_id, req?.record_index, req?.model_name],
    queryFn: () => lershaClient.getExplanation(req!),
    enabled: !!req,
    staleTime: Infinity,
  });
}
