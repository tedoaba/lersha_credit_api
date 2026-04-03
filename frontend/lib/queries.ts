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

// ── Results list ────────────────────────────────────────────────────────────

export function useResults(options: { limit?: number; model_name?: string } = {}) {
  return useQuery({
    queryKey: ["results", options],
    queryFn: () => lershaClient.getResults(options),
    staleTime: 60_000,
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
