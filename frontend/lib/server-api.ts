/**
 * Server-side data helper for the ISR dashboard page.
 * Called from a React Server Component — runs only on the server.
 * API key is read from process.env.API_KEY and never sent to the browser.
 */

import type { ResultsResponse } from "./types";

export async function getServerSideJobSummary(): Promise<{
  total: number;
  completed: number;
  recent: ResultsResponse["records"];
} | null> {
  const baseUrl = process.env.API_BASE_URL ?? "http://localhost:8006";
  const apiKey = process.env.API_KEY ?? "";

  try {
    const res = await fetch(`${baseUrl}/v1/results/?limit=500`, {
      headers: { "X-API-Key": apiKey },
      next: { revalidate: 60 },
    });

    if (!res.ok) return null;
    const data: ResultsResponse = await res.json();

    const recent = data.records
      .slice()
      .sort((a, b) => {
        const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0;
        const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0;
        return tb - ta;
      })
      .slice(0, 10);

    return {
      total: data.total,
      completed: data.total,
      recent,
    };
  } catch {
    return null;
  }
}
