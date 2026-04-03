"use client";

import Link from "next/link";
import { useJobStore } from "@/lib/stores";
import { useJobStatus } from "@/lib/queries";
import PredictionForm from "@/components/PredictionForm";
import JobStatusBadge from "@/components/JobStatusBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function PredictPage() {
  const { activeJobId, clearActiveJobId } = useJobStore();
  const { data: job, isLoading } = useJobStatus(activeJobId);

  return (
    <div className="max-w-xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Submit Prediction</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Select a data source and submit a credit scoring job.
        </p>
      </div>

      <PredictionForm />

      {activeJobId && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center justify-between">
              <span>Job Status</span>
              <button
                onClick={clearActiveJobId}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                ✕ Clear
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-muted-foreground truncate">
                {activeJobId}
              </span>
              {isLoading ? (
                <span className="text-xs text-muted-foreground">Connecting…</span>
              ) : job ? (
                <JobStatusBadge status={job.status} />
              ) : null}
            </div>

            {job?.error && (
              <p className="text-sm text-destructive">Error: {job.error}</p>
            )}

            {job?.status === "completed" && (
              <Link
                href="/results"
                className="inline-flex items-center rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                View Results →
              </Link>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
