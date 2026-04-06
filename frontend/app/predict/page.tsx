"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useJobStore } from "@/lib/stores";
import { useJobStatus, useJobs } from "@/lib/queries";
import PredictionForm from "@/components/PredictionForm";
import JobStatusBadge from "@/components/JobStatusBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { JobStatus } from "@/lib/types";

const PROGRESS_MAP: Record<JobStatus, number> = {
  pending: 15,
  processing: 60,
  completed: 100,
  failed: 100,
};

function formatTimeAgo(timestamp: string | null): string {
  if (!timestamp) return "\u2014";
  const diff = Date.now() - new Date(timestamp).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function PredictPage() {
  const { activeJobId, clearActiveJobId } = useJobStore();
  const { data: job, isLoading } = useJobStatus(activeJobId);
  const { data: jobHistory } = useJobs(10);

  const [elapsed, setElapsed] = useState(0);
  const [startTime] = useState(() => Date.now());

  // Track elapsed time while job is active
  useEffect(() => {
    if (!activeJobId || !job || job.status === "completed" || job.status === "failed") return;
    const interval = setInterval(() => setElapsed(Math.floor((Date.now() - startTime) / 1000)), 1000);
    return () => clearInterval(interval);
  }, [activeJobId, job, startTime]);

  const progress = job ? PROGRESS_MAP[job.status] : 0;

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">New Credit Evaluation</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Select a data source and submit a credit scoring job.
        </p>
      </div>

      <PredictionForm />

      {/* Active Job Status */}
      {activeJobId && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center justify-between">
              <span>Active Job</span>
              <button
                type="button"
                onClick={clearActiveJobId}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Dismiss
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-muted-foreground truncate flex-1">
                {activeJobId}
              </span>
              {isLoading ? (
                <span className="text-xs text-muted-foreground">Connecting...</span>
              ) : job ? (
                <JobStatusBadge status={job.status} />
              ) : null}
            </div>

            {/* Progress bar */}
            {job && job.status !== "failed" && (
              <div className="space-y-1">
                <Progress value={progress} className="h-2" />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>
                    {job.status === "pending" && "Waiting in queue..."}
                    {job.status === "processing" && "Running inference..."}
                    {job.status === "completed" && "Complete"}
                  </span>
                  {job.status !== "completed" && (
                    <span>Elapsed: {elapsed}s</span>
                  )}
                </div>
              </div>
            )}

            {job?.error && (
              <p className="text-sm text-destructive">Error: {job.error}</p>
            )}

            {job?.status === "completed" && (
              <Link
                href="/farmers"
                className="inline-flex items-center rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                View Results
              </Link>
            )}
          </CardContent>
        </Card>
      )}

      {/* Job History */}
      {jobHistory && jobHistory.jobs.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-base font-semibold">Job History</h2>
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Job ID</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="hidden sm:table-cell">Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {jobHistory.jobs.map((j) => (
                  <TableRow key={j.job_id}>
                    <TableCell className="font-mono text-xs truncate max-w-50">
                      {j.job_id}
                    </TableCell>
                    <TableCell>
                      <JobStatusBadge status={j.status as JobStatus} />
                    </TableCell>
                    <TableCell className="hidden sm:table-cell text-xs text-muted-foreground">
                      {formatTimeAgo(j.created_at)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}
    </div>
  );
}
