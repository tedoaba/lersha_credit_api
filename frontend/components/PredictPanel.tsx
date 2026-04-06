"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { useJobStore } from "@/lib/stores";
import { useJobStatus, useJobs, useResultsPaginated } from "@/lib/queries";
import PredictionForm from "@/components/PredictionForm";
import JobStatusBadge from "@/components/JobStatusBadge";
import DecisionBadge from "@/components/DecisionBadge";
import FarmerDetailDrawer from "@/components/FarmerDetailDrawer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ChevronDown, ChevronUp } from "lucide-react";
import { groupByFarmer } from "@/lib/types";
import type { JobStatus, GroupedFarmer } from "@/lib/types";

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

function formatModelName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function PredictPanel() {
  const { activeJobId, clearActiveJobId } = useJobStore();
  const { data: job, isLoading } = useJobStatus(activeJobId);
  const { data: jobHistory } = useJobs(10);

  const [elapsed, setElapsed] = useState(0);
  const [startTime] = useState(() => Date.now());
  const [completedJobId, setCompletedJobId] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    if (job?.status === "completed" && activeJobId) {
      setCompletedJobId(activeJobId);
    }
  }, [job?.status, activeJobId]);

  useEffect(() => {
    if (!activeJobId || !job || job.status === "completed" || job.status === "failed") return;
    const interval = setInterval(() => setElapsed(Math.floor((Date.now() - startTime) / 1000)), 1000);
    return () => clearInterval(interval);
  }, [activeJobId, job, startTime]);

  const { data: jobResults, isLoading: loadingResults } = useResultsPaginated({
    page: 1,
    per_page: 100,
    job_id: completedJobId ?? undefined,
  });

  const grouped = useMemo(() => (jobResults ? groupByFarmer(jobResults.records) : []), [jobResults]);

  const modelColumns = useMemo(() => {
    const set = new Set<string>();
    for (const farmer of grouped) {
      for (const m of farmer.models) set.add(m.model_name);
    }
    return Array.from(set).sort();
  }, [grouped]);

  const [selectedFarmer, setSelectedFarmer] = useState<GroupedFarmer | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const openDrawer = useCallback((farmer: GroupedFarmer, model: string) => {
    setSelectedFarmer(farmer);
    setSelectedModel(model);
    setDrawerOpen(true);
  }, []);

  const progress = job ? PROGRESS_MAP[job.status] : 0;

  return (
    <div className="space-y-6">
      <PredictionForm />

      {/* Active Job Status */}
      {activeJobId && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center justify-between">
              <span>Active Job</span>
              <button
                type="button"
                onClick={() => {
                  clearActiveJobId();
                  if (job?.status !== "completed") setCompletedJobId(null);
                }}
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
          </CardContent>
        </Card>
      )}

      {/* Job Results — one row per farmer, model columns */}
      {completedJobId && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold">
              Prediction Results
              {grouped.length > 0 && (
                <span className="ml-2 text-sm font-normal text-muted-foreground">
                  ({grouped.length} farmer{grouped.length !== 1 ? "s" : ""} scored)
                </span>
              )}
            </h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setCompletedJobId(null)}
              className="text-xs"
            >
              Clear
            </Button>
          </div>

          {loadingResults && (
            <div className="text-sm text-muted-foreground animate-pulse">Loading results...</div>
          )}

          {grouped.length > 0 && (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-10">#</TableHead>
                    <TableHead>Farmer</TableHead>
                    <TableHead className="hidden sm:table-cell">Gender</TableHead>
                    {modelColumns.map((col) => (
                      <TableHead key={col}>{formatModelName(col)}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {grouped.map((farmer, i) => {
                    const name =
                      [farmer.first_name, farmer.middle_name, farmer.last_name]
                        .filter(Boolean)
                        .join(" ") || farmer.farmer_uid;
                    const modelMap = new Map(farmer.models.map((m) => [m.model_name, m]));

                    return (
                      <TableRow key={farmer.farmer_uid}>
                        <TableCell className="text-xs text-muted-foreground">{i + 1}</TableCell>
                        <TableCell className="font-medium">{name}</TableCell>
                        <TableCell className="hidden sm:table-cell text-sm text-muted-foreground">
                          {farmer.gender ?? "\u2014"}
                        </TableCell>
                        {modelColumns.map((col) => {
                          const model = modelMap.get(col);
                          return (
                            <TableCell key={col}>
                              {model ? (
                                <button
                                  type="button"
                                  title={`View ${formatModelName(col)} result`}
                                  onClick={() => openDrawer(farmer, col)}
                                  className="cursor-pointer hover:opacity-70 transition-opacity"
                                >
                                  <DecisionBadge decision={model.predicted_class_name} confidence={model.confidence_score} showIcon={false} />
                                </button>
                              ) : (
                                <span className="text-xs text-muted-foreground">{"\u2014"}</span>
                              )}
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}

          {jobResults && jobResults.total === 0 && (
            <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
              No results found for this job.
            </div>
          )}
        </div>
      )}

      {/* Job History — collapsible */}
      {jobHistory && jobHistory.jobs.length > 0 && (
        <div className="space-y-3">
          <button
            type="button"
            onClick={() => setShowHistory((v) => !v)}
            className="flex items-center gap-2 text-base font-semibold hover:text-primary transition-colors"
          >
            Job History
            {showHistory ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>

          {showHistory && (
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
          )}
        </div>
      )}

      <FarmerDetailDrawer
        farmer={selectedFarmer}
        modelName={selectedModel}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  );
}
